"""Command-line interface for FairwayCut."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from fairwaycut import __version__
from fairwaycut.core.config import Config
from fairwaycut.core.models import SwingPhase


from fairwaycut.ui import console, print_banner, print_swing_summary, RichProgressHandler


@click.group()
@click.version_option(version=__version__)
def main():
    """FairwayCut - Golf swing detection and video generation."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON report path")
@click.option(
    "--mode", "-m",
    type=click.Choice(["audio", "hybrid", "lite", "full"]),
    default="hybrid",
    help="Processing mode: audio (fastest), hybrid (audio + targeted pose), lite (full video, fast model), full (full video, accurate model)"
)
@click.option("--start", "-s", type=float, default=0.0, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, help="End time in seconds (default: video end)")
@click.option("--min-gap", type=float, default=3.0, help="Minimum gap between swings (seconds)")
@click.option("--export-3d-poses/--no-3d-poses", default=False, help="Include normalized 3D poses in JSON report")
@click.option("--plot-3d/--no-plot-3d", default=False, help="Generate 3D swing analysis plot")
@click.option(
    "--camera-3d",
    type=click.Choice(["front", "dtl", "isometric"]),
    default="isometric",
    help="Camera angle for 3D plot"
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(
    video_path: str,
    output: Optional[str],
    mode: str,
    start: float,
    end: Optional[float],
    min_gap: float,
    export_3d_poses: bool,
    plot_3d: bool,
    camera_3d: str,
    verbose: bool,
):
    """
    Analyze a video for golf swings.
    
    Processing modes (--mode):
    
    \b
      audio  - Audio detection only (fastest, ~10 seconds)
      hybrid - Audio + pose around impacts (recommended, fast + accurate)
      lite   - Full video with lite pose model (slower, ~5-10 min)
      full   - Full video with full pose model (slowest, ~15-30 min)
    
    The 'hybrid' mode is recommended for most use cases - it's fast and
    catches most swings. Use 'lite' or 'full' if audio might miss swings.
    
    Time range examples:
    
    \b
      --start 60 --end 180    # Analyze 1:00 to 3:00
      --start 0 --end 120     # First 2 minutes
      --start 300             # From 5:00 to end
    
    3D Export options:
    
    \b
      --export-3d-poses  Include normalized 3D world coordinates in JSON
      --plot-3d          Generate multi-view 3D swing analysis image
      --camera-3d        Camera angle for 3D plot (front/dtl/isometric)
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner(__version__)
    
    video_path = Path(video_path)
    click.echo(f"📹 Analyzing: {video_path.name}")
    
    # Map CLI mode to ProcessingMode
    mode_map = {
        "audio": ProcessingMode.AUDIO,
        "hybrid": ProcessingMode.HYBRID,
        "lite": ProcessingMode.LITE,
        "full": ProcessingMode.FULL,
    }
    processing_mode = mode_map[mode]
    
    mode_descriptions = {
        "audio": "🔊 Audio-only mode (fastest)",
        "hybrid": "🎯 Hybrid mode - audio + targeted pose (recommended)",
        "lite": "🦴 Full video with lite model (slower)",
        "full": "🦴 Full video with full model (slowest, most accurate)",
    }
    console.print(mode_descriptions[mode])
    
    # Display time range
    if start > 0 or end is not None:
        start_str = f"{int(start // 60)}:{int(start % 60):02d}"
        end_str = f"{int(end // 60)}:{int(end % 60):02d}" if end else "end"
        console.print(f"⏱️  Time range: [cyan]{start_str}[/cyan] → [cyan]{end_str}[/cyan]")
    
    # Load config
    config = Config.default()
    config.audio.min_gap_sec = min_gap
    
    # Import here to avoid slow startup
    from fairwaycut.fusion.detector import detect_swings
    
    # Progress display using Rich
    handler = RichProgressHandler(console, verbose=verbose)
    
    try:
        # Run detection
        # Run detection with live progress
        with handler.live() as h:
            result = detect_swings(
                video_path,
                mode=processing_mode,
                config=config,
                progress_callback=h.callback,
                start_time=start,
                end_time=end,
            )
        
        console.print()
        console.print(f"🎯 Found [bold green]{len(result.swings)}[/bold green] swings")
        
        # Print swing summary
        if result.swings:
            print_swing_summary(result.swings)
        
        # Save report
        if output:
            output_path = Path(output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_report.json"
        
        report = result.to_dict()
        report["source_file"] = str(video_path)
        report["version"] = __version__
        
        # Add 3D poses to report if requested
        if export_3d_poses and result.pose_result and result.pose_result.frames:
            console.print("🎮 Normalizing 3D poses...")
            from fairwaycut.pose.normalizer import PoseNormalizer, normalize_poses_for_export
            
            # Export normalized poses for each swing
            for i, swing in enumerate(result.swings):
                # Get frames for this swing
                swing_frames = [
                    f for f in result.pose_result.frames
                    if swing.start_time <= f.timestamp <= swing.end_time
                ]
                
                if swing_frames:
                    poses_3d = normalize_poses_for_export(swing_frames)
                    report["swings"][i]["poses_3d"] = poses_3d
            
            console.print(f"  ✓ Added 3D poses for {len(result.swings)} swings")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        console.print(f"\n📝 Report saved: [bold user]{output_path}[/bold user]")
        
        # Generate 3D plot if requested
        if plot_3d and result.pose_result and result.swings:
            console.print("📊 Generating 3D analysis plot...")
            from fairwaycut.pose.normalizer import PoseNormalizer
            from fairwaycut.visualization.plot3d import plot_swing_3d, save_swing_3d_plot
            
            normalizer = PoseNormalizer()
            
            for swing in result.swings:
                # Get frames for this swing
                swing_frames = [
                    f for f in result.pose_result.frames
                    if swing.start_time <= f.timestamp <= swing.end_time
                ]
                
                if swing_frames:
                    normalized = normalizer.normalize_sequence(swing_frames)
                    normalizer.reset()
                    
                    if normalized:
                        plot_path = output_path.parent / f"{output_path.stem}_swing{swing.swing_id}_3d.png"
                        save_swing_3d_plot(
                            normalized,
                            swing,
                            plot_path,
                            camera_views=["front", "dtl", "isometric"],
                        )
                        console.print(f"  ✓ Saved 3D plot: [bold user]{plot_path.name}[/bold user]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output video path")
@click.option("--skeleton/--no-skeleton", default=True, help="Show pose skeleton")
@click.option("--waveform/--no-waveform", default=True, help="Show audio waveform")
@click.option("--phase-label/--no-phase-label", default=True, help="Show swing phase labels")
@click.option(
    "--mode", "-m",
    type=click.Choice(["audio", "segments", "lite", "full"]),
    default="segments",
    help="Processing mode for detection"
)
@click.option("--view-3d/--no-view-3d", default=False, help="Enable 3D pose viewer")
@click.option(
    "--layout-3d",
    type=click.Choice(["inset", "side-by-side"]),
    default="inset",
    help="3D view layout: inset (picture-in-picture) or side-by-side"
)
@click.option(
    "--camera-3d",
    type=click.Choice(["front", "dtl", "isometric"]),
    default="isometric",
    help="3D camera angle: front, dtl (down-the-line), or isometric"
)
@click.option("--show-voxel/--no-voxel", default=False, help="Show voxel motion volume in 3D view")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def demo(
    video_path: str,
    output: Optional[str],
    skeleton: bool,
    waveform: bool,
    phase_label: bool,
    mode: str,
    view_3d: bool,
    layout_3d: str,
    camera_3d: str,
    show_voxel: bool,
    verbose: bool,
):
    """
    Generate a demo video with overlays.
    
    Creates a new video with pose skeleton, audio waveform,
    and swing phase labels overlaid on the original footage.
    
    Use --mode to control speed/accuracy tradeoff:
    
    \b
      audio    - Audio overlay only (fastest, no skeleton)
      segments - Pose around impacts (recommended)
      lite     - Full pose with lite model
      full     - Full pose with full model (best quality)
    
    3D Visualization (--view-3d):
    
    \b
      --view-3d              Enable 3D pose viewer
      --layout-3d inset      Picture-in-picture (default)
      --layout-3d side-by-side  Video + 3D view side by side
      --camera-3d front      Face-on view
      --camera-3d dtl        Down-the-line view
      --camera-3d isometric  3/4 view (default)
      --show-voxel           Show voxel motion trail
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner(__version__)
    
    video_path = Path(video_path)
    console.print(f"📹 Processing: [bold]{video_path.name}[/bold]")
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_demo.mp4"
    
    console.print(f"🎬 Output: [bold user]{output_path}[/bold user]")
    
    # Map mode
    mode_map = {
        "audio": ProcessingMode.AUDIO_ONLY,
        "segments": ProcessingMode.POSE_SEGMENTS,
        "lite": ProcessingMode.POSE_LITE,
        "full": ProcessingMode.POSE_FULL,
    }
    processing_mode = mode_map[mode]
    
    # Disable skeleton for audio-only mode
    if mode == "audio":
        skeleton = False
        console.print("🔊 Audio-only mode (no skeleton overlay)")
    
    # Show options
    overlays = []
    if skeleton:
        overlays.append("skeleton")
    if waveform:
        overlays.append("waveform")
    if phase_label and mode != "audio":
        overlays.append("phase labels")
    console.print(f"🎨 Overlays: {', '.join(overlays) if overlays else 'waveform only'}")
    
    # Import modules
    from fairwaycut.audio.extraction import extract_audio_from_video
    from fairwaycut.fusion.detector import detect_swings
    from fairwaycut.video.generator import DemoVideoGenerator, DemoVideoOptions, View3DOptions
    from fairwaycut.core.config import Config
    
    config = Config.default()
    
    # Setup 3D options if enabled
    view_3d_options = None
    if view_3d:
        view_3d_options = View3DOptions(
            enabled=True,
            layout=layout_3d,
            camera_view=camera_3d,
            show_voxel=show_voxel,
        )
        console.print(f"🎮 3D View: {layout_3d} layout, {camera_3d} camera" + 
                      (" + voxel" if show_voxel else ""))
    
    # Progress display using Rich
    handler = RichProgressHandler(console, verbose=verbose)

    try:
        # Run detection with live progress
        with handler.live() as h:
            h.console.print("\n🎵 Extracting audio...")
            audio = extract_audio_from_video(video_path)
            
            mode_desc = {
                "audio": "audio only",
                "segments": "pose around impacts",
                "lite": "full video (lite model)",
                "full": "full video (full model)",
            }
            h.console.print(f"🔍 Detecting swings ({mode_desc[mode]})...")

            result = detect_swings(
                video_path,
                mode=processing_mode,
                config=config,
                progress_callback=h.callback,
            )
        
        console.print(f"\n🎯 Found [bold green]{len(result.swings)}[/bold green] swings")
        
        # Step 3: Generate demo video
        console.print("🎬 Generating demo video...")
        
        options = DemoVideoOptions(
            show_skeleton=skeleton,
            show_waveform=waveform,
            show_phase_label=phase_label,
            show_timestamp=True,
            show_impact_marker=True,
            view_3d=view_3d_options,
        )
        
        generator = DemoVideoGenerator(options=options)
        
        # Simple progress for video generation since it's not hooked into the handler
        from rich.progress import track
        
        # We need a custom callback wrapper for the generator because it expects (current, total)
        # We'll use a manual progress bar here or just wrap the generator if possible.
        # But DemoVideoGenerator.generate doesn't yield, it takes a callback.
        
        # Let's use a new Progress instance for this separate long-running task
        from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as video_progress:
            task = video_progress.add_task("Rendering...", total=100) # Total is unknown initially or 100%?
            # Generator callback provides current frame index, but we might not know total frames easily ahead of time?
            # Actually generator.generate usually knows.
            
            def video_progress_callback(current: int, total: int):
                video_progress.update(task, completed=current, total=total)
            
            generator.generate(
                video_path,
                output_path,
                result,
                audio,
                progress_callback=video_progress_callback,
            )
        
        console.print()
        console.print(f"✅ Demo video saved: [bold user]{output_path}[/bold user]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for clips")
@click.option(
    "--mode", "-m",
    type=click.Choice(["audio", "hybrid", "lite", "full"]),
    default="audio",
    help="Processing mode: audio (fastest), hybrid (audio + pose), lite, full"
)
@click.option("--start", "-s", type=float, default=0.0, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, help="End time in seconds (default: video end)")
@click.option("--pre-impact", type=float, default=3.0, help="Seconds before impact to include in clip")
@click.option("--post-impact", type=float, default=2.0, help="Seconds after impact to include in clip")
@click.option("--with-overlays/--no-overlays", default=False, help="Add visual overlays to clips")
@click.option("--view-3d/--no-view-3d", default=False, help="Enable 3D pose viewer (requires --with-overlays)")
@click.option(
    "--layout-3d",
    type=click.Choice(["inset", "side-by-side"]),
    default="inset",
    help="3D view layout (requires --with-overlays --view-3d)"
)
@click.option(
    "--camera-3d",
    type=click.Choice(["front", "dtl", "isometric"]),
    default="isometric",
    help="3D camera angle (requires --with-overlays --view-3d)"
)
@click.option("--show-voxel/--no-voxel", default=False, help="Show voxel motion volume (requires --with-overlays --view-3d)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def extract(
    video_path: str,
    output_dir: Optional[str],
    mode: str,
    start: float,
    end: Optional[float],
    pre_impact: float,
    post_impact: float,
    with_overlays: bool,
    view_3d: bool,
    layout_3d: str,
    camera_3d: str,
    show_voxel: bool,
    verbose: bool,
):
    """
    Extract individual swing clips from a video.
    
    Detects swings and saves each as a separate video file.
    
    \b
    Modes:
      audio  - Audio detection only (fastest, no skeleton overlay)
      hybrid - Audio + pose around impacts (recommended for overlays)
      lite   - Full video pose with lite model
      full   - Full video pose with full model
    
    \b
    3D Options (when using --with-overlays):
      --view-3d              Enable 3D pose viewer in clips
      --layout-3d inset      Picture-in-picture (default)
      --layout-3d side-by-side  Video + 3D view side by side
      --camera-3d front/dtl/isometric  Camera angle
      --show-voxel           Show voxel motion trail
    
    \b
    Examples:
      fairwaycut extract video.mp4                        # Quick extraction
      fairwaycut extract video.mp4 --with-overlays -m hybrid  # With skeleton
      fairwaycut extract video.mp4 --with-overlays --view-3d  # With 3D view
      fairwaycut extract video.mp4 -s 60 -e 180           # Only 1:00-3:00
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner(__version__)
    
    video_path = Path(video_path)
    console.print(f"📹 Processing: [bold]{video_path.name}[/bold]")
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = video_path.parent / f"{video_path.stem}_swings"
    
    output_path.mkdir(parents=True, exist_ok=True)
    console.print(f"📁 Output directory: [bold user]{output_path}[/bold user]")
    
    # Display time range
    if start > 0 or end is not None:
        start_str = f"{int(start // 60)}:{int(start % 60):02d}"
        end_str = f"{int(end // 60)}:{int(end % 60):02d}" if end else "end"
        console.print(f"⏱️  Time range: [cyan]{start_str}[/cyan] → [cyan]{end_str}[/cyan]")
    
    # Map mode
    mode_map = {
        "audio": ProcessingMode.AUDIO,
        "hybrid": ProcessingMode.HYBRID,
        "lite": ProcessingMode.LITE,
        "full": ProcessingMode.FULL,
    }
    processing_mode = mode_map[mode]
    
    # Import modules
    from fairwaycut.audio.extraction import extract_audio_from_video
    from fairwaycut.fusion.detector import detect_swings
    from fairwaycut.core.config import Config
    from moviepy import VideoFileClip
    
    config = Config.default()
    config.fusion.pre_impact_sec = pre_impact
    config.fusion.post_impact_sec = post_impact
    
    # Progress display using Rich
    handler = RichProgressHandler(console, verbose=verbose)

    try:
        # Step 1: Detect swings
        mode_desc = {"audio": "audio", "hybrid": "hybrid", "lite": "lite", "full": "full"}
        
        with handler.live() as h:
             h.console.print(f"🔍 Detecting swings (mode: {mode_desc[mode]})...")
             
             result = detect_swings(
                video_path,
                mode=processing_mode,
                config=config,
                start_time=start,
                end_time=end,
                progress_callback=h.callback,
            )
        
        console.print(f"🎯 Found [bold green]{len(result.swings)}[/bold green] swings")
        
        if not result.swings:
            console.print("No swings detected. Try adjusting parameters.", style="yellow")
            return
        
        # Step 2: Extract clips
        console.print("✂️ Extracting clips...")
        
        if with_overlays:
            # Use demo generator for overlay clips
            from fairwaycut.video.generator import generate_all_swing_clips, DemoVideoOptions, View3DOptions
            
            audio = extract_audio_from_video(video_path)
            
            # Setup 3D options if enabled
            view_3d_options = None
            if view_3d:
                view_3d_options = View3DOptions(
                    enabled=True,
                    layout=layout_3d,
                    camera_view=camera_3d,
                    show_voxel=show_voxel,
                )
                console.print(f"🎮 3D View: {layout_3d} layout, {camera_3d} camera" + 
                              (" + voxel" if show_voxel else ""))
            
            options = DemoVideoOptions(
                show_skeleton=True,
                show_waveform=True,
                show_phase_label=True,
                view_3d=view_3d_options,
            )
            
            clips = generate_all_swing_clips(
                video_path,
                output_path,
                result,
                audio,
                options=options,
            )
            
            
            console.print(f"\n✅ Extracted [bold green]{len(clips)}[/bold green] clips with overlays")
        else:
            # Simple extraction without overlays
            with VideoFileClip(str(video_path)) as video:
                # Use track for simple progress
                from rich.progress import track
                
                for swing in track(result.swings, description="Saving clips..."):
                    clip_path = output_path / f"swing_{swing.swing_id:03d}.mp4"
                    
                    # Extract subclip
                    subclip = video.subclipped(swing.start_time, swing.end_time)
                    # Suppress moviepy output unless verbose, but we are inside a progress bar so suppress it anyway or it breaks UI
                    subclip.write_videofile(
                        str(clip_path),
                        codec="libx264",
                        audio_codec="aac",
                        logger=None, # Suppress moviepy bar
                    )
                    
                    if verbose:
                        console.print(f"  ✓ Swing #{swing.swing_id}: {clip_path.name}")
            
            console.print(f"\n✅ Extracted [bold green]{len(result.swings)}[/bold green] clips")
        
        # Save manifest
        manifest_path = output_path / "manifest.json"
        manifest = {
            "source_file": str(video_path),
            "num_swings": len(result.swings),
            "swings": [s.to_dict() for s in result.swings],
            "version": __version__,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        console.print(f"📝 Manifest saved: [bold user]{manifest_path}[/bold user]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output image path")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def plot(
    video_path: str,
    output: Optional[str],
    verbose: bool,
):
    """
    Generate analysis plots for a video.
    
    Creates a visualization showing the audio waveform, envelope,
    spectral flux, and detected impact points.
    """
    print_banner(__version__)
    
    video_path = Path(video_path)
    console.print(f"📹 Analyzing: [bold]{video_path.name}[/bold]")
    
    # Import modules
    from fairwaycut.audio.extraction import extract_audio_from_video
    from fairwaycut.audio.detection import detect_impacts_adaptive_snr
    from fairwaycut.visualization.plots import plot_analysis, save_figure
    
    try:
        # Extract audio
        console.print("🎵 Extracting audio...")
        audio = extract_audio_from_video(video_path)
        
        # Detect impacts
        console.print("🔍 Detecting impacts...")
        result = detect_impacts_adaptive_snr(audio)
        
        console.print(f"🎯 Found [bold green]{len(result.events)}[/bold green] impacts")
        
        # Generate plot
        console.print("📊 Generating plot...")
        fig = plot_analysis(audio, result)
        
        # Save
        if output:
            output_path = Path(output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_analysis.png"
        
        save_figure(fig, output_path)
        
        console.print(f"✅ Plot saved: [bold user]{output_path}[/bold user]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
def info():
    """Display version and system information."""
    print_banner(__version__)
    
    console.print("[bold]System Information:[/bold]")
    console.print(f"  Python: {sys.version.split()[0]}")
    console.print(f"  FairwayCut: {__version__}")
    
    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    
    deps = [
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("moviepy", "moviepy"),
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy"),
        ("click", "click"),
    ]
    
    for name, module in deps:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "unknown")
            console.print(f"  ✓ {name}: {version}", style="green")
        except ImportError:
            console.print(f"  ✗ {name}: not installed", style="red")


if __name__ == "__main__":
    main()

