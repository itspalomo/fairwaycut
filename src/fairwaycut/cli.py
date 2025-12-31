"""Command-line interface for FairwayCut."""

import json
import sys
from pathlib import Path
from typing import Optional



from fairwaycut import __version__
from fairwaycut.core.config import Config
from fairwaycut.core.models import SwingPhase
import rich_click as click

# Rich Click Configuration
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.STYLE_HELPTEXT = "dim"
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_SWITCH = "bold green"
click.rich_click.STYLE_METAVAR = "bold yellow"
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.SHOW_ARGUMENTS = True

# Define option groups for cleaner help
click.rich_click.OPTION_GROUPS = {
    "fairwaycut analyze": [
        {"name": "Input/Output", "options": ["--output", "--verbose"]},
        {"name": "Processing Options", "options": ["--mode", "--start", "--end", "--min-gap"]},
        {"name": "3D Export Options", "options": ["--export-3d-poses", "--plot-3d", "--interactive-3d", "--camera-3d"]},
    ],
    "fairwaycut extract": [
        {"name": "Input/Output", "options": ["--output-dir", "--verbose"]},
        {"name": "Processing Options", "options": ["--mode", "--start", "--end", "--pre-impact", "--post-impact"]},
        {"name": "Export Details", "options": ["--with-overlays", "--export-3d"]},
    ],
}


from fairwaycut.ui import console, print_banner, print_swing_summary, RichProgressHandler


@click.group()
@click.version_option(version=__version__)
def main():
    """FairwayCut - Golf swing detection and video generation."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output JSON report path",
    show_default="video directory/<name>_report.json",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["audio", "hybrid", "lite", "full"]),
    default="hybrid",
    show_default=True,
    help="Processing mode: audio (fastest), hybrid (audio + targeted pose), lite (full video, fast model), full (full video, accurate model)",
)
@click.option("--start", "-s", type=float, default=0.0, show_default=True, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, show_default="video end", help="End time in seconds")
@click.option("--min-gap", type=float, default=3.0, show_default=True, help="Minimum gap between swings (seconds)")
@click.option("--export-3d-poses/--no-3d-poses", default=False, show_default=True, help="Include normalized 3D poses in JSON report")
@click.option("--plot-3d/--no-plot-3d", default=False, show_default=True, help="Generate 3D swing analysis plot (static PNG)")
@click.option("--interactive-3d/--no-interactive-3d", default=False, show_default=True, help="Generate interactive 3D HTML viewer (rotatable)")
@click.option(
    "--camera-3d",
    type=click.Choice(["front", "dtl", "isometric"]),
    default="isometric",
    show_default=True,
    help="Camera angle for 3D plot",
)
@click.option("--verbose", "-v", is_flag=True, show_default=True, help="Verbose output")
def analyze(
    video_path: str,
    output: Optional[str],
    mode: str,
    start: float,
    end: Optional[float],
    min_gap: float,
    export_3d_poses: bool,
    plot_3d: bool,
    interactive_3d: bool,
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
      --plot-3d          Generate static multi-view 3D swing analysis image
      --interactive-3d   Generate interactive HTML viewer (rotatable/zoomable)
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
            from rich.progress import track
            for i, swing in enumerate(track(result.swings, description="Normalizing poses...")):
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
            
            from rich.progress import track
            for swing in track(result.swings, description="Generating 3D plots..."):
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
                        if verbose:
                            console.print(f"  ✓ Saved 3D plot: [bold user]{plot_path.name}[/bold user]")
            if not verbose:
                console.print(f"  ✓ Saved {len(result.swings)} 3D plots")
        
        # Generate interactive 3D HTML if requested
        if interactive_3d and result.pose_result and result.swings:
            console.print("🎮 Generating interactive 3D viewer...")
            from fairwaycut.pose.normalizer import PoseNormalizer
            from fairwaycut.visualization.interactive import generate_interactive_swing_viewer
            
            normalizer = PoseNormalizer()
            
            from rich.progress import track
            for swing in track(result.swings, description="Generating interactive viewers..."):
                swing_frames = [
                    f for f in result.pose_result.frames
                    if swing.start_time <= f.timestamp <= swing.end_time
                ]
                
                if swing_frames:
                    normalized = normalizer.normalize_sequence(swing_frames)
                    normalizer.reset()
                    
                    if normalized:
                        html_path = output_path.parent / f"{output_path.stem}_swing{swing.swing_id}_3d.html"
                        generate_interactive_swing_viewer(
                            normalized,
                            swing,
                            html_path,
                            title=f"Swing #{swing.swing_id}",
                        )
            
            console.print(f"  ✓ Saved interactive viewers")
            console.print(f"    Open in browser to rotate/zoom/pan")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    help="Output directory for clips",
    show_default="video directory/<name>_swings",
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["audio", "hybrid", "lite", "full"]),
    default="audio",
    show_default=True,
    help="Processing mode: audio (fastest), hybrid (audio + pose), lite, full",
)
@click.option("--start", "-s", type=float, default=0.0, show_default=True, help="Start time in seconds")
@click.option("--end", "-e", type=float, default=None, show_default="video end", help="End time in seconds")
@click.option("--pre-impact", type=float, default=3.0, show_default=True, help="Seconds before impact to include in clip")
@click.option("--post-impact", type=float, default=2.0, show_default=True, help="Seconds after impact to include in clip")
@click.option("--with-overlays/--no-overlays", default=False, show_default=True, help="Add visual overlays to clips")
@click.option("--export-3d/--no-export-3d", default=False, show_default=True, help="Export interactive 3D viewer for each swing")
@click.option("--verbose", "-v", is_flag=True, show_default=True, help="Verbose output")
def extract(
    video_path: str,
    output_dir: Optional[str],
    mode: str,
    start: float,
    end: Optional[float],
    pre_impact: float,
    post_impact: float,
    with_overlays: bool,
    export_3d: bool,
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
    
      fairwaycut extract video.mp4                        # Quick extraction
      fairwaycut extract video.mp4 --with-overlays -m hybrid  # With skeleton
      fairwaycut extract video.mp4 --export-3d            # With interactive 3D export
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
            # Use overlay generator for clips
            from fairwaycut.video.generator import generate_all_swing_clips, DemoVideoOptions
            
            audio = extract_audio_from_video(video_path)
            

            
            options = DemoVideoOptions(
                show_skeleton=True,
                show_waveform=True,
                show_phase_label=True,
            )
            
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
            
            # Create a custom progress display for generation
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                # Main task for swings
                swing_task = progress.add_task(f"Generating clips (0/{len(result.swings)})", total=len(result.swings))
                # Sub-task for frames (initially hidden or just waiting)
                frame_task = progress.add_task("Rendering frames", total=100, visible=False)
                
                def generation_progress(status: str, current: int, total: int):
                    if status.endswith("_frames"):
                        # Update frame progress
                        if not progress.tasks[frame_task].visible:
                            progress.update(frame_task, visible=True)
                        progress.update(frame_task, completed=current, total=total)
                    elif status.startswith("swing_"):
                        # Start of a new swing
                        progress.update(swing_task, completed=current, description=f"Generating clips ({current+1}/{total})")
                        # Reset frame task for new swing
                        progress.update(frame_task, completed=0, total=100, visible=True)
                
                clips = generate_all_swing_clips(
                    video_path,
                    output_path,
                    result,
                    audio,
                    options=options,
                    progress_callback=generation_progress,
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
        
        # Step 3: Export 3D viewer if requested
        if export_3d and result.pose_result and result.pose_result.frames:
            console.print("🎮 Generating 3D viewers...")
            from fairwaycut.pose.normalizer import PoseNormalizer
            from fairwaycut.visualization.interactive import generate_interactive_swing_viewer
            
            normalizer = PoseNormalizer()
            
            from rich.progress import track
            for i, swing in enumerate(track(result.swings, description="Generating 3D viewers...")):
                swing_frames = [
                    f for f in result.pose_result.frames
                    if swing.start_time <= f.timestamp <= swing.end_time
                ]
                
                if swing_frames:
                    normalized = normalizer.normalize_sequence(swing_frames)
                    normalizer.reset()
                    
                    if normalized:
                        html_path = output_path / f"swing_{swing.swing_id:03d}_3d.html"
                        generate_interactive_swing_viewer(
                            normalized,
                            swing,
                            html_path,
                            title=f"Swing #{swing.swing_id}",
                        )
            
            console.print(f"✅ Generated 3D viewers in [bold user]{output_path}[/bold user]")
        
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
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output image path",
    show_default="video directory/<name>_analysis.png",
)
@click.option("--verbose", "-v", is_flag=True, show_default=True, help="Verbose output")
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
