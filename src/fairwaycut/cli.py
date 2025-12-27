"""Command-line interface for FairwayCut."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from fairwaycut import __version__
from fairwaycut.core.config import Config
from fairwaycut.core.models import SwingPhase


def print_banner():
    """Print FairwayCut banner."""
    click.secho("""
╔═══════════════════════════════════════════╗
║  FairwayCut - Golf Swing Auto-Segmentation ║
║  Version {}                            ║
╚═══════════════════════════════════════════╝
""".format(__version__), fg="green")


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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(
    video_path: str,
    output: Optional[str],
    mode: str,
    start: float,
    end: Optional[float],
    min_gap: float,
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
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner()
    
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
    click.echo(mode_descriptions[mode])
    
    # Display time range
    if start > 0 or end is not None:
        start_str = f"{int(start // 60)}:{int(start % 60):02d}"
        end_str = f"{int(end // 60)}:{int(end % 60):02d}" if end else "end"
        click.echo(f"⏱️  Time range: {start_str} → {end_str}")
    
    # Load config
    config = Config.default()
    config.audio.min_gap_sec = min_gap
    
    # Import here to avoid slow startup
    from fairwaycut.fusion.detector import detect_swings
    
    # Progress display
    current_stage = ""
    
    def progress_callback(stage: str, current: int, total: int):
        nonlocal current_stage
        if stage != current_stage:
            current_stage = stage
            stage_names = {
                "audio_extraction": "🎵 Extracting audio...",
                "audio_detection": "🔍 Detecting impacts...",
                "pose_estimation": "🦴 Estimating poses...",
                "fusion": "🔗 Fusing signals...",
                "complete": "✅ Analysis complete!",
            }
            click.echo(stage_names.get(stage, stage))
        
        if verbose and total > 0:
            pct = (current / total) * 100
            click.echo(f"   Progress: {pct:.0f}% ({current}/{total})", nl=False)
            click.echo("\r", nl=False)
    
    try:
        # Run detection
        result = detect_swings(
            video_path,
            mode=processing_mode,
            config=config,
            progress_callback=progress_callback if verbose else None,
            start_time=start,
            end_time=end,
        )
        
        click.echo()
        click.secho(f"🎯 Found {len(result.swings)} swings", fg="green", bold=True)
        
        # Print swing summary
        if result.swings:
            click.echo("\n📊 Swing Summary:")
            click.echo("-" * 60)
            for swing in result.swings:
                time_str = f"{swing.impact_time:.1f}s"
                conf_str = f"{swing.combined_confidence:.0%}"
                click.echo(f"  Swing #{swing.swing_id}: impact at {time_str} (confidence: {conf_str})")
        
        # Save report
        if output:
            output_path = Path(output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_report.json"
        
        report = result.to_dict()
        report["source_file"] = str(video_path)
        report["version"] = __version__
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        click.echo(f"\n📝 Report saved: {output_path}")
        
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def demo(
    video_path: str,
    output: Optional[str],
    skeleton: bool,
    waveform: bool,
    phase_label: bool,
    mode: str,
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
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner()
    
    video_path = Path(video_path)
    click.echo(f"📹 Processing: {video_path.name}")
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_demo.mp4"
    
    click.echo(f"🎬 Output: {output_path}")
    
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
        click.echo("🔊 Audio-only mode (no skeleton overlay)")
    
    # Show options
    overlays = []
    if skeleton:
        overlays.append("skeleton")
    if waveform:
        overlays.append("waveform")
    if phase_label and mode != "audio":
        overlays.append("phase labels")
    click.echo(f"🎨 Overlays: {', '.join(overlays) if overlays else 'waveform only'}")
    
    # Import modules
    from fairwaycut.audio.extraction import extract_audio_from_video
    from fairwaycut.fusion.detector import detect_swings
    from fairwaycut.video.generator import DemoVideoGenerator, DemoVideoOptions
    from fairwaycut.core.config import Config
    
    config = Config.default()
    
    try:
        # Step 1: Extract audio
        click.echo("\n🎵 Extracting audio...")
        audio = extract_audio_from_video(video_path)
        
        # Step 2: Detect swings
        mode_desc = {
            "audio": "audio only",
            "segments": "pose around impacts",
            "lite": "full video (lite model)",
            "full": "full video (full model)",
        }
        click.echo(f"🔍 Detecting swings ({mode_desc[mode]})...")
        
        def detection_progress(stage: str, current: int, total: int):
            if verbose and stage == "pose_estimation" and total > 0:
                pct = (current / total) * 100
                click.echo(f"   Pose estimation: {pct:.0f}%", nl=False)
                click.echo("\r", nl=False)
        
        result = detect_swings(
            video_path,
            mode=processing_mode,
            config=config,
            progress_callback=detection_progress if verbose else None,
        )
        
        click.echo(f"\n🎯 Found {len(result.swings)} swings")
        
        # Step 3: Generate demo video
        click.echo("🎬 Generating demo video...")
        
        options = DemoVideoOptions(
            show_skeleton=skeleton,
            show_waveform=waveform,
            show_phase_label=phase_label,
            show_timestamp=True,
            show_impact_marker=True,
        )
        
        generator = DemoVideoGenerator(options=options)
        
        def video_progress(current: int, total: int):
            if total > 0:
                pct = (current / total) * 100
                click.echo(f"   Rendering: {pct:.0f}% ({current}/{total} frames)", nl=False)
                click.echo("\r", nl=False)
        
        generator.generate(
            video_path,
            output_path,
            result,
            audio,
            progress_callback=video_progress,
        )
        
        click.echo()
        click.secho(f"✅ Demo video saved: {output_path}", fg="green", bold=True)
        
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
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
    Examples:
      fairwaycut extract video.mp4                        # Quick extraction
      fairwaycut extract video.mp4 --with-overlays -m hybrid  # With skeleton
      fairwaycut extract video.mp4 -s 60 -e 180           # Only 1:00-3:00
    """
    from fairwaycut.core.config import ProcessingMode
    
    print_banner()
    
    video_path = Path(video_path)
    click.echo(f"📹 Processing: {video_path.name}")
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = video_path.parent / f"{video_path.stem}_swings"
    
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"📁 Output directory: {output_path}")
    
    # Display time range
    if start > 0 or end is not None:
        start_str = f"{int(start // 60)}:{int(start % 60):02d}"
        end_str = f"{int(end // 60)}:{int(end % 60):02d}" if end else "end"
        click.echo(f"⏱️  Time range: {start_str} → {end_str}")
    
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
    
    try:
        # Step 1: Detect swings
        mode_desc = {"audio": "audio", "hybrid": "hybrid", "lite": "lite", "full": "full"}
        click.echo(f"🔍 Detecting swings (mode: {mode_desc[mode]})...")
        
        result = detect_swings(
            video_path,
            mode=processing_mode,
            config=config,
            start_time=start,
            end_time=end,
        )
        
        click.echo(f"🎯 Found {len(result.swings)} swings")
        
        if not result.swings:
            click.echo("No swings detected. Try adjusting parameters.")
            return
        
        # Step 2: Extract clips
        click.echo("✂️ Extracting clips...")
        
        if with_overlays:
            # Use demo generator for overlay clips
            from fairwaycut.video.generator import generate_all_swing_clips, DemoVideoOptions
            
            audio = extract_audio_from_video(video_path)
            options = DemoVideoOptions(
                show_skeleton=True,
                show_waveform=True,
                show_phase_label=True,
            )
            
            clips = generate_all_swing_clips(
                video_path,
                output_path,
                result,
                audio,
                options=options,
            )
            
            click.echo(f"\n✅ Extracted {len(clips)} clips with overlays")
        else:
            # Simple extraction without overlays
            with VideoFileClip(str(video_path)) as video:
                for swing in result.swings:
                    clip_path = output_path / f"swing_{swing.swing_id:03d}.mp4"
                    
                    # Extract subclip
                    subclip = video.subclipped(swing.start_time, swing.end_time)
                    subclip.write_videofile(
                        str(clip_path),
                        codec="libx264",
                        audio_codec="aac",
                        logger=None if not verbose else "bar",
                    )
                    
                    click.echo(f"  ✓ Swing #{swing.swing_id}: {clip_path.name}")
            
            click.echo(f"\n✅ Extracted {len(result.swings)} clips")
        
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
        
        click.echo(f"📝 Manifest saved: {manifest_path}")
        
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
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
    print_banner()
    
    video_path = Path(video_path)
    click.echo(f"📹 Analyzing: {video_path.name}")
    
    # Import modules
    from fairwaycut.audio.extraction import extract_audio_from_video
    from fairwaycut.audio.detection import detect_impacts_adaptive_snr
    from fairwaycut.visualization.plots import plot_analysis, save_figure
    
    try:
        # Extract audio
        click.echo("🎵 Extracting audio...")
        audio = extract_audio_from_video(video_path)
        
        # Detect impacts
        click.echo("🔍 Detecting impacts...")
        result = detect_impacts_adaptive_snr(audio)
        
        click.echo(f"🎯 Found {len(result.events)} impacts")
        
        # Generate plot
        click.echo("📊 Generating plot...")
        fig = plot_analysis(audio, result)
        
        # Save
        if output:
            output_path = Path(output)
        else:
            output_path = video_path.parent / f"{video_path.stem}_analysis.png"
        
        save_figure(fig, output_path)
        
        click.secho(f"✅ Plot saved: {output_path}", fg="green", bold=True)
        
    except Exception as e:
        click.secho(f"❌ Error: {e}", fg="red", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
def info():
    """Display version and system information."""
    print_banner()
    
    click.echo("System Information:")
    click.echo(f"  Python: {sys.version}")
    click.echo(f"  FairwayCut: {__version__}")
    
    # Check dependencies
    click.echo("\nDependencies:")
    
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
            click.secho(f"  ✓ {name}: {version}", fg="green")
        except ImportError:
            click.secho(f"  ✗ {name}: not installed", fg="red")


if __name__ == "__main__":
    main()

