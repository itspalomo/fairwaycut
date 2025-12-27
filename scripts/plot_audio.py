#!/usr/bin/env python
"""Plot audio waveform and envelope from a video file."""

import sys
from pathlib import Path

import click

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio import extract_audio_from_video
from fairwaycut.detection import detect_impacts, detect_impacts_adaptive
from fairwaycut.visualization import (
    plot_waveform,
    plot_envelope,
    plot_analysis,
    save_figure,
)


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to save output plots",
)
@click.option(
    "--threshold",
    type=float,
    default=-20.0,
    help="Detection threshold in dB (default: -20)",
)
@click.option(
    "--min-gap",
    type=float,
    default=3.0,
    help="Minimum gap between detections in seconds (default: 3.0)",
)
@click.option(
    "--adaptive/--no-adaptive",
    default=True,
    help="Use adaptive thresholding (default: True)",
)
@click.option(
    "--waveform-only",
    is_flag=True,
    help="Only plot the waveform (no detection)",
)
@click.option(
    "--show",
    is_flag=True,
    help="Show plots interactively instead of saving",
)
def main(
    video_path: str,
    output_dir: str,
    threshold: float,
    min_gap: float,
    adaptive: bool,
    waveform_only: bool,
    show: bool,
):
    """
    Plot audio waveform and envelope from a video file.

    VIDEO_PATH: Path to the video file to analyze.

    Example usage:
        uv run python scripts/plot_audio.py samples/IMG_6644.MOV
        uv run python scripts/plot_audio.py samples/IMG_6644.MOV --waveform-only
        uv run python scripts/plot_audio.py samples/IMG_6644.MOV --threshold -15 --min-gap 2.5
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    click.echo(f"📹 Processing: {video_path.name}")
    click.echo("   Extracting audio...")

    try:
        audio = extract_audio_from_video(video_path)
    except Exception as e:
        click.echo(f"❌ Error extracting audio: {e}", err=True)
        sys.exit(1)

    click.echo(f"   Duration: {audio.duration:.2f}s | Sample rate: {audio.sample_rate} Hz")

    # Generate base filename
    base_name = video_path.stem

    if waveform_only:
        # Just plot waveform
        click.echo("   Generating waveform plot...")
        fig = plot_waveform(audio)
        
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            output_path = output_dir / f"{base_name}_waveform.png"
            save_figure(fig, output_path)
            click.echo(f"✅ Saved: {output_path}")
    else:
        # Run detection and create analysis plots
        click.echo("   Running impact detection...")
        
        if adaptive:
            result = detect_impacts_adaptive(audio, min_gap_sec=min_gap)
            click.echo(f"   Mode: Adaptive (threshold={result.parameters['threshold_db']:.1f} dB)")
        else:
            result = detect_impacts(audio, threshold_db=threshold, min_gap_sec=min_gap)
            click.echo(f"   Mode: Fixed threshold ({threshold} dB)")

        click.echo(f"   Detected {len(result.events)} potential impacts")

        # Print detected events
        if result.events:
            click.echo("\n   Detected events:")
            for i, event in enumerate(result.events, 1):
                click.echo(
                    f"     #{i}: {event.timestamp:.2f}s "
                    f"(confidence: {event.confidence:.2f}, amplitude: {event.amplitude_db:.1f} dB)"
                )

        # Generate plots
        click.echo("\n   Generating plots...")
        
        # Analysis plot (combined waveform + envelope)
        fig_analysis = plot_analysis(audio, result)
        
        # Envelope plot with peaks
        fig_envelope = plot_envelope(audio, result)
        
        # Waveform plot
        fig_waveform = plot_waveform(audio)

        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            # Save all plots
            analysis_path = output_dir / f"{base_name}_analysis.png"
            envelope_path = output_dir / f"{base_name}_envelope.png"
            waveform_path = output_dir / f"{base_name}_waveform.png"
            
            save_figure(fig_analysis, analysis_path)
            save_figure(fig_envelope, envelope_path)
            save_figure(fig_waveform, waveform_path)
            
            click.echo(f"✅ Saved: {analysis_path}")
            click.echo(f"✅ Saved: {envelope_path}")
            click.echo(f"✅ Saved: {waveform_path}")

    click.echo("\n🏌️ Done!")


if __name__ == "__main__":
    main()

