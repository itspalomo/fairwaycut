#!/usr/bin/env python
"""Detect golf ball impact events from a video file."""

import json
import sys
from pathlib import Path
from datetime import datetime

import click

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio import extract_audio_from_video
from fairwaycut.detection import (
    detect_impacts,
    detect_impacts_adaptive,
    detect_impacts_transient,
    detect_impacts_adaptive_snr,
)
from fairwaycut.visualization import (
    plot_analysis,
    plot_detection_summary,
    save_figure,
)


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to save output files",
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
    "--prominence",
    type=float,
    default=10.0,
    help="Minimum peak prominence in dB (default: 10.0)",
)
@click.option(
    "--adaptive/--no-adaptive",
    default=False,
    help="Use adaptive thresholding (default: False)",
)
@click.option(
    "--method",
    type=click.Choice(["snr", "transient", "adaptive", "basic"]),
    default="snr",
    help="Detection method: snr (adaptive SNR, best), transient, adaptive, basic",
)
@click.option(
    "--onset-threshold",
    type=float,
    default=3.0,
    help="Minimum onset strength for transient detection (default: 3.0)",
)
@click.option(
    "--snr-threshold",
    type=float,
    default=2.5,
    help="SNR threshold for adaptive detection (default: 2.5)",
)
@click.option(
    "--json-only",
    is_flag=True,
    help="Only output JSON report (no plots)",
)
@click.option(
    "--stdout",
    is_flag=True,
    help="Output JSON to stdout instead of file",
)
def main(
    video_path: str,
    output_dir: str,
    threshold: float,
    min_gap: float,
    prominence: float,
    adaptive: bool,
    method: str,
    onset_threshold: float,
    snr_threshold: float,
    json_only: bool,
    stdout: bool,
):
    """
    Detect golf ball impact events from a video file.

    VIDEO_PATH: Path to the video file to analyze.

    The tool analyzes the audio track to find transient impact sounds
    that indicate ball strikes. Results are saved as JSON and optionally
    as visualization plots.

    Example usage:
        uv run python scripts/detect_impacts.py samples/IMG_6644.MOV
        uv run python scripts/detect_impacts.py samples/IMG_6644.MOV --threshold -15 --min-gap 2.5
        uv run python scripts/detect_impacts.py samples/IMG_6644.MOV --json-only --stdout
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not stdout:
        click.echo(f"📹 Processing: {video_path.name}")
        click.echo("   Extracting audio...")

    try:
        audio = extract_audio_from_video(video_path)
    except Exception as e:
        if stdout:
            print(json.dumps({"error": str(e)}))
        else:
            click.echo(f"❌ Error extracting audio: {e}", err=True)
        sys.exit(1)

    if not stdout:
        click.echo(f"   Duration: {audio.duration:.2f}s | Sample rate: {audio.sample_rate} Hz")
        click.echo("   Running impact detection...")

    # Run detection
    if method == "snr":
        result = detect_impacts_adaptive_snr(
            audio,
            min_gap_sec=min_gap,
            snr_threshold=snr_threshold,
            # Use low onset threshold to catch hits in noisy sections
            # SNR filtering handles the noise discrimination
            min_onset=0.5,
            amplitude_threshold_db=threshold,
        )
        if not stdout:
            click.echo(f"   Method: Adaptive SNR (threshold={snr_threshold})")
    elif method == "transient":
        result = detect_impacts_transient(
            audio,
            min_gap_sec=min_gap,
            onset_threshold=onset_threshold,
            amplitude_threshold_db=threshold,
        )
        if not stdout:
            click.echo(f"   Method: Transient (onset threshold={onset_threshold})")
    elif method == "adaptive" or adaptive:
        result = detect_impacts_adaptive(
            audio,
            min_gap_sec=min_gap,
        )
        if not stdout:
            click.echo(f"   Method: Adaptive")
    else:
        result = detect_impacts(
            audio,
            threshold_db=threshold,
            min_gap_sec=min_gap,
            prominence_db=prominence,
        )
        if not stdout:
            click.echo(f"   Method: Basic (threshold={threshold} dB)")

    if not stdout:
        click.echo(f"   Detected {len(result.events)} potential impacts")

    # Prepare report
    report = {
        "source_file": str(video_path.absolute()),
        "analysis_timestamp": datetime.now().isoformat(),
        "audio_info": {
            "duration_seconds": audio.duration,
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
        },
        "detection": result.to_dict(),
    }

    if stdout:
        # Output JSON to stdout
        print(json.dumps(report, indent=2))
    else:
        # Save JSON report
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = video_path.stem
        
        json_path = output_dir / f"{base_name}_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"✅ Report saved: {json_path}")

        # Print detected events
        if result.events:
            click.echo("\n   Detected events:")
            for i, event in enumerate(result.events, 1):
                extras = ""
                if hasattr(event, 'onset_strength') and event.onset_strength > 0:
                    extras = f", onset: {event.onset_strength:.1f}, flux: {event.spectral_flux:.2f}"
                click.echo(
                    f"     #{i}: {event.timestamp:.2f}s "
                    f"(conf: {event.confidence:.2f}, amp: {event.amplitude_db:.1f} dB{extras})"
                )

        # Generate plots unless json_only
        if not json_only:
            click.echo("\n   Generating visualizations...")
            
            # Analysis plot
            fig_analysis = plot_analysis(audio, result)
            analysis_path = output_dir / f"{base_name}_analysis.png"
            save_figure(fig_analysis, analysis_path)
            click.echo(f"✅ Saved: {analysis_path}")
            
            # Detection summary
            fig_summary = plot_detection_summary(result)
            summary_path = output_dir / f"{base_name}_summary.png"
            save_figure(fig_summary, summary_path)
            click.echo(f"✅ Saved: {summary_path}")

        click.echo("\n🏌️ Detection complete!")


if __name__ == "__main__":
    main()

