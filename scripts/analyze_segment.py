#!/usr/bin/env python
"""Analyze a specific time segment of audio in detail."""

import sys
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio import extract_audio_from_video, AudioData
from fairwaycut.detection import detect_impacts_adaptive, detect_impacts_transient, DetectionResult
from fairwaycut.visualization import save_figure, apply_style, WAVEFORM_COLOR, ENVELOPE_COLOR, PEAK_COLOR, THRESHOLD_COLOR


def plot_segment_analysis(
    audio: AudioData,
    result: DetectionResult,
    start_sec: float,
    end_sec: float,
    title: str = None,
) -> plt.Figure:
    """Create detailed analysis plot for a time segment."""
    apply_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Calculate sample indices for the segment
    start_sample = int(start_sec * audio.sample_rate)
    end_sample = int(end_sec * audio.sample_rate)
    
    # Get waveform segment
    samples_segment = audio.samples[start_sample:end_sample]
    times_waveform = np.linspace(start_sec, end_sec, len(samples_segment))
    
    # Get envelope segment
    mask = (result.envelope_times >= start_sec) & (result.envelope_times <= end_sec)
    envelope_times_seg = result.envelope_times[mask]
    envelope_db_seg = result.envelope_db[mask]
    envelope_seg = result.envelope[mask]
    
    # Filter events in this segment
    events_in_segment = [e for e in result.events if start_sec <= e.timestamp <= end_sec]
    
    # Plot 1: Raw waveform
    axes[0].plot(times_waveform, samples_segment, color=WAVEFORM_COLOR, linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Waveform (Zoomed)")
    axes[0].grid(True, alpha=0.3)
    
    # Mark events on waveform
    for event in events_in_segment:
        axes[0].axvline(x=event.timestamp, color=PEAK_COLOR, linestyle="-", alpha=0.7, linewidth=2)
        axes[0].text(event.timestamp, axes[0].get_ylim()[1] * 0.9, 
                    f"{event.timestamp:.1f}s", 
                    ha="center", fontsize=8, color=PEAK_COLOR)
    
    # Plot 2: Linear envelope (not dB)
    axes[1].plot(envelope_times_seg, envelope_seg, color=ENVELOPE_COLOR, linewidth=1)
    axes[1].set_ylabel("RMS Energy")
    axes[1].set_title("RMS Envelope (Linear Scale)")
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(envelope_times_seg, 0, envelope_seg, color=ENVELOPE_COLOR, alpha=0.3)
    
    # Mark events
    for event in events_in_segment:
        axes[1].axvline(x=event.timestamp, color=PEAK_COLOR, linestyle="-", alpha=0.7, linewidth=2)
    
    # Plot 3: dB envelope with threshold
    axes[2].plot(envelope_times_seg, envelope_db_seg, color=ENVELOPE_COLOR, linewidth=1)
    
    threshold = result.parameters.get("threshold_db", -20)
    axes[2].axhline(y=threshold, color=THRESHOLD_COLOR, linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Threshold ({threshold:.1f} dB)")
    
    # Mark detected peaks
    for event in events_in_segment:
        axes[2].scatter([event.timestamp], [event.amplitude_db], 
                       color=PEAK_COLOR, s=150, zorder=5, marker="v")
        axes[2].annotate(f"{event.timestamp:.1f}s\n{event.amplitude_db:.1f}dB\nconf:{event.confidence:.2f}",
                        xy=(event.timestamp, event.amplitude_db),
                        xytext=(0, 15), textcoords="offset points",
                        ha="center", fontsize=8, color=PEAK_COLOR,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="#16213e", edgecolor=PEAK_COLOR))
    
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude (dB)")
    axes[2].set_title("RMS Envelope (dB Scale) with Detections")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(start_sec, end_sec)
    
    source_name = Path(audio.source_file).name
    fig.suptitle(
        title or f"Segment Analysis: {source_name} [{start_sec:.0f}s - {end_sec:.0f}s]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    fig.tight_layout()
    return fig


def print_segment_events(result: DetectionResult, start_sec: float, end_sec: float):
    """Print detailed info about events in a segment."""
    events = [e for e in result.events if start_sec <= e.timestamp <= end_sec]
    
    print(f"\n📊 Events in segment [{start_sec:.0f}s - {end_sec:.0f}s]:")
    print("-" * 60)
    
    if not events:
        print("   No events detected in this segment")
        return
    
    for i, event in enumerate(events, 1):
        status = "🎯" if event.confidence >= 0.9 else "⚠️" if event.confidence >= 0.7 else "❓"
        print(f"   {status} #{i}: {event.timestamp:.2f}s")
        print(f"       Amplitude: {event.amplitude_db:.1f} dB")
        print(f"       Confidence: {event.confidence:.2f}")
        if hasattr(event, 'onset_strength') and event.onset_strength > 0:
            print(f"       Onset: {event.onset_strength:.1f} | Flux: {event.spectral_flux:.2f}")
        print()


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--start", "-s",
    type=float,
    default=0.0,
    help="Start time in seconds",
)
@click.option(
    "--end", "-e",
    type=float,
    default=60.0,
    help="End time in seconds",
)
@click.option(
    "-o", "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to save output plots",
)
@click.option(
    "--min-gap",
    type=float,
    default=3.0,
    help="Minimum gap between detections in seconds",
)
@click.option(
    "--transient/--adaptive",
    default=True,
    help="Use transient detection (default) or adaptive",
)
def main(video_path: str, start: float, end: float, output_dir: str, min_gap: float, transient: bool):
    """
    Analyze a specific time segment in detail.
    
    Example:
        uv run python scripts/analyze_segment.py samples/IMG_6644.MOV --start 0 --end 60
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    click.echo(f"📹 Processing: {video_path.name}")
    click.echo(f"   Segment: {start:.0f}s - {end:.0f}s")
    click.echo("   Extracting audio...")
    
    audio = extract_audio_from_video(video_path)
    click.echo(f"   Duration: {audio.duration:.2f}s | Sample rate: {audio.sample_rate} Hz")
    
    # Validate segment bounds
    end = min(end, audio.duration)
    
    click.echo("   Running detection...")
    if transient:
        result = detect_impacts_transient(audio, min_gap_sec=min_gap)
        click.echo("   Method: Transient (onset strength)")
    else:
        result = detect_impacts_adaptive(audio, min_gap_sec=min_gap)
        click.echo("   Method: Adaptive")
    
    # Print segment analysis
    print_segment_events(result, start, end)
    
    # Generate zoomed plot
    click.echo("   Generating segment plot...")
    fig = plot_segment_analysis(audio, result, start, end)
    
    base_name = video_path.stem
    output_path = output_dir / f"{base_name}_segment_{int(start)}s-{int(end)}s.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    click.echo(f"✅ Saved: {output_path}")
    
    click.echo("\n🏌️ Done!")


if __name__ == "__main__":
    main()

