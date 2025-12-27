#!/usr/bin/env python
"""Analyze transient characteristics to distinguish ball impacts from other sounds."""

import sys
from pathlib import Path

import click
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio import extract_audio_from_video, AudioData
from fairwaycut.visualization import save_figure, apply_style, WAVEFORM_COLOR, ENVELOPE_COLOR, PEAK_COLOR, THRESHOLD_COLOR


def analyze_onset_characteristics(
    audio: AudioData,
    start_sec: float,
    end_sec: float,
    hop_length: int = 256,
) -> dict:
    """
    Analyze onset/transient characteristics of audio segment.
    
    Returns multiple signal representations that help distinguish
    impulsive sounds (ball hits) from sustained sounds (motors, voices).
    """
    # Get audio segment
    start_sample = int(start_sec * audio.sample_rate)
    end_sample = int(end_sec * audio.sample_rate)
    segment = audio.samples[start_sample:end_sample]
    
    # 1. Onset strength - measures how "attacky" each frame is
    onset_env = librosa.onset.onset_strength(
        y=segment, 
        sr=audio.sample_rate,
        hop_length=hop_length,
    )
    
    # 2. Spectral flux - measures spectral change rate
    spec = np.abs(librosa.stft(segment, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])  # Pad to match length
    
    # 3. RMS envelope
    rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
    
    # 4. Zero crossing rate - impulsive sounds have high ZCR
    zcr = librosa.feature.zero_crossing_rate(segment, hop_length=hop_length)[0]
    
    # Time axis
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=audio.sample_rate, hop_length=hop_length)
    times = times + start_sec  # Offset to absolute time
    
    return {
        "times": times,
        "onset_strength": onset_env,
        "spectral_flux": spectral_flux[:len(times)],
        "rms": rms[:len(times)],
        "zcr": zcr[:len(times)],
    }


def plot_transient_analysis(
    audio: AudioData,
    start_sec: float,
    end_sec: float,
    known_events: list[tuple[float, str]] = None,
) -> plt.Figure:
    """
    Create detailed transient analysis plot.
    
    known_events: list of (timestamp, label) tuples for annotation
    """
    apply_style()
    
    analysis = analyze_onset_characteristics(audio, start_sec, end_sec)
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    times = analysis["times"]
    
    # Get waveform segment for top plot
    start_sample = int(start_sec * audio.sample_rate)
    end_sample = int(end_sec * audio.sample_rate)
    samples_segment = audio.samples[start_sample:end_sample]
    waveform_times = np.linspace(start_sec, end_sec, len(samples_segment))
    
    # Plot 1: Raw waveform
    axes[0].plot(waveform_times, samples_segment, color=WAVEFORM_COLOR, linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Waveform")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: RMS Envelope
    axes[1].plot(times, analysis["rms"], color=ENVELOPE_COLOR, linewidth=1)
    axes[1].fill_between(times, 0, analysis["rms"], color=ENVELOPE_COLOR, alpha=0.3)
    axes[1].set_ylabel("RMS Energy")
    axes[1].set_title("RMS Envelope (Energy)")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Onset Strength (KEY for transients!)
    axes[2].plot(times, analysis["onset_strength"], color="#ffc93c", linewidth=1)
    axes[2].fill_between(times, 0, analysis["onset_strength"], color="#ffc93c", alpha=0.3)
    axes[2].set_ylabel("Onset Strength")
    axes[2].set_title("Onset Strength (Attack Detection) - HIGH = Impulsive/Transient")
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Spectral Flux
    axes[3].plot(times, analysis["spectral_flux"], color="#00d9ff", linewidth=1)
    axes[3].fill_between(times, 0, analysis["spectral_flux"], color="#00d9ff", alpha=0.3)
    axes[3].set_ylabel("Spectral Flux")
    axes[3].set_title("Spectral Flux (Frequency Change Rate)")
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Zero Crossing Rate
    axes[4].plot(times, analysis["zcr"], color="#ff6b6b", linewidth=1)
    axes[4].fill_between(times, 0, analysis["zcr"], color="#ff6b6b", alpha=0.3)
    axes[4].set_ylabel("ZCR")
    axes[4].set_title("Zero Crossing Rate")
    axes[4].set_xlabel("Time (seconds)")
    axes[4].grid(True, alpha=0.3)
    
    # Add event markers on all plots
    if known_events:
        for timestamp, label in known_events:
            if start_sec <= timestamp <= end_sec:
                for ax in axes:
                    ax.axvline(x=timestamp, color=PEAK_COLOR, linestyle="--", alpha=0.7, linewidth=1.5)
                # Add label on top plot
                axes[0].text(timestamp, axes[0].get_ylim()[1] * 0.95, label, 
                           ha="center", fontsize=9, color=PEAK_COLOR,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="#16213e", edgecolor=PEAK_COLOR))
    
    for ax in axes:
        ax.set_xlim(start_sec, end_sec)
    
    source_name = Path(audio.source_file).name
    fig.suptitle(
        f"Transient Analysis: {source_name} [{start_sec:.0f}s - {end_sec:.0f}s]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    
    fig.tight_layout()
    return fig


def compute_transient_score(
    audio: AudioData,
    timestamp: float,
    window_sec: float = 0.1,
    hop_length: int = 256,
) -> dict:
    """
    Compute a transient/impulsiveness score for a detected event.
    
    Higher scores indicate more impulsive (ball-hit-like) sounds.
    Lower scores indicate sustained sounds (motor, voice).
    """
    # Analyze a window around the event
    start_sec = max(0, timestamp - window_sec)
    end_sec = min(audio.duration, timestamp + window_sec)
    
    analysis = analyze_onset_characteristics(audio, start_sec, end_sec, hop_length)
    
    # Find the peak in onset strength near the timestamp
    times = analysis["times"]
    onset = analysis["onset_strength"]
    rms = analysis["rms"]
    
    # Find index closest to timestamp
    idx = np.argmin(np.abs(times - timestamp))
    
    # Get local peak values (within small window around idx)
    window = 5  # frames
    local_onset = onset[max(0, idx-window):min(len(onset), idx+window)]
    local_rms = rms[max(0, idx-window):min(len(rms), idx+window)]
    
    peak_onset = np.max(local_onset) if len(local_onset) > 0 else 0
    peak_rms = np.max(local_rms) if len(local_rms) > 0 else 0
    
    # Compute ratio: onset strength relative to RMS
    # High ratio = transient (sharp attack relative to overall energy)
    # Low ratio = sustained (energy without sharp attack)
    if peak_rms > 0:
        transient_ratio = peak_onset / (peak_rms * 100)
    else:
        transient_ratio = 0
    
    return {
        "timestamp": timestamp,
        "peak_onset_strength": float(peak_onset),
        "peak_rms": float(peak_rms),
        "transient_ratio": float(transient_ratio),
        "is_likely_impact": transient_ratio > 0.5,  # Tunable threshold
    }


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
def main(video_path: str, start: float, end: float, output_dir: str):
    """
    Analyze transient characteristics to distinguish ball impacts.
    
    Example:
        uv run python scripts/analyze_transients.py samples/IMG_6644.MOV --start 20 --end 45
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    click.echo(f"📹 Processing: {video_path.name}")
    click.echo(f"   Segment: {start:.0f}s - {end:.0f}s")
    click.echo("   Extracting audio...")
    
    audio = extract_audio_from_video(video_path)
    click.echo(f"   Duration: {audio.duration:.2f}s")
    
    # Known events from user feedback
    known_events = [
        (26.0, "Golf Cart"),
        (34.8, "REAL HIT"),
    ]
    
    # Filter to segment
    known_in_segment = [(t, l) for t, l in known_events if start <= t <= end]
    
    # Analyze transient characteristics for known events
    click.echo("\n📊 Transient Analysis of Known Events:")
    click.echo("-" * 60)
    
    for timestamp, label in known_events:
        if start <= timestamp <= end:
            scores = compute_transient_score(audio, timestamp)
            status = "✅" if scores["is_likely_impact"] else "❌"
            click.echo(f"\n   {label} @ {timestamp:.1f}s:")
            click.echo(f"     Peak Onset Strength: {scores['peak_onset_strength']:.2f}")
            click.echo(f"     Peak RMS: {scores['peak_rms']:.4f}")
            click.echo(f"     Transient Ratio: {scores['transient_ratio']:.2f}")
            click.echo(f"     {status} Likely Impact: {scores['is_likely_impact']}")
    
    # Generate plot
    click.echo("\n   Generating transient analysis plot...")
    fig = plot_transient_analysis(audio, start, end, known_in_segment)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = video_path.stem
    output_path = output_dir / f"{base_name}_transients_{int(start)}s-{int(end)}s.png"
    save_figure(fig, output_path)
    click.echo(f"✅ Saved: {output_path}")
    
    click.echo("\n🏌️ Done!")


if __name__ == "__main__":
    main()

