"""Visualization utilities for audio analysis and impact detection."""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fairwaycut.audio import AudioData, get_waveform_times
from fairwaycut.detection import DetectionResult


# Style configuration
PLOT_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eaeaea",
    "text.color": "#eaeaea",
    "xtick.color": "#eaeaea",
    "ytick.color": "#eaeaea",
    "grid.color": "#0f3460",
    "grid.alpha": 0.5,
}

WAVEFORM_COLOR = "#4ecca3"
ENVELOPE_COLOR = "#e94560"
PEAK_COLOR = "#ffc93c"
THRESHOLD_COLOR = "#ff6b6b"


def apply_style():
    """Apply custom plot style."""
    plt.rcParams.update(PLOT_STYLE)
    plt.rcParams["font.family"] = "monospace"


def plot_waveform(
    audio: AudioData,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (14, 4),
) -> Figure:
    """
    Plot the raw audio waveform.

    Args:
        audio: AudioData containing the audio samples.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    times = get_waveform_times(audio)
    ax.plot(times, audio.samples, color=WAVEFORM_COLOR, linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Audio Waveform - {Path(audio.source_file).name}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, audio.duration)

    fig.tight_layout()
    return fig


def plot_envelope(
    audio: AudioData,
    result: DetectionResult,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (14, 4),
    show_threshold: bool = True,
) -> Figure:
    """
    Plot the audio envelope with detected peaks.

    Args:
        audio: AudioData containing the audio samples.
        result: DetectionResult from impact detection.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).
        show_threshold: Whether to show the detection threshold line.

    Returns:
        Matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot envelope
    ax.plot(
        result.envelope_times,
        result.envelope_db,
        color=ENVELOPE_COLOR,
        linewidth=1,
        label="RMS Envelope (dB)",
    )

    # Plot threshold line
    if show_threshold:
        threshold = result.parameters.get("threshold_db", -20)
        ax.axhline(
            y=threshold,
            color=THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Threshold ({threshold:.1f} dB)",
        )

    # Mark detected peaks
    if result.events:
        peak_times = [e.timestamp for e in result.events]
        peak_amplitudes = [e.amplitude_db for e in result.events]
        ax.scatter(
            peak_times,
            peak_amplitudes,
            color=PEAK_COLOR,
            s=100,
            zorder=5,
            marker="v",
            label=f"Detected Impacts ({len(result.events)})",
        )

        # Add vertical lines at impact times
        for t in peak_times:
            ax.axvline(x=t, color=PEAK_COLOR, linestyle=":", alpha=0.3, linewidth=1)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(title or f"Audio Envelope - {Path(audio.source_file).name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, audio.duration)

    fig.tight_layout()
    return fig


def plot_analysis(
    audio: AudioData,
    result: DetectionResult,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (14, 10),
) -> Figure:
    """
    Create a comprehensive analysis plot with waveform, envelope, and spectral flux.

    Args:
        audio: AudioData containing the audio samples.
        result: DetectionResult from impact detection.
        title: Optional title for the plot.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    import librosa
    
    apply_style()

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Top plot: Waveform
    times = get_waveform_times(audio)
    axes[0].plot(times, audio.samples, color=WAVEFORM_COLOR, linewidth=0.3, alpha=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Raw Waveform")
    axes[0].grid(True, alpha=0.3)

    # Mark detected impacts on waveform
    if result.events:
        for event in result.events:
            axes[0].axvline(x=event.timestamp, color=PEAK_COLOR, linestyle="-", alpha=0.5, linewidth=1)

    # Middle plot: Envelope
    axes[1].plot(
        result.envelope_times,
        result.envelope_db,
        color=ENVELOPE_COLOR,
        linewidth=1,
    )

    # Plot threshold
    threshold = result.parameters.get("threshold_db", -20)
    axes[1].axhline(
        y=threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    # Mark peaks on envelope
    if result.events:
        peak_times = [e.timestamp for e in result.events]
        peak_amplitudes = [e.amplitude_db for e in result.events]
        axes[1].scatter(
            peak_times,
            peak_amplitudes,
            color=PEAK_COLOR,
            s=80,
            zorder=5,
            marker="v",
        )

    axes[1].set_ylabel("Amplitude (dB)")
    axes[1].set_title(f"RMS Envelope - {len(result.events)} impacts detected")
    axes[1].grid(True, alpha=0.3)

    # Bottom plot: Spectral Flux
    hop_length = result.parameters.get("hop_length", 256)
    spec = np.abs(librosa.stft(audio.samples, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])
    flux_times = librosa.frames_to_time(np.arange(len(spectral_flux)), sr=audio.sample_rate, hop_length=hop_length)
    
    # Color for spectral flux
    FLUX_COLOR = "#00d9ff"
    
    axes[2].plot(flux_times, spectral_flux, color=FLUX_COLOR, linewidth=0.8, alpha=0.9)
    axes[2].fill_between(flux_times, 0, spectral_flux, color=FLUX_COLOR, alpha=0.3)
    
    # Mark peaks on spectral flux
    if result.events:
        peak_times = [e.timestamp for e in result.events]
        # Find flux values at peak times
        peak_flux_values = []
        for pt in peak_times:
            idx = np.argmin(np.abs(flux_times - pt))
            peak_flux_values.append(spectral_flux[idx] if idx < len(spectral_flux) else 0)
        
        axes[2].scatter(
            peak_times,
            peak_flux_values,
            color=PEAK_COLOR,
            s=80,
            zorder=5,
            marker="v",
        )
    
    # Show flux threshold if available
    flux_threshold = result.parameters.get("spectral_flux_threshold", 1.0)
    axes[2].axhline(
        y=flux_threshold,
        color=THRESHOLD_COLOR,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Spectral Flux")
    axes[2].set_title(f"Spectral Flux (threshold={flux_threshold:.1f})")
    axes[2].grid(True, alpha=0.3)
    
    # Set x-axis limits for all plots
    for ax in axes:
        ax.set_xlim(0, audio.duration)

    # Add overall title
    source_name = Path(audio.source_file).name
    fig.suptitle(
        title or f"FairwayCut Analysis - {source_name}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout()
    return fig


def plot_detection_summary(
    result: DetectionResult,
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot a summary of detected events with confidence scores.

    Args:
        result: DetectionResult from impact detection.
        figsize: Figure size as (width, height).

    Returns:
        Matplotlib Figure object.
    """
    apply_style()

    if not result.events:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No impacts detected",
            ha="center", va="center",
            fontsize=16,
            color="#e94560",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    events = result.events
    indices = range(1, len(events) + 1)
    confidences = [e.confidence for e in events]
    amplitudes = [e.amplitude_db for e in events]

    # Left plot: Confidence scores
    bars1 = axes[0].bar(indices, confidences, color=WAVEFORM_COLOR, alpha=0.8)
    axes[0].set_xlabel("Impact #")
    axes[0].set_ylabel("Confidence Score")
    axes[0].set_title("Detection Confidence")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, conf in zip(bars1, confidences):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{conf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Right plot: Amplitude
    bars2 = axes[1].bar(indices, amplitudes, color=ENVELOPE_COLOR, alpha=0.8)
    axes[1].set_xlabel("Impact #")
    axes[1].set_ylabel("Amplitude (dB)")
    axes[1].set_title("Peak Amplitude")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, amp in zip(bars2, amplitudes):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{amp:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        f"Detection Summary - {len(events)} Impacts",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig


def save_figure(fig: Figure, output_path: str | Path, dpi: int = 150) -> Path:
    """
    Save a figure to a file.

    Args:
        fig: Matplotlib Figure to save.
        output_path: Path to save the figure to.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

