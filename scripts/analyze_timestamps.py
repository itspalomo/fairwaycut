#!/usr/bin/env python
"""Analyze specific timestamps to understand audio characteristics."""

import sys
from pathlib import Path

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio import extract_audio_from_video

# Known events from user feedback
KNOWN_EVENTS = [
    (34.78, "YOUR HIT - quiet section"),
    (646.52, "Old detection (ball machine)"),
    (658.0, "YOUR HIT (~10:58) - MISSING"),
    (692.35, "Detected - close to 693s"),
    (693.0, "YOUR HIT (~11:33)"),
]

def analyze_timestamp(audio, timestamp, window_sec=0.2, hop_length=256, local_window_sec=10.0):
    """Analyze audio characteristics at a specific timestamp including local SNR."""
    from scipy.ndimage import median_filter
    
    sr = audio.sample_rate
    
    # Get segment around timestamp for peak detection
    start_sec = max(0, timestamp - window_sec)
    end_sec = min(audio.duration, timestamp + window_sec)
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    segment = audio.samples[start_sample:end_sample]
    
    # Compute onset strength for segment
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=hop_length)
    
    # Compute spectral flux for segment
    spec = np.abs(librosa.stft(segment, hop_length=hop_length))
    if spec.shape[1] > 1:
        spectral_flux_seg = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    else:
        spectral_flux_seg = np.array([0])
    
    # Compute RMS
    rms = librosa.feature.rms(y=segment, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(audio.samples))
    
    # Compute local background for SNR (need wider window)
    bg_start = max(0, timestamp - local_window_sec/2)
    bg_end = min(audio.duration, timestamp + local_window_sec/2)
    bg_start_sample = int(bg_start * sr)
    bg_end_sample = int(bg_end * sr)
    bg_segment = audio.samples[bg_start_sample:bg_end_sample]
    
    bg_spec = np.abs(librosa.stft(bg_segment, hop_length=hop_length))
    if bg_spec.shape[1] > 1:
        bg_flux = np.sqrt(np.mean(np.diff(bg_spec, axis=1)**2, axis=0))
        local_bg = float(np.median(bg_flux))
    else:
        local_bg = 0.1
    
    peak_flux = float(np.max(spectral_flux_seg)) if len(spectral_flux_seg) > 0 else 0
    snr = peak_flux / (local_bg + 1e-6)
    
    return {
        "peak_onset": float(np.max(onset_env)) if len(onset_env) > 0 else 0,
        "peak_flux": peak_flux,
        "local_bg": local_bg,
        "snr": snr,
        "peak_rms_db": float(np.max(rms_db)) if len(rms_db) > 0 else -60,
    }


def main():
    video_path = Path("samples/IMG_6644.MOV")
    
    print("📹 Analyzing known timestamps...")
    print("   Extracting audio...")
    audio = extract_audio_from_video(video_path)
    
    print(f"\n{'='*95}")
    print(f"{'Timestamp':<10} {'Event':<28} {'Onset':>7} {'Flux':>7} {'BG':>6} {'SNR':>6} {'Amp':>8}")
    print(f"{'='*95}")
    
    for timestamp, label in KNOWN_EVENTS:
        stats = analyze_timestamp(audio, timestamp)
        
        # Determine if it would pass SNR thresholds
        passes_onset = stats["peak_onset"] >= 1.5
        passes_snr = stats["snr"] >= 2.5
        passes_amp = stats["peak_rms_db"] >= -15.0
        
        status = "✅" if (passes_onset and passes_snr and passes_amp) else "❌"
        
        print(f"{timestamp:>7.1f}s   {label:<28} {stats['peak_onset']:>7.2f} {stats['peak_flux']:>7.2f} {stats['local_bg']:>6.2f} {stats['snr']:>6.1f} {stats['peak_rms_db']:>8.1f}  {status}")
    
    print(f"{'='*95}")
    print("\nSNR thresholds: onset >= 1.5, SNR >= 2.5, amplitude >= -15 dB")
    print("✅ = Would be detected, ❌ = Would be filtered")


if __name__ == "__main__":
    main()

