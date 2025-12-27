#!/usr/bin/env python
"""Debug SNR detection around specific timestamps."""

import sys
from pathlib import Path
import numpy as np
import librosa
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fairwaycut.audio import extract_audio_from_video

def main():
    video_path = Path("samples/IMG_6644.MOV")
    
    print("📹 Debugging SNR detection...")
    audio = extract_audio_from_video(video_path)
    
    hop_length = 256
    local_window_sec = 10.0
    
    # Compute spectral flux for full audio
    spec = np.abs(librosa.stft(audio.samples, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])
    
    # Compute onset strength
    onset_env = librosa.onset.onset_strength(y=audio.samples, sr=audio.sample_rate, hop_length=hop_length)
    
    # Align lengths
    min_len = min(len(onset_env), len(spectral_flux))
    spectral_flux = spectral_flux[:min_len]
    onset_env = onset_env[:min_len]
    
    # Time axis
    times = librosa.frames_to_time(np.arange(min_len), sr=audio.sample_rate, hop_length=hop_length)
    
    # Compute local background
    window_frames = int(local_window_sec * audio.sample_rate / hop_length)
    if window_frames % 2 == 0:
        window_frames += 1
    local_background = median_filter(spectral_flux, size=window_frames, mode='reflect')
    
    # Compute SNR
    snr = spectral_flux / (local_background + 1e-6)
    
    # Look at region around 658s
    target_times = [655, 658, 660, 665, 670, 690, 693]
    
    print(f"\n{'Time':>8} {'Flux':>8} {'BG':>8} {'SNR':>8} {'Onset':>8} {'Is Peak':>8}")
    print("=" * 60)
    
    # Find peaks in SNR
    peaks, _ = find_peaks(snr, height=2.0, distance=int(3.0 * audio.sample_rate / hop_length))
    peak_times = times[peaks]
    
    for t in target_times:
        idx = np.argmin(np.abs(times - t))
        flux_val = spectral_flux[idx]
        bg_val = local_background[idx]
        snr_val = snr[idx]
        onset_val = onset_env[idx]
        
        # Check if this is near a peak
        nearest_peak = peak_times[np.argmin(np.abs(peak_times - t))] if len(peak_times) > 0 else -1
        is_peak = "✅" if abs(nearest_peak - t) < 2 else f"~{nearest_peak:.1f}s"
        
        print(f"{t:>8.1f} {flux_val:>8.2f} {bg_val:>8.2f} {snr_val:>8.2f} {onset_val:>8.2f} {is_peak:>8}")
    
    # Find actual peaks in 640-700 region
    mask = (times >= 640) & (times <= 700)
    region_peaks = peaks[(times[peaks] >= 640) & (times[peaks] <= 700)]
    
    print(f"\n📍 Actual SNR peaks found in 640-700s region:")
    for p in region_peaks:
        print(f"   {times[p]:.1f}s - SNR: {snr[p]:.2f}, Flux: {spectral_flux[p]:.2f}, Onset: {onset_env[p]:.2f}")


if __name__ == "__main__":
    main()

