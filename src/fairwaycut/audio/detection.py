"""Impact event detection for golf swing analysis."""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
import librosa

from fairwaycut.core.models import AudioData, ImpactEvent, DetectionResult
from fairwaycut.audio.extraction import compute_envelope


def detect_impacts(
    audio: AudioData,
    threshold_db: float = -20.0,
    min_gap_sec: float = 3.0,
    frame_length: int = 512,
    hop_length: int = 256,
    prominence_db: float = 10.0,
) -> DetectionResult:
    """
    Detect impact events in audio using peak detection on the RMS envelope.

    Args:
        audio: AudioData containing the audio samples.
        threshold_db: Minimum amplitude threshold in dB for peak detection.
        min_gap_sec: Minimum time gap between detected events in seconds.
        frame_length: Length of the analysis frame in samples.
        hop_length: Number of samples between successive frames.
        prominence_db: Minimum prominence of peaks in dB.

    Returns:
        DetectionResult containing detected events and analysis data.
    """
    # Compute RMS envelope
    envelope, times = compute_envelope(audio, frame_length, hop_length)

    # Convert to dB scale
    envelope_db = librosa.amplitude_to_db(envelope, ref=np.max(envelope))

    # Calculate minimum distance between peaks in frames
    min_distance_frames = int(min_gap_sec * audio.sample_rate / hop_length)

    # Find peaks in the envelope
    peaks, properties = find_peaks(
        envelope_db,
        height=threshold_db,
        distance=min_distance_frames,
        prominence=prominence_db,
    )

    # Extract peak heights
    peak_heights = properties.get("peak_heights", envelope_db[peaks])

    # Compute confidence scores based on prominence and height
    if len(peaks) > 0:
        prominences = properties.get("prominences", np.ones(len(peaks)) * prominence_db)
        height_scores = (peak_heights - threshold_db) / (-threshold_db) 
        prominence_scores = prominences / (prominence_db * 2)
        confidence_scores = np.clip((height_scores + prominence_scores) / 2, 0, 1)
    else:
        confidence_scores = np.array([])

    # Create ImpactEvent objects
    events = []
    for i, peak_idx in enumerate(peaks):
        event = ImpactEvent(
            timestamp=times[peak_idx],
            confidence=float(confidence_scores[i]),
            amplitude_db=float(peak_heights[i]),
        )
        events.append(event)

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    parameters = {
        "threshold_db": threshold_db,
        "min_gap_sec": min_gap_sec,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "prominence_db": prominence_db,
        "sample_rate": audio.sample_rate,
    }

    return DetectionResult(
        events=events,
        parameters=parameters,
        envelope=envelope,
        envelope_times=times,
        envelope_db=envelope_db,
    )


def detect_impacts_adaptive(
    audio: AudioData,
    min_gap_sec: float = 3.0,
    frame_length: int = 512,
    hop_length: int = 256,
    threshold_percentile: float = 90.0,
    prominence_factor: float = 1.5,
) -> DetectionResult:
    """
    Detect impact events using adaptive thresholding based on audio statistics.

    This method automatically determines the threshold based on the audio's
    noise floor and dynamic range, making it more robust to different
    recording conditions.

    Args:
        audio: AudioData containing the audio samples.
        min_gap_sec: Minimum time gap between detected events in seconds.
        frame_length: Length of the analysis frame in samples.
        hop_length: Number of samples between successive frames.
        threshold_percentile: Percentile of envelope to use as threshold.
        prominence_factor: Factor applied to std deviation for prominence.

    Returns:
        DetectionResult containing detected events and analysis data.
    """
    # Compute RMS envelope
    envelope, times = compute_envelope(audio, frame_length, hop_length)

    # Convert to dB scale (relative to max)
    envelope_db = librosa.amplitude_to_db(envelope, ref=np.max(envelope))

    # Adaptive threshold based on percentile
    threshold_db = float(np.percentile(envelope_db, threshold_percentile))

    # Adaptive prominence based on standard deviation
    envelope_std = float(np.std(envelope_db))
    prominence_db = envelope_std * prominence_factor

    # Calculate minimum distance between peaks in frames
    min_distance_frames = int(min_gap_sec * audio.sample_rate / hop_length)

    # Find peaks in the envelope
    peaks, properties = find_peaks(
        envelope_db,
        height=threshold_db,
        distance=min_distance_frames,
        prominence=prominence_db,
    )

    # Extract peak heights
    peak_heights = properties.get("peak_heights", envelope_db[peaks])

    # Compute confidence scores
    if len(peaks) > 0:
        prominences = properties.get("prominences", np.ones(len(peaks)) * prominence_db)
        height_scores = (peak_heights - threshold_db) / (-threshold_db) if threshold_db < 0 else np.ones(len(peaks))
        prominence_scores = prominences / (prominence_db * 2)
        confidence_scores = np.clip((height_scores + prominence_scores) / 2, 0, 1)
    else:
        confidence_scores = np.array([])

    # Create ImpactEvent objects
    events = []
    for i, peak_idx in enumerate(peaks):
        event = ImpactEvent(
            timestamp=times[peak_idx],
            confidence=float(confidence_scores[i]),
            amplitude_db=float(peak_heights[i]),
        )
        events.append(event)

    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)

    parameters = {
        "threshold_db": threshold_db,
        "min_gap_sec": min_gap_sec,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "prominence_db": prominence_db,
        "threshold_percentile": threshold_percentile,
        "prominence_factor": prominence_factor,
        "sample_rate": audio.sample_rate,
        "adaptive": True,
    }

    return DetectionResult(
        events=events,
        parameters=parameters,
        envelope=envelope,
        envelope_times=times,
        envelope_db=envelope_db,
    )


def detect_impacts_transient(
    audio: AudioData,
    min_gap_sec: float = 3.0,
    hop_length: int = 256,
    onset_threshold: float = 3.0,
    spectral_flux_threshold: float = 1.0,
    amplitude_threshold_db: float = -12.0,
) -> DetectionResult:
    """
    Detect impact events using transient analysis (spectral flux + onset strength).
    
    This method is more accurate at distinguishing true ball impacts from
    sustained sounds like motors, voices, or background noise. It uses
    spectral flux as the primary signal (best for detecting the sharp 
    frequency changes of ball impacts) and onset strength as validation.

    Args:
        audio: AudioData containing the audio samples.
        min_gap_sec: Minimum time gap between detected events in seconds.
        hop_length: Number of samples between successive frames.
        onset_threshold: Minimum onset strength for validation.
        spectral_flux_threshold: Minimum spectral flux for detection.
        amplitude_threshold_db: Minimum amplitude in dB (relative to max).

    Returns:
        DetectionResult containing detected events and analysis data.
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=hop_length,
    )
    
    # Compute spectral flux - PRIMARY detection signal
    spec = np.abs(librosa.stft(audio.samples, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])
    
    # Ensure same length
    min_len = min(len(onset_env), len(spectral_flux))
    onset_env = onset_env[:min_len]
    spectral_flux = spectral_flux[:min_len]
    
    # Compute RMS envelope for amplitude check
    rms = librosa.feature.rms(y=audio.samples, hop_length=hop_length)[0]
    rms = rms[:min_len]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))
    
    # Time axis
    times = librosa.frames_to_time(np.arange(min_len), sr=audio.sample_rate, hop_length=hop_length)
    
    # Calculate minimum distance between peaks in frames
    min_distance_frames = int(min_gap_sec * audio.sample_rate / hop_length)
    
    # Use spectral flux as PRIMARY detection signal
    peaks_flux, props_flux = find_peaks(
        spectral_flux,
        height=spectral_flux_threshold,
        distance=min_distance_frames,
        prominence=spectral_flux_threshold * 0.3,
    )
    
    # Filter peaks by additional criteria
    events = []
    for peak_idx in peaks_flux:
        peak_onset = onset_env[peak_idx] if peak_idx < len(onset_env) else 0
        peak_flux = spectral_flux[peak_idx]
        peak_amplitude_db = rms_db[peak_idx] if peak_idx < len(rms_db) else -50
        
        # Check criteria
        passes_flux = peak_flux >= spectral_flux_threshold
        passes_onset = peak_onset >= onset_threshold * 0.5
        passes_amplitude = peak_amplitude_db >= amplitude_threshold_db
        
        if passes_flux and passes_amplitude and passes_onset:
            flux_score = min(1.0, peak_flux / (spectral_flux_threshold * 3))
            onset_score = min(1.0, peak_onset / (onset_threshold * 2)) if passes_onset else 0.3
            amp_score = min(1.0, (peak_amplitude_db - amplitude_threshold_db) / 12)
            
            confidence = flux_score * 0.5 + onset_score * 0.2 + amp_score * 0.3
            
            event = ImpactEvent(
                timestamp=times[peak_idx],
                confidence=float(confidence),
                amplitude_db=float(peak_amplitude_db),
                onset_strength=float(peak_onset),
                spectral_flux=float(peak_flux),
                is_transient=passes_onset,
            )
            events.append(event)
    
    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)
    
    parameters = {
        "min_gap_sec": min_gap_sec,
        "hop_length": hop_length,
        "onset_threshold": onset_threshold,
        "spectral_flux_threshold": spectral_flux_threshold,
        "amplitude_threshold_db": amplitude_threshold_db,
        "sample_rate": audio.sample_rate,
        "method": "transient_flux",
    }
    
    # For compatibility, compute standard envelope
    envelope, env_times = compute_envelope(audio, frame_length=512, hop_length=hop_length)
    envelope_db = librosa.amplitude_to_db(envelope, ref=np.max(envelope))
    
    return DetectionResult(
        events=events,
        parameters=parameters,
        envelope=envelope[:len(env_times)],
        envelope_times=env_times,
        envelope_db=envelope_db[:len(env_times)],
    )


def detect_impacts_adaptive_snr(
    audio: AudioData,
    min_gap_sec: float = 3.0,
    hop_length: int = 256,
    snr_threshold: float = 2.5,
    local_window_sec: float = 10.0,
    min_flux: float = 0.8,
    min_onset: float = 0.5,
    amplitude_threshold_db: float = -15.0,
) -> DetectionResult:
    """
    Detect impacts using local signal-to-noise ratio (SNR) adaptive thresholding.
    
    This method adapts to changing noise conditions by comparing peaks to their
    local background. Works well in both quiet and noisy environments.

    Args:
        audio: AudioData containing the audio samples.
        min_gap_sec: Minimum time gap between detected events in seconds.
        hop_length: Number of samples between successive frames.
        snr_threshold: Minimum ratio of peak to local background.
        local_window_sec: Window size for computing local background (seconds).
        min_flux: Absolute minimum spectral flux (prevents noise floor detections).
        min_onset: Minimum onset strength to validate transient.
        amplitude_threshold_db: Minimum amplitude in dB (relative to max).

    Returns:
        DetectionResult containing detected events and analysis data.
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=hop_length,
    )
    
    # Compute spectral flux
    spec = np.abs(librosa.stft(audio.samples, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])
    
    # Ensure same length
    min_len = min(len(onset_env), len(spectral_flux))
    onset_env = onset_env[:min_len]
    spectral_flux = spectral_flux[:min_len]
    
    # Compute RMS envelope for amplitude check
    rms = librosa.feature.rms(y=audio.samples, hop_length=hop_length)[0]
    rms = rms[:min_len]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max(rms))
    
    # Time axis
    times = librosa.frames_to_time(np.arange(min_len), sr=audio.sample_rate, hop_length=hop_length)
    
    # Compute local background using rolling median
    window_frames = int(local_window_sec * audio.sample_rate / hop_length)
    if window_frames % 2 == 0:
        window_frames += 1
    
    local_background = median_filter(spectral_flux, size=window_frames, mode='reflect')
    
    # Compute local SNR
    snr = spectral_flux / (local_background + 1e-6)
    
    # Calculate minimum distance between peaks in frames
    min_distance_frames = int(min_gap_sec * audio.sample_rate / hop_length)
    
    # Find peaks in SNR signal
    peaks, props = find_peaks(
        snr,
        height=snr_threshold,
        distance=min_distance_frames,
        prominence=snr_threshold * 0.3,
    )
    
    # Filter peaks by additional criteria
    events = []
    for peak_idx in peaks:
        peak_onset = onset_env[peak_idx] if peak_idx < len(onset_env) else 0
        peak_flux = spectral_flux[peak_idx]
        peak_snr = snr[peak_idx]
        peak_amplitude_db = rms_db[peak_idx] if peak_idx < len(rms_db) else -50
        local_bg = local_background[peak_idx]
        
        # Adaptive onset threshold based on noise level
        quiet_threshold = 0.2
        noisy_threshold = 0.4
        
        if local_bg < quiet_threshold:
            adaptive_onset = max(min_onset, 1.5)
        elif local_bg > noisy_threshold:
            adaptive_onset = min_onset
        else:
            t = (local_bg - quiet_threshold) / (noisy_threshold - quiet_threshold)
            adaptive_onset = 1.5 * (1 - t) + min_onset * t
        
        # Check criteria
        passes_snr = peak_snr >= snr_threshold
        passes_min_flux = peak_flux >= min_flux
        passes_onset = peak_onset >= adaptive_onset
        passes_amplitude = peak_amplitude_db >= amplitude_threshold_db
        
        if passes_snr and passes_min_flux and passes_onset and passes_amplitude:
            snr_score = min(1.0, (peak_snr - snr_threshold) / (snr_threshold * 2))
            onset_score = min(1.0, peak_onset / 6.0)
            amp_score = min(1.0, (peak_amplitude_db - amplitude_threshold_db) / 15)
            
            confidence = snr_score * 0.5 + onset_score * 0.25 + amp_score * 0.25
            
            event = ImpactEvent(
                timestamp=times[peak_idx],
                confidence=float(confidence),
                amplitude_db=float(peak_amplitude_db),
                onset_strength=float(peak_onset),
                spectral_flux=float(peak_flux),
                is_transient=True,
            )
            events.append(event)
    
    # Sort by timestamp
    events.sort(key=lambda e: e.timestamp)
    
    parameters = {
        "min_gap_sec": min_gap_sec,
        "hop_length": hop_length,
        "snr_threshold": snr_threshold,
        "local_window_sec": local_window_sec,
        "min_flux": min_flux,
        "min_onset": min_onset,
        "amplitude_threshold_db": amplitude_threshold_db,
        "sample_rate": audio.sample_rate,
        "method": "adaptive_snr",
    }
    
    # For compatibility, compute standard envelope
    envelope, env_times = compute_envelope(audio, frame_length=512, hop_length=hop_length)
    envelope_db = librosa.amplitude_to_db(envelope, ref=np.max(envelope))
    
    return DetectionResult(
        events=events,
        parameters=parameters,
        envelope=envelope[:len(env_times)],
        envelope_times=env_times,
        envelope_db=envelope_db[:len(env_times)],
    )

