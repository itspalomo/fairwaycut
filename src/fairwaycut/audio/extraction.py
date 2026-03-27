"""Audio extraction and analysis utilities for golf swing detection."""

import subprocess
from pathlib import Path

import librosa
import numpy as np

from fairwaycut.core.models import AudioData


def _format_ffmpeg_time(timestamp: float) -> str:
    """Format a timestamp for ffmpeg command arguments."""
    return f"{timestamp:.6f}"


def _has_audio_stream(video_path: Path) -> bool:
    """Return True when the input file contains at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def extract_audio_from_video(
    video_path: str | Path,
    start_time: float | None = None,
    end_time: float | None = None,
    sample_rate: int = 44100,
) -> AudioData:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the video file.
        start_time: Optional absolute start time in seconds.
        end_time: Optional absolute end time in seconds.
        sample_rate: Target output sample rate.

    Returns:
        AudioData containing the extracted audio samples and metadata.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video has no audio track.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    normalized_start = max(0.0, start_time or 0.0)
    normalized_end = end_time if end_time is None else max(normalized_start, end_time)

    if not _has_audio_stream(video_path):
        raise ValueError(f"Video file has no audio track: {video_path}")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]

    if normalized_start > 0:
        command.extend(["-ss", _format_ffmpeg_time(normalized_start)])

    command.extend(["-i", str(video_path)])

    if normalized_end is not None:
        command.extend(["-t", _format_ffmpeg_time(normalized_end - normalized_start)])

    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
    )

    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Failed to extract audio from {video_path}: {stderr}")

    samples = np.frombuffer(result.stdout, dtype=np.float32).copy()
    duration = len(samples) / sample_rate if sample_rate > 0 else 0.0

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        source_file=str(video_path),
        start_time=normalized_start,
    )


def load_audio_file(audio_path: str | Path, target_sr: int | None = None) -> AudioData:
    """
    Load an audio file directly.

    Args:
        audio_path: Path to the audio file.
        target_sr: Target sample rate. If None, uses the file's native sample rate.

    Returns:
        AudioData containing the audio samples and metadata.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    samples, sample_rate = librosa.load(str(audio_path), sr=target_sr, mono=True)
    duration = len(samples) / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        source_file=str(audio_path),
        start_time=0.0,
    )


def compute_envelope(
    audio: AudioData,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the RMS envelope of the audio signal.

    Args:
        audio: AudioData containing the audio samples.
        frame_length: Length of the analysis frame in samples.
        hop_length: Number of samples between successive frames.

    Returns:
        Tuple of (envelope, times) where envelope is the RMS energy
        and times are the corresponding timestamps in seconds.
    """
    # Compute RMS envelope
    envelope = librosa.feature.rms(
        y=audio.samples,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # Convert frame indices to time
    times = librosa.frames_to_time(
        np.arange(len(envelope)),
        sr=audio.sample_rate,
        hop_length=hop_length,
    ) + audio.start_time

    return envelope, times


def compute_envelope_db(
    audio: AudioData,
    frame_length: int = 2048,
    hop_length: int = 512,
    ref: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the RMS envelope of the audio signal in decibels.

    Args:
        audio: AudioData containing the audio samples.
        frame_length: Length of the analysis frame in samples.
        hop_length: Number of samples between successive frames.
        ref: Reference value for dB conversion.

    Returns:
        Tuple of (envelope_db, times) where envelope_db is the RMS energy in dB
        and times are the corresponding timestamps in seconds.
    """
    envelope, times = compute_envelope(audio, frame_length, hop_length)

    # Convert to dB scale
    envelope_db = librosa.amplitude_to_db(envelope, ref=ref)

    return envelope_db, times


def get_waveform_times(audio: AudioData) -> np.ndarray:
    """
    Get the time array for the audio waveform.

    Args:
        audio: AudioData containing the audio samples.

    Returns:
        Array of timestamps in seconds for each sample.
    """
    if audio.sample_rate <= 0 or len(audio.samples) == 0:
        return np.array([], dtype=float)

    return np.arange(len(audio.samples), dtype=float) / audio.sample_rate + audio.start_time


def compute_spectral_flux(
    audio: AudioData,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spectral flux of the audio signal.
    
    Spectral flux measures the rate of change of the spectrum,
    which is useful for detecting transient events like ball impacts.

    Args:
        audio: AudioData containing the audio samples.
        hop_length: Number of samples between successive frames.

    Returns:
        Tuple of (spectral_flux, times) arrays.
    """
    spec = np.abs(librosa.stft(audio.samples, hop_length=hop_length))
    spectral_flux = np.sqrt(np.mean(np.diff(spec, axis=1)**2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])
    
    times = librosa.frames_to_time(
        np.arange(len(spectral_flux)),
        sr=audio.sample_rate,
        hop_length=hop_length,
    ) + audio.start_time
    
    return spectral_flux, times


def compute_onset_strength(
    audio: AudioData,
    hop_length: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute onset strength envelope.
    
    Onset strength indicates how likely a given frame is to be 
    the start of a musical/audio event.

    Args:
        audio: AudioData containing the audio samples.
        hop_length: Number of samples between successive frames.

    Returns:
        Tuple of (onset_envelope, times) arrays.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=hop_length,
    )
    
    times = librosa.frames_to_time(
        np.arange(len(onset_env)),
        sr=audio.sample_rate,
        hop_length=hop_length,
    ) + audio.start_time
    
    return onset_env, times
