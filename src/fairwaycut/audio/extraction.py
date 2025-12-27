"""Audio extraction and analysis utilities for golf swing detection."""

import tempfile
from pathlib import Path

import numpy as np
import librosa
from moviepy import VideoFileClip

from fairwaycut.core.models import AudioData


def extract_audio_from_video(video_path: str | Path) -> AudioData:
    """
    Extract audio from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        AudioData containing the extracted audio samples and metadata.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video has no audio track.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Extract audio using moviepy
    with VideoFileClip(str(video_path)) as video:
        if video.audio is None:
            raise ValueError(f"Video file has no audio track: {video_path}")

        # Create a temporary file for the extracted audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name

        # Write audio to temporary file
        video.audio.write_audiofile(
            temp_audio_path,
            fps=44100,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )

    # Load the audio with librosa for analysis
    samples, sample_rate = librosa.load(temp_audio_path, sr=None, mono=True)

    # Clean up temporary file
    Path(temp_audio_path).unlink(missing_ok=True)

    duration = len(samples) / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        source_file=str(video_path),
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
    )

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
    return np.linspace(0, audio.duration, len(audio.samples))


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
    )
    
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
    )
    
    return onset_env, times

