"""Audio extraction and impact detection for golf swing analysis."""

from fairwaycut.audio.extraction import (
    extract_audio_from_video,
    load_audio_file,
    compute_envelope,
    compute_envelope_db,
    get_waveform_times,
)
from fairwaycut.audio.detection import (
    detect_impacts,
    detect_impacts_adaptive,
    detect_impacts_transient,
    detect_impacts_adaptive_snr,
)

__all__ = [
    "extract_audio_from_video",
    "load_audio_file",
    "compute_envelope",
    "compute_envelope_db",
    "get_waveform_times",
    "detect_impacts",
    "detect_impacts_adaptive",
    "detect_impacts_transient",
    "detect_impacts_adaptive_snr",
]

