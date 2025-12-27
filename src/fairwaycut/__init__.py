"""FairwayCut - Golf swing auto-segmentation with pose estimation and audio analysis."""

__version__ = "0.2.0"

# Re-export core models for backwards compatibility
from fairwaycut.core.models import (
    AudioData,
    FramePose,
    ImpactEvent,
    SwingEvent,
    SwingPhase,
    DetectionResult,
)

# Re-export commonly used functions
from fairwaycut.audio import (
    extract_audio_from_video,
    load_audio_file,
    detect_impacts,
    detect_impacts_adaptive,
    detect_impacts_transient,
    detect_impacts_adaptive_snr,
)

__all__ = [
    "__version__",
    # Core models
    "AudioData",
    "FramePose",
    "ImpactEvent",
    "SwingEvent",
    "SwingPhase",
    "DetectionResult",
    # Audio functions
    "extract_audio_from_video",
    "load_audio_file",
    "detect_impacts",
    "detect_impacts_adaptive",
    "detect_impacts_transient",
    "detect_impacts_adaptive_snr",
]
