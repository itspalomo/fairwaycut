"""Multi-modal fusion for swing detection combining audio and pose signals."""

from fairwaycut.fusion.detector import (
    SwingDetector,
    detect_swings,
    detect_swings_audio_only,
    detect_swings_hybrid,
    detect_swings_full_pose,
    MODE_DESCRIPTIONS,
)
from fairwaycut.core.config import ProcessingMode

__all__ = [
    "SwingDetector",
    "detect_swings",
    "detect_swings_audio_only",
    "detect_swings_hybrid",
    "detect_swings_full_pose",
    "ProcessingMode",
    "MODE_DESCRIPTIONS",
]

