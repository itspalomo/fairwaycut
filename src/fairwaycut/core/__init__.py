"""Core models and configuration for FairwayCut."""

from fairwaycut.core.models import (
    AudioData,
    FramePose,
    ImpactEvent,
    SwingEvent,
    SwingPhase,
    DetectionResult,
)
from fairwaycut.core.config import (
    Config,
    VideoConfig,
    VisualizationConfig,
    VisualizationStyle,
)

__all__ = [
    # Models
    "AudioData",
    "FramePose",
    "ImpactEvent",
    "SwingEvent",
    "SwingPhase",
    "DetectionResult",
    # Config
    "Config",
    "VideoConfig",
    "VisualizationConfig",
    "VisualizationStyle",
]

