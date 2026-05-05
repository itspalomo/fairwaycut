"""Configuration management for FairwayCut."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import json


class ProcessingMode(Enum):
    """Processing modes with different speed/accuracy tradeoffs.
    
    AUDIO: Fastest - audio detection only, no pose estimation
    HYBRID: Fast - audio detection + pose estimation around detected impacts only
    LITE: Slower - full video pose estimation with lite model
    FULL: Slowest - full video pose estimation with full accuracy model
    """
    AUDIO = "audio"      # Audio only, no pose
    HYBRID = "hybrid"    # Audio + targeted pose around impacts (best balance)
    LITE = "lite"        # Full video, lite pose model
    FULL = "full"        # Full video, full pose model
    
    # Backwards compatibility aliases
    @classmethod
    def from_string(cls, value: str) -> "ProcessingMode":
        """Convert string to ProcessingMode with backwards compatibility."""
        # Handle old names
        aliases = {
            "audio_only": cls.AUDIO,
            "pose_segments": cls.HYBRID,
            "segments": cls.HYBRID,
            "pose_lite": cls.LITE,
            "pose_full": cls.FULL,
        }
        if value in aliases:
            return aliases[value]
        return cls(value)


@dataclass
class AudioConfig:
    """Configuration for audio analysis."""
    
    # Detection parameters
    threshold_db: float = -20.0
    min_gap_sec: float = 3.0
    frame_length: int = 512
    hop_length: int = 256
    prominence_db: float = 10.0
    
    # Adaptive SNR parameters
    snr_threshold: float = 2.5
    local_window_sec: float = 10.0
    min_flux: float = 0.8
    min_onset: float = 0.5
    amplitude_threshold_db: float = -15.0


@dataclass
class PoseConfig:
    """Configuration for pose estimation.
    
    Performance notes:
    - model_complexity=0 (lite) is ~3x faster than full, good enough for swing detection
    - model_complexity=1 (full) is more accurate but slower
    - model_complexity=2 (heavy) is most accurate but significantly slower
    - process_every_n_frames=2 skips every other frame for 2x speed
    - max_frame_size=640 resizes large frames for faster processing
    """
    
    # MediaPipe settings
    model_complexity: int = 0  # 0=lite/fast, 1=full, 2=heavy/slow
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_frame_size: int = 640  # Resize frames larger than this
    
    # Processing
    process_every_n_frames: int = 1  # Skip frames for performance (1=all, 2=half)
    
    # Swing phase detection
    velocity_smoothing_window: int = 5
    phase_transition_threshold: float = 0.3


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""

    # Temporal alignment
    max_alignment_offset_sec: float = 0.5  # Max audio-pose alignment drift

    # Audio-confidence floor — independent of pose. Real swings on the
    # sample reel sit at 0.84+; weak/spurious onsets are below 0.5.
    min_audio_confidence: float = 0.5

    # Segment boundaries
    pre_impact_sec: float = 3.0  # Time before impact to include
    post_impact_sec: float = 2.0  # Time after impact to include

    # Swing-motion gate (applied whenever pose was attempted for an event).
    # An audio impact with no real swing pattern in the pose track gets
    # dropped — this is the main filter that turns hybrid into a real
    # validator rather than a soft re-weighting of audio detections.
    require_swing_motion: bool = True
    # Wrist-speed peak (normalized-coords/sec) the segment must exceed to
    # qualify as a swing. Real swings on the sample reel peak at 15+; 2.0
    # is well below that but well above idle/setup motion.
    min_peak_wrist_speed: float = 2.0
    # Require backswing + downswing + impact phase pattern. The phase
    # detector is currently noisy on real footage, so we only check that
    # the required phase set was emitted, not its timing.
    require_complete_swing: bool = True


@dataclass
class VideoConfig:
    """Configuration for video generation."""

    # Output settings
    output_fps: float = 30.0
    output_codec: str = "libx264"
    output_quality: int = 23  # CRF value (lower = better quality)

    # Audio waveform overlay
    waveform_height: int = 80

    # Wrist-speed HUD: scaling factor from normalized-coords/sec to m/s.
    # The pose normalizer outputs y in roughly [0, 1] across the frame; for
    # a typical person filling ~2 m of frame height, multiplying normalized
    # speed by ~2.0 gives an honest m/s readout. Tunable per-rig.
    wrist_speed_scale_mps: float = 2.0


@dataclass
class Config:
    """Main configuration container."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "audio": self.audio.__dict__,
            "pose": self.pose.__dict__,
            "fusion": self.fusion.__dict__,
            "video": {
                k: list(v) if isinstance(v, tuple) else v 
                for k, v in self.video.__dict__.items()
            },
            "output_dir": str(self.output_dir),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        config = cls()
        
        if "audio" in data:
            for k, v in data["audio"].items():
                setattr(config.audio, k, v)
        
        if "pose" in data:
            for k, v in data["pose"].items():
                setattr(config.pose, k, v)
        
        if "fusion" in data:
            for k, v in data["fusion"].items():
                setattr(config.fusion, k, v)
        
        if "video" in data:
            for k, v in data["video"].items():
                if isinstance(v, list):
                    v = tuple(v)
                setattr(config.video, k, v)
        
        if "output_dir" in data:
            config.output_dir = Path(data["output_dir"])
        
        return config
    
    @classmethod
    def default(cls) -> "Config":
        """Return default configuration."""
        return cls()

