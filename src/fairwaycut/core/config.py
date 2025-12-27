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
    
    # Fusion weights
    audio_weight: float = 0.6
    pose_weight: float = 0.4
    
    # Temporal alignment
    max_alignment_offset_sec: float = 0.5  # Max audio-pose alignment drift
    
    # Confidence thresholds
    min_combined_confidence: float = 0.4
    
    # Segment boundaries
    pre_impact_sec: float = 3.0  # Time before impact to include
    post_impact_sec: float = 2.0  # Time after impact to include


class VisualizationStyle(Enum):
    """Visualization style presets for skeleton rendering.
    
    MINIMAL: Clean glow skeleton, no trails - fast rendering
    STANDARD: Glow + short trails + phase-aware colors
    CINEMATIC: Full effects with long trails, depth coloring, particles
    LEGACY: Basic lines and circles (no glow effects) - fastest
    """
    MINIMAL = "minimal"
    STANDARD = "standard"
    CINEMATIC = "cinematic"
    LEGACY = "legacy"


@dataclass
class VisualizationConfig:
    """Configuration for skeleton visualization effects."""
    
    # Style preset
    style: VisualizationStyle = VisualizationStyle.STANDARD
    
    # Glow settings
    enable_glow: bool = True
    glow_intensity: float = 1.0  # 0.5-2.0 multiplier
    
    # Motion trails
    enable_trails: bool = True
    trail_length: int = 12  # Number of frames to show
    trail_landmarks: list[int] = None  # Default: wrists [15, 16]
    
    # Depth coloring
    enable_depth_coloring: bool = False
    near_color: tuple[int, int, int] = (0, 100, 255)   # Warm BGR (close)
    far_color: tuple[int, int, int] = (255, 100, 0)    # Cool BGR (far)
    
    # Velocity effects
    enable_velocity_intensity: bool = True
    
    # Phase colors
    enable_phase_colors: bool = True
    
    # Joint styling
    joint_style: str = "circle"  # "circle", "diamond", "hexagon"
    
    # Bone styling
    bone_style: str = "line"  # "line", "capsule"
    
    # Impact effects
    enable_impact_particles: bool = True
    
    def __post_init__(self):
        if self.trail_landmarks is None:
            self.trail_landmarks = [15, 16]  # Wrists
    
    @classmethod
    def from_style(cls, style: VisualizationStyle) -> "VisualizationConfig":
        """Create config from a preset style."""
        if style == VisualizationStyle.LEGACY:
            return cls(
                style=style,
                enable_glow=False,
                enable_trails=False,
                enable_depth_coloring=False,
                enable_velocity_intensity=False,
                enable_phase_colors=False,
                enable_impact_particles=False,
            )
        elif style == VisualizationStyle.MINIMAL:
            return cls(
                style=style,
                enable_glow=True,
                enable_trails=False,
                enable_depth_coloring=False,
                enable_velocity_intensity=False,
                enable_phase_colors=True,
                joint_style="circle",
                bone_style="line",
            )
        elif style == VisualizationStyle.CINEMATIC:
            return cls(
                style=style,
                enable_glow=True,
                glow_intensity=1.2,
                enable_trails=True,
                trail_length=18,
                enable_depth_coloring=True,
                enable_velocity_intensity=True,
                enable_phase_colors=True,
                joint_style="diamond",
                bone_style="capsule",
                enable_impact_particles=True,
            )
        else:  # STANDARD
            return cls(style=style)


@dataclass
class VideoConfig:
    """Configuration for video generation."""
    
    # Output settings
    output_fps: float = 30.0
    output_codec: str = "libx264"
    output_quality: int = 23  # CRF value (lower = better quality)
    
    # Overlay settings
    skeleton_color: tuple[int, int, int] = (0, 255, 128)  # BGR green
    skeleton_thickness: int = 2
    landmark_radius: int = 4
    
    # Enhanced visualization
    visualization_style: VisualizationStyle = VisualizationStyle.STANDARD
    visualization_config: VisualizationConfig = None
    
    # Audio waveform overlay
    waveform_height: int = 80
    waveform_color: tuple[int, int, int] = (78, 204, 163)  # Teal
    waveform_bg_color: tuple[int, int, int] = (22, 33, 62)  # Dark blue
    
    # Phase label
    phase_font_scale: float = 1.0
    phase_color: tuple[int, int, int] = (255, 255, 255)
    
    def __post_init__(self):
        if self.visualization_config is None:
            self.visualization_config = VisualizationConfig.from_style(
                self.visualization_style
            )


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

