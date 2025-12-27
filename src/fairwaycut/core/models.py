"""Core data models for FairwayCut."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class SwingPhase(Enum):
    """Phases of a golf swing."""
    
    IDLE = "idle"
    ADDRESS = "address"
    BACKSWING = "backswing"
    TOP = "top"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"


@dataclass
class AudioData:
    """Container for extracted audio data."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    source_file: str

    @property
    def num_samples(self) -> int:
        """Return the number of audio samples."""
        return len(self.samples)
    
    def get_segment(self, start_sec: float, end_sec: float) -> "AudioData":
        """Extract a segment of audio between start and end times."""
        start_idx = int(start_sec * self.sample_rate)
        end_idx = int(end_sec * self.sample_rate)
        start_idx = max(0, start_idx)
        end_idx = min(len(self.samples), end_idx)
        
        return AudioData(
            samples=self.samples[start_idx:end_idx],
            sample_rate=self.sample_rate,
            duration=end_sec - start_sec,
            source_file=self.source_file,
        )


@dataclass
class Landmark:
    """A single pose landmark with 3D coordinates and visibility."""
    
    x: float  # Normalized x coordinate (0-1)
    y: float  # Normalized y coordinate (0-1)
    z: float  # Depth estimate
    visibility: float  # Confidence of detection (0-1)
    
    def to_pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return int(self.x * width), int(self.y * height)


@dataclass
class FramePose:
    """Pose data for a single video frame."""
    
    frame_index: int
    timestamp: float  # Time in seconds
    landmarks: list[Landmark]  # 33 MediaPipe pose landmarks
    confidence: float = 0.0  # Overall pose detection confidence
    
    @property
    def is_valid(self) -> bool:
        """Check if pose was successfully detected."""
        return len(self.landmarks) > 0 and self.confidence > 0.5
    
    def get_landmark(self, index: int) -> Optional[Landmark]:
        """Get a specific landmark by index."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None


@dataclass
class ImpactEvent:
    """A detected impact event from audio analysis."""

    timestamp: float  # Time in seconds
    confidence: float  # Confidence score (0-1)
    amplitude_db: float  # Peak amplitude in dB
    onset_strength: float = 0.0  # Onset/attack strength
    spectral_flux: float = 0.0  # Spectral change rate
    is_transient: bool = True  # Whether this appears to be a true transient

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": round(self.timestamp, 3),
            "confidence": round(self.confidence, 3),
            "amplitude_db": round(self.amplitude_db, 2),
            "onset_strength": round(self.onset_strength, 2),
            "spectral_flux": round(self.spectral_flux, 2),
            "is_transient": bool(self.is_transient),
        }


@dataclass
class SwingEvent:
    """A verified swing event combining audio and pose data."""
    
    swing_id: int  # Sequential swing number
    impact_time: float  # Time of ball impact in seconds
    
    # Time boundaries
    start_time: float  # Start of swing segment
    end_time: float  # End of swing segment
    
    # Detection confidence
    audio_confidence: float = 0.0  # Confidence from audio detection
    pose_confidence: float = 0.0  # Confidence from pose detection
    combined_confidence: float = 0.0  # Fused confidence score
    
    # Phase timing (optional, populated by pose analysis)
    address_time: Optional[float] = None
    backswing_start: Optional[float] = None
    top_time: Optional[float] = None
    downswing_start: Optional[float] = None
    follow_through_start: Optional[float] = None
    finish_time: Optional[float] = None
    
    # Metadata
    source_file: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "swing_id": self.swing_id,
            "impact_time": round(self.impact_time, 3),
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "audio_confidence": round(self.audio_confidence, 3),
            "pose_confidence": round(self.pose_confidence, 3),
            "combined_confidence": round(self.combined_confidence, 3),
            "phases": {
                "address": round(self.address_time, 3) if self.address_time else None,
                "backswing_start": round(self.backswing_start, 3) if self.backswing_start else None,
                "top": round(self.top_time, 3) if self.top_time else None,
                "downswing_start": round(self.downswing_start, 3) if self.downswing_start else None,
                "impact": round(self.impact_time, 3),
                "follow_through_start": round(self.follow_through_start, 3) if self.follow_through_start else None,
                "finish": round(self.finish_time, 3) if self.finish_time else None,
            },
        }
    
    @property
    def duration(self) -> float:
        """Total duration of the swing segment."""
        return self.end_time - self.start_time


@dataclass
class DetectionResult:
    """Result of impact detection from audio analysis."""

    events: list[ImpactEvent]
    parameters: dict
    envelope: np.ndarray
    envelope_times: np.ndarray
    envelope_db: np.ndarray

    @property
    def timestamps(self) -> list[float]:
        """Return list of event timestamps."""
        return [e.timestamp for e in self.events]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "events": [e.to_dict() for e in self.events],
            "parameters": self.parameters,
            "num_events": len(self.events),
        }


@dataclass
class PoseAnalysisResult:
    """Result of pose estimation on a video."""
    
    frames: list[FramePose]
    fps: float
    total_frames: int
    video_duration: float
    source_file: str
    
    @property
    def valid_frames(self) -> list[FramePose]:
        """Return only frames with valid pose detection."""
        return [f for f in self.frames if f.is_valid]
    
    @property
    def detection_rate(self) -> float:
        """Percentage of frames with successful pose detection."""
        if not self.frames:
            return 0.0
        return len(self.valid_frames) / len(self.frames)
    
    def get_frame_at_time(self, timestamp: float) -> Optional[FramePose]:
        """Get the pose data for a specific timestamp."""
        frame_idx = int(timestamp * self.fps)
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None


@dataclass
class FusionResult:
    """Result of multi-modal swing detection."""
    
    swings: list[SwingEvent]
    audio_result: DetectionResult
    pose_result: Optional[PoseAnalysisResult]
    parameters: dict = field(default_factory=dict)
    
    @property
    def num_swings(self) -> int:
        """Number of detected swings."""
        return len(self.swings)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "num_swings": self.num_swings,
            "swings": [s.to_dict() for s in self.swings],
            "audio_events": len(self.audio_result.events),
            "pose_detection_rate": self.pose_result.detection_rate if self.pose_result else None,
            "parameters": self.parameters,
        }

