"""Base protocol for pose estimation backends.

This module defines the common interface that all pose estimation backends
must implement, enabling platform-specific optimizations while maintaining
a unified API for the rest of the application.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from fairwaycut.core.models import FramePose, PoseAnalysisResult


class PoseBackend(ABC):
    """
    Abstract base class for pose estimation backends.
    
    All pose estimation implementations (Apple Vision, MediaPipe, etc.)
    must implement this interface to ensure compatibility with the
    rest of the FairwayCut system.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend (e.g., 'apple_vision', 'mediapipe')."""
        pass
    
    @property
    @abstractmethod
    def num_landmarks(self) -> int:
        """Return the number of landmarks this backend produces."""
        pass
    
    @property
    @abstractmethod
    def supports_gpu(self) -> bool:
        """Return True if this backend uses GPU/hardware acceleration."""
        pass
    
    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> FramePose:
        """
        Process a single video frame and extract pose landmarks.
        
        Args:
            frame: BGR image as numpy array (OpenCV format).
            frame_index: Index of the frame in the video.
            timestamp: Timestamp of the frame in seconds.
        
        Returns:
            FramePose with extracted landmarks (normalized 0-1 coordinates).
        """
        pass
    
    @abstractmethod
    def process_video(
        self,
        video_path: str | Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        process_every_n: int = 1,
        max_frames: Optional[int] = None,
    ) -> PoseAnalysisResult:
        """
        Process an entire video and extract poses for all frames.
        
        Args:
            video_path: Path to the video file.
            progress_callback: Optional callback(current_frame, total_frames).
            process_every_n: Process every Nth frame (for performance).
            max_frames: Maximum number of frames to process (None = all).
        
        Returns:
            PoseAnalysisResult containing all frame poses.
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video cannot be opened.
        """
        pass
    
    @abstractmethod
    def process_video_segment(
        self,
        video_path: str | Path,
        start_time: float,
        end_time: float,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        process_every_n: int = 1,
    ) -> PoseAnalysisResult:
        """
        Process a segment of a video between start and end times.
        
        Args:
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.
            process_every_n: Process every Nth frame.
        
        Returns:
            PoseAnalysisResult for the segment.
        """
        pass

    def process_video_segments(
        self,
        video_path: str | Path,
        segments: list[tuple[float, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        process_every_n: int = 1,
    ) -> list[PoseAnalysisResult]:
        """
        Process multiple video segments.

        Backends can override this to reuse a single open capture/session.
        The default implementation falls back to processing each segment
        independently.
        """
        results: list[PoseAnalysisResult] = []
        total_segments = len(segments)

        for index, (start_time, end_time) in enumerate(segments):
            segment_progress = None
            if progress_callback is not None:
                segment_progress = lambda current, total, idx=index: progress_callback(  # noqa: E731
                    idx * max(total, 1) + current,
                    max(total_segments, 1) * max(total, 1),
                )

            results.append(
                self.process_video_segment(
                    video_path,
                    start_time,
                    end_time,
                    progress_callback=segment_progress,
                    process_every_n=process_every_n,
                )
            )

        return results
    
    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()


# Standard landmark indices for cross-backend compatibility
# Maps conceptual body parts to a standardized index scheme
# Backends should map their native landmarks to these indices
STANDARD_LANDMARK_INDICES = {
    # Face
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    
    # Torso
    "neck": 5,
    "left_shoulder": 6,
    "right_shoulder": 7,
    "left_hip": 8,
    "right_hip": 9,
    
    # Left arm
    "left_elbow": 10,
    "left_wrist": 11,
    
    # Right arm
    "right_elbow": 12,
    "right_wrist": 13,
    
    # Left leg
    "left_knee": 14,
    "left_ankle": 15,
    
    # Right leg
    "right_knee": 16,
    "right_ankle": 17,
}

# Minimum landmarks required for golf swing analysis
GOLF_REQUIRED_LANDMARKS = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]
