"""Pose estimation wrapper for golf swing analysis.

This module provides a unified interface for pose estimation, automatically
selecting the best available backend for the current platform:

- macOS: Apple Vision Framework (hardware-accelerated via Neural Engine)
- Other platforms: MediaPipe (CPU-based, cross-platform)

Performance Notes:
- Apple Vision on macOS: 30-60+ FPS on Apple Silicon
- MediaPipe CPU: 3-10 FPS depending on model complexity
- For long videos, use process_every_n > 1 or segment-based processing
"""

from pathlib import Path
from typing import Optional, Callable

from fairwaycut.core.models import FramePose, PoseAnalysisResult
from fairwaycut.pose.backends import (
    PoseBackend,
    create_backend,
    get_available_backends,
)


class PoseEstimator:
    """
    Unified pose estimation interface for golf swing analysis.
    
    This class automatically selects the best available pose estimation
    backend for the current platform, providing hardware acceleration
    on macOS via Apple Vision while falling back to MediaPipe elsewhere.
    
    The API remains consistent regardless of which backend is used.
    
    Example:
        # Auto-select best backend for platform
        with PoseEstimator() as estimator:
            result = estimator.process_video("swing.mp4")
        
        # Force a specific backend
        with PoseEstimator(backend="mediapipe") as estimator:
            result = estimator.process_video("swing.mp4")
    """
    
    def __init__(
        self,
        backend: Optional[str] = None,
        prefer_native: bool = True,
        # Common options
        max_frame_size: int = 640,
        min_confidence: float = 0.5,
        # MediaPipe-specific options (ignored by Apple Vision)
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the pose estimator with automatic backend selection.
        
        Args:
            backend: Explicitly request a backend ("apple_vision" or "mediapipe").
                    If None, automatically selects the best available backend.
            prefer_native: If True, prefer platform-native backends.
            max_frame_size: Maximum frame dimension (resizes larger frames).
            min_confidence: Minimum confidence threshold for landmarks (Apple Vision).
            model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy).
            min_detection_confidence: MediaPipe detection confidence threshold.
            min_tracking_confidence: MediaPipe tracking confidence threshold.
        """
        # Create the backend - factory filters kwargs automatically
        self._backend = create_backend(
            prefer_native=prefer_native,
            backend_name=backend,
            max_frame_size=max_frame_size,
            min_confidence=min_confidence,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    
    @property
    def backend_name(self) -> str:
        """Return the name of the active backend."""
        return self._backend.name
    
    @property
    def num_landmarks(self) -> int:
        """Return the number of landmarks produced by the active backend."""
        return self._backend.num_landmarks
    
    @property
    def supports_gpu(self) -> bool:
        """Return True if the active backend uses GPU/hardware acceleration."""
        return self._backend.supports_gpu
    
    def process_frame(
        self,
        frame,
        frame_index: int,
        timestamp: float,
    ) -> FramePose:
        """
        Process a single video frame and extract pose landmarks.
        
        Args:
            frame: BGR image as numpy array.
            frame_index: Index of the frame in the video.
            timestamp: Timestamp of the frame in seconds.
        
        Returns:
            FramePose with extracted landmarks (normalized 0-1 coordinates).
        """
        return self._backend.process_frame(frame, frame_index, timestamp)
    
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
        return self._backend.process_video(
            video_path,
            progress_callback=progress_callback,
            process_every_n=process_every_n,
            max_frames=max_frames,
        )
    
    def process_video_segment(
        self,
        video_path: str | Path,
        start_time: float,
        end_time: float,
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
        return self._backend.process_video_segment(
            video_path,
            start_time,
            end_time,
            process_every_n=process_every_n,
        )
    
    def close(self):
        """Release backend resources."""
        self._backend.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def estimate_poses(
    video_path: str | Path,
    backend: Optional[str] = None,
    process_every_n: int = 1,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    **kwargs,
) -> PoseAnalysisResult:
    """
    Convenience function to estimate poses from a video.
    
    Args:
        video_path: Path to the video file.
        backend: Backend to use ("apple_vision", "mediapipe", or None for auto).
        process_every_n: Process every Nth frame for performance.
        progress_callback: Optional callback for progress updates.
        **kwargs: Additional arguments passed to PoseEstimator.
    
    Returns:
        PoseAnalysisResult with all detected poses.
    
    Example:
        # Auto-select backend
        result = estimate_poses("swing.mp4")
        
        # Force MediaPipe
        result = estimate_poses("swing.mp4", backend="mediapipe")
    """
    with PoseEstimator(backend=backend, **kwargs) as estimator:
        return estimator.process_video(
            video_path,
            progress_callback=progress_callback,
            process_every_n=process_every_n,
        )
