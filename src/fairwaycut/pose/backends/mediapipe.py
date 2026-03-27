"""MediaPipe Pose estimation backend.

Performance Notes:
- MediaPipe Python on macOS does NOT support GPU acceleration (Metal/CoreML)
- Processing speed on CPU is typically 3-10 FPS depending on model complexity
- For long videos, use process_every_n > 1 or segment-based processing
- The lite model (complexity=0) is ~3x faster than full model

This backend serves as the cross-platform fallback when native acceleration
is not available (e.g., on Windows, Linux, or when Apple Vision is unavailable).
"""

from pathlib import Path
from typing import Optional, Callable
import urllib.request

import cv2
import numpy as np

from fairwaycut.core.models import FramePose, Landmark, PoseAnalysisResult
from fairwaycut.pose.backends.base import PoseBackend


# Check MediaPipe availability
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    python = None
    vision = None


# Model URLs - lite is ~3x faster, full is more accurate
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
POSE_MODEL_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_FULL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


def get_model_path(model_complexity: int = 1) -> Path:
    """Get the path to the pose landmarker model, downloading if necessary."""
    cache_dir = Path.home() / ".cache" / "fairwaycut" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Select model based on complexity
    if model_complexity == 0:
        url = POSE_MODEL_LITE_URL
        filename = "pose_landmarker_lite.task"
    elif model_complexity == 2:
        url = POSE_MODEL_URL  # Heavy model
        filename = "pose_landmarker_heavy.task"
    else:
        url = POSE_MODEL_FULL_URL
        filename = "pose_landmarker_full.task"
    
    model_path = cache_dir / filename
    
    if not model_path.exists():
        print(f"Downloading pose model to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    
    return model_path


def is_available() -> bool:
    """Check if MediaPipe backend is available."""
    return MEDIAPIPE_AVAILABLE


class MediaPipeBackend(PoseBackend):
    """
    MediaPipe-based pose estimation backend.
    
    This backend uses Google's MediaPipe for pose detection. It provides
    33 body landmarks and works on all platforms, but runs on CPU only
    on macOS (no Metal/CoreML support in Python bindings).
    
    Performance tips:
    - Use model_complexity=0 (lite) for ~3x faster processing
    - Use max_frame_size to resize large frames before processing
    - Use process_every_n to skip frames (interpolate for smooth overlays)
    - Process only segments around detected impacts, not full video
    """
    
    def __init__(
        self,
        model_complexity: int = 0,  # Default to lite for speed
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        max_frame_size: int = 640,  # Resize frames for faster processing
    ):
        """
        Initialize the MediaPipe pose backend.
        
        Args:
            model_complexity: 0 (lite/fast), 1 (full), or 2 (heavy/accurate).
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            enable_segmentation: Whether to output segmentation mask.
            max_frame_size: Maximum frame dimension (resizes larger frames).
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )
        
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.max_frame_size = max_frame_size
        
        # Get model path (downloads if needed)
        model_path = get_model_path(model_complexity)
        
        # Initialize MediaPipe Pose Landmarker with new Tasks API
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=enable_segmentation,
            num_poses=1,  # Only detect one person for golf
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
    
    @property
    def name(self) -> str:
        """Return the name of this backend."""
        return "mediapipe"
    
    @property
    def num_landmarks(self) -> int:
        """Return the number of landmarks this backend produces (MediaPipe: 33)."""
        return 33
    
    @property
    def supports_gpu(self) -> bool:
        """MediaPipe Python does not support GPU on macOS."""
        return False
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if larger than max_frame_size for faster processing."""
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_frame_size:
            return frame
        
        scale = self.max_frame_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def process_frame(
        self,
        frame: np.ndarray,
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
        # Resize for faster processing
        resized = self._resize_frame(frame)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame (timestamp in milliseconds)
        timestamp_ms = int(timestamp * 1000)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        landmarks = []
        confidence = 0.0
        
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            # Get first detected pose
            pose_landmarks = results.pose_landmarks[0]
            
            # Extract all 33 landmarks (coordinates are normalized 0-1)
            for lm in pose_landmarks:
                landmarks.append(Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0,
                ))
            
            # Calculate overall confidence as average visibility
            visibilities = [lm.visibility for lm in landmarks]
            confidence = float(np.mean(visibilities))
        
        return FramePose(
            frame_index=frame_index,
            timestamp=timestamp,
            landmarks=landmarks,
            confidence=confidence,
        )
    
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
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frames: list[FramePose] = []
        frame_index = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if max_frames and frame_index >= max_frames:
                    break
                
                # Calculate timestamp
                timestamp = frame_index / fps if fps > 0 else 0
                
                # Process frame (or skip if using process_every_n)
                if frame_index % process_every_n == 0:
                    pose = self.process_frame(frame, frame_index, timestamp)
                    frames.append(pose)
                else:
                    # Create empty placeholder for skipped frames
                    frames.append(FramePose(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        landmarks=[],
                        confidence=0.0,
                    ))
                
                # Progress callback
                if progress_callback and frame_index % 30 == 0:
                    progress_callback(frame_index, total_frames)
                
                frame_index += 1
        
        finally:
            cap.release()
        
        return PoseAnalysisResult(
            frames=frames,
            fps=fps,
            total_frames=len(frames),
            video_duration=duration,
            source_file=str(video_path),
        )
    
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
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames: list[FramePose] = []
        frame_index = start_frame
        
        try:
            while cap.isOpened() and frame_index < end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_index / fps if fps > 0 else 0
                
                if (frame_index - start_frame) % process_every_n == 0:
                    pose = self.process_frame(frame, frame_index, timestamp)
                    frames.append(pose)
                else:
                    frames.append(FramePose(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        landmarks=[],
                        confidence=0.0,
                    ))
                
                frame_index += 1
                
                # Progress callback
                if progress_callback and frame_index % 30 == 0:
                    segment_frames = end_frame - start_frame
                    current_frames = frame_index - start_frame
                    progress_callback(current_frames, segment_frames)
        
        finally:
            cap.release()
        
        return PoseAnalysisResult(
            frames=frames,
            fps=fps,
            total_frames=len(frames),
            video_duration=end_time - start_time,
            source_file=str(video_path),
        )

    def process_video_segments(
        self,
        video_path: str | Path,
        segments: list[tuple[float, float]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        process_every_n: int = 1,
    ) -> list[PoseAnalysisResult]:
        """Process multiple video segments while reusing one open capture."""
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if not segments:
            return []

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_segment_frames = sum(
            max(0, int(end_time * fps) - int(start_time * fps))
            for start_time, end_time in segments
        )
        processed_frames = 0
        results: list[PoseAnalysisResult] = []

        try:
            for start_time, end_time in segments:
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                frames: list[FramePose] = []
                frame_index = start_frame

                while cap.isOpened() and frame_index < end_frame:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    timestamp = frame_index / fps if fps > 0 else 0

                    if (frame_index - start_frame) % process_every_n == 0:
                        pose = self.process_frame(frame, frame_index, timestamp)
                        frames.append(pose)
                    else:
                        frames.append(
                            FramePose(
                                frame_index=frame_index,
                                timestamp=timestamp,
                                landmarks=[],
                                confidence=0.0,
                            )
                        )

                    frame_index += 1
                    processed_frames += 1

                    if progress_callback and processed_frames % 30 == 0:
                        progress_callback(processed_frames, total_segment_frames)

                results.append(
                    PoseAnalysisResult(
                        frames=frames,
                        fps=fps,
                        total_frames=len(frames),
                        video_duration=end_time - start_time,
                        source_file=str(video_path),
                    )
                )
        finally:
            cap.release()

        return results
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
