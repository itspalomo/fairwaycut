"""Apple Vision Framework pose estimation backend.

This backend uses Apple's Vision framework via PyObjC for hardware-accelerated
pose estimation on macOS and iOS devices. It leverages the Neural Engine and
Metal for significantly faster processing compared to CPU-based alternatives.

Performance Notes:
- Uses Apple Neural Engine + Metal for hardware acceleration
- Typically 30-60+ FPS on Apple Silicon (vs 3-10 FPS with MediaPipe CPU)
- Provides 19 body landmarks (fewer than MediaPipe's 33, but sufficient for golf)
- Only available on macOS 11+ and iOS 14+

Requirements:
- macOS 11.0+ or iOS 14.0+
- pyobjc-framework-Vision package
"""

from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

from fairwaycut.core.models import FramePose, Landmark, PoseAnalysisResult
from fairwaycut.pose.backends.base import PoseBackend


# Check Apple Vision availability
APPLE_VISION_AVAILABLE = False
Vision = None
Quartz = None

try:
    import platform
    if platform.system() == "Darwin":
        import Vision
        import Quartz
        from Foundation import NSData
        APPLE_VISION_AVAILABLE = True
except ImportError:
    pass


# Apple Vision joint names as they appear in PyObjC
APPLE_VISION_JOINT_NAMES = [
    "head_joint",            # nose/head
    "left_eye_joint",        # left eye
    "right_eye_joint",       # right eye
    "left_ear_joint",        # left ear
    "right_ear_joint",       # right ear
    "left_shoulder_1_joint", # left shoulder
    "right_shoulder_1_joint",# right shoulder
    "left_forearm_joint",    # left elbow
    "right_forearm_joint",   # right elbow
    "left_hand_joint",       # left wrist
    "right_hand_joint",      # right wrist
    "left_upLeg_joint",      # left hip
    "right_upLeg_joint",     # right hip
    "left_leg_joint",        # left knee
    "right_leg_joint",       # right knee
    "left_foot_joint",       # left ankle
    "right_foot_joint",      # right ankle
    "neck_1_joint",          # neck
    "root",                  # center of hips (pelvis)
]

# Mapping from Apple Vision joint names to MediaPipe landmark indices (0-32)
# This allows Apple Vision output to be compatible with existing overlay code
# MediaPipe has 33 landmarks, Apple Vision has 19 - we map what we have
APPLE_TO_MEDIAPIPE_INDEX = {
    "head_joint": 0,             # MediaPipe: nose
    "left_eye_joint": 2,         # MediaPipe: left_eye (use middle eye, skip inner/outer)
    "right_eye_joint": 5,        # MediaPipe: right_eye
    "left_ear_joint": 7,         # MediaPipe: left_ear
    "right_ear_joint": 8,        # MediaPipe: right_ear
    "left_shoulder_1_joint": 11, # MediaPipe: left_shoulder
    "right_shoulder_1_joint": 12,# MediaPipe: right_shoulder
    "left_forearm_joint": 13,    # MediaPipe: left_elbow
    "right_forearm_joint": 14,   # MediaPipe: right_elbow
    "left_hand_joint": 15,       # MediaPipe: left_wrist
    "right_hand_joint": 16,      # MediaPipe: right_wrist
    "left_upLeg_joint": 23,      # MediaPipe: left_hip
    "right_upLeg_joint": 24,     # MediaPipe: right_hip
    "left_leg_joint": 25,        # MediaPipe: left_knee
    "right_leg_joint": 26,       # MediaPipe: right_knee
    "left_foot_joint": 27,       # MediaPipe: left_ankle
    "right_foot_joint": 28,      # MediaPipe: right_ankle
    "neck_1_joint": None,        # No direct MediaPipe equivalent (skip)
    "root": None,                # No direct MediaPipe equivalent (skip)
}

# Total MediaPipe landmarks for compatibility
MEDIAPIPE_LANDMARK_COUNT = 33


def is_available() -> bool:
    """Check if Apple Vision backend is available."""
    return APPLE_VISION_AVAILABLE


class AppleVisionBackend(PoseBackend):
    """
    Apple Vision Framework pose estimation backend.
    
    This backend uses Apple's native Vision framework for pose detection,
    providing hardware-accelerated inference on the Neural Engine and Metal.
    It's significantly faster than CPU-based alternatives on Apple Silicon.
    
    The backend provides 19 body landmarks which cover all joints needed
    for golf swing analysis (shoulders, elbows, wrists, hips, knees, ankles).
    """
    
    def __init__(
        self,
        max_frame_size: int = 640,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the Apple Vision pose backend.
        
        Args:
            max_frame_size: Maximum frame dimension (resizes larger frames).
            min_confidence: Minimum confidence threshold for landmarks.
        """
        if not APPLE_VISION_AVAILABLE:
            raise ImportError(
                "Apple Vision framework is not available. "
                "This backend requires macOS 11+ plus the PyObjC Vision and Quartz frameworks. "
                "Install with: pip install pyobjc-framework-Vision pyobjc-framework-Quartz"
            )
        
        self.max_frame_size = max_frame_size
        self.min_confidence = min_confidence
    
    @property
    def name(self) -> str:
        """Return the name of this backend."""
        return "apple_vision"
    
    @property
    def num_landmarks(self) -> int:
        """Return the number of landmarks (33 for MediaPipe compatibility)."""
        return MEDIAPIPE_LANDMARK_COUNT
    
    @property
    def supports_gpu(self) -> bool:
        """Apple Vision uses Neural Engine and Metal for acceleration."""
        return True
    
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
    
    def _numpy_to_cgimage(self, frame: np.ndarray):
        """Convert numpy BGR array to CGImage for Vision framework."""
        # Convert BGR to RGBA (32-bit for proper byte alignment)
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        h, w, c = rgba_frame.shape
        
        # Create CGImage from numpy array
        bytes_per_row = w * 4  # 4 bytes per pixel (RGBA)
        color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        
        # Create data provider from numpy bytes
        data = rgba_frame.tobytes()
        ns_data = NSData.dataWithBytes_length_(data, len(data))
        provider = Quartz.CGDataProviderCreateWithCFData(ns_data)
        
        cg_image = Quartz.CGImageCreate(
            w, h,
            8,   # bits per component
            32,  # bits per pixel (RGBA)
            bytes_per_row,
            color_space,
            Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNoneSkipLast,
            provider,
            None,  # decode array
            False,  # should interpolate
            Quartz.kCGRenderingIntentDefault,
        )
        
        return cg_image
    
    def _extract_landmarks_from_observation(
        self,
        observation,
        frame_height: int,
    ) -> tuple[list[Landmark], float]:
        """
        Extract landmarks from a VNHumanBodyPoseObservation.
        
        Outputs landmarks in MediaPipe-compatible format (33 landmarks)
        for compatibility with existing overlay and analysis code.
        
        Args:
            observation: VNHumanBodyPoseObservation from Vision framework.
            frame_height: Height of the frame (for Y coordinate flip).
        
        Returns:
            Tuple of (landmarks list in MediaPipe format, average confidence).
        """
        # Initialize all 33 MediaPipe landmarks as empty placeholders
        landmarks = [
            Landmark(x=0.0, y=0.0, z=0.0, visibility=0.0)
            for _ in range(MEDIAPIPE_LANDMARK_COUNT)
        ]
        confidences = []
        
        # Extract Apple Vision joints and map to MediaPipe indices
        for joint_name in APPLE_VISION_JOINT_NAMES:
            mediapipe_idx = APPLE_TO_MEDIAPIPE_INDEX.get(joint_name)
            if mediapipe_idx is None:
                continue  # Skip joints without MediaPipe equivalent
            
            try:
                point, error = observation.recognizedPointForJointName_error_(
                    joint_name, None
                )
                
                if point and point.confidence() >= self.min_confidence:
                    # Vision framework uses bottom-left origin, normalize to 0-1
                    # and flip Y to match standard image coordinates (top-left origin)
                    x = point.location().x
                    y = 1.0 - point.location().y  # Flip Y coordinate
                    
                    landmarks[mediapipe_idx] = Landmark(
                        x=x,
                        y=y,
                        z=0.0,  # Apple Vision 2D doesn't provide depth
                        visibility=float(point.confidence()),
                    )
                    confidences.append(point.confidence())
            except Exception:
                pass  # Keep placeholder for this joint
        
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        return landmarks, avg_confidence
    
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
        h, w = resized.shape[:2]
        
        # Convert to CGImage
        cg_image = self._numpy_to_cgimage(resized)
        
        if cg_image is None:
            return FramePose(
                frame_index=frame_index,
                timestamp=timestamp,
                landmarks=[],
                confidence=0.0,
            )
        
        # Create pose detection request
        request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        
        # Create request handler
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None
        )
        
        # Perform the request
        success, error = handler.performRequests_error_([request], None)
        
        landmarks = []
        confidence = 0.0
        
        if success and request.results() and len(request.results()) > 0:
            # Get the first detected pose
            observation = request.results()[0]
            landmarks, confidence = self._extract_landmarks_from_observation(
                observation, h
            )
        
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
        """Release any resources (Apple Vision doesn't require explicit cleanup)."""
        pass
