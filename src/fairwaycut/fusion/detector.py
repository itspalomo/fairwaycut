"""Multi-modal swing detection combining audio and pose signals."""

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Optional, Callable

from fairwaycut.core.models import (
    AudioData,
    ImpactEvent,
    SwingEvent,
    DetectionResult,
    PoseAnalysisResult,
    FusionResult,
)
from fairwaycut.core.config import Config, FusionConfig, ProcessingMode
from fairwaycut.audio.extraction import extract_audio_from_video
from fairwaycut.audio.detection import detect_impacts_adaptive_snr
from fairwaycut.pose.backends import get_available_backends
from fairwaycut.pose.estimator import PoseEstimator
from fairwaycut.pose.swing_phases import SwingPhaseDetector, SwingPhasesResult


# Processing mode descriptions for CLI help
MODE_DESCRIPTIONS = {
    ProcessingMode.AUDIO: "Audio detection only - fastest, no pose estimation",
    ProcessingMode.HYBRID: "Audio + targeted pose around impacts - fast, good accuracy",
    ProcessingMode.LITE: "Full video with lite pose model - slower, catches missed audio",
    ProcessingMode.FULL: "Full video with full pose model - slowest, best accuracy",
}


@dataclass(frozen=True)
class MergedPoseWindow:
    """A merged pose-processing window plus the events it covers."""

    start_time: float
    end_time: float
    event_indices: tuple[int, ...]


class SwingDetector:
    """
    Multi-modal swing detector combining audio impact detection with pose estimation.
    
    This detector uses audio to find candidate impact times, then validates
    and enriches those detections with pose-based swing phase analysis.
    The fusion approach provides more accurate swing detection than either
    modality alone.
    """
    
    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        audio_weight: float = 0.6,
        pose_weight: float = 0.4,
    ):
        """
        Initialize the swing detector.
        
        Args:
            config: FusionConfig with detection parameters.
            audio_weight: Weight for audio-based confidence (0-1).
            pose_weight: Weight for pose-based confidence (0-1).
        """
        self.config = config or FusionConfig()
        self.audio_weight = audio_weight
        self.pose_weight = pose_weight
        
        # Normalize weights
        total = self.audio_weight + self.pose_weight
        self.audio_weight /= total
        self.pose_weight /= total
    
    def detect(
        self,
        audio_result: DetectionResult,
        pose_result: Optional[PoseAnalysisResult] = None,
        pre_impact_sec: float = 3.0,
        post_impact_sec: float = 2.0,
    ) -> FusionResult:
        """
        Detect swings by fusing audio impacts with pose data.
        
        Args:
            audio_result: DetectionResult from audio impact detection.
            pose_result: Optional PoseAnalysisResult from pose estimation.
            pre_impact_sec: Seconds to include before impact.
            post_impact_sec: Seconds to include after impact.
        
        Returns:
            FusionResult with verified swing events.
        """
        swings = []
        phase_detector = SwingPhaseDetector()
        
        for i, audio_event in enumerate(audio_result.events):
            # Define segment boundaries
            start_time = max(0, audio_event.timestamp - pre_impact_sec)
            end_time = audio_event.timestamp + post_impact_sec
            
            # Initialize confidence scores
            audio_confidence = audio_event.confidence
            pose_confidence = 0.0
            
            # Phase information
            phases_result: Optional[SwingPhasesResult] = None
            
            # If we have pose data, validate and enrich
            if pose_result is not None and pose_result.frames:
                # Extract poses for this segment
                segment_poses = self._get_segment_poses(
                    pose_result, start_time, end_time
                )
                
                if segment_poses:
                    # Create mini pose result for phase detection
                    segment_result = PoseAnalysisResult(
                        frames=segment_poses,
                        fps=pose_result.fps,
                        total_frames=len(segment_poses),
                        video_duration=end_time - start_time,
                        source_file=pose_result.source_file,
                    )
                    
                    # Detect swing phases with audio hint
                    phases_result = phase_detector.detect_phases(
                        segment_result,
                        expected_impact_time=audio_event.timestamp,
                    )
                    
                    # Calculate pose confidence
                    pose_confidence = self._calculate_pose_confidence(
                        phases_result, segment_result
                    )
            
            # Combine confidences
            combined_confidence = (
                self.audio_weight * audio_confidence +
                self.pose_weight * pose_confidence
            )
            
            # Create swing event
            swing = SwingEvent(
                swing_id=i + 1,
                impact_time=audio_event.timestamp,
                start_time=start_time,
                end_time=end_time,
                audio_confidence=audio_confidence,
                pose_confidence=pose_confidence,
                combined_confidence=combined_confidence,
                source_file=audio_result.parameters.get("source_file", ""),
            )
            
            # Add phase timing if available
            if phases_result and phases_result.transitions:
                swing = self._add_phase_timing(swing, phases_result)
            
            swings.append(swing)
        
        # Filter low confidence swings
        min_confidence = self.config.min_combined_confidence
        swings = [s for s in swings if s.combined_confidence >= min_confidence]
        
        return FusionResult(
            swings=swings,
            audio_result=audio_result,
            pose_result=pose_result,
            parameters={
                "audio_weight": self.audio_weight,
                "pose_weight": self.pose_weight,
                "pre_impact_sec": pre_impact_sec,
                "post_impact_sec": post_impact_sec,
                "min_combined_confidence": min_confidence,
            },
        )
    
    def _get_segment_poses(
        self,
        pose_result: PoseAnalysisResult,
        start_time: float,
        end_time: float,
    ) -> list:
        """Extract poses within a time segment."""
        return [
            f for f in pose_result.frames
            if start_time <= f.timestamp <= end_time
        ]
    
    def _calculate_pose_confidence(
        self,
        phases_result: SwingPhasesResult,
        segment_result: PoseAnalysisResult,
    ) -> float:
        """Calculate confidence score from pose analysis."""
        if not phases_result.has_complete_swing:
            # Partial detection - lower confidence
            return phases_result.confidence * 0.5
        
        # Full swing detected
        base_confidence = phases_result.confidence
        
        # Boost if pose detection rate is good
        detection_rate = segment_result.detection_rate
        if detection_rate > 0.8:
            base_confidence *= 1.1
        elif detection_rate < 0.5:
            base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def _add_phase_timing(
        self,
        swing: SwingEvent,
        phases_result: SwingPhasesResult,
    ) -> SwingEvent:
        """Add phase timing information to swing event."""
        for transition in phases_result.transitions:
            phase = transition.to_phase
            timestamp = transition.timestamp
            
            from fairwaycut.core.models import SwingPhase
            
            if phase == SwingPhase.ADDRESS:
                swing.address_time = timestamp
            elif phase == SwingPhase.BACKSWING:
                swing.backswing_start = timestamp
            elif phase == SwingPhase.TOP:
                swing.top_time = timestamp
            elif phase == SwingPhase.DOWNSWING:
                swing.downswing_start = timestamp
            elif phase == SwingPhase.FOLLOW_THROUGH:
                swing.follow_through_start = timestamp
            elif phase == SwingPhase.FINISH:
                swing.finish_time = timestamp
        
        return swing
    
    def detect_with_segments(
        self,
        audio_result: DetectionResult,
        segment_poses: dict[int, PoseAnalysisResult],
        pre_impact_sec: float = 3.0,
        post_impact_sec: float = 2.0,
    ) -> FusionResult:
        """
        Detect swings using pre-computed pose segments (optimized approach).
        
        Instead of processing the full video, this uses pose data that was
        extracted only around detected audio impacts.
        
        Args:
            audio_result: DetectionResult from audio impact detection.
            segment_poses: Dict mapping event index to PoseAnalysisResult for that segment.
            pre_impact_sec: Seconds to include before impact.
            post_impact_sec: Seconds to include after impact.
        
        Returns:
            FusionResult with verified swing events.
        """
        swings = []
        phase_detector = SwingPhaseDetector()
        
        for i, audio_event in enumerate(audio_result.events):
            # Define segment boundaries
            start_time = max(0, audio_event.timestamp - pre_impact_sec)
            end_time = audio_event.timestamp + post_impact_sec
            
            # Initialize confidence scores
            audio_confidence = audio_event.confidence
            pose_confidence = 0.0
            
            # Phase information
            phases_result: Optional[SwingPhasesResult] = None
            
            # Get pose data for this segment if available
            if i in segment_poses:
                segment_result = segment_poses[i]
                
                if segment_result and segment_result.frames:
                    # Detect swing phases with audio hint
                    phases_result = phase_detector.detect_phases(
                        segment_result,
                        expected_impact_time=audio_event.timestamp,
                    )
                    
                    # Calculate pose confidence
                    pose_confidence = self._calculate_pose_confidence(
                        phases_result, segment_result
                    )
            
            # Combine confidences
            combined_confidence = (
                self.audio_weight * audio_confidence +
                self.pose_weight * pose_confidence
            )
            
            # Create swing event
            swing = SwingEvent(
                swing_id=i + 1,
                impact_time=audio_event.timestamp,
                start_time=start_time,
                end_time=end_time,
                audio_confidence=audio_confidence,
                pose_confidence=pose_confidence,
                combined_confidence=combined_confidence,
                source_file=audio_result.parameters.get("source_file", ""),
            )
            
            # Add phase timing if available
            if phases_result and phases_result.transitions:
                swing = self._add_phase_timing(swing, phases_result)
            
            swings.append(swing)
        
        # Filter low confidence swings
        min_confidence = self.config.min_combined_confidence
        swings = [s for s in swings if s.combined_confidence >= min_confidence]
        
        # Combine all segment poses into one result for reference
        all_frames = []
        for pose_result in segment_poses.values():
            if pose_result:
                all_frames.extend(pose_result.frames)
        
        combined_pose_result = None
        if all_frames:
            first_result = next(iter(segment_poses.values()))
            combined_pose_result = PoseAnalysisResult(
                frames=all_frames,
                fps=first_result.fps if first_result else 30.0,
                total_frames=len(all_frames),
                video_duration=sum(p.video_duration for p in segment_poses.values() if p),
                source_file=first_result.source_file if first_result else "",
            )
        
        return FusionResult(
            swings=swings,
            audio_result=audio_result,
            pose_result=combined_pose_result,
            parameters={
                "audio_weight": self.audio_weight,
                "pose_weight": self.pose_weight,
                "pre_impact_sec": pre_impact_sec,
                "post_impact_sec": post_impact_sec,
                "min_combined_confidence": min_confidence,
                "optimization": "segment_based",
                "segments_processed": len(segment_poses),
            },
        )


def detect_swings(
    video_path: str | Path,
    mode: ProcessingMode = ProcessingMode.HYBRID,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    audio: Optional[AudioData] = None,
) -> FusionResult:
    """
    Detect swings in a video using multi-modal analysis.
    
    This is the main entry point for swing detection, with different
    processing modes for speed/accuracy tradeoffs:
    
    - AUDIO: Fastest, audio detection only
    - HYBRID: Fast, audio + pose only around detected impacts (recommended)
    - LITE: Slower, full video with lite pose model
    - FULL: Slowest, full video with full pose model
    
    Args:
        video_path: Path to the video file.
        mode: ProcessingMode - determines speed/accuracy tradeoff.
        config: Optional configuration object.
        progress_callback: Optional callback(stage, current, total).
        start_time: Start time in seconds (default: 0).
        end_time: End time in seconds (default: None = video end).
        audio: Optional pre-decoded audio segment to reuse.
    
    Returns:
        FusionResult with detected swings.
    """
    video_path = Path(video_path)
    config = config or Config.default()
    
    # Stage 1: Audio extraction and analysis (always done)
    if audio is None:
        if progress_callback:
            progress_callback("audio_extraction", 0, 1)

        audio = extract_audio_from_video(
            video_path,
            start_time=start_time if start_time > 0 else None,
            end_time=end_time,
        )

        if progress_callback:
            progress_callback("audio_extraction", 1, 1)
    else:
        requested_end = end_time if end_time is not None else audio.end_time
        actual_requested_end = min(requested_end, audio.end_time)
        if start_time > audio.start_time or actual_requested_end < audio.end_time:
            audio = audio.get_segment(start_time, actual_requested_end)

    actual_end = audio.end_time
    actual_start = audio.start_time
    
    if progress_callback:
        progress_callback("audio_detection", 0, 1)
    
    audio_result = detect_impacts_adaptive_snr(
        audio,
        min_gap_sec=config.audio.min_gap_sec,
        hop_length=config.audio.hop_length,
        snr_threshold=config.audio.snr_threshold,
        local_window_sec=config.audio.local_window_sec,
        min_flux=config.audio.min_flux,
        min_onset=config.audio.min_onset,
        amplitude_threshold_db=config.audio.amplitude_threshold_db,
    )
    if progress_callback:
        progress_callback("audio_detection", 1, 1)
    
    # Filter events to time range
    audio_result.events = [
        e for e in audio_result.events 
        if actual_start <= e.timestamp <= actual_end
    ]
    
    # Add source file to parameters
    audio_result.parameters["source_file"] = str(video_path)
    audio_result.parameters["mode"] = mode.value
    audio_result.parameters["start_time"] = actual_start
    audio_result.parameters["end_time"] = actual_end
    
    # Stage 2: Pose estimation (based on mode)
    pose_result: Optional[PoseAnalysisResult] = None
    segment_poses: dict[int, PoseAnalysisResult] = {}
    pose_routing: dict[str, object] = {}
    
    if mode == ProcessingMode.AUDIO:
        # No pose estimation - just use audio results
        pass
    
    elif mode == ProcessingMode.HYBRID:
        # Pose estimation only around detected audio impacts
        if audio_result.events:
            segment_poses, pose_routing = _process_pose_segments(
                video_path,
                audio_result,
                config,
                progress_callback,
                start_time=actual_start,
                end_time=actual_end,
            )
    
    elif mode in (ProcessingMode.LITE, ProcessingMode.FULL):
        # Full video pose estimation
        model_complexity = 0 if mode == ProcessingMode.LITE else 1
        pose_result, pose_routing = _process_full_video_pose(
            video_path,
            model_complexity,
            config,
            progress_callback,
            start_time=actual_start,
            end_time=actual_end,
        )
    
    # Stage 3: Fusion
    if progress_callback:
        progress_callback("fusion", 0, 1)
    
    detector = SwingDetector(
        config=config.fusion,
        audio_weight=config.fusion.audio_weight,
        pose_weight=config.fusion.pose_weight,
    )
    
    # Choose fusion method based on what data we have
    if pose_result is not None:
        # Full video pose - use standard fusion
        result = detector.detect(
            audio_result,
            pose_result,
            pre_impact_sec=config.fusion.pre_impact_sec,
            post_impact_sec=config.fusion.post_impact_sec,
        )
    elif segment_poses:
        # Segment-based pose - use segment fusion
        result = detector.detect_with_segments(
            audio_result,
            segment_poses,
            pre_impact_sec=config.fusion.pre_impact_sec,
            post_impact_sec=config.fusion.post_impact_sec,
        )
    else:
        # Audio only - create result without pose
        result = detector.detect(
            audio_result,
            None,
            pre_impact_sec=config.fusion.pre_impact_sec,
            post_impact_sec=config.fusion.post_impact_sec,
        )
    
    if progress_callback:
        progress_callback("fusion", 1, 1)
    
    if progress_callback:
        progress_callback("complete", 1, 1)

    result.parameters.update(
        {
            "mode": mode.value,
            "analysis_start_time": actual_start,
            "analysis_end_time": actual_end,
            **pose_routing,
        }
    )

    return result


def _process_pose_segments(
    video_path: Path,
    audio_result: DetectionResult,
    config: Config,
    progress_callback: Optional[Callable[[str, int, int], None]],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> tuple[dict[int, PoseAnalysisResult], dict[str, object]]:
    """Process pose estimation only around detected audio impacts."""
    segment_poses: dict[int, PoseAnalysisResult] = {}
    
    # Filter events to time range
    events = audio_result.events
    if end_time:
        events = [e for e in events if start_time <= e.timestamp <= end_time]
    
    if progress_callback:
        progress_callback("pose_estimation", 0, len(events))
    
    estimator = PoseEstimator(
        model_complexity=0,  # Use lite for segment mode
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
        max_frame_size=480,  # Smaller for speed
    )
    routing = _build_pose_routing_metadata(estimator)
    video_end = end_time if end_time is not None else max(
        (event.timestamp for event in events),
        default=start_time,
    )
    
    try:
        windows = _build_merged_pose_windows(
            events=events,
            pre_impact_sec=config.fusion.pre_impact_sec,
            post_impact_sec=config.fusion.post_impact_sec,
            video_start=start_time,
            video_end=video_end,
        )
        merged_results = estimator.process_video_segments(
            video_path,
            [(window.start_time, window.end_time) for window in windows],
            progress_callback=_create_pose_progress_callback(progress_callback),
            process_every_n=max(1, config.pose.process_every_n_frames),
        )

        for window, merged_result in zip(windows, merged_results):
            for event_index in window.event_indices:
                event = events[event_index]
                seg_start = max(0.0, event.timestamp - config.fusion.pre_impact_sec)
                seg_end = min(video_end, event.timestamp + config.fusion.post_impact_sec)
                segment_poses[event_index] = _slice_pose_result(
                    merged_result,
                    seg_start,
                    seg_end,
                )

        if progress_callback:
            progress_callback("pose_estimation", len(events), len(events))

        routing["segments_processed"] = len(segment_poses)
        routing["merged_pose_windows"] = len(windows)
    finally:
        estimator.close()

    return segment_poses, routing


def _process_full_video_pose(
    video_path: Path,
    model_complexity: int,
    config: Config,
    progress_callback: Optional[Callable[[str, int, int], None]],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> tuple[PoseAnalysisResult, dict[str, object]]:
    """Process pose estimation on full video or time range."""
    if progress_callback:
        progress_callback("pose_estimation", 0, 1)

    estimator = PoseEstimator(
        model_complexity=model_complexity,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
        max_frame_size=config.pose.max_frame_size,
    )
    routing = _build_pose_routing_metadata(estimator)

    try:
        if start_time > 0 or end_time is not None:
            pose_result = estimator.process_video_segment(
                video_path,
                start_time=start_time,
                end_time=end_time if end_time is not None else float("inf"),
                progress_callback=_create_pose_progress_callback(progress_callback),
                process_every_n=config.pose.process_every_n_frames,
            )
        else:
            pose_result = estimator.process_video(
                video_path,
                progress_callback=_create_pose_progress_callback(progress_callback),
                process_every_n=config.pose.process_every_n_frames,
            )
    finally:
        estimator.close()

    return pose_result, routing


def _create_pose_progress_callback(
    progress_callback: Optional[Callable[[str, int, int], None]],
) -> Optional[Callable[[int, int], None]]:
    """Adapt pose-backend progress updates to the CLI stage callback."""
    if progress_callback is None:
        return None

    def pose_progress(current: int, total: int):
        progress_callback("pose_estimation", current, total)

    return pose_progress


def _build_pose_routing_metadata(estimator: PoseEstimator) -> dict[str, object]:
    """Describe the automatically selected pose backend for reporting."""
    metadata: dict[str, object] = {
        "pose_backend": estimator.backend_name,
        "pose_backend_accelerated": estimator.supports_gpu,
    }

    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and estimator.backend_name == "mediapipe"
        and "apple_vision" not in get_available_backends()
    ):
        metadata["pose_backend_recommendation"] = (
            "Apple Vision support is missing from the current environment. "
            "Run `uv sync` to refresh macOS dependencies and enable hardware-accelerated pose estimation."
        )

    return metadata


def _build_merged_pose_windows(
    events: list[ImpactEvent],
    pre_impact_sec: float,
    post_impact_sec: float,
    video_start: float,
    video_end: float,
) -> list[MergedPoseWindow]:
    """Merge overlapping pose windows so nearby impacts share one decode pass."""
    windows: list[MergedPoseWindow] = []

    for event_index, event in enumerate(events):
        window_start = max(video_start, event.timestamp - pre_impact_sec)
        window_end = min(video_end, event.timestamp + post_impact_sec)

        if not windows or window_start > windows[-1].end_time:
            windows.append(
                MergedPoseWindow(
                    start_time=window_start,
                    end_time=window_end,
                    event_indices=(event_index,),
                )
            )
            continue

        previous = windows[-1]
        windows[-1] = MergedPoseWindow(
            start_time=previous.start_time,
            end_time=max(previous.end_time, window_end),
            event_indices=previous.event_indices + (event_index,),
        )

    return windows


def _slice_pose_result(
    pose_result: PoseAnalysisResult,
    start_time: float,
    end_time: float,
) -> PoseAnalysisResult:
    """Slice a merged pose result back into the per-event time window."""
    frames = [
        frame
        for frame in pose_result.frames
        if start_time <= frame.timestamp <= end_time
    ]
    return PoseAnalysisResult(
        frames=frames,
        fps=pose_result.fps,
        total_frames=len(frames),
        video_duration=max(0.0, end_time - start_time),
        source_file=pose_result.source_file,
    )


def detect_swings_audio_only(
    video_path: str | Path,
    config: Optional[Config] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> FusionResult:
    """
    Detect swings using audio analysis only (fastest).
    
    Args:
        video_path: Path to the video file.
        config: Optional configuration object.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = video end).
    
    Returns:
        FusionResult with detected swings (pose_result will be None).
    """
    return detect_swings(
        video_path, 
        mode=ProcessingMode.AUDIO, 
        config=config,
        start_time=start_time,
        end_time=end_time,
    )


def detect_swings_hybrid(
    video_path: str | Path,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> FusionResult:
    """
    Detect swings with hybrid approach: audio + targeted pose (recommended).
    
    This is a good balance of speed and accuracy - uses audio to find
    candidate swings, then validates with pose estimation around impacts.
    
    Args:
        video_path: Path to the video file.
        config: Optional configuration object.
        progress_callback: Optional progress callback.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = video end).
    
    Returns:
        FusionResult with detected swings.
    """
    return detect_swings(
        video_path, 
        mode=ProcessingMode.HYBRID, 
        config=config,
        progress_callback=progress_callback,
        start_time=start_time,
        end_time=end_time,
    )


def detect_swings_full_pose(
    video_path: str | Path,
    use_lite_model: bool = True,
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> FusionResult:
    """
    Detect swings with full video pose estimation (comprehensive but slow).
    
    This processes the entire video with pose estimation, which can catch
    swings that audio detection might miss (e.g., if audio cuts out).
    
    Args:
        video_path: Path to the video file.
        use_lite_model: Use lite model (faster) or full model (more accurate).
        config: Optional configuration object.
        progress_callback: Optional progress callback.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = video end).
    
    Returns:
        FusionResult with detected swings.
    """
    mode = ProcessingMode.LITE if use_lite_model else ProcessingMode.FULL
    return detect_swings(
        video_path, 
        mode=mode, 
        config=config,
        progress_callback=progress_callback,
        start_time=start_time,
        end_time=end_time,
    )
