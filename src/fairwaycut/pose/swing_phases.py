"""Golf swing phase detection using pose landmarks."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from fairwaycut.core.models import FramePose, PoseAnalysisResult, SwingPhase
from fairwaycut.pose.landmarks import (
    POSE_LANDMARKS,
    get_wrist_speed,
    get_hand_height,
    get_hip_rotation,
    get_shoulder_rotation,
)


@dataclass
class PhaseTransition:
    """A detected transition between swing phases."""
    
    from_phase: SwingPhase
    to_phase: SwingPhase
    frame_index: int
    timestamp: float
    confidence: float


@dataclass
class SwingPhasesResult:
    """Result of swing phase detection for a video segment."""
    
    transitions: list[PhaseTransition]
    frame_phases: list[SwingPhase]  # Phase for each frame
    impact_frame: Optional[int] = None
    impact_timestamp: Optional[float] = None
    confidence: float = 0.0
    
    @property
    def has_complete_swing(self) -> bool:
        """Check if a complete swing was detected."""
        phases_detected = set(t.to_phase for t in self.transitions)
        required = {SwingPhase.BACKSWING, SwingPhase.DOWNSWING, SwingPhase.IMPACT}
        return required.issubset(phases_detected)


class SwingPhaseDetector:
    """
    Detect golf swing phases from pose data.
    
    Phases detected:
    - IDLE: No significant motion
    - ADDRESS: Stable stance, preparing to swing
    - BACKSWING: Hands rising, shoulder rotation
    - TOP: Top of backswing, hands at highest point
    - DOWNSWING: Rapid hand descent toward ball
    - IMPACT: Moment of ball contact (maximum hand speed)
    - FOLLOW_THROUGH: Post-impact deceleration
    - FINISH: End of swing, stable position
    """
    
    def __init__(
        self,
        velocity_smoothing_window: int = 5,
        min_backswing_height: float = 0.1,
        min_downswing_speed: float = 0.5,
        impact_speed_percentile: float = 95.0,
    ):
        """
        Initialize the swing phase detector.
        
        Args:
            velocity_smoothing_window: Window size for velocity smoothing.
            min_backswing_height: Minimum hand height change for backswing.
            min_downswing_speed: Minimum wrist speed for downswing detection.
            impact_speed_percentile: Percentile of speed to identify impact.
        """
        self.velocity_smoothing_window = velocity_smoothing_window
        self.min_backswing_height = min_backswing_height
        self.min_downswing_speed = min_downswing_speed
        self.impact_speed_percentile = impact_speed_percentile
    
    def detect_phases(
        self,
        pose_result: PoseAnalysisResult,
        expected_impact_time: Optional[float] = None,
    ) -> SwingPhasesResult:
        """
        Detect swing phases from pose analysis result.
        
        Args:
            pose_result: PoseAnalysisResult from pose estimation.
            expected_impact_time: Optional hint for impact time from audio.
        
        Returns:
            SwingPhasesResult with detected phases and transitions.
        """
        frames = pose_result.frames
        
        if len(frames) < 10:
            return SwingPhasesResult(
                transitions=[],
                frame_phases=[SwingPhase.IDLE] * len(frames),
                confidence=0.0,
            )
        
        # Extract motion features
        wrist_speeds = self._compute_wrist_speeds(frames)
        hand_heights = self._compute_hand_heights(frames)
        
        # Smooth the signals
        if len(wrist_speeds) >= self.velocity_smoothing_window:
            wrist_speeds = self._smooth_signal(wrist_speeds)
        if len(hand_heights) >= self.velocity_smoothing_window:
            hand_heights = self._smooth_signal(hand_heights)
        
        # Detect impact point (maximum speed)
        impact_frame = self._detect_impact(
            wrist_speeds, 
            frames, 
            expected_impact_time
        )
        
        # Detect other phase transitions relative to impact
        transitions, frame_phases = self._detect_transitions(
            frames,
            wrist_speeds,
            hand_heights,
            impact_frame,
        )
        
        # Calculate overall confidence
        confidence = self._compute_confidence(frames, transitions, impact_frame)
        
        impact_timestamp = None
        if impact_frame is not None and impact_frame < len(frames):
            impact_timestamp = frames[impact_frame].timestamp
        
        return SwingPhasesResult(
            transitions=transitions,
            frame_phases=frame_phases,
            impact_frame=impact_frame,
            impact_timestamp=impact_timestamp,
            confidence=confidence,
        )
    
    def _compute_wrist_speeds(self, frames: list[FramePose]) -> np.ndarray:
        """Compute wrist speed for each frame."""
        speeds = []
        for i in range(len(frames)):
            speed = get_wrist_speed(frames, i, window=1)
            speeds.append(speed if speed is not None else 0.0)
        return np.array(speeds)
    
    def _compute_hand_heights(self, frames: list[FramePose]) -> np.ndarray:
        """Compute hand height for each frame."""
        heights = []
        for frame in frames:
            # Average of both hands
            left_h = get_hand_height(frame, use_left=True)
            right_h = get_hand_height(frame, use_left=False)
            
            if left_h is not None and right_h is not None:
                heights.append((left_h + right_h) / 2)
            elif left_h is not None:
                heights.append(left_h)
            elif right_h is not None:
                heights.append(right_h)
            else:
                heights.append(0.0)
        
        return np.array(heights)
    
    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to signal."""
        window = min(self.velocity_smoothing_window, len(signal))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return signal
        return savgol_filter(signal, window, polyorder=2)
    
    def _detect_impact(
        self,
        wrist_speeds: np.ndarray,
        frames: list[FramePose],
        expected_impact_time: Optional[float],
    ) -> Optional[int]:
        """Detect the impact frame (maximum wrist speed)."""
        if len(wrist_speeds) == 0:
            return None
        
        # If we have an expected impact time from audio, search around it
        if expected_impact_time is not None:
            # Find frame closest to expected time
            timestamps = [f.timestamp for f in frames]
            search_center = np.argmin(np.abs(np.array(timestamps) - expected_impact_time))
            
            # Search window: +/- 0.5 seconds
            fps = len(frames) / (frames[-1].timestamp - frames[0].timestamp) if len(frames) > 1 else 30
            search_window = int(0.5 * fps)
            
            start_idx = max(0, search_center - search_window)
            end_idx = min(len(wrist_speeds), search_center + search_window)
            
            # Find peak in search window
            segment_speeds = wrist_speeds[start_idx:end_idx]
            if len(segment_speeds) > 0:
                local_peak = np.argmax(segment_speeds)
                return start_idx + local_peak
        
        # Fall back to finding global maximum
        peaks, properties = find_peaks(
            wrist_speeds,
            height=np.percentile(wrist_speeds, self.impact_speed_percentile),
            distance=int(len(wrist_speeds) * 0.1),  # Min 10% video length between peaks
        )
        
        if len(peaks) > 0:
            # Return the highest peak
            peak_heights = properties['peak_heights']
            return peaks[np.argmax(peak_heights)]
        
        # Last resort: global maximum
        return int(np.argmax(wrist_speeds))
    
    def _detect_transitions(
        self,
        frames: list[FramePose],
        wrist_speeds: np.ndarray,
        hand_heights: np.ndarray,
        impact_frame: Optional[int],
    ) -> tuple[list[PhaseTransition], list[SwingPhase]]:
        """Detect phase transitions throughout the swing."""
        n_frames = len(frames)
        frame_phases = [SwingPhase.IDLE] * n_frames
        transitions = []
        
        if impact_frame is None:
            return transitions, frame_phases
        
        # Work backwards from impact to find backswing phases
        # Work forwards from impact to find follow-through
        
        # Find top of backswing (hand height peak before impact)
        pre_impact = hand_heights[:impact_frame]
        if len(pre_impact) > 5:
            top_frame = np.argmax(pre_impact)
        else:
            top_frame = max(0, impact_frame - 10)
        
        # Find start of backswing (where hand height starts increasing)
        pre_top = hand_heights[:top_frame]
        if len(pre_top) > 5:
            # Find where derivative becomes positive
            deriv = np.diff(pre_top)
            positive_deriv = np.where(deriv > 0.01)[0]
            if len(positive_deriv) > 0:
                backswing_start = positive_deriv[0]
            else:
                backswing_start = max(0, top_frame - 30)
        else:
            backswing_start = 0
        
        # Find end of follow-through (where speed drops significantly)
        post_impact = wrist_speeds[impact_frame:]
        if len(post_impact) > 5:
            # Find where speed drops below 20% of impact speed
            threshold = wrist_speeds[impact_frame] * 0.2
            below_threshold = np.where(post_impact < threshold)[0]
            if len(below_threshold) > 0:
                finish_frame = impact_frame + below_threshold[0]
            else:
                finish_frame = min(n_frames - 1, impact_frame + 30)
        else:
            finish_frame = min(n_frames - 1, impact_frame + 15)
        
        # Assign phases to frames
        # Address: before backswing
        for i in range(0, backswing_start):
            frame_phases[i] = SwingPhase.ADDRESS
        
        # Backswing: from start to top
        for i in range(backswing_start, top_frame):
            frame_phases[i] = SwingPhase.BACKSWING
        
        # Top: at peak
        if top_frame < n_frames:
            frame_phases[top_frame] = SwingPhase.TOP
        
        # Downswing: from top to impact
        for i in range(top_frame + 1, impact_frame):
            frame_phases[i] = SwingPhase.DOWNSWING
        
        # Impact: at impact frame
        if impact_frame < n_frames:
            frame_phases[impact_frame] = SwingPhase.IMPACT
        
        # Follow-through: from impact to finish
        for i in range(impact_frame + 1, finish_frame):
            frame_phases[i] = SwingPhase.FOLLOW_THROUGH
        
        # Finish: after follow-through
        for i in range(finish_frame, n_frames):
            frame_phases[i] = SwingPhase.FINISH
        
        # Create transition events
        if backswing_start > 0:
            transitions.append(PhaseTransition(
                from_phase=SwingPhase.ADDRESS,
                to_phase=SwingPhase.BACKSWING,
                frame_index=backswing_start,
                timestamp=frames[backswing_start].timestamp,
                confidence=0.8,
            ))
        
        if top_frame > backswing_start:
            transitions.append(PhaseTransition(
                from_phase=SwingPhase.BACKSWING,
                to_phase=SwingPhase.TOP,
                frame_index=top_frame,
                timestamp=frames[top_frame].timestamp,
                confidence=0.9,
            ))
        
        transitions.append(PhaseTransition(
            from_phase=SwingPhase.TOP,
            to_phase=SwingPhase.DOWNSWING,
            frame_index=top_frame + 1,
            timestamp=frames[min(top_frame + 1, n_frames - 1)].timestamp,
            confidence=0.9,
        ))
        
        transitions.append(PhaseTransition(
            from_phase=SwingPhase.DOWNSWING,
            to_phase=SwingPhase.IMPACT,
            frame_index=impact_frame,
            timestamp=frames[impact_frame].timestamp,
            confidence=0.95,
        ))
        
        if finish_frame > impact_frame:
            transitions.append(PhaseTransition(
                from_phase=SwingPhase.IMPACT,
                to_phase=SwingPhase.FOLLOW_THROUGH,
                frame_index=impact_frame + 1,
                timestamp=frames[min(impact_frame + 1, n_frames - 1)].timestamp,
                confidence=0.85,
            ))
            
            transitions.append(PhaseTransition(
                from_phase=SwingPhase.FOLLOW_THROUGH,
                to_phase=SwingPhase.FINISH,
                frame_index=finish_frame,
                timestamp=frames[finish_frame].timestamp,
                confidence=0.8,
            ))
        
        return transitions, frame_phases
    
    def _compute_confidence(
        self,
        frames: list[FramePose],
        transitions: list[PhaseTransition],
        impact_frame: Optional[int],
    ) -> float:
        """Compute overall confidence in the phase detection."""
        if impact_frame is None or len(transitions) == 0:
            return 0.0
        
        # Factor 1: Pose detection quality
        valid_frames = [f for f in frames if f.is_valid]
        pose_quality = len(valid_frames) / len(frames) if frames else 0
        
        # Factor 2: Number of phases detected
        phases_detected = len(set(t.to_phase for t in transitions))
        phase_completeness = min(1.0, phases_detected / 5)  # Expect ~5 transitions
        
        # Factor 3: Average transition confidence
        avg_transition_conf = np.mean([t.confidence for t in transitions])
        
        # Weighted combination
        confidence = (
            pose_quality * 0.3 +
            phase_completeness * 0.3 +
            avg_transition_conf * 0.4
        )
        
        return float(confidence)


def detect_swing_phases(
    pose_result: PoseAnalysisResult,
    expected_impact_time: Optional[float] = None,
) -> SwingPhasesResult:
    """
    Convenience function to detect swing phases from pose data.
    
    Args:
        pose_result: PoseAnalysisResult from pose estimation.
        expected_impact_time: Optional hint for impact time from audio.
    
    Returns:
        SwingPhasesResult with detected phases.
    """
    detector = SwingPhaseDetector()
    return detector.detect_phases(pose_result, expected_impact_time)

