"""Pose estimation and swing phase detection using MediaPipe."""

from fairwaycut.pose.estimator import PoseEstimator
from fairwaycut.pose.landmarks import (
    POSE_LANDMARKS,
    GOLF_RELEVANT_LANDMARKS,
    normalize_landmarks,
    get_landmark_velocity,
)
from fairwaycut.pose.swing_phases import (
    SwingPhaseDetector,
    detect_swing_phases,
)

__all__ = [
    "PoseEstimator",
    "POSE_LANDMARKS",
    "GOLF_RELEVANT_LANDMARKS",
    "normalize_landmarks",
    "get_landmark_velocity",
    "SwingPhaseDetector",
    "detect_swing_phases",
]

