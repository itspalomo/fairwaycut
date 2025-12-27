"""Landmark utilities and constants for MediaPipe pose estimation."""

from typing import Optional
import numpy as np

from fairwaycut.core.models import FramePose, Landmark


# MediaPipe Pose landmark indices
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
POSE_LANDMARKS = {
    # Face
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    
    # Upper body
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    
    # Lower body
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Landmarks most relevant for golf swing analysis
GOLF_RELEVANT_LANDMARKS = {
    # Hands and arms (critical for club tracking)
    "left_wrist": 15,
    "right_wrist": 16,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_shoulder": 11,
    "right_shoulder": 12,
    
    # Core and hips (rotation tracking)
    "left_hip": 23,
    "right_hip": 24,
    
    # Stability reference
    "left_ankle": 27,
    "right_ankle": 28,
    "nose": 0,
}

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
    (9, 10),  # Mouth
    
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),  # Hips
    
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),  # Wrist to fingers
    (17, 19),
    
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),  # Wrist to fingers
    (18, 20),
    
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Golf-specific simplified skeleton (cleaner visualization)
GOLF_SKELETON_CONNECTIONS = [
    # Torso
    (11, 12),  # Shoulders
    (11, 23), (12, 24),  # Shoulders to hips
    (23, 24),  # Hips
    
    # Arms (most important for swing)
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    
    # Legs
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
]


def normalize_landmarks(
    pose: FramePose,
    reference_landmark: int = 23,  # Left hip as default anchor
) -> list[Landmark]:
    """
    Normalize landmark positions relative to a reference point.
    
    This helps compare poses across different frame positions.
    
    Args:
        pose: FramePose with raw landmarks.
        reference_landmark: Index of landmark to use as origin.
    
    Returns:
        List of normalized landmarks.
    """
    if not pose.landmarks or reference_landmark >= len(pose.landmarks):
        return pose.landmarks
    
    ref = pose.landmarks[reference_landmark]
    normalized = []
    
    for lm in pose.landmarks:
        normalized.append(Landmark(
            x=lm.x - ref.x,
            y=lm.y - ref.y,
            z=lm.z - ref.z,
            visibility=lm.visibility,
        ))
    
    return normalized


def get_landmark_velocity(
    poses: list[FramePose],
    landmark_index: int,
    frame_index: int,
    window: int = 1,
) -> Optional[tuple[float, float, float]]:
    """
    Calculate velocity of a landmark between frames.
    
    Args:
        poses: List of FramePose objects.
        landmark_index: Index of the landmark to track.
        frame_index: Current frame index.
        window: Number of frames to look back for velocity calculation.
    
    Returns:
        Tuple of (vx, vy, vz) velocity components, or None if cannot calculate.
    """
    if frame_index < window or frame_index >= len(poses):
        return None
    
    current = poses[frame_index]
    previous = poses[frame_index - window]
    
    if not current.is_valid or not previous.is_valid:
        return None
    
    curr_lm = current.get_landmark(landmark_index)
    prev_lm = previous.get_landmark(landmark_index)
    
    if curr_lm is None or prev_lm is None:
        return None
    
    # Time difference
    dt = current.timestamp - previous.timestamp
    if dt <= 0:
        return None
    
    vx = (curr_lm.x - prev_lm.x) / dt
    vy = (curr_lm.y - prev_lm.y) / dt
    vz = (curr_lm.z - prev_lm.z) / dt
    
    return (vx, vy, vz)


def get_wrist_speed(
    poses: list[FramePose],
    frame_index: int,
    window: int = 1,
) -> Optional[float]:
    """
    Calculate average wrist speed (both hands) at a frame.
    
    This is a key metric for swing analysis - high wrist speed
    indicates downswing and impact phases.
    
    Args:
        poses: List of FramePose objects.
        frame_index: Current frame index.
        window: Frames to look back for velocity.
    
    Returns:
        Average wrist speed, or None if cannot calculate.
    """
    left_vel = get_landmark_velocity(poses, POSE_LANDMARKS["left_wrist"], frame_index, window)
    right_vel = get_landmark_velocity(poses, POSE_LANDMARKS["right_wrist"], frame_index, window)
    
    speeds = []
    if left_vel:
        speeds.append(np.sqrt(left_vel[0]**2 + left_vel[1]**2))
    if right_vel:
        speeds.append(np.sqrt(right_vel[0]**2 + right_vel[1]**2))
    
    if not speeds:
        return None
    
    return float(np.mean(speeds))


def get_hip_rotation(pose: FramePose) -> Optional[float]:
    """
    Estimate hip rotation angle from pose.
    
    Uses the angle of the line between left and right hip
    relative to the horizontal axis.
    
    Args:
        pose: FramePose with landmarks.
    
    Returns:
        Rotation angle in degrees, or None if cannot calculate.
    """
    if not pose.is_valid:
        return None
    
    left_hip = pose.get_landmark(POSE_LANDMARKS["left_hip"])
    right_hip = pose.get_landmark(POSE_LANDMARKS["right_hip"])
    
    if left_hip is None or right_hip is None:
        return None
    
    # Calculate angle from horizontal
    dx = right_hip.x - left_hip.x
    dy = right_hip.y - left_hip.y
    
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return float(angle)


def get_shoulder_rotation(pose: FramePose) -> Optional[float]:
    """
    Estimate shoulder rotation angle from pose.
    
    Args:
        pose: FramePose with landmarks.
    
    Returns:
        Rotation angle in degrees, or None if cannot calculate.
    """
    if not pose.is_valid:
        return None
    
    left_shoulder = pose.get_landmark(POSE_LANDMARKS["left_shoulder"])
    right_shoulder = pose.get_landmark(POSE_LANDMARKS["right_shoulder"])
    
    if left_shoulder is None or right_shoulder is None:
        return None
    
    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return float(angle)


def get_hand_height(pose: FramePose, use_left: bool = True) -> Optional[float]:
    """
    Get hand height relative to hip (useful for swing phase detection).
    
    Args:
        pose: FramePose with landmarks.
        use_left: Whether to use left hand (True) or right hand (False).
    
    Returns:
        Vertical distance of wrist from hip (negative = below hip).
    """
    if not pose.is_valid:
        return None
    
    wrist_key = "left_wrist" if use_left else "right_wrist"
    hip_key = "left_hip" if use_left else "right_hip"
    
    wrist = pose.get_landmark(POSE_LANDMARKS[wrist_key])
    hip = pose.get_landmark(POSE_LANDMARKS[hip_key])
    
    if wrist is None or hip is None:
        return None
    
    # In MediaPipe, y increases downward, so lower y = higher position
    return float(hip.y - wrist.y)


def compute_pose_smoothness(
    poses: list[FramePose],
    start_idx: int,
    end_idx: int,
    landmark_index: int = 15,  # Left wrist
) -> float:
    """
    Compute smoothness of motion for a landmark over a range of frames.
    
    Lower values indicate smoother motion (less jitter).
    
    Args:
        poses: List of FramePose objects.
        start_idx: Starting frame index.
        end_idx: Ending frame index.
        landmark_index: Landmark to analyze.
    
    Returns:
        Smoothness metric (lower = smoother).
    """
    if end_idx <= start_idx or end_idx > len(poses):
        return float('inf')
    
    velocities = []
    for i in range(start_idx + 1, end_idx):
        vel = get_landmark_velocity(poses, landmark_index, i)
        if vel:
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            velocities.append(speed)
    
    if len(velocities) < 2:
        return float('inf')
    
    # Jerk (rate of change of velocity) as smoothness metric
    accelerations = np.diff(velocities)
    return float(np.std(accelerations))

