"""Temporal smoothing for pose coordinates.

This module provides smoothing algorithms to reduce jitter in pose data
while preserving important motion characteristics like peak velocities.

The primary algorithm is Savitzky-Golay filtering, which fits polynomials
to data windows and is well-suited for pose trajectory smoothing.
"""

from typing import Optional
import numpy as np
from scipy.signal import savgol_filter

from fairwaycut.pose.normalizer import NormalizedPose


def smooth_pose_trajectory(
    positions: np.ndarray,
    window: int = 7,
    order: int = 2,
) -> np.ndarray:
    """Smooth pose trajectory with Savitzky-Golay filter.
    
    Applies the filter independently to each joint and axis.
    
    Args:
        positions: Array of shape (num_frames, num_joints, 3)
        window: Filter window size (must be odd, typically 5-11)
        order: Polynomial order (typically 2-3)
        
    Returns:
        Smoothed positions with same shape
    """
    if positions.shape[0] < window:
        return positions  # Not enough frames
    
    # Ensure window is odd
    if window % 2 == 0:
        window += 1
    
    smoothed = np.zeros_like(positions)
    num_joints = positions.shape[1]
    
    for joint in range(num_joints):
        for axis in range(3):
            smoothed[:, joint, axis] = savgol_filter(
                positions[:, joint, axis], window, order
            )
    
    return smoothed


def smooth_velocities(
    velocities: np.ndarray,
    window: int = 5,
    order: int = 2,
) -> np.ndarray:
    """Smooth velocity data.
    
    Args:
        velocities: Array of shape (num_frames, num_joints)
        window: Filter window size
        order: Polynomial order
        
    Returns:
        Smoothed velocities
    """
    if velocities.shape[0] < window:
        return velocities
    
    if window % 2 == 0:
        window += 1
    
    smoothed = np.zeros_like(velocities)
    num_joints = velocities.shape[1]
    
    for joint in range(num_joints):
        smoothed[:, joint] = savgol_filter(
            velocities[:, joint], window, order
        )
    
    return smoothed


class PoseSmoother:
    """Temporal smoother for pose sequences.
    
    Supports multiple smoothing strategies:
    - Savitzky-Golay: Best for preserving peaks, good for swing analysis
    - Exponential Moving Average: Simpler, more lag but smoother
    - Velocity-adaptive: Less smoothing during fast motion
    """
    
    def __init__(
        self,
        window: int = 7,
        order: int = 2,
        method: str = "savgol",
        velocity_adaptive: bool = False,
        velocity_threshold: float = 0.5,
    ):
        """Initialize the pose smoother.
        
        Args:
            window: Filter window size (frames)
            order: Polynomial order for Savitzky-Golay
            method: Smoothing method - "savgol" or "ema"
            velocity_adaptive: If True, reduce smoothing during fast motion
            velocity_threshold: Velocity above which to reduce smoothing
        """
        self.window = window if window % 2 == 1 else window + 1
        self.order = order
        self.method = method
        self.velocity_adaptive = velocity_adaptive
        self.velocity_threshold = velocity_threshold
    
    def smooth(
        self,
        poses: list[NormalizedPose],
    ) -> list[NormalizedPose]:
        """Smooth a sequence of normalized poses.
        
        Args:
            poses: List of NormalizedPose objects
            
        Returns:
            List of smoothed NormalizedPose objects
        """
        if len(poses) < self.window:
            return poses
        
        # Extract joint positions into array
        positions = np.array([p.joints for p in poses])  # (N, 13, 3)
        velocities = np.array([p.velocities for p in poses])  # (N, 13)
        
        # Apply smoothing
        if self.method == "savgol":
            smoothed_positions = self._savgol_smooth(positions, velocities)
        elif self.method == "ema":
            smoothed_positions = self._ema_smooth(positions)
        else:
            smoothed_positions = positions
        
        # Smooth velocities as well
        smoothed_velocities = smooth_velocities(velocities, self.window, self.order)
        
        # Create new NormalizedPose objects with smoothed data
        smoothed_poses = []
        for i, pose in enumerate(poses):
            smoothed_poses.append(NormalizedPose(
                joints=smoothed_positions[i],
                timestamp=pose.timestamp,
                phase=pose.phase,
                velocities=smoothed_velocities[i],
                confidence=pose.confidence,
                frame_index=pose.frame_index,
            ))
        
        return smoothed_poses
    
    def _savgol_smooth(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """Apply Savitzky-Golay smoothing.
        
        Args:
            positions: Joint positions (N, 13, 3)
            velocities: Joint velocities (N, 13)
            
        Returns:
            Smoothed positions
        """
        if not self.velocity_adaptive:
            return smooth_pose_trajectory(positions, self.window, self.order)
        
        # Velocity-adaptive smoothing: blend between strong and weak smoothing
        # based on velocity magnitude
        
        # Strong smoothing (larger window)
        strong_window = min(self.window + 4, len(positions))
        if strong_window % 2 == 0:
            strong_window -= 1
        strong_smoothed = smooth_pose_trajectory(positions, strong_window, self.order)
        
        # Weak smoothing (smaller window)
        weak_window = max(3, self.window - 2)
        if weak_window % 2 == 0:
            weak_window += 1
        weak_smoothed = smooth_pose_trajectory(positions, weak_window, self.order)
        
        # Blend based on velocity
        max_vel_per_frame = np.max(velocities, axis=1)  # (N,)
        
        # Normalize to [0, 1] blend factor (0 = strong smoothing, 1 = weak smoothing)
        blend = np.clip(max_vel_per_frame / self.velocity_threshold, 0, 1)
        
        # Reshape for broadcasting
        blend = blend[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        
        # Blend: high velocity -> weak smoothing (preserve motion)
        smoothed = (1 - blend) * strong_smoothed + blend * weak_smoothed
        
        return smoothed
    
    def _ema_smooth(
        self,
        positions: np.ndarray,
        alpha: float = 0.3,
    ) -> np.ndarray:
        """Apply exponential moving average smoothing.
        
        Args:
            positions: Joint positions (N, 13, 3)
            alpha: Smoothing factor (0 = max smoothing, 1 = no smoothing)
            
        Returns:
            Smoothed positions
        """
        smoothed = np.zeros_like(positions)
        smoothed[0] = positions[0]
        
        for i in range(1, len(positions)):
            smoothed[i] = alpha * positions[i] + (1 - alpha) * smoothed[i - 1]
        
        return smoothed


def compute_derivatives(
    poses: list[NormalizedPose],
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration from pose sequence.
    
    Uses central differences for interior points, forward/backward
    differences at boundaries.
    
    Args:
        poses: List of NormalizedPose objects
        fps: Frames per second
        
    Returns:
        Tuple of (velocities, accelerations) arrays, both shape (N, 13, 3)
    """
    if len(poses) < 2:
        zeros = np.zeros((len(poses), 13, 3))
        return zeros, zeros
    
    positions = np.array([p.joints for p in poses])  # (N, 13, 3)
    dt = 1.0 / fps
    
    # Velocity: first derivative
    velocity = np.zeros_like(positions)
    
    # Forward difference for first frame
    velocity[0] = (positions[1] - positions[0]) / dt
    
    # Central difference for interior frames
    for i in range(1, len(positions) - 1):
        velocity[i] = (positions[i + 1] - positions[i - 1]) / (2 * dt)
    
    # Backward difference for last frame
    velocity[-1] = (positions[-1] - positions[-2]) / dt
    
    # Acceleration: second derivative
    acceleration = np.zeros_like(positions)
    
    # Forward difference for first frame
    acceleration[0] = (velocity[1] - velocity[0]) / dt
    
    # Central difference for interior frames
    for i in range(1, len(velocity) - 1):
        acceleration[i] = (velocity[i + 1] - velocity[i - 1]) / (2 * dt)
    
    # Backward difference for last frame
    acceleration[-1] = (velocity[-1] - velocity[-2]) / dt
    
    return velocity, acceleration


def find_peak_velocity_frames(
    poses: list[NormalizedPose],
    joint_indices: Optional[list[int]] = None,
    top_n: int = 5,
) -> list[int]:
    """Find frames with highest joint velocities.
    
    Useful for identifying key moments in the swing (impact, etc.).
    
    Args:
        poses: List of NormalizedPose objects
        joint_indices: Joints to consider (default: wrists [7, 8])
        top_n: Number of top frames to return
        
    Returns:
        List of frame indices sorted by velocity (highest first)
    """
    if not poses:
        return []
    
    if joint_indices is None:
        joint_indices = [7, 8]  # Wrists
    
    velocities = np.array([p.velocities for p in poses])  # (N, 13)
    
    # Sum velocities for specified joints
    joint_velocities = velocities[:, joint_indices].sum(axis=1)  # (N,)
    
    # Get top N indices
    top_indices = np.argsort(joint_velocities)[::-1][:top_n]
    
    return top_indices.tolist()


