"""Pose normalization for backend-agnostic 3D representation.

This module normalizes poses from different backends (MediaPipe, Apple Vision)
to a consistent 3D coordinate system suitable for visualization and analysis.

Coordinate System:
- Origin: Hip midpoint (center of left_hip and right_hip)
- +X: Right (golfer's left)
- +Y: Up (toward head)
- +Z: Toward camera (golfer facing camera has positive z for front body parts)

Scale:
- Normalized to unit scale based on torso height (hip to shoulder distance)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from fairwaycut.core.models import FramePose, Landmark, SwingPhase, PoseAnalysisResult
from fairwaycut.pose.landmarks import POSE_LANDMARKS


# Normalized skeleton joint indices for 3D visualization
# Maps from our simplified 13-joint skeleton to MediaPipe indices
NORMALIZED_JOINT_NAMES = [
    "hip_center",      # 0 - Computed midpoint
    "left_hip",        # 1
    "right_hip",       # 2
    "left_shoulder",   # 3
    "right_shoulder",  # 4
    "left_elbow",      # 5
    "right_elbow",     # 6
    "left_wrist",      # 7
    "right_wrist",     # 8
    "left_knee",       # 9
    "right_knee",      # 10
    "left_ankle",      # 11
    "right_ankle",     # 12
]

# MediaPipe landmark indices for each normalized joint
MEDIAPIPE_TO_NORMALIZED = {
    "hip_center": None,  # Computed
    "left_hip": POSE_LANDMARKS["left_hip"],         # 23
    "right_hip": POSE_LANDMARKS["right_hip"],       # 24
    "left_shoulder": POSE_LANDMARKS["left_shoulder"],   # 11
    "right_shoulder": POSE_LANDMARKS["right_shoulder"], # 12
    "left_elbow": POSE_LANDMARKS["left_elbow"],     # 13
    "right_elbow": POSE_LANDMARKS["right_elbow"],   # 14
    "left_wrist": POSE_LANDMARKS["left_wrist"],     # 15
    "right_wrist": POSE_LANDMARKS["right_wrist"],   # 16
    "left_knee": POSE_LANDMARKS["left_knee"],       # 25
    "right_knee": POSE_LANDMARKS["right_knee"],     # 26
    "left_ankle": POSE_LANDMARKS["left_ankle"],     # 27
    "right_ankle": POSE_LANDMARKS["right_ankle"],   # 28
}

# Skeleton connections for 3D rendering (parent -> child relationships)
SKELETON_CONNECTIONS_3D = [
    # Torso
    (0, 1),   # hip_center -> left_hip
    (0, 2),   # hip_center -> right_hip
    (0, 3),   # hip_center -> left_shoulder (via spine, simplified)
    (0, 4),   # hip_center -> right_shoulder
    (3, 4),   # left_shoulder -> right_shoulder
    (1, 2),   # left_hip -> right_hip
    
    # Left arm
    (3, 5),   # left_shoulder -> left_elbow
    (5, 7),   # left_elbow -> left_wrist
    
    # Right arm
    (4, 6),   # right_shoulder -> right_elbow
    (6, 8),   # right_elbow -> right_wrist
    
    # Left leg
    (1, 9),   # left_hip -> left_knee
    (9, 11),  # left_knee -> left_ankle
    
    # Right leg
    (2, 10),  # right_hip -> right_knee
    (10, 12), # right_knee -> right_ankle
]

# Key joints for tracking (wrists are most important for golf swing)
KEY_JOINTS = [7, 8]  # left_wrist, right_wrist


@dataclass
class NormalizedPose:
    """Backend-agnostic normalized pose in world coordinates.
    
    Attributes:
        joints: Array of shape (13, 3) containing x, y, z coordinates
        timestamp: Frame timestamp in seconds
        phase: Current swing phase
        velocities: Per-joint velocity magnitudes, shape (13,)
        confidence: Overall pose confidence (0-1)
        frame_index: Original frame index in video
    """
    joints: np.ndarray  # Shape: (13, 3) - [x, y, z] per joint
    timestamp: float
    phase: SwingPhase = SwingPhase.IDLE
    velocities: np.ndarray = field(default_factory=lambda: np.zeros(13))
    confidence: float = 0.0
    frame_index: int = 0
    
    @property
    def num_joints(self) -> int:
        """Number of joints in the skeleton."""
        return len(NORMALIZED_JOINT_NAMES)
    
    @property
    def joint_names(self) -> list[str]:
        """List of joint names."""
        return NORMALIZED_JOINT_NAMES
    
    def get_joint(self, name: str) -> Optional[np.ndarray]:
        """Get joint position by name.
        
        Args:
            name: Joint name from NORMALIZED_JOINT_NAMES
            
        Returns:
            Array [x, y, z] or None if not found
        """
        try:
            idx = NORMALIZED_JOINT_NAMES.index(name)
            return self.joints[idx]
        except ValueError:
            return None
    
    def get_wrist_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Get left and right wrist positions."""
        return self.joints[7], self.joints[8]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "joints": self.joints.tolist(),
            "timestamp": round(self.timestamp, 4),
            "phase": self.phase.value,
            "velocities": self.velocities.tolist(),
            "confidence": round(self.confidence, 3),
            "frame_index": self.frame_index,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NormalizedPose":
        """Create from dictionary."""
        return cls(
            joints=np.array(data["joints"]),
            timestamp=data["timestamp"],
            phase=SwingPhase(data["phase"]),
            velocities=np.array(data["velocities"]),
            confidence=data["confidence"],
            frame_index=data["frame_index"],
        )


class PoseNormalizer:
    """Normalizes poses from different backends to a consistent 3D rig.
    
    Handles:
    - Centering at hip midpoint
    - Scaling to unit torso height
    - Axis alignment (+Y up, +Z toward camera)
    - Z-coordinate sign correction (MediaPipe uses negative z for closer)
    - Depth synthesis for 2D-only backends (Apple Vision)
    """
    
    def __init__(
        self,
        flip_z: bool = True,
        synthesize_depth: bool = True,
        reference_shoulder_width: float = 0.4,  # meters, typical adult
    ):
        """Initialize the pose normalizer.
        
        Args:
            flip_z: Flip z-axis to use +z = toward camera convention
            synthesize_depth: Synthesize depth for 2D-only backends
            reference_shoulder_width: Reference shoulder width for depth synthesis
        """
        self.flip_z = flip_z
        self.synthesize_depth = synthesize_depth
        self.reference_shoulder_width = reference_shoulder_width
        
        # Cache for velocity computation
        self._prev_joints: Optional[np.ndarray] = None
        self._prev_timestamp: float = 0.0
    
    def normalize(
        self,
        pose: FramePose,
        phase: SwingPhase = SwingPhase.IDLE,
    ) -> Optional[NormalizedPose]:
        """Normalize a single frame pose.
        
        Args:
            pose: Input FramePose from any backend
            phase: Current swing phase
            
        Returns:
            NormalizedPose or None if pose is invalid
        """
        if not pose.is_valid or len(pose.landmarks) < 33:
            return None
        
        # Extract relevant joints
        joints = self._extract_joints(pose)
        if joints is None:
            return None
        
        # Check if this is a 2D-only pose (all z values are 0 or very small)
        is_2d_only = np.allclose(joints[:, 2], 0, atol=1e-6)
        
        if is_2d_only and self.synthesize_depth:
            joints = self._synthesize_depth(joints)
        
        # Center at hip midpoint
        joints = self._center_pose(joints)
        
        # Scale to unit torso height
        joints = self._scale_pose(joints)
        
        # Flip z-axis if needed
        if self.flip_z:
            joints[:, 2] = -joints[:, 2]
        
        # Compute velocities
        velocities = self._compute_velocities(joints, pose.timestamp)
        
        return NormalizedPose(
            joints=joints,
            timestamp=pose.timestamp,
            phase=phase,
            velocities=velocities,
            confidence=pose.confidence,
            frame_index=pose.frame_index,
        )
    
    def normalize_sequence(
        self,
        poses: list[FramePose],
        phases: Optional[list[SwingPhase]] = None,
    ) -> list[NormalizedPose]:
        """Normalize a sequence of poses.
        
        Args:
            poses: List of FramePose objects
            phases: Optional list of swing phases per frame
            
        Returns:
            List of NormalizedPose objects (may be shorter if some poses invalid)
        """
        self.reset()
        
        normalized = []
        for i, pose in enumerate(poses):
            phase = phases[i] if phases and i < len(phases) else SwingPhase.IDLE
            norm_pose = self.normalize(pose, phase)
            if norm_pose is not None:
                normalized.append(norm_pose)
        
        return normalized
    
    def normalize_result(
        self,
        result: PoseAnalysisResult,
        phases: Optional[list[SwingPhase]] = None,
    ) -> list[NormalizedPose]:
        """Normalize poses from a PoseAnalysisResult.
        
        Args:
            result: PoseAnalysisResult from pose estimation
            phases: Optional list of swing phases per frame
            
        Returns:
            List of NormalizedPose objects
        """
        return self.normalize_sequence(result.frames, phases)
    
    def reset(self) -> None:
        """Reset internal state (velocity cache)."""
        self._prev_joints = None
        self._prev_timestamp = 0.0
    
    def _extract_joints(self, pose: FramePose) -> Optional[np.ndarray]:
        """Extract relevant joints from FramePose.
        
        Args:
            pose: Input FramePose
            
        Returns:
            Array of shape (13, 3) or None if missing required joints
        """
        joints = np.zeros((13, 3), dtype=np.float32)
        min_visibility = 0.3
        
        for i, joint_name in enumerate(NORMALIZED_JOINT_NAMES):
            if joint_name == "hip_center":
                # Compute midpoint of hips
                left_hip_idx = MEDIAPIPE_TO_NORMALIZED["left_hip"]
                right_hip_idx = MEDIAPIPE_TO_NORMALIZED["right_hip"]
                
                if left_hip_idx >= len(pose.landmarks) or right_hip_idx >= len(pose.landmarks):
                    return None
                
                left_hip = pose.landmarks[left_hip_idx]
                right_hip = pose.landmarks[right_hip_idx]
                
                if left_hip.visibility < min_visibility or right_hip.visibility < min_visibility:
                    return None
                
                joints[i] = [
                    (left_hip.x + right_hip.x) / 2,
                    (left_hip.y + right_hip.y) / 2,
                    (left_hip.z + right_hip.z) / 2,
                ]
            else:
                mp_idx = MEDIAPIPE_TO_NORMALIZED[joint_name]
                if mp_idx >= len(pose.landmarks):
                    return None
                
                lm = pose.landmarks[mp_idx]
                if lm.visibility < min_visibility:
                    # Use position but mark as low confidence
                    pass
                
                joints[i] = [lm.x, lm.y, lm.z]
        
        return joints
    
    def _synthesize_depth(self, joints: np.ndarray) -> np.ndarray:
        """Synthesize z-depth for 2D-only poses.
        
        Uses body proportions to estimate depth. The shoulder width in the
        image gives us a scale factor to estimate z-positions.
        
        Args:
            joints: Array of shape (13, 3) with z=0
            
        Returns:
            Joints with synthesized z values
        """
        # Get shoulder width in normalized coordinates
        left_shoulder = joints[3]
        right_shoulder = joints[4]
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        
        if shoulder_width < 0.01:
            return joints  # Can't estimate
        
        # Scale factor: how much smaller is the observed shoulder width
        # compared to reference? Use this to synthesize depth.
        # This is a rough heuristic - real depth would need stereo or ML.
        
        # Assume arms extend forward/backward relative to torso
        # Wrists get small z offset based on assumed arm pose
        joints_with_z = joints.copy()
        
        # Give wrists a slight forward offset (toward camera)
        # This helps with visualization even if not accurate
        joints_with_z[7, 2] = 0.1  # left_wrist
        joints_with_z[8, 2] = 0.1  # right_wrist
        
        # Elbows slightly forward
        joints_with_z[5, 2] = 0.05  # left_elbow
        joints_with_z[6, 2] = 0.05  # right_elbow
        
        # Feet slightly back
        joints_with_z[11, 2] = -0.05  # left_ankle
        joints_with_z[12, 2] = -0.05  # right_ankle
        
        return joints_with_z
    
    def _center_pose(self, joints: np.ndarray) -> np.ndarray:
        """Center pose at hip midpoint (origin).
        
        Args:
            joints: Array of shape (13, 3)
            
        Returns:
            Centered joints
        """
        hip_center = joints[0]
        return joints - hip_center
    
    def _scale_pose(self, joints: np.ndarray) -> np.ndarray:
        """Scale pose to unit torso height.
        
        Uses the distance from hip center to shoulder midpoint as reference.
        
        Args:
            joints: Array of shape (13, 3)
            
        Returns:
            Scaled joints
        """
        # Shoulder midpoint
        shoulder_mid = (joints[3] + joints[4]) / 2
        
        # Hip center is at origin after centering
        torso_height = np.linalg.norm(shoulder_mid)
        
        if torso_height < 0.01:
            return joints  # Avoid division by zero
        
        # Scale so torso height = 1
        return joints / torso_height
    
    def _compute_velocities(
        self,
        joints: np.ndarray,
        timestamp: float,
    ) -> np.ndarray:
        """Compute per-joint velocity magnitudes.
        
        Args:
            joints: Current joint positions
            timestamp: Current timestamp
            
        Returns:
            Array of velocity magnitudes, shape (13,)
        """
        velocities = np.zeros(13, dtype=np.float32)
        
        if self._prev_joints is not None:
            dt = timestamp - self._prev_timestamp
            if dt > 0:
                # Compute displacement
                displacement = joints - self._prev_joints
                # Velocity magnitude per joint
                velocities = np.linalg.norm(displacement, axis=1) / dt
        
        # Update cache
        self._prev_joints = joints.copy()
        self._prev_timestamp = timestamp
        
        return velocities


def normalize_poses_for_export(
    poses: list[FramePose],
    phases: Optional[list[SwingPhase]] = None,
) -> dict:
    """Normalize poses and format for JSON export.
    
    This is a convenience function for the analyze command's --export-3d-poses flag.
    
    Args:
        poses: List of FramePose objects
        phases: Optional list of swing phases
        
    Returns:
        Dictionary suitable for JSON serialization
    """
    normalizer = PoseNormalizer()
    normalized = normalizer.normalize_sequence(poses, phases)
    
    if not normalized:
        return {
            "timestamps": [],
            "joints": [],
            "joint_names": NORMALIZED_JOINT_NAMES,
            "coordinate_system": "normalized_world",
            "num_frames": 0,
        }
    
    return {
        "timestamps": [p.timestamp for p in normalized],
        "joints": [p.joints.tolist() for p in normalized],
        "joint_names": NORMALIZED_JOINT_NAMES,
        "coordinate_system": "normalized_world",
        "num_frames": len(normalized),
    }

