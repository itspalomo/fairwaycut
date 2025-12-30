"""3D skeleton visualization for golf swing analysis.

This module provides Matplotlib-based 3D rendering of pose skeletons
with support for multiple camera views, phase-aware coloring, and
velocity-based effects.

Key Features:
- Canned camera presets: front, down-the-line (DTL), isometric
- Phase-aware color themes (reuses overlays.py themes)
- Velocity-based joint sizing and intensity
- Efficient rendering to numpy arrays for video composition
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fairwaycut.core.models import SwingPhase
from fairwaycut.pose.normalizer import (
    NormalizedPose,
    SKELETON_CONNECTIONS_3D,
    NORMALIZED_JOINT_NAMES,
    KEY_JOINTS,
)


# Camera presets: (elevation, azimuth, description)
CAMERA_PRESETS = {
    "front": (0, 0, "Face-on view"),
    "dtl": (10, 90, "Down-the-line view"),
    "isometric": (30, -60, "3/4 isometric view"),
    "top": (90, 0, "Bird's eye view"),
}

# Phase colors (RGB, 0-1 range for matplotlib)
# Converted from BGR overlays.py colors
PHASE_COLORS = {
    SwingPhase.IDLE: (0.5, 0.5, 0.5),           # Gray
    SwingPhase.ADDRESS: (0.39, 0.78, 1.0),      # Light blue
    SwingPhase.BACKSWING: (0.78, 0.39, 1.0),    # Purple
    SwingPhase.TOP: (1.0, 0.2, 1.0),            # Magenta
    SwingPhase.DOWNSWING: (1.0, 0.55, 0.0),     # Orange
    SwingPhase.IMPACT: (1.0, 1.0, 0.0),         # Yellow
    SwingPhase.FOLLOW_THROUGH: (0.0, 1.0, 0.78), # Green-cyan
    SwingPhase.FINISH: (0.0, 0.59, 1.0),        # Blue
}

# Default color for unknown phases
DEFAULT_COLOR = (0.0, 1.0, 0.5)  # Neon green


@dataclass
class Viewer3DOptions:
    """Configuration options for 3D viewer."""
    
    # Camera
    camera_view: str = "isometric"
    
    # Figure size (in inches, will be converted to pixels)
    figsize: tuple[float, float] = (4, 4)
    dpi: int = 72
    
    # Colors
    use_phase_colors: bool = True
    default_bone_color: tuple[float, float, float] = DEFAULT_COLOR
    default_joint_color: tuple[float, float, float] = DEFAULT_COLOR
    background_color: str = "black"
    
    # Skeleton rendering
    bone_linewidth: float = 2.5
    joint_size: float = 30.0
    
    # Velocity effects
    velocity_scaling: bool = True
    velocity_min_scale: float = 0.8
    velocity_max_scale: float = 1.5
    
    # Axis limits (normalized coordinates)
    xlim: tuple[float, float] = (-1.5, 1.5)
    ylim: tuple[float, float] = (-1.5, 1.5)
    zlim: tuple[float, float] = (-0.5, 2.0)
    
    # Grid and labels
    show_grid: bool = False
    show_axes: bool = False
    show_labels: bool = False


class SwingViewer3D:
    """3D skeleton viewer for golf swing visualization.
    
    Renders NormalizedPose objects to numpy arrays suitable for
    video composition. Optimized for repeated rendering by reusing
    figure/axes objects.
    """
    
    def __init__(self, options: Optional[Viewer3DOptions] = None):
        """Initialize the 3D viewer.
        
        Args:
            options: Viewer configuration options
        """
        self.options = options or Viewer3DOptions()
        
        # Validate camera view
        if self.options.camera_view not in CAMERA_PRESETS:
            raise ValueError(
                f"Unknown camera view: {self.options.camera_view}. "
                f"Valid options: {list(CAMERA_PRESETS.keys())}"
            )
        
        # Initialize figure and axes (reused across frames)
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[Axes3D] = None
        self._initialized = False
        
        # Disable interactive mode
        plt.ioff()
    
    def _init_figure(self) -> None:
        """Initialize or reinitialize figure and axes."""
        if self._fig is not None:
            plt.close(self._fig)
        
        self._fig = plt.figure(
            figsize=self.options.figsize,
            dpi=self.options.dpi,
            facecolor=self.options.background_color,
        )
        
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_facecolor(self.options.background_color)
        
        # Set camera view
        elev, azim, _ = CAMERA_PRESETS[self.options.camera_view]
        self._ax.view_init(elev=elev, azim=azim)
        
        # Set axis limits
        self._ax.set_xlim(self.options.xlim)
        self._ax.set_ylim(self.options.zlim)  # Y in 3D plot = Z in our coords (height)
        self._ax.set_zlim(self.options.ylim)  # Z in 3D plot = Y in our coords (depth)
        
        # Configure axes appearance
        if not self.options.show_axes:
            self._ax.set_axis_off()
        
        if not self.options.show_grid:
            self._ax.grid(False)
            # Hide panes
            self._ax.xaxis.pane.fill = False
            self._ax.yaxis.pane.fill = False
            self._ax.zaxis.pane.fill = False
            self._ax.xaxis.pane.set_edgecolor('none')
            self._ax.yaxis.pane.set_edgecolor('none')
            self._ax.zaxis.pane.set_edgecolor('none')
        
        self._initialized = True
    
    def set_camera(self, view: str) -> None:
        """Change camera view.
        
        Args:
            view: Camera preset name (front, dtl, isometric, top)
        """
        if view not in CAMERA_PRESETS:
            raise ValueError(f"Unknown camera view: {view}")
        
        self.options.camera_view = view
        
        if self._ax is not None:
            elev, azim, _ = CAMERA_PRESETS[view]
            self._ax.view_init(elev=elev, azim=azim)
    
    def render_frame(
        self,
        pose: NormalizedPose,
        trail_poses: Optional[list[NormalizedPose]] = None,
    ) -> np.ndarray:
        """Render a single pose frame to numpy array.
        
        Args:
            pose: NormalizedPose to render
            trail_poses: Optional list of previous poses for motion trail
            
        Returns:
            BGR numpy array suitable for OpenCV/video composition
        """
        if not self._initialized:
            self._init_figure()
        
        # Clear previous frame content
        self._ax.cla()
        
        # Reapply settings after clear
        elev, azim, _ = CAMERA_PRESETS[self.options.camera_view]
        self._ax.view_init(elev=elev, azim=azim)
        self._ax.set_xlim(self.options.xlim)
        self._ax.set_ylim(self.options.zlim)
        self._ax.set_zlim(self.options.ylim)
        
        if not self.options.show_axes:
            self._ax.set_axis_off()
        
        self._ax.set_facecolor(self.options.background_color)
        
        # Get colors based on phase
        if self.options.use_phase_colors:
            bone_color = PHASE_COLORS.get(pose.phase, self.options.default_bone_color)
            joint_color = PHASE_COLORS.get(pose.phase, self.options.default_joint_color)
        else:
            bone_color = self.options.default_bone_color
            joint_color = self.options.default_joint_color
        
        # Draw motion trail if provided
        if trail_poses:
            self._draw_trail(trail_poses, bone_color)
        
        # Draw skeleton
        self._draw_skeleton(pose, bone_color, joint_color)
        
        # Convert to numpy array
        return self._fig_to_numpy()
    
    def _draw_skeleton(
        self,
        pose: NormalizedPose,
        bone_color: tuple[float, float, float],
        joint_color: tuple[float, float, float],
    ) -> None:
        """Draw skeleton bones and joints.
        
        Args:
            pose: Pose to draw
            bone_color: RGB color for bones
            joint_color: RGB color for joints
        """
        joints = pose.joints
        velocities = pose.velocities
        
        # Map our coordinate system to matplotlib 3D:
        # Our: X=right, Y=up, Z=toward camera
        # Matplotlib: X=right, Y=depth, Z=up
        # So we swap Y and Z
        
        # Draw bones
        for start_idx, end_idx in SKELETON_CONNECTIONS_3D:
            start = joints[start_idx]
            end = joints[end_idx]
            
            xs = [start[0], end[0]]
            ys = [start[2], end[2]]  # Our Z -> matplotlib Y (depth)
            zs = [start[1], end[1]]  # Our Y -> matplotlib Z (up)
            
            self._ax.plot(
                xs, ys, zs,
                color=bone_color,
                linewidth=self.options.bone_linewidth,
                solid_capstyle='round',
            )
        
        # Draw joints
        xs = joints[:, 0]
        ys = joints[:, 2]  # Z -> depth
        zs = joints[:, 1]  # Y -> up
        
        # Calculate joint sizes based on velocity
        if self.options.velocity_scaling and velocities is not None:
            # Normalize velocities to [0, 1]
            max_vel = velocities.max() if velocities.max() > 0 else 1.0
            normalized_vel = velocities / max_vel
            
            # Scale sizes
            scale_range = self.options.velocity_max_scale - self.options.velocity_min_scale
            scales = self.options.velocity_min_scale + normalized_vel * scale_range
            sizes = self.options.joint_size * scales
        else:
            sizes = np.full(len(joints), self.options.joint_size)
        
        # Highlight key joints (wrists) with larger size
        for key_idx in KEY_JOINTS:
            sizes[key_idx] *= 1.5
        
        self._ax.scatter(
            xs, ys, zs,
            c=[joint_color],
            s=sizes,
            depthshade=True,
            edgecolors='white',
            linewidths=0.5,
        )
    
    def _draw_trail(
        self,
        trail_poses: list[NormalizedPose],
        color: tuple[float, float, float],
    ) -> None:
        """Draw motion trail for wrist joints.
        
        Args:
            trail_poses: List of previous poses (oldest first)
            color: Base color for trail
        """
        if len(trail_poses) < 2:
            return
        
        # Draw trail for wrists
        for wrist_idx in KEY_JOINTS:  # [7, 8] = left/right wrist
            trail_points = []
            for pose in trail_poses:
                point = pose.joints[wrist_idx]
                # Swap Y and Z for matplotlib
                trail_points.append([point[0], point[2], point[1]])
            
            trail_points = np.array(trail_points)
            
            # Draw with fading alpha
            num_segments = len(trail_points) - 1
            for i in range(num_segments):
                alpha = (i + 1) / num_segments * 0.5  # Fade from 0 to 0.5
                segment_color = (*color, alpha)
                
                self._ax.plot(
                    trail_points[i:i+2, 0],
                    trail_points[i:i+2, 1],
                    trail_points[i:i+2, 2],
                    color=segment_color,
                    linewidth=1.5,
                )
    
    def _fig_to_numpy(self) -> np.ndarray:
        """Convert figure to BGR numpy array.
        
        Returns:
            BGR numpy array for OpenCV
        """
        self._fig.canvas.draw()
        
        # Get RGBA buffer
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        
        # Convert RGBA to BGR (OpenCV format)
        bgr = buf[:, :, [2, 1, 0]]
        
        return bgr.copy()  # Return a copy to avoid buffer issues
    
    def get_frame_size(self) -> tuple[int, int]:
        """Get output frame dimensions.
        
        Returns:
            (width, height) in pixels
        """
        width = int(self.options.figsize[0] * self.options.dpi)
        height = int(self.options.figsize[1] * self.options.dpi)
        return width, height
    
    def close(self) -> None:
        """Release resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self._initialized = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def render_pose_3d(
    pose: NormalizedPose,
    camera_view: str = "isometric",
    figsize: tuple[float, float] = (4, 4),
    dpi: int = 72,
) -> np.ndarray:
    """Convenience function to render a single pose.
    
    Args:
        pose: NormalizedPose to render
        camera_view: Camera preset
        figsize: Figure size in inches
        dpi: Dots per inch
        
    Returns:
        BGR numpy array
    """
    options = Viewer3DOptions(
        camera_view=camera_view,
        figsize=figsize,
        dpi=dpi,
    )
    
    with SwingViewer3D(options) as viewer:
        return viewer.render_frame(pose)


def create_multi_view_render(
    pose: NormalizedPose,
    views: list[str] = ["front", "dtl", "isometric"],
    single_size: tuple[float, float] = (3, 3),
    dpi: int = 72,
) -> np.ndarray:
    """Render pose from multiple camera angles side-by-side.
    
    Args:
        pose: NormalizedPose to render
        views: List of camera view names
        single_size: Size of each view in inches
        dpi: Dots per inch
        
    Returns:
        BGR numpy array with views horizontally concatenated
    """
    frames = []
    
    for view in views:
        options = Viewer3DOptions(
            camera_view=view,
            figsize=single_size,
            dpi=dpi,
        )
        
        with SwingViewer3D(options) as viewer:
            frame = viewer.render_frame(pose)
            frames.append(frame)
    
    # Concatenate horizontally
    return np.concatenate(frames, axis=1)

