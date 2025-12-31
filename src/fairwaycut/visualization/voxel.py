"""Voxelized motion volume visualization.

This module creates voxel-based representations of golf swing motion,
providing a volumetric feel to the swing trajectory. Voxels are colored
by velocity magnitude or swing phase.

The voxel grid accumulates joint positions over a time window and
renders them as a 3D volume that shows the swept space of the swing.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fairwaycut.core.models import SwingPhase
from fairwaycut.pose.normalizer import NormalizedPose, KEY_JOINTS
from fairwaycut.visualization.viewer3d import PHASE_COLORS, CAMERA_PRESETS


# Velocity colormap (low to high): blue -> cyan -> green -> yellow -> red
VELOCITY_CMAP = plt.cm.plasma


@dataclass
class VoxelOptions:
    """Configuration options for voxel visualization."""
    
    # Grid settings
    grid_size: int = 12  # 12x12x12 voxels
    
    # History
    history_frames: int = 15  # Frames to accumulate
    
    # Coloring
    color_by: str = "velocity"  # "velocity" or "phase"
    
    # Rendering
    figsize: tuple[float, float] = (2, 2)
    dpi: int = 72
    background_color: str = "black"
    
    # Camera
    camera_view: str = "isometric"
    
    # Voxel appearance
    alpha: float = 0.6
    edge_alpha: float = 0.2
    
    # Coordinate bounds (in normalized pose space)
    x_range: tuple[float, float] = (-1.5, 1.5)
    y_range: tuple[float, float] = (-0.5, 2.0)  # Height
    z_range: tuple[float, float] = (-1.0, 1.0)  # Depth
    
    # Joint filtering
    track_joints: Optional[list[int]] = None  # None = all joints


class VoxelMotionVolume:
    """Voxelized motion volume for swing visualization.
    
    Accumulates joint positions over time and renders them as a
    3D voxel grid, providing a volumetric representation of the
    swing motion.
    """
    
    def __init__(self, options: Optional[VoxelOptions] = None):
        """Initialize the voxel motion volume.
        
        Args:
            options: Voxel configuration options
        """
        self.options = options or VoxelOptions()
        
        # Initialize grid
        size = self.options.grid_size
        self._grid = np.zeros((size, size, size), dtype=np.float32)
        self._phase_grid = np.zeros((size, size, size), dtype=np.int32)
        
        # History buffer
        self._history: deque[tuple[np.ndarray, float, SwingPhase]] = deque(
            maxlen=self.options.history_frames
        )
        
        # Figure for rendering
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[Axes3D] = None
        self._initialized = False
        
        plt.ioff()
    
    def _init_figure(self) -> None:
        """Initialize figure and axes."""
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
        if self.options.camera_view in CAMERA_PRESETS:
            elev, azim, _ = CAMERA_PRESETS[self.options.camera_view]
            self._ax.view_init(elev=elev, azim=azim)
        
        self._initialized = True
    
    def add_pose(self, pose: NormalizedPose) -> None:
        """Add a pose to the motion volume.
        
        Args:
            pose: NormalizedPose to add
        """
        # Determine which joints to track
        if self.options.track_joints is not None:
            joint_indices = self.options.track_joints
        else:
            joint_indices = list(range(len(pose.joints)))
        
        # Extract positions and velocities
        positions = pose.joints[joint_indices]
        velocities = pose.velocities[joint_indices]
        
        # Add to history
        self._history.append((positions, velocities.max(), pose.phase))
        
        # Update grid
        self._update_grid()
    
    def _update_grid(self) -> None:
        """Update voxel grid from history."""
        # Reset grid
        self._grid.fill(0)
        self._phase_grid.fill(0)
        
        size = self.options.grid_size
        
        for positions, max_vel, phase in self._history:
            for pos in positions:
                # Convert world coordinates to grid indices
                ix, iy, iz = self._world_to_voxel(pos)
                
                # Check bounds
                if 0 <= ix < size and 0 <= iy < size and 0 <= iz < size:
                    # Store max velocity seen at this voxel
                    self._grid[ix, iy, iz] = max(self._grid[ix, iy, iz], max_vel)
                    # Store phase (latest wins)
                    self._phase_grid[ix, iy, iz] = phase.value if hasattr(phase, 'value') else 0
    
    def _world_to_voxel(self, pos: np.ndarray) -> tuple[int, int, int]:
        """Convert world coordinates to voxel indices.
        
        Args:
            pos: [x, y, z] in normalized pose coordinates
            
        Returns:
            (ix, iy, iz) voxel indices
        """
        size = self.options.grid_size
        
        # Normalize to [0, 1] within bounds
        x_norm = (pos[0] - self.options.x_range[0]) / (self.options.x_range[1] - self.options.x_range[0])
        y_norm = (pos[1] - self.options.y_range[0]) / (self.options.y_range[1] - self.options.y_range[0])
        z_norm = (pos[2] - self.options.z_range[0]) / (self.options.z_range[1] - self.options.z_range[0])
        
        # Convert to indices
        ix = int(x_norm * (size - 1))
        iy = int(y_norm * (size - 1))
        iz = int(z_norm * (size - 1))
        
        return ix, iy, iz
    
    def render(self) -> np.ndarray:
        """Render the voxel volume to numpy array.
        
        Returns:
            BGR numpy array for OpenCV
        """
        if not self._initialized:
            self._init_figure()
        
        # Clear previous content
        self._ax.cla()
        
        # Reapply settings
        if self.options.camera_view in CAMERA_PRESETS:
            elev, azim, _ = CAMERA_PRESETS[self.options.camera_view]
            self._ax.view_init(elev=elev, azim=azim)
        
        self._ax.set_axis_off()
        self._ax.set_facecolor(self.options.background_color)
        
        # Get filled voxels
        filled = self._grid > 0
        
        if not filled.any():
            # No voxels to render, return black frame
            return self._fig_to_numpy()
        
        # Create color array
        colors = self._compute_colors(filled)
        
        # Render voxels
        self._ax.voxels(
            filled,
            facecolors=colors,
            edgecolors=np.clip(colors * 0.5, 0, 1),  # Darker edges
            shade=True,
            alpha=self.options.alpha,
        )
        
        return self._fig_to_numpy()
    
    def _compute_colors(self, filled: np.ndarray) -> np.ndarray:
        """Compute colors for voxels.
        
        Args:
            filled: Boolean array of filled voxels
            
        Returns:
            RGBA color array matching grid shape
        """
        size = self.options.grid_size
        colors = np.zeros((size, size, size, 4), dtype=np.float32)
        
        if self.options.color_by == "velocity":
            # Color by velocity using colormap
            max_vel = self._grid.max() if self._grid.max() > 0 else 1.0
            normalized = self._grid / max_vel
            
            # Apply colormap
            for ix in range(size):
                for iy in range(size):
                    for iz in range(size):
                        if filled[ix, iy, iz]:
                            colors[ix, iy, iz] = VELOCITY_CMAP(normalized[ix, iy, iz])
        
        elif self.options.color_by == "phase":
            # Color by swing phase
            for ix in range(size):
                for iy in range(size):
                    for iz in range(size):
                        if filled[ix, iy, iz]:
                            # Get phase color
                            phase_val = self._phase_grid[ix, iy, iz]
                            try:
                                phase = SwingPhase(phase_val) if isinstance(phase_val, str) else list(SwingPhase)[phase_val]
                                rgb = PHASE_COLORS.get(phase, (0.5, 1.0, 0.5))
                            except (ValueError, IndexError):
                                rgb = (0.5, 1.0, 0.5)
                            
                            colors[ix, iy, iz] = (*rgb, self.options.alpha)
        
        else:
            # Default green
            colors[filled] = (0.5, 1.0, 0.5, self.options.alpha)
        
        return colors
    
    def _fig_to_numpy(self) -> np.ndarray:
        """Convert figure to BGR numpy array."""
        self._fig.canvas.draw()
        buf = np.asarray(self._fig.canvas.buffer_rgba())
        bgr = buf[:, :, [2, 1, 0]]
        return bgr.copy()
    
    def get_frame_size(self) -> tuple[int, int]:
        """Get output frame dimensions.
        
        Returns:
            (width, height) in pixels
        """
        width = int(self.options.figsize[0] * self.options.dpi)
        height = int(self.options.figsize[1] * self.options.dpi)
        return width, height
    
    def reset(self) -> None:
        """Reset the motion volume (clear history and grid)."""
        self._history.clear()
        self._grid.fill(0)
        self._phase_grid.fill(0)
    
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


class WristTrailVolume(VoxelMotionVolume):
    """Specialized voxel volume that only tracks wrist motion.
    
    This provides a cleaner visualization focused on the club/hand path,
    which is the most important trajectory in golf swing analysis.
    """
    
    def __init__(self, options: Optional[VoxelOptions] = None):
        """Initialize wrist trail volume.
        
        Args:
            options: Voxel configuration options
        """
        if options is None:
            options = VoxelOptions()
        
        # Override to track only wrists
        options.track_joints = KEY_JOINTS  # [7, 8] = left/right wrist
        
        super().__init__(options)


def render_motion_volume(
    poses: list[NormalizedPose],
    grid_size: int = 12,
    color_by: str = "velocity",
    camera_view: str = "isometric",
) -> np.ndarray:
    """Convenience function to render a motion volume from a pose sequence.
    
    Args:
        poses: List of NormalizedPose objects
        grid_size: Voxel grid size
        color_by: "velocity" or "phase"
        camera_view: Camera preset name
        
    Returns:
        BGR numpy array
    """
    options = VoxelOptions(
        grid_size=grid_size,
        color_by=color_by,
        camera_view=camera_view,
        history_frames=len(poses),
    )
    
    with VoxelMotionVolume(options) as volume:
        for pose in poses:
            volume.add_pose(pose)
        
        return volume.render()


