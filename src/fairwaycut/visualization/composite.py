"""Composite rendering for 2D video + 3D visualization.

This module combines the original 2D video frame with 3D skeleton views
and optional voxel insets to create enhanced visualization outputs.

Layout Options:
- Side-by-side: Video on left, 3D view on right
- Inset: 3D view overlaid as picture-in-picture in corner
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import cv2
import numpy as np

from fairwaycut.pose.normalizer import NormalizedPose
from fairwaycut.visualization.viewer3d import SwingViewer3D, Viewer3DOptions
from fairwaycut.visualization.voxel import VoxelMotionVolume, VoxelOptions, WristTrailVolume


class LayoutMode(Enum):
    """Layout modes for composite rendering."""
    SIDE_BY_SIDE = "side-by-side"
    INSET = "inset"


@dataclass
class CompositeOptions:
    """Configuration options for composite rendering."""
    
    # Layout
    layout: LayoutMode = LayoutMode.INSET
    
    # Side-by-side specific
    video_width_ratio: float = 0.6  # Video takes 60% of width
    
    # Inset specific
    inset_position: str = "top-right"  # top-right, top-left, bottom-right, bottom-left
    inset_size_ratio: float = 0.3  # Inset is 30% of video height
    inset_padding: int = 10  # Padding from edges
    inset_border_width: int = 2
    inset_border_color: tuple[int, int, int] = (100, 100, 100)  # Gray border
    
    # Voxel inset (sub-inset within 3D view)
    show_voxel: bool = False
    voxel_position: str = "bottom-right"
    voxel_size_ratio: float = 0.25  # 25% of 3D view size
    
    # 3D viewer options
    viewer_3d_options: Optional[Viewer3DOptions] = None
    
    # Voxel options
    voxel_options: Optional[VoxelOptions] = None
    
    # Background for expanded canvas (side-by-side)
    background_color: tuple[int, int, int] = (20, 20, 20)


class CompositeRenderer:
    """Combines 2D video frames with 3D visualization.
    
    Manages SwingViewer3D and VoxelMotionVolume instances and composites
    their output onto video frames.
    """
    
    def __init__(self, options: Optional[CompositeOptions] = None):
        """Initialize composite renderer.
        
        Args:
            options: Composite configuration options
        """
        self.options = options or CompositeOptions()
        
        # Initialize 3D viewer
        viewer_opts = self.options.viewer_3d_options or Viewer3DOptions()
        self._viewer = SwingViewer3D(viewer_opts)
        
        # Initialize voxel volume if enabled
        self._voxel: Optional[VoxelMotionVolume] = None
        if self.options.show_voxel:
            voxel_opts = self.options.voxel_options or VoxelOptions()
            self._voxel = WristTrailVolume(voxel_opts)
        
        # Trail history for motion trails
        self._trail_history: list[NormalizedPose] = []
        self._max_trail_length = 15
    
    def render(
        self,
        video_frame: np.ndarray,
        pose: Optional[NormalizedPose],
    ) -> np.ndarray:
        """Render composite frame.
        
        Args:
            video_frame: Original video frame (BGR)
            pose: Optional NormalizedPose for 3D rendering
            
        Returns:
            Composite BGR frame
        """
        if pose is None:
            # No pose data, return original frame
            if self.options.layout == LayoutMode.SIDE_BY_SIDE:
                return self._create_side_by_side_no_pose(video_frame)
            return video_frame
        
        # Update trail history
        self._trail_history.append(pose)
        if len(self._trail_history) > self._max_trail_length:
            self._trail_history.pop(0)
        
        # Update voxel volume
        if self._voxel is not None:
            self._voxel.add_pose(pose)
        
        # Render 3D view
        trail = self._trail_history[:-1] if len(self._trail_history) > 1 else None
        view_3d = self._viewer.render_frame(pose, trail_poses=trail)
        
        # Render voxel if enabled
        voxel_frame = None
        if self._voxel is not None:
            voxel_frame = self._voxel.render()
        
        # Composite based on layout
        if self.options.layout == LayoutMode.SIDE_BY_SIDE:
            return self._composite_side_by_side(video_frame, view_3d, voxel_frame)
        else:
            return self._composite_inset(video_frame, view_3d, voxel_frame)
    
    def _composite_side_by_side(
        self,
        video_frame: np.ndarray,
        view_3d: np.ndarray,
        voxel_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Composite video and 3D view side by side.
        
        Layout: [Video | 3D View (with optional voxel inset)]
        
        Args:
            video_frame: Original video frame
            view_3d: Rendered 3D view
            voxel_frame: Optional voxel rendering
            
        Returns:
            Composite frame
        """
        vh, vw = video_frame.shape[:2]
        
        # Calculate dimensions
        video_width = int(vw * self.options.video_width_ratio)
        panel_3d_width = vw - video_width
        
        # Resize video to fit
        video_resized = cv2.resize(video_frame, (video_width, vh))
        
        # Resize 3D view to fit panel
        view_3d_resized = cv2.resize(view_3d, (panel_3d_width, vh))
        
        # Add voxel inset to 3D panel if enabled
        if voxel_frame is not None:
            view_3d_resized = self._add_inset(
                view_3d_resized,
                voxel_frame,
                self.options.voxel_position,
                self.options.voxel_size_ratio,
            )
        
        # Concatenate horizontally
        composite = np.concatenate([video_resized, view_3d_resized], axis=1)
        
        return composite
    
    def _composite_inset(
        self,
        video_frame: np.ndarray,
        view_3d: np.ndarray,
        voxel_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Composite 3D view as inset over video.
        
        Args:
            video_frame: Original video frame
            view_3d: Rendered 3D view
            voxel_frame: Optional voxel rendering
            
        Returns:
            Composite frame with 3D inset
        """
        # Add voxel to 3D view first if enabled
        if voxel_frame is not None:
            view_3d = self._add_inset(
                view_3d,
                voxel_frame,
                self.options.voxel_position,
                self.options.voxel_size_ratio,
            )
        
        # Add 3D view as inset on video
        composite = self._add_inset(
            video_frame,
            view_3d,
            self.options.inset_position,
            self.options.inset_size_ratio,
            border_width=self.options.inset_border_width,
            border_color=self.options.inset_border_color,
        )
        
        return composite
    
    def _add_inset(
        self,
        background: np.ndarray,
        inset: np.ndarray,
        position: str,
        size_ratio: float,
        border_width: int = 0,
        border_color: tuple[int, int, int] = (100, 100, 100),
    ) -> np.ndarray:
        """Add an inset image to a background image.
        
        Args:
            background: Background image
            inset: Inset image to overlay
            position: Position string (top-right, top-left, bottom-right, bottom-left)
            size_ratio: Size of inset relative to background height
            border_width: Width of border around inset
            border_color: BGR color of border
            
        Returns:
            Background with inset overlaid
        """
        result = background.copy()
        bh, bw = background.shape[:2]
        
        # Calculate inset size
        inset_height = int(bh * size_ratio)
        inset_width = int(inset.shape[1] * inset_height / inset.shape[0])
        
        # Resize inset
        inset_resized = cv2.resize(inset, (inset_width, inset_height))
        
        # Calculate position
        padding = self.options.inset_padding
        
        if "top" in position:
            y = padding
        else:  # bottom
            y = bh - inset_height - padding
        
        if "right" in position:
            x = bw - inset_width - padding
        else:  # left
            x = padding
        
        # Draw border if specified
        if border_width > 0:
            cv2.rectangle(
                result,
                (x - border_width, y - border_width),
                (x + inset_width + border_width, y + inset_height + border_width),
                border_color,
                border_width,
            )
        
        # Overlay inset
        result[y:y + inset_height, x:x + inset_width] = inset_resized
        
        return result
    
    def _create_side_by_side_no_pose(self, video_frame: np.ndarray) -> np.ndarray:
        """Create side-by-side layout when no pose is available.
        
        Args:
            video_frame: Original video frame
            
        Returns:
            Frame with empty 3D panel
        """
        vh, vw = video_frame.shape[:2]
        
        video_width = int(vw * self.options.video_width_ratio)
        panel_3d_width = vw - video_width
        
        # Resize video
        video_resized = cv2.resize(video_frame, (video_width, vh))
        
        # Create empty 3D panel
        empty_panel = np.full(
            (vh, panel_3d_width, 3),
            self.options.background_color,
            dtype=np.uint8,
        )
        
        # Add "No Pose Data" text
        text = "No Pose Data"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (panel_3d_width - text_size[0]) // 2
        text_y = (vh + text_size[1]) // 2
        cv2.putText(
            empty_panel,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (100, 100, 100),
            thickness,
            cv2.LINE_AA,
        )
        
        return np.concatenate([video_resized, empty_panel], axis=1)
    
    def get_output_size(self, video_size: tuple[int, int]) -> tuple[int, int]:
        """Get output frame dimensions.
        
        Args:
            video_size: (width, height) of input video
            
        Returns:
            (width, height) of output composite
        """
        vw, vh = video_size
        
        if self.options.layout == LayoutMode.SIDE_BY_SIDE:
            # Width increases, height stays same
            return vw, vh
        else:
            # Inset mode: same size as input
            return vw, vh
    
    def reset(self) -> None:
        """Reset renderer state (clears history)."""
        self._trail_history.clear()
        if self._voxel is not None:
            self._voxel.reset()
    
    def close(self) -> None:
        """Release resources."""
        self._viewer.close()
        if self._voxel is not None:
            self._voxel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def composite_frame(
    video_frame: np.ndarray,
    pose: Optional[NormalizedPose],
    layout: str = "inset",
    camera_view: str = "isometric",
    show_voxel: bool = False,
) -> np.ndarray:
    """Convenience function to composite a single frame.
    
    Note: For video processing, use CompositeRenderer class directly
    to maintain state across frames (trails, voxel accumulation).
    
    Args:
        video_frame: Original video frame (BGR)
        pose: Optional NormalizedPose
        layout: "inset" or "side-by-side"
        camera_view: Camera preset name
        show_voxel: Whether to show voxel inset
        
    Returns:
        Composite BGR frame
    """
    layout_mode = LayoutMode(layout)
    
    viewer_opts = Viewer3DOptions(camera_view=camera_view)
    
    options = CompositeOptions(
        layout=layout_mode,
        show_voxel=show_voxel,
        viewer_3d_options=viewer_opts,
    )
    
    with CompositeRenderer(options) as renderer:
        return renderer.render(video_frame, pose)

