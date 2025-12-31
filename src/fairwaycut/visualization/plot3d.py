"""Static 3D analysis plots for swing visualization.

This module generates static multi-panel 3D analysis figures showing
the swing trajectory from multiple camera angles. Used by the analyze
command's --plot-3d flag.

Unlike viewer3d.py (optimized for video frame rendering), this module
focuses on publication-quality static figures with annotations.
"""

from pathlib import Path
from typing import Optional
import numpy as np

# Use Agg backend for headless rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fairwaycut.core.models import SwingEvent, SwingPhase
from fairwaycut.pose.normalizer import (
    NormalizedPose,
    SKELETON_CONNECTIONS_3D,
    NORMALIZED_JOINT_NAMES,
    KEY_JOINTS,
)
from fairwaycut.visualization.viewer3d import PHASE_COLORS, CAMERA_PRESETS


# Velocity colormap for wrist trails
VELOCITY_CMAP = plt.cm.plasma


def plot_swing_3d(
    poses: list[NormalizedPose],
    swing: Optional[SwingEvent] = None,
    output_path: Optional[Path] = None,
    camera_views: list[str] = ["front", "dtl", "isometric"],
    title: Optional[str] = None,
    figsize: tuple[float, float] = (15, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Generate multi-panel 3D swing analysis figure.
    
    Creates a figure with multiple camera angle panels showing:
    - Full swing trajectory with ghost poses at key phases
    - Wrist path trace colored by velocity
    - Phase annotations
    
    Args:
        poses: List of NormalizedPose objects for the swing
        swing: Optional SwingEvent with metadata
        output_path: Optional path to save figure
        camera_views: List of camera view names
        title: Optional figure title
        figsize: Figure size in inches
        dpi: Dots per inch for output
        
    Returns:
        Matplotlib Figure object
    """
    if not poses:
        raise ValueError("No poses provided")
    
    num_views = len(camera_views)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')
    
    # Add title
    if title:
        fig.suptitle(title, color='white', fontsize=14, fontweight='bold', y=0.98)
    elif swing:
        fig.suptitle(
            f"Swing #{swing.swing_id} | Impact: {swing.impact_time:.2f}s | "
            f"Confidence: {swing.combined_confidence:.0%}",
            color='white',
            fontsize=12,
            y=0.98,
        )
    
    # Create subplots for each view
    axes = []
    for i, view in enumerate(camera_views):
        ax = fig.add_subplot(1, num_views, i + 1, projection='3d')
        ax.set_facecolor('black')
        axes.append(ax)
        
        # Set camera
        if view in CAMERA_PRESETS:
            elev, azim, description = CAMERA_PRESETS[view]
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(view.upper(), color='white', fontsize=10, pad=5)
        
        # Configure axes
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.5, 2.0)
        
        # Hide axes
        ax.set_axis_off()
        
        # Hide panes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
    
    # Find key poses (address, top, impact, finish)
    key_poses = _find_key_poses(poses)
    
    # Draw on each axis
    for ax in axes:
        # Draw wrist trails
        _draw_wrist_trails(ax, poses)
        
        # Draw key pose skeletons
        for label, pose in key_poses.items():
            if pose is not None:
                alpha = 0.9 if label == "impact" else 0.5
                _draw_skeleton_3d(ax, pose, alpha=alpha, label=label)
    
    # Add phase legend
    _add_phase_legend(fig)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=dpi,
            facecolor='black',
            edgecolor='none',
            bbox_inches='tight',
        )
    
    return fig


def _find_key_poses(poses: list[NormalizedPose]) -> dict[str, Optional[NormalizedPose]]:
    """Find key poses in the swing sequence.
    
    Args:
        poses: List of NormalizedPose objects
        
    Returns:
        Dictionary mapping phase names to poses
    """
    key_poses = {
        "address": None,
        "top": None,
        "impact": None,
        "finish": None,
    }
    
    # Find poses by phase
    phase_map = {
        SwingPhase.ADDRESS: "address",
        SwingPhase.TOP: "top",
        SwingPhase.IMPACT: "impact",
        SwingPhase.FINISH: "finish",
    }
    
    for pose in poses:
        if pose.phase in phase_map:
            key_name = phase_map[pose.phase]
            # Take first occurrence of each phase
            if key_poses[key_name] is None:
                key_poses[key_name] = pose
    
    # Fallback: if no phases detected, use positions in sequence
    if all(v is None for v in key_poses.values()):
        n = len(poses)
        if n >= 4:
            key_poses["address"] = poses[0]
            key_poses["top"] = poses[n // 3]
            key_poses["impact"] = poses[2 * n // 3]
            key_poses["finish"] = poses[-1]
        elif n >= 2:
            key_poses["address"] = poses[0]
            key_poses["finish"] = poses[-1]
        elif n == 1:
            key_poses["impact"] = poses[0]
    
    return key_poses


def _draw_skeleton_3d(
    ax: Axes3D,
    pose: NormalizedPose,
    alpha: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Draw a 3D skeleton on the axes.
    
    Args:
        ax: Matplotlib 3D axes
        pose: NormalizedPose to draw
        alpha: Opacity
        label: Optional label for annotation
    """
    joints = pose.joints
    color = PHASE_COLORS.get(pose.phase, (0.5, 1.0, 0.5))
    
    # Map coordinates: our Y (up) -> matplotlib Z, our Z (depth) -> matplotlib Y
    
    # Draw bones
    for start_idx, end_idx in SKELETON_CONNECTIONS_3D:
        start = joints[start_idx]
        end = joints[end_idx]
        
        xs = [start[0], end[0]]
        ys = [start[2], end[2]]  # Our Z -> mpl Y
        zs = [start[1], end[1]]  # Our Y -> mpl Z
        
        ax.plot(xs, ys, zs, color=color, linewidth=1.5, alpha=alpha)
    
    # Draw joints
    xs = joints[:, 0]
    ys = joints[:, 2]
    zs = joints[:, 1]
    
    ax.scatter(xs, ys, zs, c=[color], s=15, alpha=alpha, edgecolors='white', linewidths=0.3)
    
    # Add label annotation
    if label:
        # Position label near head (approximate)
        head_pos = joints[0]  # Hip center, adjust as needed
        ax.text(
            head_pos[0],
            head_pos[2] + 0.3,
            head_pos[1] + 1.2,
            label.upper(),
            color=color,
            fontsize=8,
            ha='center',
            alpha=alpha,
        )


def _draw_wrist_trails(
    ax: Axes3D,
    poses: list[NormalizedPose],
) -> None:
    """Draw wrist motion trails colored by velocity.
    
    Args:
        ax: Matplotlib 3D axes
        poses: List of poses
    """
    if len(poses) < 2:
        return
    
    for wrist_idx in KEY_JOINTS:  # [7, 8] = left/right wrist
        # Extract wrist positions
        positions = np.array([p.joints[wrist_idx] for p in poses])
        velocities = np.array([p.velocities[wrist_idx] for p in poses])
        
        # Map coordinates
        xs = positions[:, 0]
        ys = positions[:, 2]  # Z -> Y
        zs = positions[:, 1]  # Y -> Z
        
        # Normalize velocities for coloring
        max_vel = velocities.max() if velocities.max() > 0 else 1.0
        normalized_vel = velocities / max_vel
        
        # Draw segments with velocity coloring
        for i in range(len(positions) - 1):
            color = VELOCITY_CMAP(normalized_vel[i])
            ax.plot(
                xs[i:i+2],
                ys[i:i+2],
                zs[i:i+2],
                color=color,
                linewidth=2,
                alpha=0.8,
            )


def _add_phase_legend(fig: plt.Figure) -> None:
    """Add phase color legend to figure.
    
    Args:
        fig: Matplotlib figure
    """
    # Create legend elements
    from matplotlib.patches import Patch
    
    legend_elements = []
    phase_names = ["ADDRESS", "BACKSWING", "TOP", "DOWNSWING", "IMPACT", "FOLLOW-THROUGH", "FINISH"]
    phases = [SwingPhase.ADDRESS, SwingPhase.BACKSWING, SwingPhase.TOP, 
              SwingPhase.DOWNSWING, SwingPhase.IMPACT, SwingPhase.FOLLOW_THROUGH, SwingPhase.FINISH]
    
    for name, phase in zip(phase_names, phases):
        color = PHASE_COLORS.get(phase, (0.5, 0.5, 0.5))
        legend_elements.append(Patch(facecolor=color, edgecolor='white', label=name, linewidth=0.5))
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(legend_elements),
        frameon=False,
        fontsize=8,
        labelcolor='white',
        handlelength=1.5,
        handleheight=1,
    )


def plot_swing_comparison(
    swings_data: list[tuple[list[NormalizedPose], SwingEvent]],
    output_path: Optional[Path] = None,
    camera_view: str = "isometric",
    figsize: tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> plt.Figure:
    """Generate comparison plot of multiple swings.
    
    Args:
        swings_data: List of (poses, swing_event) tuples
        output_path: Optional path to save figure
        camera_view: Camera view for all panels
        figsize: Figure size in inches
        dpi: Dots per inch
        
    Returns:
        Matplotlib Figure object
    """
    num_swings = len(swings_data)
    if num_swings == 0:
        raise ValueError("No swings provided")
    
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor='black')
    fig.suptitle("Swing Comparison", color='white', fontsize=14, fontweight='bold')
    
    for i, (poses, swing) in enumerate(swings_data):
        ax = fig.add_subplot(1, num_swings, i + 1, projection='3d')
        ax.set_facecolor('black')
        
        # Set camera
        if camera_view in CAMERA_PRESETS:
            elev, azim, _ = CAMERA_PRESETS[camera_view]
            ax.view_init(elev=elev, azim=azim)
        
        ax.set_title(
            f"Swing #{swing.swing_id}\n{swing.combined_confidence:.0%}",
            color='white',
            fontsize=9,
        )
        
        # Configure axes
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-0.5, 2.0)
        ax.set_axis_off()
        
        # Draw trails and key poses
        _draw_wrist_trails(ax, poses)
        
        key_poses = _find_key_poses(poses)
        for label, pose in key_poses.items():
            if pose is not None:
                _draw_skeleton_3d(ax, pose, alpha=0.7)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor='black', bbox_inches='tight')
    
    return fig


def save_swing_3d_plot(
    poses: list[NormalizedPose],
    swing: SwingEvent,
    output_path: Path,
    **kwargs,
) -> Path:
    """Convenience function to save a 3D swing plot.
    
    Args:
        poses: List of NormalizedPose objects
        swing: SwingEvent metadata
        output_path: Path to save figure
        **kwargs: Additional arguments passed to plot_swing_3d
        
    Returns:
        Path to saved figure
    """
    fig = plot_swing_3d(poses, swing, output_path, **kwargs)
    plt.close(fig)
    return output_path

