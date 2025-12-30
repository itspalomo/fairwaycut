"""Interactive 3D swing visualization using Plotly.

This module generates interactive HTML visualizations that can be
opened in a browser for full rotation/zoom/pan capabilities.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from fairwaycut.core.models import SwingPhase, SwingEvent
from fairwaycut.pose.normalizer import (
    NormalizedPose,
    SKELETON_CONNECTIONS_3D,
    NORMALIZED_JOINT_NAMES,
    KEY_JOINTS,
)

# Phase colors (RGB hex for plotly)
PHASE_COLORS_HEX = {
    SwingPhase.IDLE: "#808080",
    SwingPhase.ADDRESS: "#64C8FF",
    SwingPhase.BACKSWING: "#C864FF",
    SwingPhase.TOP: "#FF33FF",
    SwingPhase.DOWNSWING: "#FF8C00",
    SwingPhase.IMPACT: "#FFFF00",
    SwingPhase.FOLLOW_THROUGH: "#00FFC8",
    SwingPhase.FINISH: "#0096FF",
}


def generate_interactive_swing_viewer(
    poses: list[NormalizedPose],
    swing: Optional[SwingEvent] = None,
    output_path: Optional[Path] = None,
    title: str = "Golf Swing 3D Viewer",
) -> str:
    """Generate an interactive HTML 3D swing viewer.
    
    Args:
        poses: List of NormalizedPose objects
        swing: Optional SwingEvent with metadata
        output_path: Path to save HTML file (if None, returns HTML string)
        title: Title for the viewer
        
    Returns:
        HTML string if output_path is None, else path to saved file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "Plotly is required for interactive visualization. "
            "Install with: pip install plotly"
        )
    
    if not poses:
        raise ValueError("No poses provided")
    
    # Create figure
    fig = go.Figure()
    
    # Add ground plane
    ground_size = 1.5
    fig.add_trace(go.Mesh3d(
        x=[-ground_size, ground_size, ground_size, -ground_size],
        y=[-ground_size, -ground_size, ground_size, ground_size],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color='rgba(100, 100, 100, 0.3)',
        name='Ground',
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Add grid lines on ground
    for i in np.linspace(-ground_size, ground_size, 7):
        fig.add_trace(go.Scatter3d(
            x=[-ground_size, ground_size],
            y=[i, i],
            z=[0, 0],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=[i, i],
            y=[-ground_size, ground_size],
            z=[0, 0],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    # Add axis markers
    arrow_len = 0.5
    # X axis (right) - Red
    fig.add_trace(go.Scatter3d(
        x=[0, arrow_len], y=[0, 0], z=[0, 0],
        mode='lines+text',
        line=dict(color='red', width=4),
        text=['', 'RIGHT'],
        textposition='top center',
        textfont=dict(color='red', size=10),
        name='X axis',
        showlegend=False,
    ))
    # Y axis (toward camera) - Green
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, arrow_len], z=[0, 0],
        mode='lines+text',
        line=dict(color='green', width=4),
        text=['', 'CAMERA'],
        textposition='top center',
        textfont=dict(color='green', size=10),
        name='Y axis',
        showlegend=False,
    ))
    # Z axis (up) - Blue
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, arrow_len],
        mode='lines+text',
        line=dict(color='cyan', width=4),
        text=['', 'UP'],
        textposition='top center',
        textfont=dict(color='cyan', size=10),
        name='Z axis',
        showlegend=False,
    ))
    
    # Sample poses for visualization (show key frames)
    num_poses = len(poses)
    if num_poses > 10:
        # Sample evenly + always include first and last
        indices = [0] + list(np.linspace(1, num_poses - 2, 8).astype(int)) + [num_poses - 1]
    else:
        indices = list(range(num_poses))
    
    # Draw wrist trails (full trajectory)
    for wrist_idx in KEY_JOINTS:
        positions = np.array([p.joints[wrist_idx] for p in poses])
        velocities = np.array([p.velocities[wrist_idx] for p in poses])
        
        # Normalize velocities for color
        max_vel = velocities.max() if velocities.max() > 0 else 1.0
        colors = velocities / max_vel
        
        # Create color scale based on velocity
        color_vals = [f'rgb({int(255*v)}, {int(100*(1-v))}, {int(50)})' for v in colors]
        
        wrist_name = "Left Wrist" if wrist_idx == 7 else "Right Wrist"
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 2],  # Our Z -> plotly Y
            z=positions[:, 1],  # Our Y -> plotly Z
            mode='lines+markers',
            line=dict(color='orange', width=3),
            marker=dict(size=2, color=colors, colorscale='Hot'),
            name=f'{wrist_name} Path',
            hovertemplate=f'{wrist_name}<br>Velocity: %{{marker.color:.2f}}<extra></extra>',
        ))
    
    # Draw skeleton at sampled frames
    for i, idx in enumerate(indices):
        pose = poses[idx]
        alpha = 0.3 + 0.7 * (i / len(indices))  # Fade in over time
        phase_color = PHASE_COLORS_HEX.get(pose.phase, "#00FF80")
        
        joints = pose.joints
        
        # Draw bones
        for start_idx, end_idx in SKELETON_CONNECTIONS_3D:
            start = joints[start_idx]
            end = joints[end_idx]
            
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[2], end[2]],  # Z -> Y
                z=[start[1], end[1]],  # Y -> Z
                mode='lines',
                line=dict(color=phase_color, width=3),
                opacity=alpha,
                showlegend=False,
                hoverinfo='skip',
            ))
        
        # Draw joints
        fig.add_trace(go.Scatter3d(
            x=joints[:, 0],
            y=joints[:, 2],
            z=joints[:, 1],
            mode='markers',
            marker=dict(size=4, color=phase_color),
            opacity=alpha,
            name=f'Frame {idx} ({pose.phase.name})',
            hovertemplate='Joint<br>X: %{x:.2f}<br>Y: %{z:.2f}<br>Z: %{y:.2f}<extra></extra>',
        ))
    
    # Configure layout
    camera = dict(
        eye=dict(x=1.5, y=-1.5, z=0.8),  # DTL-ish view
        up=dict(x=0, y=0, z=1),
    )
    
    fig.update_layout(
        title=dict(
            text=title + (f"<br><sub>Impact: {swing.impact_time:.2f}s | Confidence: {swing.combined_confidence:.0%}</sub>" if swing else ""),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis=dict(title='Left ← → Right', range=[-1.5, 1.5], showgrid=False, showbackground=False),
            yaxis=dict(title='← Camera', range=[-1.5, 1.5], showgrid=False, showbackground=False),
            zaxis=dict(title='Height', range=[-0.5, 2.0], showgrid=False, showbackground=False),
            camera=camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8),
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        plot_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    
    # Add phase color legend as annotations
    phase_legend = "<br>".join([
        f"<span style='color:{color}'>{phase.name}</span>"
        for phase, color in PHASE_COLORS_HEX.items()
        if phase != SwingPhase.IDLE
    ])
    
    # Generate HTML
    html_content = fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToAdd': ['orbitRotation', 'resetCameraDefault3d'],
        },
    )
    
    # Add instructions
    instructions = """
    <div style="position: fixed; bottom: 10px; left: 10px; background: rgba(0,0,0,0.7); 
                color: white; padding: 10px; border-radius: 5px; font-size: 12px; z-index: 1000;">
        <b>Controls:</b><br>
        🖱️ Drag to rotate<br>
        🔍 Scroll to zoom<br>
        ⇧+Drag to pan<br>
        Double-click to reset
    </div>
    """
    html_content = html_content.replace('</body>', instructions + '</body>')
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        return str(output_path)
    
    return html_content


def generate_swing_animation(
    poses: list[NormalizedPose],
    swing: Optional[SwingEvent] = None,
    output_path: Optional[Path] = None,
    fps: float = 30.0,
) -> str:
    """Generate an animated 3D swing viewer with playback controls.
    
    Args:
        poses: List of NormalizedPose objects
        swing: Optional SwingEvent with metadata
        output_path: Path to save HTML file
        fps: Frames per second for animation
        
    Returns:
        Path to saved HTML file
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly required. Install with: pip install plotly")
    
    if not poses:
        raise ValueError("No poses provided")
    
    # Create frames for animation
    frames = []
    
    for i, pose in enumerate(poses):
        joints = pose.joints
        phase_color = PHASE_COLORS_HEX.get(pose.phase, "#00FF80")
        
        # Collect all traces for this frame
        frame_data = []
        
        # Bones
        for start_idx, end_idx in SKELETON_CONNECTIONS_3D:
            start = joints[start_idx]
            end = joints[end_idx]
            
            frame_data.append(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[2], end[2]],
                z=[start[1], end[1]],
                mode='lines',
                line=dict(color=phase_color, width=4),
            ))
        
        # Joints
        frame_data.append(go.Scatter3d(
            x=joints[:, 0],
            y=joints[:, 2],
            z=joints[:, 1],
            mode='markers',
            marker=dict(size=6, color=phase_color),
        ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(i),
            traces=list(range(len(frame_data))),
        ))
    
    # Initial frame
    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
    )
    
    # Add ground plane (static)
    ground_size = 1.5
    for i in np.linspace(-ground_size, ground_size, 7):
        fig.add_trace(go.Scatter3d(
            x=[-ground_size, ground_size], y=[i, i], z=[0, 0],
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=[i, i], y=[-ground_size, ground_size], z=[0, 0],
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False, hoverinfo='skip',
        ))
    
    # Animation controls
    frame_duration = 1000 / fps
    
    fig.update_layout(
        title="Golf Swing Animation" + (f" - Impact: {swing.impact_time:.2f}s" if swing else ""),
        scene=dict(
            xaxis=dict(title='X', range=[-1.5, 1.5]),
            yaxis=dict(title='Depth', range=[-1.5, 1.5]),
            zaxis=dict(title='Height', range=[-0.5, 2.0]),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.8)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8),
        ),
        paper_bgcolor='rgb(20, 20, 20)',
        font=dict(color='white'),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=0,
                x=0.1,
                xanchor='right',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            fromcurrent=True,
                            transition=dict(duration=0),
                        )],
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=False),
                            mode='immediate',
                            transition=dict(duration=0),
                        )],
                    ),
                ],
            ),
        ],
        sliders=[dict(
            active=0,
            yanchor='top',
            xanchor='left',
            currentvalue=dict(
                font=dict(size=12),
                prefix='Frame: ',
                visible=True,
                xanchor='right',
            ),
            transition=dict(duration=0),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[[str(i)], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate',
                        transition=dict(duration=0),
                    )],
                    label=str(i),
                    method='animate',
                )
                for i in range(len(frames))
            ],
        )],
    )
    
    html = fig.to_html(include_plotlyjs=True, full_html=True)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        return str(output_path)
    
    return html

