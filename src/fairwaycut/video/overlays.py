"""Video overlay utilities for pose skeleton and audio visualization."""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import cv2
import numpy as np

from fairwaycut.core.models import (
    FramePose,
    Landmark,
    AudioData,
    SwingPhase,
    DetectionResult,
    ImpactEvent,
)
from fairwaycut.pose.landmarks import (
    GOLF_SKELETON_CONNECTIONS,
    POSE_CONNECTIONS,
    POSE_LANDMARKS,
)
from fairwaycut.video.effects import (
    create_transparent_layer,
    apply_glow_effect,
    draw_glowing_line,
    draw_glowing_circle,
    draw_diamond,
    draw_hexagon,
    draw_motion_trail,
    draw_capsule_bone,
    create_particle_burst,
    depth_to_color,
    velocity_to_intensity,
    interpolate_color,
    alpha_composite,
)


# Default colors (BGR format)
DEFAULT_SKELETON_COLOR = (0, 255, 128)  # Green
DEFAULT_LANDMARK_COLOR = (0, 200, 255)  # Orange
DEFAULT_WAVEFORM_COLOR = (163, 204, 78)  # Teal (BGR)
DEFAULT_WAVEFORM_BG = (62, 33, 22)  # Dark blue (BGR)
DEFAULT_IMPACT_COLOR = (0, 255, 255)  # Yellow
DEFAULT_PHASE_COLOR = (255, 255, 255)  # White


# ============================================================================
# Rendering Mode and Color Theme System
# ============================================================================

class RenderMode(Enum):
    """Rendering mode presets."""
    MINIMAL = "minimal"      # Clean glow, no trails (fast)
    STANDARD = "standard"    # Glow + short trails + phase colors
    CINEMATIC = "cinematic"  # Full effects, long trails, depth, particles


@dataclass
class ColorTheme:
    """Color theme for skeleton visualization."""
    
    primary: tuple[int, int, int]       # Main skeleton color
    secondary: tuple[int, int, int]     # Secondary/accent color
    glow: tuple[int, int, int]          # Glow color
    joint_primary: tuple[int, int, int] # Key joint color (wrists, etc)
    joint_secondary: tuple[int, int, int]  # Other joint color
    trail: tuple[int, int, int]         # Motion trail color
    
    @classmethod
    def default(cls) -> "ColorTheme":
        """Default neon green theme."""
        return cls(
            primary=(128, 255, 0),       # Bright green
            secondary=(0, 255, 200),     # Cyan
            glow=(0, 255, 128),          # Green glow
            joint_primary=(0, 255, 255), # Yellow for key joints
            joint_secondary=(0, 200, 255),  # Orange
            trail=(0, 255, 180),         # Trail color
        )


# Phase-specific color themes (BGR format)
PHASE_THEMES: dict[SwingPhase, ColorTheme] = {
    SwingPhase.IDLE: ColorTheme(
        primary=(128, 128, 128),      # Gray
        secondary=(100, 100, 100),
        glow=(80, 80, 80),
        joint_primary=(150, 150, 150),
        joint_secondary=(120, 120, 120),
        trail=(100, 100, 100),
    ),
    SwingPhase.ADDRESS: ColorTheme(
        primary=(255, 200, 100),      # Light blue/cyan
        secondary=(255, 180, 50),
        glow=(255, 220, 150),
        joint_primary=(255, 255, 100),
        joint_secondary=(255, 200, 80),
        trail=(255, 200, 120),
    ),
    SwingPhase.BACKSWING: ColorTheme(
        primary=(255, 100, 200),      # Purple/magenta
        secondary=(255, 50, 150),
        glow=(255, 120, 220),
        joint_primary=(255, 150, 255),
        joint_secondary=(255, 100, 200),
        trail=(255, 80, 180),
    ),
    SwingPhase.TOP: ColorTheme(
        primary=(255, 50, 255),       # Bright magenta
        secondary=(200, 0, 200),
        glow=(255, 100, 255),
        joint_primary=(255, 150, 255),
        joint_secondary=(255, 80, 255),
        trail=(255, 50, 220),
    ),
    SwingPhase.DOWNSWING: ColorTheme(
        primary=(0, 140, 255),        # Orange/amber
        secondary=(0, 100, 255),
        glow=(0, 180, 255),
        joint_primary=(0, 200, 255),
        joint_secondary=(0, 150, 255),
        trail=(0, 120, 255),
    ),
    SwingPhase.IMPACT: ColorTheme(
        primary=(0, 255, 255),        # Bright yellow/gold
        secondary=(0, 220, 255),
        glow=(100, 255, 255),
        joint_primary=(255, 255, 255),  # White flash
        joint_secondary=(150, 255, 255),
        trail=(0, 255, 255),
    ),
    SwingPhase.FOLLOW_THROUGH: ColorTheme(
        primary=(200, 255, 0),        # Green/teal
        secondary=(150, 255, 50),
        glow=(180, 255, 100),
        joint_primary=(200, 255, 100),
        joint_secondary=(150, 255, 50),
        trail=(180, 255, 80),
    ),
    SwingPhase.FINISH: ColorTheme(
        primary=(255, 150, 0),        # Teal/blue
        secondary=(255, 100, 50),
        glow=(255, 180, 100),
        joint_primary=(255, 200, 100),
        joint_secondary=(255, 150, 50),
        trail=(255, 130, 80),
    ),
}


# ============================================================================
# Pose History Buffer for Motion Trails
# ============================================================================

@dataclass
class PoseHistoryEntry:
    """Single entry in pose history."""
    pose: FramePose
    timestamp: float
    velocities: dict[int, float] = field(default_factory=dict)  # landmark_idx -> velocity


class PoseHistory:
    """
    Circular buffer storing recent poses for motion trail rendering.
    
    Calculates velocities between frames for intensity modulation.
    """
    
    def __init__(self, max_size: int = 15):
        """
        Initialize pose history buffer.
        
        Args:
            max_size: Maximum number of poses to store.
        """
        self.max_size = max_size
        self._buffer: deque[PoseHistoryEntry] = deque(maxlen=max_size)
        self._last_positions: dict[int, tuple[float, float, float]] = {}
    
    def add(self, pose: FramePose, timestamp: float) -> None:
        """
        Add a new pose to the history.
        
        Args:
            pose: FramePose to add.
            timestamp: Current timestamp.
        """
        velocities = {}
        
        if pose.is_valid and self._last_positions:
            # Calculate velocities for key landmarks
            for idx, lm in enumerate(pose.landmarks):
                if idx in self._last_positions:
                    prev = self._last_positions[idx]
                    dx = lm.x - prev[0]
                    dy = lm.y - prev[1]
                    dz = lm.z - prev[2]
                    velocities[idx] = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Update last positions
        if pose.is_valid:
            self._last_positions = {
                idx: (lm.x, lm.y, lm.z)
                for idx, lm in enumerate(pose.landmarks)
            }
        
        self._buffer.append(PoseHistoryEntry(
            pose=pose,
            timestamp=timestamp,
            velocities=velocities,
        ))
    
    def get_trail_points(
        self,
        landmark_idx: int,
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        """
        Get pixel coordinates for a landmark's motion trail.
        
        Args:
            landmark_idx: Index of the landmark to track.
            width: Frame width for coordinate conversion.
            height: Frame height for coordinate conversion.
        
        Returns:
            List of (x, y) pixel coordinates from oldest to newest.
        """
        points = []
        for entry in self._buffer:
            if entry.pose.is_valid and landmark_idx < len(entry.pose.landmarks):
                lm = entry.pose.landmarks[landmark_idx]
                if lm.visibility > 0.5:
                    points.append(lm.to_pixel(width, height))
        return points
    
    def get_velocity(self, landmark_idx: int) -> float:
        """
        Get the current velocity of a landmark.
        
        Args:
            landmark_idx: Index of the landmark.
        
        Returns:
            Velocity magnitude (0 if not available).
        """
        if not self._buffer:
            return 0.0
        latest = self._buffer[-1]
        return latest.velocities.get(landmark_idx, 0.0)
    
    def get_max_velocity(self) -> float:
        """Get the maximum velocity across all landmarks in current frame."""
        if not self._buffer:
            return 0.0
        latest = self._buffer[-1]
        return max(latest.velocities.values()) if latest.velocities else 0.0
    
    def clear(self) -> None:
        """Clear the history buffer."""
        self._buffer.clear()
        self._last_positions.clear()
    
    def __len__(self) -> int:
        return len(self._buffer)


# ============================================================================
# Skeleton Renderer Class
# ============================================================================

@dataclass
class SkeletonRendererOptions:
    """Configuration options for SkeletonRenderer."""
    
    # Rendering mode
    mode: RenderMode = RenderMode.STANDARD
    
    # Basic settings
    thickness: int = 3
    joint_radius: int = 5
    min_visibility: float = 0.5
    golf_mode: bool = True
    
    # Glow settings
    enable_glow: bool = True
    glow_passes: list[tuple[int, float]] = field(
        default_factory=lambda: [(31, 0.3), (15, 0.5), (7, 0.7)]
    )
    
    # Trail settings
    enable_trails: bool = True
    trail_length: int = 12
    trail_landmarks: list[int] = field(
        default_factory=lambda: [15, 16]  # Wrists by default
    )
    
    # Depth coloring
    enable_depth_coloring: bool = False
    near_color: tuple[int, int, int] = (0, 100, 255)   # Warm (close)
    far_color: tuple[int, int, int] = (255, 100, 0)    # Cool (far)
    
    # Velocity effects
    enable_velocity_intensity: bool = True
    velocity_min: float = 0.01
    velocity_max: float = 0.15
    
    # Phase coloring
    enable_phase_colors: bool = True
    
    # Joint styling
    joint_style: str = "circle"  # "circle", "diamond", "hexagon"
    key_joints: list[int] = field(
        default_factory=lambda: [15, 16, 13, 14, 11, 12]  # Wrists, elbows, shoulders
    )
    
    # Bone styling
    bone_style: str = "line"  # "line", "capsule"
    
    # Impact effects
    enable_impact_particles: bool = True
    
    @classmethod
    def from_mode(cls, mode: RenderMode) -> "SkeletonRendererOptions":
        """Create options from a preset mode."""
        if mode == RenderMode.MINIMAL:
            return cls(
                mode=mode,
                enable_glow=True,
                enable_trails=False,
                enable_depth_coloring=False,
                enable_velocity_intensity=False,
                enable_phase_colors=True,
                joint_style="circle",
                bone_style="line",
                glow_passes=[(15, 0.4), (7, 0.6)],
            )
        elif mode == RenderMode.CINEMATIC:
            return cls(
                mode=mode,
                enable_glow=True,
                enable_trails=True,
                trail_length=18,
                enable_depth_coloring=True,
                enable_velocity_intensity=True,
                enable_phase_colors=True,
                joint_style="diamond",
                bone_style="capsule",
                enable_impact_particles=True,
                glow_passes=[(41, 0.25), (21, 0.4), (11, 0.6), (5, 0.8)],
            )
        else:  # STANDARD
            return cls(mode=mode)


class SkeletonRenderer:
    """
    Advanced skeleton renderer with multi-pass effects.
    
    Provides neon glow, motion trails, depth coloring, phase-aware theming,
    and velocity-based intensity modulation.
    """
    
    def __init__(
        self,
        options: Optional[SkeletonRendererOptions] = None,
    ):
        """
        Initialize the skeleton renderer.
        
        Args:
            options: Rendering options (uses defaults if None).
        """
        self.options = options or SkeletonRendererOptions()
        self.history = PoseHistory(max_size=self.options.trail_length)
        self._current_theme = ColorTheme.default()
    
    def set_mode(self, mode: RenderMode) -> None:
        """Switch rendering mode."""
        self.options = SkeletonRendererOptions.from_mode(mode)
        self.history = PoseHistory(max_size=self.options.trail_length)
    
    def render(
        self,
        frame: np.ndarray,
        pose: FramePose,
        phase: SwingPhase = SwingPhase.IDLE,
        is_impact: bool = False,
        custom_theme: Optional[ColorTheme] = None,
    ) -> np.ndarray:
        """
        Render skeleton with all effects onto frame.
        
        Args:
            frame: Input frame (BGR).
            pose: Current pose to render.
            phase: Current swing phase for theming.
            is_impact: Whether this is an impact frame.
            custom_theme: Optional custom color theme override.
        
        Returns:
            Frame with skeleton overlay.
        """
        if not pose.is_valid:
            return frame
        
        h, w = frame.shape[:2]
        
        # Update pose history for trails
        self.history.add(pose, pose.timestamp)
        
        # Select color theme
        if custom_theme:
            theme = custom_theme
        elif self.options.enable_phase_colors:
            theme = PHASE_THEMES.get(phase, ColorTheme.default())
        else:
            theme = ColorTheme.default()
        
        self._current_theme = theme
        
        # Create skeleton layer (BGRA for alpha compositing)
        skeleton_layer = create_transparent_layer(w, h)
        
        # Get connections
        connections = (
            GOLF_SKELETON_CONNECTIONS if self.options.golf_mode
            else POSE_CONNECTIONS
        )
        
        # Calculate global intensity based on max velocity
        max_vel = self.history.get_max_velocity()
        global_intensity = 1.0
        if self.options.enable_velocity_intensity:
            global_intensity = velocity_to_intensity(
                max_vel,
                self.options.velocity_min,
                self.options.velocity_max,
                min_intensity=0.7,
                max_intensity=1.5,
            )
        
        # === Pass 1: Draw motion trails ===
        if self.options.enable_trails and len(self.history) > 2:
            skeleton_layer = self._draw_trails(skeleton_layer, w, h, theme)
        
        # === Pass 2: Draw bones ===
        skeleton_layer = self._draw_bones(
            skeleton_layer, pose, connections, w, h, theme, global_intensity
        )
        
        # === Pass 3: Draw joints ===
        skeleton_layer = self._draw_joints(
            skeleton_layer, pose, w, h, theme, global_intensity
        )
        
        # === Pass 4: Impact particles ===
        if is_impact and self.options.enable_impact_particles:
            # Add particles at wrists
            for wrist_idx in [15, 16]:
                if wrist_idx < len(pose.landmarks):
                    wrist = pose.landmarks[wrist_idx]
                    if wrist.visibility > self.options.min_visibility:
                        pt = wrist.to_pixel(w, h)
                        skeleton_layer = create_particle_burst(
                            skeleton_layer,
                            pt,
                            theme.joint_primary,
                            num_particles=20,
                            radius=50,
                            particle_size=4,
                        )
        
        # === Final: Apply glow and composite ===
        if self.options.enable_glow:
            result = apply_glow_effect(
                frame,
                skeleton_layer,
                glow_passes=self.options.glow_passes,
                blend_mode="additive",
            )
        else:
            result = alpha_composite(frame, skeleton_layer)
        
        return result
    
    def _draw_trails(
        self,
        layer: np.ndarray,
        w: int,
        h: int,
        theme: ColorTheme,
    ) -> np.ndarray:
        """Draw motion trails for tracked landmarks."""
        for landmark_idx in self.options.trail_landmarks:
            points = self.history.get_trail_points(landmark_idx, w, h)
            if len(points) >= 2:
                layer = draw_motion_trail(
                    layer,
                    points,
                    theme.trail,
                    max_thickness=self.options.thickness + 2,
                    min_alpha=30,
                    max_alpha=180,
                )
        return layer
    
    def _draw_bones(
        self,
        layer: np.ndarray,
        pose: FramePose,
        connections: list[tuple[int, int]],
        w: int,
        h: int,
        theme: ColorTheme,
        intensity: float,
    ) -> np.ndarray:
        """Draw skeleton bones."""
        # Sort by depth if depth coloring enabled
        if self.options.enable_depth_coloring:
            # Sort connections by average Z (back to front)
            def get_avg_z(conn: tuple[int, int]) -> float:
                start_idx, end_idx = conn
                if start_idx >= len(pose.landmarks) or end_idx >= len(pose.landmarks):
                    return 0.0
                return (pose.landmarks[start_idx].z + pose.landmarks[end_idx].z) / 2
            
            connections = sorted(connections, key=get_avg_z, reverse=True)
        
        for start_idx, end_idx in connections:
            if start_idx >= len(pose.landmarks) or end_idx >= len(pose.landmarks):
                continue
            
            start_lm = pose.landmarks[start_idx]
            end_lm = pose.landmarks[end_idx]
            
            # Check visibility
            if (start_lm.visibility < self.options.min_visibility or
                end_lm.visibility < self.options.min_visibility):
                continue
            
            start_pt = start_lm.to_pixel(w, h)
            end_pt = end_lm.to_pixel(w, h)
            
            # Determine bone color
            if self.options.enable_depth_coloring:
                avg_z = (start_lm.z + end_lm.z) / 2
                bone_color = depth_to_color(
                    avg_z,
                    self.options.near_color,
                    self.options.far_color,
                )
            else:
                bone_color = theme.primary
            
            # Modulate color by intensity
            if intensity > 1.0:
                bone_color = tuple(
                    min(255, int(c * intensity)) for c in bone_color
                )
            
            # Draw bone
            if self.options.bone_style == "capsule":
                layer = draw_capsule_bone(
                    layer, start_pt, end_pt, bone_color,
                    radius=self.options.thickness + 2,
                )
            else:
                layer = draw_glowing_line(
                    layer, start_pt, end_pt, bone_color,
                    thickness=self.options.thickness,
                    glow_radius=6,
                )
        
        return layer
    
    def _draw_joints(
        self,
        layer: np.ndarray,
        pose: FramePose,
        w: int,
        h: int,
        theme: ColorTheme,
        intensity: float,
    ) -> np.ndarray:
        """Draw skeleton joints."""
        for idx, lm in enumerate(pose.landmarks):
            if lm.visibility < self.options.min_visibility:
                continue
            
            pt = lm.to_pixel(w, h)
            
            # Determine if this is a key joint
            is_key = idx in self.options.key_joints
            
            # Get joint-specific velocity for pulsing effect
            velocity = self.history.get_velocity(idx)
            vel_scale = 1.0
            if self.options.enable_velocity_intensity:
                vel_scale = velocity_to_intensity(
                    velocity,
                    self.options.velocity_min,
                    self.options.velocity_max,
                    min_intensity=0.8,
                    max_intensity=1.3,
                )
            
            # Calculate radius with velocity scaling
            base_radius = self.options.joint_radius
            if is_key:
                base_radius = int(base_radius * 1.3)
            radius = int(base_radius * vel_scale)
            
            # Select color
            if self.options.enable_depth_coloring:
                joint_color = depth_to_color(
                    lm.z,
                    self.options.near_color,
                    self.options.far_color,
                )
            else:
                joint_color = theme.joint_primary if is_key else theme.joint_secondary
            
            # Draw joint based on style
            if self.options.joint_style == "diamond" and is_key:
                layer = draw_diamond(layer, pt, radius, joint_color, glow_radius=4)
            elif self.options.joint_style == "hexagon" and is_key:
                layer = draw_hexagon(layer, pt, radius, joint_color, glow_radius=4)
            else:
                layer = draw_glowing_circle(
                    layer, pt, radius, joint_color, glow_radius=4
                )
        
        return layer
    
    def reset(self) -> None:
        """Reset the renderer state (clears history)."""
        self.history.clear()


def draw_pose_skeleton(
    frame: np.ndarray,
    pose: FramePose,
    color: tuple[int, int, int] = DEFAULT_SKELETON_COLOR,
    landmark_color: Optional[tuple[int, int, int]] = None,
    thickness: int = 2,
    landmark_radius: int = 4,
    min_visibility: float = 0.5,
    golf_mode: bool = True,
) -> np.ndarray:
    """
    Draw pose skeleton overlay on a video frame.
    
    Args:
        frame: Input frame (will be modified in place).
        pose: FramePose with landmarks to draw.
        color: BGR color for skeleton lines.
        landmark_color: BGR color for landmarks (None = same as skeleton).
        thickness: Line thickness.
        landmark_radius: Radius of landmark circles.
        min_visibility: Minimum visibility threshold to draw landmark.
        golf_mode: Use simplified golf-specific skeleton.
    
    Returns:
        Frame with skeleton overlay.
    """
    if not pose.is_valid:
        return frame
    
    h, w = frame.shape[:2]
    landmark_color = landmark_color or color
    
    # Choose connection set
    connections = GOLF_SKELETON_CONNECTIONS if golf_mode else POSE_CONNECTIONS
    
    # Draw connections
    for start_idx, end_idx in connections:
        if start_idx >= len(pose.landmarks) or end_idx >= len(pose.landmarks):
            continue
        
        start_lm = pose.landmarks[start_idx]
        end_lm = pose.landmarks[end_idx]
        
        # Check visibility
        if start_lm.visibility < min_visibility or end_lm.visibility < min_visibility:
            continue
        
        # Convert to pixel coordinates
        start_pt = start_lm.to_pixel(w, h)
        end_pt = end_lm.to_pixel(w, h)
        
        # Draw line
        cv2.line(frame, start_pt, end_pt, color, thickness, cv2.LINE_AA)
    
    # Draw landmarks
    for lm in pose.landmarks:
        if lm.visibility < min_visibility:
            continue
        
        pt = lm.to_pixel(w, h)
        cv2.circle(frame, pt, landmark_radius, landmark_color, -1, cv2.LINE_AA)
    
    return frame


def draw_audio_waveform(
    frame: np.ndarray,
    audio: AudioData,
    current_time: float,
    window_sec: float = 5.0,
    height: int = 80,
    color: tuple[int, int, int] = DEFAULT_WAVEFORM_COLOR,
    bg_color: tuple[int, int, int] = DEFAULT_WAVEFORM_BG,
    impact_events: Optional[list[ImpactEvent]] = None,
    impact_color: tuple[int, int, int] = DEFAULT_IMPACT_COLOR,
    position: str = "bottom",
) -> np.ndarray:
    """
    Draw audio waveform overlay on a video frame.
    
    Args:
        frame: Input frame (will be modified in place).
        audio: AudioData to visualize.
        current_time: Current playback time in seconds.
        window_sec: Width of waveform window in seconds.
        height: Height of waveform strip in pixels.
        color: BGR color for waveform.
        bg_color: BGR color for background.
        impact_events: Optional list of impact events to mark.
        impact_color: BGR color for impact markers.
        position: "top" or "bottom" of frame.
    
    Returns:
        Frame with waveform overlay.
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Calculate waveform position
    if position == "top":
        y_start = 0
    else:
        y_start = frame_h - height
    
    # Create waveform background
    waveform_bg = np.full((height, frame_w, 3), bg_color, dtype=np.uint8)
    
    # Calculate time window
    half_window = window_sec / 2
    start_time = max(0, current_time - half_window)
    end_time = min(audio.duration, current_time + half_window)
    
    # Get audio samples for window
    start_sample = int(start_time * audio.sample_rate)
    end_sample = int(end_time * audio.sample_rate)
    samples = audio.samples[start_sample:end_sample]
    
    if len(samples) == 0:
        # Paste empty background and return
        frame[y_start:y_start + height, :] = waveform_bg
        return frame
    
    # Downsample to match frame width
    num_bins = frame_w
    bin_size = max(1, len(samples) // num_bins)
    
    # Calculate envelope (max amplitude per bin)
    envelope = np.zeros(num_bins)
    for i in range(min(num_bins, len(samples) // bin_size)):
        bin_start = i * bin_size
        bin_end = min(bin_start + bin_size, len(samples))
        if bin_start < bin_end:
            envelope[i] = np.max(np.abs(samples[bin_start:bin_end]))
    
    # Normalize envelope
    max_amp = np.max(envelope) if np.max(envelope) > 0 else 1
    envelope = envelope / max_amp
    
    # Draw waveform
    center_y = height // 2
    for x in range(num_bins):
        amp = int(envelope[x] * (height // 2 - 2))
        if amp > 0:
            cv2.line(
                waveform_bg,
                (x, center_y - amp),
                (x, center_y + amp),
                color,
                1,
            )
    
    # Draw current time indicator (vertical line)
    time_position = (current_time - start_time) / (end_time - start_time) if end_time > start_time else 0.5
    indicator_x = int(time_position * frame_w)
    cv2.line(waveform_bg, (indicator_x, 0), (indicator_x, height), (255, 255, 255), 2)
    
    # Draw impact markers
    if impact_events:
        for event in impact_events:
            if start_time <= event.timestamp <= end_time:
                event_position = (event.timestamp - start_time) / (end_time - start_time)
                event_x = int(event_position * frame_w)
                
                # Draw triangle marker
                pts = np.array([
                    [event_x, 5],
                    [event_x - 5, 0],
                    [event_x + 5, 0],
                ], np.int32)
                cv2.fillPoly(waveform_bg, [pts], impact_color)
                
                # Draw vertical line
                cv2.line(waveform_bg, (event_x, 5), (event_x, height - 5), impact_color, 1)
    
    # Paste waveform onto frame
    frame[y_start:y_start + height, :] = waveform_bg
    
    return frame


def draw_swing_phase_label(
    frame: np.ndarray,
    phase: SwingPhase,
    color: tuple[int, int, int] = DEFAULT_PHASE_COLOR,
    font_scale: float = 1.0,
    position: tuple[int, int] = (20, 40),
    show_background: bool = True,
) -> np.ndarray:
    """
    Draw swing phase label on a video frame.
    
    Args:
        frame: Input frame (will be modified in place).
        phase: Current swing phase.
        color: BGR color for text.
        font_scale: Font size scale.
        position: (x, y) position for label.
        show_background: Whether to show semi-transparent background.
    
    Returns:
        Frame with phase label.
    """
    # Phase display names
    phase_names = {
        SwingPhase.IDLE: "IDLE",
        SwingPhase.ADDRESS: "ADDRESS",
        SwingPhase.BACKSWING: "BACKSWING",
        SwingPhase.TOP: "TOP",
        SwingPhase.DOWNSWING: "DOWNSWING",
        SwingPhase.IMPACT: "IMPACT",
        SwingPhase.FOLLOW_THROUGH: "FOLLOW-THROUGH",
        SwingPhase.FINISH: "FINISH",
    }
    
    # Phase colors (BGR)
    phase_colors = {
        SwingPhase.IDLE: (128, 128, 128),
        SwingPhase.ADDRESS: (255, 200, 0),
        SwingPhase.BACKSWING: (0, 255, 255),
        SwingPhase.TOP: (0, 200, 255),
        SwingPhase.DOWNSWING: (0, 128, 255),
        SwingPhase.IMPACT: (0, 255, 0),
        SwingPhase.FOLLOW_THROUGH: (255, 128, 0),
        SwingPhase.FINISH: (255, 0, 128),
    }
    
    label = phase_names.get(phase, "UNKNOWN")
    label_color = phase_colors.get(phase, color)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background
    if show_background:
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            (0, 0, 0),
            -1,
        )
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            label_color,
            2,
        )
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw text
    cv2.putText(frame, label, (x, y), font, font_scale, label_color, thickness, cv2.LINE_AA)
    
    return frame


def draw_impact_marker(
    frame: np.ndarray,
    is_impact: bool,
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 30,
    position: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Draw impact indicator on frame (visual flash/circle).
    
    Args:
        frame: Input frame (will be modified in place).
        is_impact: Whether this is an impact frame.
        color: BGR color for impact indicator.
        radius: Radius of impact circle.
        position: Position for indicator (None = center of frame).
    
    Returns:
        Frame with impact indicator.
    """
    if not is_impact:
        return frame
    
    h, w = frame.shape[:2]
    
    if position is None:
        position = (w // 2, h // 2)
    
    # Draw pulsing circle effect
    cv2.circle(frame, position, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, position, radius - 10, color, 2, cv2.LINE_AA)
    
    # Add glow effect
    overlay = frame.copy()
    cv2.circle(overlay, position, radius + 20, color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    return frame


def draw_timestamp(
    frame: np.ndarray,
    timestamp: float,
    color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    position: str = "top_right",
) -> np.ndarray:
    """
    Draw timestamp on video frame.
    
    Args:
        frame: Input frame.
        timestamp: Time in seconds.
        color: BGR color for text.
        font_scale: Font scale.
        position: "top_left", "top_right", "bottom_left", "bottom_right".
    
    Returns:
        Frame with timestamp.
    """
    h, w = frame.shape[:2]
    
    # Format timestamp
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    ms = int((timestamp % 1) * 100)
    text = f"{minutes:02d}:{seconds:02d}.{ms:02d}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    padding = 10
    
    if position == "top_left":
        x, y = padding, text_h + padding
    elif position == "top_right":
        x, y = w - text_w - padding, text_h + padding
    elif position == "bottom_left":
        x, y = padding, h - padding
    else:  # bottom_right
        x, y = w - text_w - padding, h - padding
    
    # Draw background
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + 5),
        (0, 0, 0),
        -1,
    )
    
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def draw_confidence_bar(
    frame: np.ndarray,
    confidence: float,
    label: str = "Confidence",
    position: tuple[int, int] = (20, 80),
    width: int = 150,
    height: int = 15,
) -> np.ndarray:
    """
    Draw a confidence bar on the frame.
    
    Args:
        frame: Input frame.
        confidence: Confidence value (0-1).
        label: Label text.
        position: (x, y) position.
        width: Bar width.
        height: Bar height.
    
    Returns:
        Frame with confidence bar.
    """
    x, y = position
    
    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Fill based on confidence
    fill_width = int(width * min(1.0, max(0.0, confidence)))
    
    # Color gradient (red -> yellow -> green)
    if confidence < 0.5:
        color = (0, int(255 * confidence * 2), 255)
    else:
        color = (0, 255, int(255 * (1 - confidence) * 2))
    
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
    
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height), (200, 200, 200), 1)
    
    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_text = f"{label}: {confidence:.0%}"
    cv2.putText(frame, label_text, (x, y - 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

