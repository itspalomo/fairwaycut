"""Visual effects utilities for skeleton visualization.

This module provides reusable visual effects like glow, blur, and blending
for creating stunning pose skeleton overlays.
"""

from typing import Optional
import cv2
import numpy as np


def create_glow_layer(
    image: np.ndarray,
    blur_size: int = 21,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Create a glowing layer from an image by applying Gaussian blur.
    
    Args:
        image: Source image (BGR or BGRA).
        blur_size: Size of Gaussian blur kernel (must be odd).
        intensity: Brightness multiplier for the glow.
    
    Returns:
        Blurred glow layer.
    """
    # Ensure odd blur size
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    
    if intensity != 1.0:
        blurred = np.clip(blurred * intensity, 0, 255).astype(np.uint8)
    
    return blurred


def apply_glow_effect(
    base_frame: np.ndarray,
    skeleton_layer: np.ndarray,
    glow_passes: list[tuple[int, float]] = None,
    blend_mode: str = "additive",
) -> np.ndarray:
    """
    Apply multi-pass glow effect to skeleton layer and composite onto base frame.
    
    Args:
        base_frame: Background frame to composite onto.
        skeleton_layer: BGRA image with skeleton drawn on transparent background.
        glow_passes: List of (blur_size, intensity) tuples for each glow pass.
                    Default: [(41, 0.4), (21, 0.6), (11, 0.8)]
        blend_mode: "additive", "screen", or "overlay".
    
    Returns:
        Composited frame with glow effect.
    """
    if glow_passes is None:
        glow_passes = [(41, 0.4), (21, 0.6), (11, 0.8)]
    
    result = base_frame.copy()
    
    # Extract RGB from skeleton layer (ignoring alpha for glow)
    if skeleton_layer.shape[2] == 4:
        skeleton_rgb = skeleton_layer[:, :, :3]
        skeleton_alpha = skeleton_layer[:, :, 3:4] / 255.0
    else:
        skeleton_rgb = skeleton_layer
        skeleton_alpha = np.ones((skeleton_layer.shape[0], skeleton_layer.shape[1], 1))
    
    # Apply each glow pass
    for blur_size, intensity in glow_passes:
        glow = create_glow_layer(skeleton_rgb, blur_size, intensity)
        result = blend_layers(result, glow, blend_mode, opacity=0.7)
    
    # Composite the sharp skeleton on top
    result = alpha_composite(result, skeleton_layer)
    
    return result


def blend_layers(
    base: np.ndarray,
    overlay: np.ndarray,
    mode: str = "additive",
    opacity: float = 1.0,
) -> np.ndarray:
    """
    Blend two layers using various blend modes.
    
    Args:
        base: Base layer (BGR).
        overlay: Overlay layer (BGR).
        mode: Blend mode - "additive", "screen", "overlay", "multiply".
        opacity: Opacity of the overlay (0-1).
    
    Returns:
        Blended result.
    """
    base_float = base.astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    
    if mode == "additive":
        # Simple additive blending (great for glow)
        blended = base_float + overlay_float * opacity
        
    elif mode == "screen":
        # Screen blend: 1 - (1-a)(1-b) - lighter result
        base_norm = base_float / 255.0
        overlay_norm = overlay_float / 255.0
        blended = (1.0 - (1.0 - base_norm) * (1.0 - overlay_norm * opacity)) * 255.0
        
    elif mode == "overlay":
        # Overlay blend: combination of multiply and screen
        base_norm = base_float / 255.0
        overlay_norm = overlay_float / 255.0
        
        mask = base_norm < 0.5
        blended = np.where(
            mask[:, :, :, np.newaxis] if len(mask.shape) == 2 else mask,
            2 * base_norm * overlay_norm * opacity,
            1.0 - 2 * (1.0 - base_norm) * (1.0 - overlay_norm * opacity)
        ) * 255.0
        
    elif mode == "multiply":
        # Multiply blend: a * b - darker result
        base_norm = base_float / 255.0
        overlay_norm = overlay_float / 255.0
        blended = base_norm * (1.0 - opacity + overlay_norm * opacity) * 255.0
        
    else:
        # Default to normal blend
        blended = base_float * (1.0 - opacity) + overlay_float * opacity
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def alpha_composite(
    background: np.ndarray,
    foreground: np.ndarray,
) -> np.ndarray:
    """
    Alpha composite foreground (BGRA) over background (BGR).
    
    Args:
        background: Background image (BGR).
        foreground: Foreground image with alpha channel (BGRA).
    
    Returns:
        Composited BGR image.
    """
    if foreground.shape[2] != 4:
        # No alpha channel, just overlay
        return foreground
    
    # Extract alpha channel and normalize
    alpha = foreground[:, :, 3:4] / 255.0
    fg_rgb = foreground[:, :, :3]
    
    # Composite
    result = background * (1 - alpha) + fg_rgb * alpha
    
    return result.astype(np.uint8)


def create_transparent_layer(
    width: int,
    height: int,
) -> np.ndarray:
    """
    Create a transparent BGRA layer.
    
    Args:
        width: Layer width.
        height: Layer height.
    
    Returns:
        Transparent BGRA numpy array.
    """
    return np.zeros((height, width, 4), dtype=np.uint8)


def draw_glowing_line(
    layer: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    glow_radius: int = 8,
) -> np.ndarray:
    """
    Draw a line with built-in glow effect on a BGRA layer.
    
    Args:
        layer: BGRA layer to draw on.
        pt1: Start point (x, y).
        pt2: End point (x, y).
        color: BGR color.
        thickness: Line thickness.
        glow_radius: Radius of glow around line.
    
    Returns:
        Layer with glowing line.
    """
    # Draw outer glow (thicker, blurred)
    glow_color = tuple(min(255, int(c * 0.5)) for c in color)
    cv2.line(layer, pt1, pt2, (*glow_color, 80), thickness + glow_radius * 2, cv2.LINE_AA)
    cv2.line(layer, pt1, pt2, (*glow_color, 120), thickness + glow_radius, cv2.LINE_AA)
    
    # Draw core line
    cv2.line(layer, pt1, pt2, (*color, 255), thickness, cv2.LINE_AA)
    
    return layer


def draw_glowing_circle(
    layer: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    glow_radius: int = 6,
    filled: bool = True,
) -> np.ndarray:
    """
    Draw a circle with built-in glow effect on a BGRA layer.
    
    Args:
        layer: BGRA layer to draw on.
        center: Center point (x, y).
        radius: Circle radius.
        color: BGR color.
        glow_radius: Radius of glow around circle.
        filled: Whether to fill the circle.
    
    Returns:
        Layer with glowing circle.
    """
    # Draw outer glow
    glow_color = tuple(min(255, int(c * 0.5)) for c in color)
    cv2.circle(layer, center, radius + glow_radius, (*glow_color, 60), -1, cv2.LINE_AA)
    cv2.circle(layer, center, radius + glow_radius // 2, (*glow_color, 100), -1, cv2.LINE_AA)
    
    # Draw core circle
    fill = -1 if filled else 2
    cv2.circle(layer, center, radius, (*color, 255), fill, cv2.LINE_AA)
    
    return layer


def draw_diamond(
    layer: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    glow_radius: int = 6,
    filled: bool = True,
) -> np.ndarray:
    """
    Draw a diamond shape with glow effect.
    
    Args:
        layer: BGRA layer to draw on.
        center: Center point (x, y).
        size: Size of diamond (half-diagonal).
        color: BGR color.
        glow_radius: Radius of glow.
        filled: Whether to fill the diamond.
    
    Returns:
        Layer with glowing diamond.
    """
    cx, cy = center
    points = np.array([
        [cx, cy - size],      # Top
        [cx + size, cy],      # Right
        [cx, cy + size],      # Bottom
        [cx - size, cy],      # Left
    ], dtype=np.int32)
    
    # Draw glow
    glow_color = tuple(min(255, int(c * 0.5)) for c in color)
    glow_points = np.array([
        [cx, cy - size - glow_radius],
        [cx + size + glow_radius, cy],
        [cx, cy + size + glow_radius],
        [cx - size - glow_radius, cy],
    ], dtype=np.int32)
    cv2.fillPoly(layer, [glow_points], (*glow_color, 60), cv2.LINE_AA)
    
    # Draw core
    if filled:
        cv2.fillPoly(layer, [points], (*color, 255), cv2.LINE_AA)
    else:
        cv2.polylines(layer, [points], True, (*color, 255), 2, cv2.LINE_AA)
    
    return layer


def draw_hexagon(
    layer: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
    glow_radius: int = 6,
    filled: bool = True,
) -> np.ndarray:
    """
    Draw a hexagon shape with glow effect.
    
    Args:
        layer: BGRA layer to draw on.
        center: Center point (x, y).
        size: Radius of hexagon.
        color: BGR color.
        glow_radius: Radius of glow.
        filled: Whether to fill the hexagon.
    
    Returns:
        Layer with glowing hexagon.
    """
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    
    points = np.array([
        [int(cx + size * np.cos(a)), int(cy + size * np.sin(a))]
        for a in angles
    ], dtype=np.int32)
    
    # Draw glow
    glow_color = tuple(min(255, int(c * 0.5)) for c in color)
    glow_size = size + glow_radius
    glow_points = np.array([
        [int(cx + glow_size * np.cos(a)), int(cy + glow_size * np.sin(a))]
        for a in angles
    ], dtype=np.int32)
    cv2.fillPoly(layer, [glow_points], (*glow_color, 60), cv2.LINE_AA)
    
    # Draw core
    if filled:
        cv2.fillPoly(layer, [points], (*color, 255), cv2.LINE_AA)
    else:
        cv2.polylines(layer, [points], True, (*color, 255), 2, cv2.LINE_AA)
    
    return layer


def interpolate_color(
    color1: tuple[int, int, int],
    color2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """
    Linearly interpolate between two BGR colors.
    
    Args:
        color1: Start color (BGR).
        color2: End color (BGR).
        t: Interpolation factor (0 = color1, 1 = color2).
    
    Returns:
        Interpolated BGR color.
    """
    t = max(0.0, min(1.0, t))
    return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))


def create_gradient_colormap(
    colors: list[tuple[int, int, int]],
    steps: int = 256,
) -> np.ndarray:
    """
    Create a gradient colormap from a list of colors.
    
    Args:
        colors: List of BGR colors to interpolate between.
        steps: Number of steps in the colormap.
    
    Returns:
        Colormap array of shape (steps, 3).
    """
    if len(colors) < 2:
        return np.array([colors[0]] * steps, dtype=np.uint8)
    
    colormap = []
    segments = len(colors) - 1
    steps_per_segment = steps // segments
    
    for i in range(segments):
        for j in range(steps_per_segment):
            t = j / steps_per_segment
            color = interpolate_color(colors[i], colors[i + 1], t)
            colormap.append(color)
    
    # Fill remaining steps
    while len(colormap) < steps:
        colormap.append(colors[-1])
    
    return np.array(colormap[:steps], dtype=np.uint8)


def depth_to_color(
    z: float,
    near_color: tuple[int, int, int] = (0, 100, 255),   # Warm orange (close)
    far_color: tuple[int, int, int] = (255, 100, 0),    # Cool blue (far)
    z_range: tuple[float, float] = (-0.5, 0.5),
) -> tuple[int, int, int]:
    """
    Map depth value to color for pseudo-3D effect.
    
    Args:
        z: Depth value from MediaPipe (typically -0.5 to 0.5).
        near_color: Color for closest points (BGR).
        far_color: Color for farthest points (BGR).
        z_range: Expected range of z values.
    
    Returns:
        BGR color for the depth.
    """
    z_min, z_max = z_range
    t = (z - z_min) / (z_max - z_min) if z_max > z_min else 0.5
    t = max(0.0, min(1.0, t))
    
    return interpolate_color(near_color, far_color, t)


def velocity_to_intensity(
    velocity: float,
    min_velocity: float = 0.0,
    max_velocity: float = 2.0,
    min_intensity: float = 0.5,
    max_intensity: float = 2.0,
) -> float:
    """
    Map velocity to glow intensity.
    
    Args:
        velocity: Current velocity magnitude.
        min_velocity: Velocity threshold for minimum intensity.
        max_velocity: Velocity threshold for maximum intensity.
        min_intensity: Minimum glow intensity.
        max_intensity: Maximum glow intensity.
    
    Returns:
        Intensity multiplier for glow effect.
    """
    if velocity <= min_velocity:
        return min_intensity
    if velocity >= max_velocity:
        return max_intensity
    
    t = (velocity - min_velocity) / (max_velocity - min_velocity)
    return min_intensity + (max_intensity - min_intensity) * t


def draw_motion_trail(
    layer: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    max_thickness: int = 4,
    min_alpha: int = 20,
    max_alpha: int = 200,
) -> np.ndarray:
    """
    Draw a motion trail with fading opacity.
    
    Args:
        layer: BGRA layer to draw on.
        points: List of (x, y) points from oldest to newest.
        color: BGR color for the trail.
        max_thickness: Thickness at the newest point.
        min_alpha: Alpha at the oldest point.
        max_alpha: Alpha at the newest point.
    
    Returns:
        Layer with motion trail.
    """
    if len(points) < 2:
        return layer
    
    n = len(points)
    for i in range(n - 1):
        # Calculate interpolation factor (0 = oldest, 1 = newest)
        t = i / (n - 1)
        
        # Interpolate thickness and alpha
        thickness = max(1, int(max_thickness * t))
        alpha = int(min_alpha + (max_alpha - min_alpha) * t)
        
        # Draw segment
        cv2.line(
            layer,
            points[i],
            points[i + 1],
            (*color, alpha),
            thickness,
            cv2.LINE_AA,
        )
    
    return layer


def create_particle_burst(
    layer: np.ndarray,
    center: tuple[int, int],
    color: tuple[int, int, int],
    num_particles: int = 12,
    radius: int = 30,
    particle_size: int = 3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Create a particle burst effect at a point.
    
    Args:
        layer: BGRA layer to draw on.
        center: Center of the burst.
        color: BGR color for particles.
        num_particles: Number of particles.
        radius: Radius of the burst.
        particle_size: Size of each particle.
        seed: Random seed for reproducibility.
    
    Returns:
        Layer with particle burst.
    """
    if seed is not None:
        np.random.seed(seed)
    
    cx, cy = center
    angles = np.random.uniform(0, 2 * np.pi, num_particles)
    distances = np.random.uniform(radius * 0.3, radius, num_particles)
    
    for angle, dist in zip(angles, distances):
        px = int(cx + dist * np.cos(angle))
        py = int(cy + dist * np.sin(angle))
        
        # Fade alpha with distance
        alpha = int(255 * (1 - dist / radius))
        
        cv2.circle(layer, (px, py), particle_size, (*color, alpha), -1, cv2.LINE_AA)
    
    return layer


def draw_capsule_bone(
    layer: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 6,
    highlight_color: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Draw a capsule-shaped bone (rounded ends) with subtle 3D shading.
    
    Args:
        layer: BGRA layer to draw on.
        pt1: Start point.
        pt2: End point.
        color: Base BGR color.
        radius: Radius of the capsule.
        highlight_color: Optional highlight color for 3D effect.
    
    Returns:
        Layer with capsule bone.
    """
    if highlight_color is None:
        # Create lighter highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
    
    # Calculate perpendicular offset for highlight
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx * dx + dy * dy)
    if length == 0:
        return layer
    
    # Draw main capsule (thick line with rounded caps)
    cv2.line(layer, pt1, pt2, (*color, 255), radius * 2, cv2.LINE_AA)
    cv2.circle(layer, pt1, radius, (*color, 255), -1, cv2.LINE_AA)
    cv2.circle(layer, pt2, radius, (*color, 255), -1, cv2.LINE_AA)
    
    # Draw highlight line (offset slightly)
    offset = radius // 3
    nx = -dy / length * offset
    ny = dx / length * offset
    
    h_pt1 = (int(pt1[0] + nx), int(pt1[1] + ny))
    h_pt2 = (int(pt2[0] + nx), int(pt2[1] + ny))
    
    cv2.line(layer, h_pt1, h_pt2, (*highlight_color, 150), radius // 2, cv2.LINE_AA)
    
    return layer

