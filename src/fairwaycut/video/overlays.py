"""Video overlay utilities for pose skeleton and audio visualization."""

from typing import Optional
import cv2
import numpy as np

from fairwaycut.core.models import (
    FramePose,
    AudioData,
    SwingPhase,
    DetectionResult,
    ImpactEvent,
)
from fairwaycut.pose.landmarks import GOLF_SKELETON_CONNECTIONS, POSE_CONNECTIONS


# Default colors (BGR format)
DEFAULT_SKELETON_COLOR = (0, 255, 128)  # Green
DEFAULT_LANDMARK_COLOR = (0, 200, 255)  # Orange
DEFAULT_WAVEFORM_COLOR = (163, 204, 78)  # Teal (BGR)
DEFAULT_WAVEFORM_BG = (62, 33, 22)  # Dark blue (BGR)
DEFAULT_IMPACT_COLOR = (0, 255, 255)  # Yellow
DEFAULT_PHASE_COLOR = (255, 255, 255)  # White


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

