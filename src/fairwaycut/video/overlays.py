"""Video overlay utilities for pose skeleton, audio waveform, and HUDs.

Visual style is ported from the reference `pose_overlay.py` in the sister
graphics repo: purple bones, neon-green wrists with white outline rings,
fading green wrist trail, single Gaussian-blur glow pass with a sharp pass
on top.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np

from fairwaycut.core.models import (
    FramePose,
    AudioData,
    SwingPhase,
    ImpactEvent,
)
from fairwaycut.pose.landmarks import (
    POSE_CONNECTIONS,
    POSE_LANDMARKS,
)


# ── Style palette (BGR) ──────────────────────────────────────────────────
# Matches /Volumes/PalomoSSD/fairwaycut-graphics/tools/pose_overlay.py
PURPLE = (255, 163, 201)        # #c9a3ff — bones + non-wrist joints
GREEN = (60, 255, 182)          # #b6ff3c — wrists + trail
WHITE = (255, 255, 255)
HUD_BG = (24, 18, 14)           # near-black panel fill
HUD_BORDER = (140, 140, 140)
HUD_LABEL = (170, 170, 170)
WAVEFORM_COLOR = (163, 204, 78)
WAVEFORM_BG = (62, 33, 22)
IMPACT_COLOR = (60, 255, 182)   # green to match wrist accent

# Bone connections drawn by the skeleton renderer. Mirrors the reference
# script's BONES list (head/torso/arms/legs).
BONES: list[tuple[int, int]] = [
    (POSE_LANDMARKS["nose"], POSE_LANDMARKS["left_shoulder"]),
    (POSE_LANDMARKS["nose"], POSE_LANDMARKS["right_shoulder"]),
    (POSE_LANDMARKS["left_shoulder"], POSE_LANDMARKS["right_shoulder"]),
    (POSE_LANDMARKS["left_shoulder"], POSE_LANDMARKS["left_hip"]),
    (POSE_LANDMARKS["right_shoulder"], POSE_LANDMARKS["right_hip"]),
    (POSE_LANDMARKS["left_hip"], POSE_LANDMARKS["right_hip"]),
    (POSE_LANDMARKS["left_shoulder"], POSE_LANDMARKS["left_elbow"]),
    (POSE_LANDMARKS["left_elbow"], POSE_LANDMARKS["left_wrist"]),
    (POSE_LANDMARKS["right_shoulder"], POSE_LANDMARKS["right_elbow"]),
    (POSE_LANDMARKS["right_elbow"], POSE_LANDMARKS["right_wrist"]),
    (POSE_LANDMARKS["left_hip"], POSE_LANDMARKS["left_knee"]),
    (POSE_LANDMARKS["left_knee"], POSE_LANDMARKS["left_ankle"]),
    (POSE_LANDMARKS["right_hip"], POSE_LANDMARKS["right_knee"]),
    (POSE_LANDMARKS["right_knee"], POSE_LANDMARKS["right_ankle"]),
]

JOINTS: list[int] = [
    POSE_LANDMARKS["nose"],
    POSE_LANDMARKS["left_shoulder"], POSE_LANDMARKS["right_shoulder"],
    POSE_LANDMARKS["left_elbow"], POSE_LANDMARKS["right_elbow"],
    POSE_LANDMARKS["left_hip"], POSE_LANDMARKS["right_hip"],
    POSE_LANDMARKS["left_knee"], POSE_LANDMARKS["right_knee"],
    POSE_LANDMARKS["left_ankle"], POSE_LANDMARKS["right_ankle"],
]

WRISTS: list[int] = [POSE_LANDMARKS["left_wrist"], POSE_LANDMARKS["right_wrist"]]
RIGHT_WRIST = POSE_LANDMARKS["right_wrist"]


@dataclass
class SkeletonRendererOptions:
    """Knobs for the skeleton renderer."""

    bone_thickness: int = 4
    joint_radius: int = 7
    wrist_radius: int = 11
    glow_blur: int = 23           # Gaussian kernel size (odd)
    trail_length: int = 50         # Right-wrist positions kept for the trail
    min_visibility: float = 0.4


class SkeletonRenderer:
    """Render pose skeleton in the project's standard look.

    Stages per frame:
      1. Draw bones, joints, wrists, and trail onto a black "glow" layer.
      2. Gaussian-blur that layer and additively blend it into the frame.
      3. Redraw the same elements sharply on top of the glow.
    """

    def __init__(self, options: Optional[SkeletonRendererOptions] = None):
        self.options = options or SkeletonRendererOptions()
        self._wrist_trail: deque[tuple[int, int]] = deque(
            maxlen=self.options.trail_length
        )
        # Buffers for HUD use (peak wrist speed in normalized units / sec).
        self._last_right_wrist: Optional[tuple[float, float, float]] = None
        self._current_speed_norm: float = 0.0
        self._peak_speed_norm: float = 0.0
        # Recent speed history for the HUD sparkline (~last 1.5 s @ 60 fps).
        self._speed_history: deque[float] = deque(maxlen=90)

    def reset(self) -> None:
        self._wrist_trail.clear()
        self._last_right_wrist = None
        self._current_speed_norm = 0.0
        self._peak_speed_norm = 0.0
        self._speed_history.clear()

    @property
    def current_wrist_speed(self) -> float:
        """Latest right-wrist speed in normalized-coords per second."""
        return self._current_speed_norm

    @property
    def peak_wrist_speed(self) -> float:
        """Peak right-wrist speed seen so far in normalized-coords per second."""
        return self._peak_speed_norm

    @property
    def recent_wrist_speeds(self) -> list[float]:
        """Trailing wrist-speed window for the HUD sparkline."""
        return list(self._speed_history)

    def _to_xy(self, lm, w: int, h: int) -> tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    def _update_speed(self, pose: FramePose) -> None:
        rw = pose.landmarks[RIGHT_WRIST]
        if rw.visibility < self.options.min_visibility:
            self._speed_history.append(self._current_speed_norm)
            return
        if self._last_right_wrist is not None:
            prev_x, prev_y, prev_t = self._last_right_wrist
            dt = pose.timestamp - prev_t
            if dt > 0:
                dx = rw.x - prev_x
                dy = rw.y - prev_y
                speed = float(np.sqrt(dx * dx + dy * dy)) / dt
                self._current_speed_norm = speed
                if speed > self._peak_speed_norm:
                    self._peak_speed_norm = speed
        self._last_right_wrist = (rw.x, rw.y, pose.timestamp)
        self._speed_history.append(self._current_speed_norm)

    def _draw_pass(
        self,
        canvas: np.ndarray,
        pose: FramePose,
        w: int,
        h: int,
        glow_pass: bool,
    ) -> None:
        opts = self.options
        bone_thick = opts.bone_thickness + (4 if glow_pass else 0)
        joint_r = opts.joint_radius + (4 if glow_pass else 0)
        wrist_r = opts.wrist_radius + (6 if glow_pass else 0)

        for a, b in BONES:
            la, lb = pose.landmarks[a], pose.landmarks[b]
            if la.visibility < opts.min_visibility or lb.visibility < opts.min_visibility:
                continue
            cv2.line(
                canvas, self._to_xy(la, w, h), self._to_xy(lb, w, h),
                PURPLE, bone_thick, cv2.LINE_AA,
            )

        for jid in JOINTS:
            lm = pose.landmarks[jid]
            if lm.visibility < opts.min_visibility:
                continue
            pt = self._to_xy(lm, w, h)
            cv2.circle(canvas, pt, joint_r, PURPLE, -1, cv2.LINE_AA)
            if not glow_pass:
                cv2.circle(canvas, pt, joint_r, WHITE, 1, cv2.LINE_AA)

        for wid in WRISTS:
            lm = pose.landmarks[wid]
            if lm.visibility < opts.min_visibility:
                continue
            pt = self._to_xy(lm, w, h)
            cv2.circle(canvas, pt, wrist_r, GREEN, -1, cv2.LINE_AA)
            if not glow_pass:
                cv2.circle(canvas, pt, wrist_r, WHITE, 2, cv2.LINE_AA)

        # Right-wrist trail — fades from black (oldest) to green (newest).
        trail = list(self._wrist_trail)
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                p0, p1 = trail[i - 1], trail[i]
                alpha = i / len(trail)
                color = (
                    int(GREEN[0] * alpha),
                    int(GREEN[1] * alpha),
                    int(GREEN[2] * alpha),
                )
                base = max(1, int(2 + alpha * (4 if glow_pass else 3)))
                thick = base + (4 if glow_pass else 0)
                cv2.line(canvas, p0, p1, color, thick, cv2.LINE_AA)

    def render(self, frame: np.ndarray, pose: FramePose) -> np.ndarray:
        if not pose.is_valid:
            return frame

        h, w = frame.shape[:2]
        opts = self.options

        # Track right-wrist for trail + speed readouts.
        rw = pose.landmarks[RIGHT_WRIST]
        if rw.visibility >= opts.min_visibility:
            self._wrist_trail.append((int(rw.x * w), int(rw.y * h)))
        self._update_speed(pose)

        # 1. Glow layer.
        glow = np.zeros_like(frame)
        self._draw_pass(glow, pose, w, h, glow_pass=True)
        k = opts.glow_blur if opts.glow_blur % 2 == 1 else opts.glow_blur + 1
        glow_blurred = cv2.GaussianBlur(glow, (k, k), 0)
        cv2.add(frame, glow_blurred, frame)

        # 2. Sharp pass on top.
        self._draw_pass(frame, pose, w, h, glow_pass=False)

        return frame


# ── HUD: liquid-glass wrist-speed panel + small badge ──────────────────

def _draw_glass_panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    *,
    tint: tuple[int, int, int] = (28, 22, 18),
    tint_alpha: float = 0.55,
    border: tuple[int, int, int] = (210, 210, 210),
    border_alpha: float = 0.35,
    highlight_alpha: float = 0.18,
) -> None:
    """Render a translucent "liquid glass" rectangle in-place.

    Uses a Gaussian-blurred ROI as a frosted backdrop, then overlays a
    subtle dark tint, an inner highlight strip at the top, and a thin
    light border. No rounded corners (cv2 lacks them natively); the
    border + blur reads as glass without them.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return

    roi = frame[y1:y2, x1:x2]

    # Frosted backdrop.
    blurred = cv2.GaussianBlur(roi, (0, 0), sigmaX=14, sigmaY=14)
    np.copyto(roi, blurred)

    # Dark tint over the blur.
    tint_layer = np.full_like(roi, tint)
    cv2.addWeighted(tint_layer, tint_alpha, roi, 1 - tint_alpha, 0, roi)

    # Top inner highlight (gradient from white-ish to transparent).
    highlight_h = max(2, (y2 - y1) // 6)
    if highlight_h > 1:
        grad = np.linspace(highlight_alpha, 0.0, highlight_h, dtype=np.float32)
        white = np.full((highlight_h, x2 - x1, 3), 255, dtype=np.uint8)
        target = roi[:highlight_h]
        for i in range(highlight_h):
            cv2.addWeighted(white[i:i+1], float(grad[i]), target[i:i+1], 1 - float(grad[i]), 0, target[i:i+1])

    # Border.
    border_layer = frame.copy()
    cv2.rectangle(border_layer, (x1, y1), (x2 - 1, y2 - 1), border, 1, cv2.LINE_AA)
    cv2.addWeighted(border_layer, border_alpha, frame, 1 - border_alpha, 0, frame)


def _draw_sparkline(
    frame: np.ndarray,
    values: list[float],
    rect: tuple[int, int, int, int],
    *,
    line_color: tuple[int, int, int] = GREEN,
    fill_color: tuple[int, int, int] = GREEN,
    fill_alpha: float = 0.18,
) -> None:
    """Draw a sparkline (line + soft fill) inside `rect = (x, y, w, h)`."""
    x, y, w, h = rect
    if w < 4 or h < 4 or len(values) < 2:
        return

    vmax = max(values)
    if vmax <= 0:
        # Flat line at the bottom.
        cv2.line(frame, (x, y + h - 1), (x + w - 1, y + h - 1), line_color, 1, cv2.LINE_AA)
        return

    n = len(values)
    pts = np.empty((n, 2), dtype=np.int32)
    for i, v in enumerate(values):
        px = x + int(i * (w - 1) / max(1, n - 1))
        py = y + h - 1 - int((v / vmax) * (h - 2))
        pts[i] = (px, py)

    # Soft fill below the line.
    fill_pts = np.vstack([
        pts,
        [[x + w - 1, y + h - 1], [x, y + h - 1]],
    ]).reshape(-1, 1, 2)
    fill_layer = frame.copy()
    cv2.fillPoly(fill_layer, [fill_pts], fill_color, cv2.LINE_AA)
    cv2.addWeighted(fill_layer, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    # Line.
    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, line_color, 2, cv2.LINE_AA)
    # Bright dot at the tip.
    cv2.circle(frame, (int(pts[-1, 0]), int(pts[-1, 1])), 3, line_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(pts[-1, 0]), int(pts[-1, 1])), 4, WHITE, 1, cv2.LINE_AA)


def draw_pose_hud(
    frame: np.ndarray,
    *,
    current_speed_mps: Optional[float] = None,
    peak_speed_mps: Optional[float] = None,
    speed_history_mps: Optional[list[float]] = None,
    landmark_count: int = 33,
    show_badge: bool = True,
) -> np.ndarray:
    """Draw a liquid-glass HUD with wrist velocity + sparkline + badge.

    `*_speed_mps` values are in metres/second (the caller scales from
    normalized pose coords). `speed_history_mps` is the trailing window
    used for the sparkline. All inputs are optional.
    """
    h, w = frame.shape[:2]
    pad = 14
    font = cv2.FONT_HERSHEY_SIMPLEX

    if show_badge:
        badge_text = "MediaPipe Pose"
        scale, thick = 0.45, 1
        (tw, th), _ = cv2.getTextSize(badge_text, font, scale, thick)
        bx, by = pad, pad
        bw, bh = tw + 28, th + 14
        _draw_glass_panel(frame, (bx, by), (bx + bw, by + bh))
        cv2.circle(frame, (bx + 12, by + bh // 2), 3, PURPLE, -1, cv2.LINE_AA)
        cv2.putText(
            frame, badge_text, (bx + 22, by + th + 7),
            font, scale, (235, 220, 255), thick, cv2.LINE_AA,
        )

        # "33 landmarks" tag, bottom-left.
        tag_text = f"{landmark_count} landmarks"
        (tw2, th2), _ = cv2.getTextSize(tag_text, font, 0.4, 1)
        bw2, bh2 = tw2 + 20, th2 + 12
        ty = h - pad - bh2
        _draw_glass_panel(frame, (pad, ty), (pad + bw2, ty + bh2))
        cv2.putText(
            frame, tag_text, (pad + 10, ty + th2 + 5),
            font, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
        )

    if current_speed_mps is None and peak_speed_mps is None:
        return frame

    panel_w = 250
    row_h = 30
    rows = (1 if current_speed_mps is not None else 0) + (1 if peak_speed_mps is not None else 0)
    spark_h = 44 if speed_history_mps else 0
    panel_h = rows * row_h + spark_h + 22
    px = w - panel_w - pad
    py = pad

    _draw_glass_panel(frame, (px, py), (px + panel_w, py + panel_h))

    def _row(label: str, value_mps: float, y_offset: int) -> int:
        cv2.putText(
            frame, label.upper(), (px + 14, py + y_offset),
            font, 0.36, (200, 200, 210), 1, cv2.LINE_AA,
        )
        value_text = f"{value_mps:5.1f}"
        (vw, vh), _ = cv2.getTextSize(value_text, font, 0.8, 2)
        cv2.putText(
            frame, value_text, (px + 14, y_offset + py + 22),
            font, 0.8, GREEN, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "m/s", (px + 14 + vw + 6, y_offset + py + 22),
            font, 0.42, (200, 200, 210), 1, cv2.LINE_AA,
        )
        return y_offset + row_h

    y_cursor = 14
    if current_speed_mps is not None:
        y_cursor = _row("wrist speed", current_speed_mps, y_cursor)
    if peak_speed_mps is not None:
        y_cursor = _row("peak", peak_speed_mps, y_cursor)

    if speed_history_mps:
        spark_pad = 14
        rect = (
            px + spark_pad,
            py + y_cursor + 4,
            panel_w - spark_pad * 2,
            spark_h - 8,
        )
        _draw_sparkline(frame, speed_history_mps, rect)

    return frame


# ── Audio waveform ──────────────────────────────────────────────────────

def _compute_waveform_envelope(samples: np.ndarray, num_bins: int) -> np.ndarray:
    envelope = np.zeros(num_bins, dtype=np.float32)
    if num_bins <= 0 or len(samples) == 0:
        return envelope

    abs_samples = np.abs(samples)
    if len(abs_samples) < num_bins:
        envelope[:len(abs_samples)] = abs_samples
        return envelope

    bin_size = max(1, len(abs_samples) // num_bins)
    usable = min(len(abs_samples), bin_size * num_bins)
    if usable <= 0:
        return envelope

    trimmed = abs_samples[:usable]
    reshaped = trimmed.reshape(-1, bin_size)
    envelope[:reshaped.shape[0]] = reshaped.max(axis=1)
    return envelope


def build_waveform_strip(
    audio: AudioData,
    current_time: float,
    frame_width: int,
    window_sec: float = 5.0,
    height: int = 80,
    color: tuple[int, int, int] = WAVEFORM_COLOR,
    bg_color: tuple[int, int, int] = WAVEFORM_BG,
    impact_events: Optional[list[ImpactEvent]] = None,
    impact_color: tuple[int, int, int] = IMPACT_COLOR,
) -> np.ndarray:
    waveform_bg = np.full((height, frame_width, 3), bg_color, dtype=np.uint8)
    if frame_width <= 0 or height <= 0:
        return waveform_bg

    half_window = window_sec / 2
    start_time = max(audio.start_time, current_time - half_window)
    end_time = min(audio.end_time, current_time + half_window)

    start_sample = int((start_time - audio.start_time) * audio.sample_rate)
    end_sample = int((end_time - audio.start_time) * audio.sample_rate)
    start_sample = max(0, start_sample)
    end_sample = min(len(audio.samples), end_sample)
    samples = audio.samples[start_sample:end_sample]

    if len(samples) == 0:
        return waveform_bg

    envelope = _compute_waveform_envelope(samples, frame_width)
    max_amp = np.max(envelope) if np.max(envelope) > 0 else 1.0
    normalized = envelope / max_amp

    max_height = max(1, height // 2 - 2)
    amplitudes = (normalized * max_height).astype(np.int32)
    center_y = height // 2
    rows = np.arange(height, dtype=np.int32)[:, None]
    mask = np.abs(rows - center_y) <= amplitudes[None, :]
    waveform_bg[mask] = color

    window_duration = end_time - start_time
    time_position = (
        (current_time - start_time) / window_duration
        if window_duration > 0
        else 0.5
    )
    indicator_x = int(np.clip(time_position * frame_width, 0, frame_width - 1))
    waveform_bg[:, max(0, indicator_x - 1):min(frame_width, indicator_x + 1)] = (255, 255, 255)

    if impact_events and window_duration > 0:
        for event in impact_events:
            if start_time <= event.timestamp <= end_time:
                event_position = (event.timestamp - start_time) / window_duration
                event_x = int(np.clip(event_position * frame_width, 0, frame_width - 1))

                pts = np.array(
                    [
                        [event_x, 5],
                        [max(0, event_x - 5), 0],
                        [min(frame_width - 1, event_x + 5), 0],
                    ],
                    np.int32,
                )
                cv2.fillPoly(waveform_bg, [pts], impact_color)
                waveform_bg[5:height - 5, event_x:event_x + 1] = impact_color

    return waveform_bg


def build_waveform_strip_sequence(
    audio: AudioData,
    timestamps: list[float],
    frame_width: int,
    window_sec: float = 5.0,
    height: int = 80,
    color: tuple[int, int, int] = WAVEFORM_COLOR,
    bg_color: tuple[int, int, int] = WAVEFORM_BG,
    impact_events: Optional[list[ImpactEvent]] = None,
    impact_color: tuple[int, int, int] = IMPACT_COLOR,
) -> list[np.ndarray]:
    return [
        build_waveform_strip(
            audio,
            current_time=timestamp,
            frame_width=frame_width,
            window_sec=window_sec,
            height=height,
            color=color,
            bg_color=bg_color,
            impact_events=impact_events,
            impact_color=impact_color,
        )
        for timestamp in timestamps
    ]


def draw_audio_waveform(
    frame: np.ndarray,
    audio: AudioData,
    current_time: float,
    window_sec: float = 5.0,
    height: int = 80,
    color: tuple[int, int, int] = WAVEFORM_COLOR,
    bg_color: tuple[int, int, int] = WAVEFORM_BG,
    impact_events: Optional[list[ImpactEvent]] = None,
    impact_color: tuple[int, int, int] = IMPACT_COLOR,
    position: str = "bottom",
) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    y_start = 0 if position == "top" else frame_h - height

    waveform_bg = build_waveform_strip(
        audio,
        current_time=current_time,
        frame_width=frame_w,
        window_sec=window_sec,
        height=height,
        color=color,
        bg_color=bg_color,
        impact_events=impact_events,
        impact_color=impact_color,
    )
    frame[y_start:y_start + height, :] = waveform_bg
    return frame


# ── Phase / impact / timestamp text overlays ────────────────────────────

PHASE_NAMES: dict[SwingPhase, str] = {
    SwingPhase.IDLE: "IDLE",
    SwingPhase.ADDRESS: "ADDRESS",
    SwingPhase.BACKSWING: "BACKSWING",
    SwingPhase.TOP: "TOP",
    SwingPhase.DOWNSWING: "DOWNSWING",
    SwingPhase.IMPACT: "IMPACT",
    SwingPhase.FOLLOW_THROUGH: "FOLLOW-THROUGH",
    SwingPhase.FINISH: "FINISH",
}


def draw_swing_phase_label(
    frame: np.ndarray,
    phase: SwingPhase,
    color: tuple[int, int, int] = WHITE,
    font_scale: float = 0.7,
    position: Optional[tuple[int, int]] = None,
    show_background: bool = True,
) -> np.ndarray:
    label = PHASE_NAMES.get(phase, "UNKNOWN")
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    if position is None:
        # Top-center keeps the badge (top-left) and HUD (top-right) clear.
        frame_h, frame_w = frame.shape[:2]
        position = ((frame_w - text_w) // 2, text_h + 24)
    x, y = position

    if show_background:
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            HUD_BG,
            -1,
        )
        cv2.rectangle(
            frame,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + baseline + padding),
            PURPLE,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_impact_marker(
    frame: np.ndarray,
    is_impact: bool,
    color: tuple[int, int, int] = GREEN,
    radius: int = 30,
    position: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if not is_impact:
        return frame

    h, w = frame.shape[:2]
    if position is None:
        position = (w // 2, h // 2)

    cv2.circle(frame, position, radius, color, 3, cv2.LINE_AA)
    cv2.circle(frame, position, radius - 10, color, 2, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.circle(overlay, position, radius + 20, color, -1)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    return frame


def draw_timestamp(
    frame: np.ndarray,
    timestamp: float,
    color: tuple[int, int, int] = WHITE,
    font_scale: float = 0.5,
    position: str = "top_right",
) -> np.ndarray:
    h, w = frame.shape[:2]
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
    else:
        x, y = w - text_w - padding, h - padding

    cv2.rectangle(
        frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5),
        HUD_BG, -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_pose_skeleton(
    frame: np.ndarray,
    pose: FramePose,
    color: tuple[int, int, int] = PURPLE,
    landmark_color: Optional[tuple[int, int, int]] = None,
    thickness: int = 2,
    landmark_radius: int = 4,
    min_visibility: float = 0.5,
    golf_mode: bool = True,
) -> np.ndarray:
    """Lightweight skeleton draw for callers that don't need glow/trails.

    Kept for backward compatibility. New code should use `SkeletonRenderer`.
    """
    if not pose.is_valid:
        return frame

    h, w = frame.shape[:2]
    landmark_color = landmark_color or color
    connections = BONES if golf_mode else POSE_CONNECTIONS

    for start_idx, end_idx in connections:
        if start_idx >= len(pose.landmarks) or end_idx >= len(pose.landmarks):
            continue
        s, e = pose.landmarks[start_idx], pose.landmarks[end_idx]
        if s.visibility < min_visibility or e.visibility < min_visibility:
            continue
        cv2.line(frame, s.to_pixel(w, h), e.to_pixel(w, h), color, thickness, cv2.LINE_AA)

    for lm in pose.landmarks:
        if lm.visibility < min_visibility:
            continue
        cv2.circle(frame, lm.to_pixel(w, h), landmark_radius, landmark_color, -1, cv2.LINE_AA)

    return frame
