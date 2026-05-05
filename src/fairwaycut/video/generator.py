"""Video generation with pose and audio overlays."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from uuid import uuid4
import cv2
import numpy as np

from fairwaycut.core.models import (
    AudioData,
    DetectionResult,
    FramePose,
    SwingEvent,
    FusionResult,
)
from fairwaycut.core.config import VideoConfig
from fairwaycut.video.extraction import (
    attach_audio_to_video,
    get_video_info,
    VideoInfo,
)
from fairwaycut.video.overlays import (
    draw_audio_waveform,
    draw_timestamp,
    draw_pose_hud,
    SkeletonRenderer,
    SkeletonRendererOptions,
    build_waveform_strip_sequence,
)


@dataclass
class DemoVideoOptions:
    """Toggles for which overlay components to render."""

    show_skeleton: bool = True
    show_hud: bool = True
    show_waveform: bool = True
    show_timestamp: bool = True

    # Waveform strip
    waveform_height: int = 80
    waveform_window_sec: float = 5.0

    # HUD
    wrist_speed_scale_mps: float = 2.0  # normalized-coords/sec → m/s

    # Skeleton renderer knobs (rarely needed)
    skeleton_renderer_options: Optional[SkeletonRendererOptions] = None

    # Output settings
    output_fps: Optional[float] = None  # None = match input
    output_codec: str = "mp4v"
    quality: int = 23


@dataclass
class SwingClipContext:
    """Shared state reused while generating multiple swing clips."""

    info: VideoInfo
    pose_by_frame: dict[int, FramePose]
    impact_events: list
    waveform_strips: Optional[list[np.ndarray]] = None


class DemoVideoGenerator:
    """Generate videos with pose + waveform + HUD overlays."""

    def __init__(
        self,
        options: Optional[DemoVideoOptions] = None,
        config: Optional[VideoConfig] = None,
    ):
        self.options = options or DemoVideoOptions()
        self.config = config

        if config:
            self.options.waveform_height = config.waveform_height
            self.options.wrist_speed_scale_mps = config.wrist_speed_scale_mps

        self._skeleton_renderer = SkeletonRenderer(
            options=self.options.skeleton_renderer_options
        )

    def _temporary_render_path(self, output_path: Path) -> Path:
        return output_path.parent / f".{output_path.stem}.{uuid4().hex}{output_path.suffix}"

    def generate(
        self,
        video_path: str | Path,
        output_path: str | Path,
        fusion_result: FusionResult,
        audio: AudioData,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = self._temporary_render_path(output_path)

        info = get_video_info(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = self.options.output_fps or info.fps
        output_height = info.height
        if self.options.show_waveform:
            output_height += self.options.waveform_height

        fourcc = cv2.VideoWriter_fourcc(*self.options.output_codec)
        writer = cv2.VideoWriter(
            str(temp_output_path), fourcc, fps, (info.width, output_height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not create output video: {output_path}")

        try:
            self._skeleton_renderer.reset()

            pose_result = fusion_result.pose_result
            audio_result = fusion_result.audio_result

            pose_by_frame: dict[int, FramePose] = {}
            if pose_result and pose_result.frames:
                for pose_frame in pose_result.frames:
                    pose_by_frame[pose_frame.frame_index] = pose_frame

            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_index / info.fps
                current_pose: Optional[FramePose] = pose_by_frame.get(frame_index)

                frame = self._apply_overlays(
                    frame, timestamp, current_pose, audio, audio_result,
                )

                if self.options.show_waveform:
                    waveform_strip = np.zeros(
                        (self.options.waveform_height, info.width, 3),
                        dtype=np.uint8,
                    )
                    draw_audio_waveform(
                        waveform_strip, audio, timestamp,
                        window_sec=self.options.waveform_window_sec,
                        height=self.options.waveform_height,
                        impact_events=audio_result.events,
                        position="top",
                    )
                    frame = self._append_waveform_strip(frame, waveform_strip)

                writer.write(frame)
                if progress_callback and frame_index % 30 == 0:
                    progress_callback(frame_index, info.total_frames)

                frame_index += 1

        finally:
            cap.release()
            writer.release()

        attach_audio_to_video(
            temp_output_path, video_path, output_path,
            start_time=0.0, end_time=info.duration,
        )
        return output_path

    def _apply_overlays(
        self,
        frame: np.ndarray,
        timestamp: float,
        pose: Optional[FramePose],
        audio: AudioData,
        audio_result: DetectionResult,
    ) -> np.ndarray:
        if self.options.show_skeleton and pose and pose.is_valid:
            frame = self._skeleton_renderer.render(frame, pose)

        if self.options.show_hud:
            scale = self.options.wrist_speed_scale_mps
            frame = draw_pose_hud(
                frame,
                current_speed_mps=self._skeleton_renderer.current_wrist_speed * scale,
                peak_speed_mps=self._skeleton_renderer.peak_wrist_speed * scale,
                speed_history_mps=[
                    s * scale for s in self._skeleton_renderer.recent_wrist_speeds
                ],
            )

        if self.options.show_timestamp:
            frame = draw_timestamp(frame, timestamp)

        return frame

    def _append_waveform_strip(
        self,
        frame: np.ndarray,
        waveform_strip: np.ndarray,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        expanded = np.zeros((h + waveform_strip.shape[0], w, 3), dtype=np.uint8)
        expanded[:h, :] = frame
        expanded[h:, :] = waveform_strip
        return expanded

    def generate_swing_clip(
        self,
        video_path: str | Path,
        output_path: str | Path,
        swing: SwingEvent,
        fusion_result: FusionResult,
        audio: AudioData,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        context: Optional[SwingClipContext] = None,
    ) -> Path:
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_output_path = self._temporary_render_path(output_path)
        info = context.info if context else get_video_info(video_path)

        start_frame = int(swing.start_time * info.fps)
        end_frame = int(swing.end_time * info.fps)

        pose_by_frame = context.pose_by_frame if context else {}
        if not pose_by_frame and fusion_result.pose_result and fusion_result.pose_result.frames:
            pose_by_frame = {
                pose_frame.frame_index: pose_frame
                for pose_frame in fusion_result.pose_result.frames
            }

        impact_events = context.impact_events if context else [
            event
            for event in fusion_result.audio_result.events
            if swing.start_time <= event.timestamp <= swing.end_time
        ]

        waveform_strips = context.waveform_strips if context else None
        if self.options.show_waveform and waveform_strips is None:
            timestamps = [
                frame_number / info.fps
                for frame_number in range(start_frame, end_frame)
            ]
            waveform_strips = build_waveform_strip_sequence(
                audio,
                timestamps=timestamps,
                frame_width=info.width,
                window_sec=self.options.waveform_window_sec,
                height=self.options.waveform_height,
                impact_events=impact_events,
            )

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        output_height = info.height
        if self.options.show_waveform:
            output_height += self.options.waveform_height

        fps = self.options.output_fps or info.fps
        fourcc = cv2.VideoWriter_fourcc(*self.options.output_codec)
        writer = cv2.VideoWriter(
            str(temp_output_path), fourcc, fps, (info.width, output_height),
        )

        try:
            self._skeleton_renderer.reset()

            frame_index = start_frame
            total_frames = end_frame - start_frame
            processed = 0

            while cap.isOpened() and frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_index / info.fps
                current_pose = pose_by_frame.get(frame_index)

                frame = self._apply_overlays(
                    frame, timestamp, current_pose, audio, fusion_result.audio_result,
                )

                if self.options.show_waveform and waveform_strips is not None:
                    frame = self._append_waveform_strip(frame, waveform_strips[processed])

                writer.write(frame)
                if progress_callback and processed % 10 == 0:
                    progress_callback(processed, total_frames)

                frame_index += 1
                processed += 1

        finally:
            cap.release()
            writer.release()

        attach_audio_to_video(
            temp_output_path, video_path, output_path,
            start_time=swing.start_time, end_time=swing.end_time,
        )
        return output_path


def generate_all_swing_clips(
    video_path: str | Path,
    output_dir: str | Path,
    fusion_result: FusionResult,
    audio: AudioData,
    options: Optional[DemoVideoOptions] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[Path]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = DemoVideoGenerator(options=options)
    clips = []
    info = get_video_info(video_path)
    pose_by_frame: dict[int, FramePose] = {}
    if fusion_result.pose_result and fusion_result.pose_result.frames:
        pose_by_frame = {
            pose_frame.frame_index: pose_frame
            for pose_frame in fusion_result.pose_result.frames
        }
    impact_events_by_swing = {
        swing.swing_id: [
            event
            for event in fusion_result.audio_result.events
            if swing.start_time <= event.timestamp <= swing.end_time
        ]
        for swing in fusion_result.swings
    }

    for i, swing in enumerate(fusion_result.swings):
        if progress_callback:
            progress_callback(f"swing_{swing.swing_id}", i, len(fusion_result.swings))

        output_path = output_dir / f"swing_{swing.swing_id:03d}.mp4"
        start_frame = int(swing.start_time * info.fps)
        end_frame = int(swing.end_time * info.fps)
        timestamps = [
            frame_number / info.fps
            for frame_number in range(start_frame, end_frame)
        ]
        waveform_strips = None
        if generator.options.show_waveform:
            waveform_strips = build_waveform_strip_sequence(
                audio,
                timestamps=timestamps,
                frame_width=info.width,
                window_sec=generator.options.waveform_window_sec,
                height=generator.options.waveform_height,
                impact_events=impact_events_by_swing[swing.swing_id],
            )
        context = SwingClipContext(
            info=info,
            pose_by_frame=pose_by_frame,
            impact_events=impact_events_by_swing[swing.swing_id],
            waveform_strips=waveform_strips,
        )

        def frame_progress(current: int, total: int):
            if progress_callback:
                progress_callback(f"swing_{swing.swing_id}_frames", current, total)

        clip_path = generator.generate_swing_clip(
            video_path,
            output_path,
            swing,
            fusion_result,
            audio,
            progress_callback=frame_progress,
            context=context,
        )
        clips.append(clip_path)

    return clips
