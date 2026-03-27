"""Video extraction and processing utilities."""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Iterator, Optional
import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Information about a video file."""
    
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str
    source_file: str
    
    @property
    def frame_duration(self) -> float:
        """Duration of a single frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0


def _format_ffmpeg_time(timestamp: float) -> str:
    """Format a timestamp for ffmpeg command arguments."""
    return f"{timestamp:.6f}"


def _has_audio_stream(video_path: Path) -> bool:
    """Return True when the input file contains at least one audio stream."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def get_video_info(video_path: str | Path) -> VideoInfo:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file.
    
    Returns:
        VideoInfo with video properties.
    
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened.
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Get codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            source_file=str(video_path),
        )
    finally:
        cap.release()


def extract_frames(
    video_path: str | Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    skip_frames: int = 0,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Extract frames from a video file as an iterator.
    
    Args:
        video_path: Path to the video file.
        start_time: Optional start time in seconds.
        end_time: Optional end time in seconds.
        skip_frames: Number of frames to skip between yields (0 = no skip).
    
    Yields:
        Tuples of (frame_index, timestamp, frame_bgr).
    
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened.
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        start_frame = 0
        end_frame = total_frames
        
        if start_time is not None and fps > 0:
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        if end_time is not None and fps > 0:
            end_frame = int(end_time * fps)
        
        frame_index = start_frame
        frames_since_yield = 0
        
        while cap.isOpened() and frame_index < end_frame:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Calculate timestamp
            timestamp = frame_index / fps if fps > 0 else 0
            
            # Yield based on skip setting
            if frames_since_yield >= skip_frames:
                yield frame_index, timestamp, frame
                frames_since_yield = 0
            else:
                frames_since_yield += 1
            
            frame_index += 1
    
    finally:
        cap.release()


def extract_frame_at_time(
    video_path: str | Path,
    timestamp: float,
) -> Optional[np.ndarray]:
    """
    Extract a single frame at a specific timestamp.
    
    Args:
        video_path: Path to the video file.
        timestamp: Time in seconds.
    
    Returns:
        Frame as BGR numpy array, or None if extraction failed.
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        return frame if ret else None
    
    finally:
        cap.release()


def extract_video_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_time: float,
    end_time: float,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
) -> Path:
    """
    Extract a video clip using ffmpeg.

    This preserves the existing re-encode behavior while avoiding MoviePy's
    Python-side overhead for each extracted swing.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        raise ValueError("Clip duration must be greater than zero")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        _format_ffmpeg_time(start_time),
        "-i",
        str(video_path),
        "-t",
        _format_ffmpeg_time(duration),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Failed to extract clip {output_path.name}: {stderr}")

    return output_path


def attach_audio_to_video(
    rendered_video_path: str | Path,
    source_video_path: str | Path,
    output_path: str | Path,
    start_time: float = 0.0,
    end_time: float | None = None,
    audio_codec: str = "aac",
) -> Path:
    """
    Mux audio from the source video into a rendered video.

    The rendered video is expected to already contain the final video frames.
    Audio is trimmed from the source video to the requested time range and
    encoded into the output container.
    """
    rendered_video_path = Path(rendered_video_path)
    source_video_path = Path(source_video_path)
    output_path = Path(output_path)

    if not rendered_video_path.exists():
        raise FileNotFoundError(f"Rendered video not found: {rendered_video_path}")
    if not source_video_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_video_path}")

    normalized_start = max(0.0, start_time)
    normalized_end = end_time if end_time is None else max(normalized_start, end_time)

    if not _has_audio_stream(source_video_path):
        if rendered_video_path != output_path:
            rendered_video_path.replace(output_path)
        return output_path

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(rendered_video_path),
    ]

    if normalized_start > 0:
        command.extend(["-ss", _format_ffmpeg_time(normalized_start)])
    if normalized_end is not None:
        command.extend(["-t", _format_ffmpeg_time(normalized_end - normalized_start)])

    command.extend(
        [
            "-i",
            str(source_video_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            audio_codec,
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"Failed to attach audio to {output_path.name}: {stderr}"
        )

    if rendered_video_path != output_path and rendered_video_path.exists():
        rendered_video_path.unlink()

    return output_path


def resize_frame(
    frame: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    maintain_aspect: bool = True,
) -> np.ndarray:
    """
    Resize a video frame.
    
    Args:
        frame: Input frame as numpy array.
        target_width: Target width (None to calculate from height).
        target_height: Target height (None to calculate from width).
        maintain_aspect: Whether to maintain aspect ratio.
    
    Returns:
        Resized frame.
    """
    h, w = frame.shape[:2]
    
    if target_width is None and target_height is None:
        return frame
    
    if maintain_aspect:
        if target_width is not None and target_height is not None:
            # Use the smaller scale factor to fit within bounds
            scale_w = target_width / w
            scale_h = target_height / h
            scale = min(scale_w, scale_h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif target_width is not None:
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
        else:
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
    else:
        new_w = target_width or w
        new_h = target_height or h
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
