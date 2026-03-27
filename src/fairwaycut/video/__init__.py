"""Video processing and overlays."""

from fairwaycut.video.extraction import (
    extract_frames,
    extract_video_clip,
    get_video_info,
    VideoInfo,
)
from fairwaycut.video.overlays import (
    draw_pose_skeleton,
    draw_audio_waveform,
    draw_swing_phase_label,
    draw_impact_marker,
    # Enhanced skeleton rendering
    SkeletonRenderer,
    SkeletonRendererOptions,
    RenderMode,
    ColorTheme,
    PoseHistory,
)
from fairwaycut.video.generator import (
    DemoVideoGenerator,
    DemoVideoOptions,
    generate_all_swing_clips,
)

__all__ = [
    # Extraction
    "extract_frames",
    "extract_video_clip",
    "get_video_info",
    "VideoInfo",
    # Basic overlays
    "draw_pose_skeleton",
    "draw_audio_waveform",
    "draw_swing_phase_label",
    "draw_impact_marker",
    # Enhanced skeleton rendering
    "SkeletonRenderer",
    "SkeletonRendererOptions",
    "RenderMode",
    "ColorTheme",
    "PoseHistory",
    # Video generation
    "DemoVideoGenerator",
    "DemoVideoOptions",
    "generate_all_swing_clips",
]
