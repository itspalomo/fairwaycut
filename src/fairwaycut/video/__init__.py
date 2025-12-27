"""Video processing, overlays, and demo video generation."""

from fairwaycut.video.extraction import (
    extract_frames,
    get_video_info,
    VideoInfo,
)
from fairwaycut.video.overlays import (
    draw_pose_skeleton,
    draw_audio_waveform,
    draw_swing_phase_label,
    draw_impact_marker,
)
from fairwaycut.video.generator import (
    DemoVideoGenerator,
    DemoVideoOptions,
    generate_demo_video,
    generate_all_swing_clips,
)

__all__ = [
    "extract_frames",
    "get_video_info",
    "VideoInfo",
    "draw_pose_skeleton",
    "draw_audio_waveform",
    "draw_swing_phase_label",
    "draw_impact_marker",
    "DemoVideoGenerator",
    "DemoVideoOptions",
    "generate_demo_video",
    "generate_all_swing_clips",
]

