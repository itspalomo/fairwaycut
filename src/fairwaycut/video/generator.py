"""Demo video generation with pose and audio overlays."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np

from fairwaycut.core.models import (
    AudioData,
    FramePose,
    SwingPhase,
    SwingEvent,
    FusionResult,
    PoseAnalysisResult,
    DetectionResult,
)
from fairwaycut.core.config import VideoConfig
from fairwaycut.video.extraction import get_video_info, VideoInfo
from fairwaycut.video.overlays import (
    draw_pose_skeleton,
    draw_audio_waveform,
    draw_swing_phase_label,
    draw_impact_marker,
    draw_timestamp,
    draw_confidence_bar,
    # New enhanced skeleton rendering
    SkeletonRenderer,
    SkeletonRendererOptions,
    RenderMode,
    ColorTheme,
)
from fairwaycut.pose.swing_phases import SwingPhasesResult

# 3D visualization imports
from fairwaycut.pose.normalizer import PoseNormalizer, NormalizedPose
from fairwaycut.visualization.composite import (
    CompositeRenderer,
    CompositeOptions,
    LayoutMode,
)
from fairwaycut.visualization.viewer3d import Viewer3DOptions
from fairwaycut.visualization.voxel import VoxelOptions


@dataclass
class View3DOptions:
    """Options for 3D pose visualization."""
    
    # Enable 3D view
    enabled: bool = False
    
    # Layout mode
    layout: str = "inset"  # "inset" or "side-by-side"
    
    # Camera view
    camera_view: str = "isometric"  # "front", "dtl", "isometric"
    
    # Voxel visualization
    show_voxel: bool = False
    
    # Inset settings
    inset_size_ratio: float = 0.3  # Size relative to video height
    inset_position: str = "top-right"
    
    # Side-by-side settings
    video_width_ratio: float = 0.6  # Video takes 60% of width


@dataclass
class DemoVideoOptions:
    """Options for demo video generation."""
    
    # What to include
    show_skeleton: bool = True
    show_waveform: bool = True
    show_phase_label: bool = True
    show_timestamp: bool = True
    show_impact_marker: bool = True
    show_confidence: bool = False
    
    # Waveform settings
    waveform_height: int = 80
    waveform_window_sec: float = 5.0
    
    # Skeleton settings (legacy - used if use_enhanced_skeleton is False)
    skeleton_color: tuple[int, int, int] = (0, 255, 128)
    skeleton_thickness: int = 2
    landmark_radius: int = 4
    golf_mode: bool = True
    
    # Enhanced skeleton settings
    use_enhanced_skeleton: bool = True
    skeleton_render_mode: RenderMode = field(default=RenderMode.STANDARD)
    skeleton_renderer_options: Optional[SkeletonRendererOptions] = None
    
    # 3D visualization settings
    view_3d: Optional[View3DOptions] = None
    
    # Output settings
    output_fps: Optional[float] = None  # None = match input
    output_codec: str = "mp4v"
    quality: int = 23


class DemoVideoGenerator:
    """
    Generate demonstration videos with pose and audio overlays.
    
    This class takes a source video and fusion results to create
    a new video with visual overlays showing:
    - Pose skeleton (with optional enhanced neon glow effects)
    - Audio waveform with impact markers
    - Current swing phase label
    - Impact flash indicator
    - Optional 3D pose viewer (side-by-side or inset)
    - Optional voxel motion volume
    """
    
    def __init__(
        self,
        options: Optional[DemoVideoOptions] = None,
        config: Optional[VideoConfig] = None,
    ):
        """
        Initialize the demo video generator.
        
        Args:
            options: DemoVideoOptions for customization.
            config: VideoConfig from global config.
        """
        self.options = options or DemoVideoOptions()
        self.config = config
        
        # Apply config overrides if provided
        if config:
            self.options.waveform_height = config.waveform_height
            self.options.skeleton_color = config.skeleton_color
            self.options.skeleton_thickness = config.skeleton_thickness
            self.options.landmark_radius = config.landmark_radius
        
        # Initialize enhanced skeleton renderer
        self._skeleton_renderer: Optional[SkeletonRenderer] = None
        if self.options.use_enhanced_skeleton:
            if self.options.skeleton_renderer_options:
                renderer_opts = self.options.skeleton_renderer_options
            else:
                renderer_opts = SkeletonRendererOptions.from_mode(
                    self.options.skeleton_render_mode
                )
                # Apply legacy options for compatibility
                renderer_opts.thickness = self.options.skeleton_thickness
                renderer_opts.joint_radius = self.options.landmark_radius
                renderer_opts.golf_mode = self.options.golf_mode
            
            self._skeleton_renderer = SkeletonRenderer(options=renderer_opts)
        
        # Initialize 3D components if enabled
        self._composite_renderer: Optional[CompositeRenderer] = None
        self._pose_normalizer: Optional[PoseNormalizer] = None
        
        if self.options.view_3d and self.options.view_3d.enabled:
            self._init_3d_components()
    
    def _init_3d_components(self) -> None:
        """Initialize 3D visualization components."""
        view_3d = self.options.view_3d
        
        # Create pose normalizer
        self._pose_normalizer = PoseNormalizer()
        
        # Create viewer options
        viewer_opts = Viewer3DOptions(
            camera_view=view_3d.camera_view,
            figsize=(3, 3),  # Reasonable size for inset/side-by-side
            dpi=72,
        )
        
        # Create voxel options if enabled
        voxel_opts = None
        if view_3d.show_voxel:
            voxel_opts = VoxelOptions(
                grid_size=12,
                history_frames=15,
                color_by="velocity",
                camera_view=view_3d.camera_view,
            )
        
        # Create composite options
        layout_mode = LayoutMode(view_3d.layout)
        composite_opts = CompositeOptions(
            layout=layout_mode,
            video_width_ratio=view_3d.video_width_ratio,
            inset_position=view_3d.inset_position,
            inset_size_ratio=view_3d.inset_size_ratio,
            show_voxel=view_3d.show_voxel,
            viewer_3d_options=viewer_opts,
            voxel_options=voxel_opts,
        )
        
        # Create composite renderer
        self._composite_renderer = CompositeRenderer(composite_opts)
    
    def _get_3d_output_dimensions(
        self,
        video_width: int,
        video_height: int,
    ) -> tuple[int, int]:
        """Calculate output dimensions when 3D view is enabled.
        
        Args:
            video_width: Original video width
            video_height: Original video height (without waveform)
            
        Returns:
            (width, height) for output video
        """
        view_3d = self.options.view_3d
        
        if view_3d.layout == "side-by-side":
            # Width stays same (video + 3D panel fit in original width)
            return video_width, video_height
        else:
            # Inset mode: same dimensions as input
            return video_width, video_height
    
    def generate(
        self,
        video_path: str | Path,
        output_path: str | Path,
        fusion_result: FusionResult,
        audio: AudioData,
        phases_by_frame: Optional[dict[int, SwingPhase]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Generate a demo video with overlays.
        
        Args:
            video_path: Path to source video.
            output_path: Path for output video.
            fusion_result: FusionResult with detection data.
            audio: AudioData for waveform visualization.
            phases_by_frame: Optional dict mapping frame index to phase.
            progress_callback: Optional callback(current_frame, total_frames).
        
        Returns:
            Path to the generated video.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get video info
        info = get_video_info(video_path)
        
        # Open source video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Determine output parameters
        fps = self.options.output_fps or info.fps
        
        # Calculate output dimensions (may add space for waveform)
        output_height = info.height
        if self.options.show_waveform:
            output_height += self.options.waveform_height
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.options.output_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (info.width, output_height),
        )
        
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Could not create output video: {output_path}")
        
        try:
            # Reset skeleton renderer for fresh history
            if self._skeleton_renderer:
                self._skeleton_renderer.reset()
            
            # Reset 3D components for fresh history
            if self._composite_renderer:
                self._composite_renderer.reset()
            if self._pose_normalizer:
                self._pose_normalizer.reset()
            
            # Get pose data if available
            pose_result = fusion_result.pose_result
            audio_result = fusion_result.audio_result
            
            # Build pose lookup by frame index for efficient access
            pose_by_frame: dict[int, FramePose] = {}
            if pose_result and pose_result.frames:
                for pose_frame in pose_result.frames:
                    pose_by_frame[pose_frame.frame_index] = pose_frame
            
            # Build impact time set for quick lookup
            impact_times = set()
            for swing in fusion_result.swings:
                impact_times.add(swing.impact_time)
            
            frame_index = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate timestamp
                timestamp = frame_index / info.fps
                
                # Check if this is near an impact
                is_impact = any(
                    abs(timestamp - t) < 0.05  # Within 50ms
                    for t in impact_times
                )
                
                # Get current phase
                current_phase = SwingPhase.IDLE
                if phases_by_frame and frame_index in phases_by_frame:
                    current_phase = phases_by_frame[frame_index]
                
                # Get pose for this frame (using frame_index lookup)
                current_pose: Optional[FramePose] = pose_by_frame.get(frame_index)
                
                # Apply overlays
                frame = self._apply_overlays(
                    frame,
                    timestamp,
                    current_pose,
                    current_phase,
                    is_impact,
                    audio,
                    audio_result,
                )
                
                # Apply 3D composite if enabled
                if self._composite_renderer and current_pose:
                    frame = self._apply_3d_overlay(frame, current_pose, current_phase)
                
                # Add waveform strip if enabled
                if self.options.show_waveform:
                    frame = self._add_waveform_strip(
                        frame,
                        info,
                        timestamp,
                        audio,
                        audio_result.events,
                    )
                
                # Write frame
                writer.write(frame)
                
                # Progress callback
                if progress_callback and frame_index % 30 == 0:
                    progress_callback(frame_index, info.total_frames)
                
                frame_index += 1
        
        finally:
            cap.release()
            writer.release()
        
        return output_path
    
    def _apply_overlays(
        self,
        frame: np.ndarray,
        timestamp: float,
        pose: Optional[FramePose],
        phase: SwingPhase,
        is_impact: bool,
        audio: AudioData,
        audio_result: DetectionResult,
    ) -> np.ndarray:
        """Apply all enabled overlays to a frame."""
        
        # Draw skeleton
        if self.options.show_skeleton and pose and pose.is_valid:
            if self._skeleton_renderer and self.options.use_enhanced_skeleton:
                # Use enhanced skeleton renderer with glow effects
                frame = self._skeleton_renderer.render(
                    frame,
                    pose,
                    phase=phase,
                    is_impact=is_impact,
                )
            else:
                # Fall back to legacy basic skeleton
                frame = draw_pose_skeleton(
                    frame,
                    pose,
                    color=self.options.skeleton_color,
                    thickness=self.options.skeleton_thickness,
                    landmark_radius=self.options.landmark_radius,
                    golf_mode=self.options.golf_mode,
                )
        
        # Draw impact marker (only for legacy mode, enhanced has built-in)
        if self.options.show_impact_marker and is_impact:
            if not self.options.use_enhanced_skeleton:
                frame = draw_impact_marker(frame, is_impact)
        
        # Draw phase label
        if self.options.show_phase_label:
            frame = draw_swing_phase_label(frame, phase)
        
        # Draw timestamp
        if self.options.show_timestamp:
            frame = draw_timestamp(frame, timestamp)
        
        # Draw confidence
        if self.options.show_confidence and pose:
            frame = draw_confidence_bar(
                frame,
                pose.confidence,
                label="Pose",
                position=(20, 100),
            )
        
        return frame
    
    def _add_waveform_strip(
        self,
        frame: np.ndarray,
        info: VideoInfo,
        timestamp: float,
        audio: AudioData,
        impact_events: list,
    ) -> np.ndarray:
        """Add waveform strip below the video frame."""
        h, w = frame.shape[:2]
        waveform_height = self.options.waveform_height
        
        # Create expanded frame
        expanded = np.zeros((h + waveform_height, w, 3), dtype=np.uint8)
        expanded[:h, :] = frame
        
        # Draw waveform in the bottom strip
        waveform_strip = expanded[h:, :]
        
        # Draw waveform
        draw_audio_waveform(
            waveform_strip,
            audio,
            timestamp,
            window_sec=self.options.waveform_window_sec,
            height=waveform_height,
            impact_events=impact_events,
            position="top",  # Fill entire strip
        )
        
        return expanded
    
    def _apply_3d_overlay(
        self,
        frame: np.ndarray,
        pose: FramePose,
        phase: SwingPhase,
    ) -> np.ndarray:
        """Apply 3D visualization overlay to frame.
        
        Args:
            frame: Video frame with 2D overlays applied
            pose: Current FramePose
            phase: Current swing phase
            
        Returns:
            Frame with 3D visualization composited
        """
        if not self._composite_renderer or not self._pose_normalizer:
            return frame
        
        # Normalize the pose
        normalized = self._pose_normalizer.normalize(pose, phase)
        
        if normalized is None:
            # Couldn't normalize, render without 3D
            return self._composite_renderer.render(frame, None)
        
        # Render composite
        return self._composite_renderer.render(frame, normalized)
    
    def generate_swing_clip(
        self,
        video_path: str | Path,
        output_path: str | Path,
        swing: SwingEvent,
        fusion_result: FusionResult,
        audio: AudioData,
        phases_result: Optional[SwingPhasesResult] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """
        Generate a demo clip for a single swing.
        
        Args:
            video_path: Path to source video.
            output_path: Path for output video.
            swing: SwingEvent to extract.
            fusion_result: FusionResult with detection data.
            audio: AudioData for waveform.
            phases_result: Optional SwingPhasesResult for phase labels.
            progress_callback: Optional progress callback.
        
        Returns:
            Path to the generated clip.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        info = get_video_info(video_path)
        
        # Calculate frame range
        start_frame = int(swing.start_time * info.fps)
        end_frame = int(swing.end_time * info.fps)
        
        # Build pose lookup by frame index for efficient access
        pose_by_frame: dict[int, FramePose] = {}
        if fusion_result.pose_result and fusion_result.pose_result.frames:
            for pose_frame in fusion_result.pose_result.frames:
                pose_by_frame[pose_frame.frame_index] = pose_frame
        
        # Build phases dict from phases_result
        phases_by_frame: dict[int, SwingPhase] = {}
        if phases_result:
            for i, phase in enumerate(phases_result.frame_phases):
                phases_by_frame[start_frame + i] = phase
        
        # Open source video
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Output dimensions
        output_height = info.height
        if self.options.show_waveform:
            output_height += self.options.waveform_height
        
        fps = self.options.output_fps or info.fps
        fourcc = cv2.VideoWriter_fourcc(*self.options.output_codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (info.width, output_height),
        )
        
        try:
            # Reset skeleton renderer for fresh history
            if self._skeleton_renderer:
                self._skeleton_renderer.reset()
            
            # Reset 3D components for fresh history
            if self._composite_renderer:
                self._composite_renderer.reset()
            if self._pose_normalizer:
                self._pose_normalizer.reset()
            
            # Get audio segment
            audio_segment = audio.get_segment(swing.start_time, swing.end_time)
            
            frame_index = start_frame
            total_frames = end_frame - start_frame
            processed = 0
            
            while cap.isOpened() and frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_index / info.fps
                is_impact = abs(timestamp - swing.impact_time) < 0.05
                
                current_phase = phases_by_frame.get(frame_index, SwingPhase.IDLE)
                
                # Get pose using frame index lookup (not array position)
                current_pose = pose_by_frame.get(frame_index)
                
                # Apply overlays
                frame = self._apply_overlays(
                    frame,
                    timestamp,
                    current_pose,
                    current_phase,
                    is_impact,
                    audio,
                    fusion_result.audio_result,
                )
                
                # Apply 3D composite if enabled
                if self._composite_renderer and current_pose:
                    frame = self._apply_3d_overlay(frame, current_pose, current_phase)
                
                if self.options.show_waveform:
                    frame = self._add_waveform_strip(
                        frame,
                        info,
                        timestamp,
                        audio,
                        [e for e in fusion_result.audio_result.events
                         if swing.start_time <= e.timestamp <= swing.end_time],
                    )
                
                writer.write(frame)
                
                if progress_callback and processed % 10 == 0:
                    progress_callback(processed, total_frames)
                
                frame_index += 1
                processed += 1
        
        finally:
            cap.release()
            writer.release()
        
        return output_path


def generate_demo_video(
    video_path: str | Path,
    output_path: str | Path,
    fusion_result: FusionResult,
    audio: AudioData,
    options: Optional[DemoVideoOptions] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """
    Convenience function to generate a demo video.
    
    Args:
        video_path: Path to source video.
        output_path: Path for output video.
        fusion_result: FusionResult with detection data.
        audio: AudioData for waveform.
        options: Optional DemoVideoOptions.
        progress_callback: Optional callback(current, total).
    
    Returns:
        Path to the generated video.
    """
    generator = DemoVideoGenerator(options=options)
    return generator.generate(
        video_path,
        output_path,
        fusion_result,
        audio,
        progress_callback=progress_callback,
    )


def generate_all_swing_clips(
    video_path: str | Path,
    output_dir: str | Path,
    fusion_result: FusionResult,
    audio: AudioData,
    options: Optional[DemoVideoOptions] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[Path]:
    """
    Generate individual clips for all detected swings.
    
    Args:
        video_path: Path to source video.
        output_dir: Directory for output clips.
        fusion_result: FusionResult with swings.
        audio: AudioData for waveform.
        options: Optional DemoVideoOptions.
        progress_callback: Optional callback(status, current, total).
    
    Returns:
        List of paths to generated clips.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = DemoVideoGenerator(options=options)
    clips = []
    
    for i, swing in enumerate(fusion_result.swings):
        if progress_callback:
            progress_callback(f"swing_{swing.swing_id}", i, len(fusion_result.swings))
        
        output_path = output_dir / f"swing_{swing.swing_id:03d}.mp4"
        
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
        )
        
        clips.append(clip_path)
    
    return clips

