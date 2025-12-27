#!/usr/bin/env python3
"""Compare all visualization styles on the same swing clip.

This script renders the same swing with all visualization modes:
- LEGACY: Basic lines (no effects)
- MINIMAL: Clean glow, no trails
- STANDARD: Glow + trails + phase colors
- CINEMATIC: Full effects with depth, particles

Usage:
    uv run python scripts/compare_viz_styles.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fairwaycut.audio.extraction import extract_audio_from_video
from fairwaycut.fusion.detector import detect_swings
from fairwaycut.core.config import ProcessingMode, Config
from fairwaycut.video.generator import DemoVideoGenerator, DemoVideoOptions
from fairwaycut.video.overlays import RenderMode, SkeletonRendererOptions


def main():
    # Configuration
    video_path = Path("samples/IMG_6644.MOV")
    output_base = Path("output/viz_comparison")
    
    # Use swing #2 (around 34s) - high confidence
    # Process 30-40s range to get that swing
    start_time = 30.0
    end_time = 40.0
    
    print("=" * 60)
    print("🎨 Visualization Style Comparison")
    print("=" * 60)
    print(f"📹 Video: {video_path}")
    print(f"⏱️  Time range: {start_time:.0f}s - {end_time:.0f}s")
    print()
    
    # Step 1: Extract audio
    print("🎵 Extracting audio...")
    audio = extract_audio_from_video(video_path)
    
    # Step 2: Detect swings with pose data
    print("🔍 Detecting swings (hybrid mode for pose data)...")
    config = Config.default()
    
    result = detect_swings(
        video_path,
        mode=ProcessingMode.HYBRID,
        config=config,
        start_time=start_time,
        end_time=end_time,
    )
    
    print(f"🎯 Found {len(result.swings)} swings in range")
    
    if not result.swings:
        print("❌ No swings found in the specified range!")
        return
    
    # Step 3: Generate clips for each visualization style
    styles = [
        ("legacy", RenderMode.MINIMAL, False),      # Legacy: no enhanced rendering
        ("minimal", RenderMode.MINIMAL, True),      # Minimal glow
        ("standard", RenderMode.STANDARD, True),    # Standard with trails
        ("cinematic", RenderMode.CINEMATIC, True),  # Full effects
    ]
    
    for style_name, render_mode, use_enhanced in styles:
        output_dir = output_base / style_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Rendering: {style_name.upper()}")
        print(f"   Output: {output_dir}")
        
        # Configure options
        if use_enhanced:
            options = DemoVideoOptions(
                show_skeleton=True,
                show_waveform=True,
                show_phase_label=True,
                show_timestamp=True,
                show_impact_marker=True,
                use_enhanced_skeleton=True,
                skeleton_render_mode=render_mode,
            )
        else:
            # Legacy mode - disable enhanced skeleton
            options = DemoVideoOptions(
                show_skeleton=True,
                show_waveform=True,
                show_phase_label=True,
                show_timestamp=True,
                show_impact_marker=True,
                use_enhanced_skeleton=False,
                skeleton_color=(0, 255, 128),  # Green
                skeleton_thickness=2,
                landmark_radius=4,
            )
        
        generator = DemoVideoGenerator(options=options)
        
        # Generate clip for each swing
        for swing in result.swings:
            output_path = output_dir / f"swing_{swing.swing_id:03d}.mp4"
            
            print(f"   → Swing #{swing.swing_id} (impact at {swing.impact_time:.1f}s)")
            
            generator.generate_swing_clip(
                video_path,
                output_path,
                swing,
                result,
                audio,
            )
    
    print("\n" + "=" * 60)
    print("✅ All visualization styles rendered!")
    print(f"📁 Output: {output_base}")
    print()
    print("Styles rendered:")
    for style_name, _, _ in styles:
        print(f"  • {style_name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

