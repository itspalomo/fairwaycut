"""Interactive 3D swing animation viewer.

Creates an elegant, Sketchfab-style animated 3D viewer that can be
opened in a browser. Features smooth skeleton playback with full
rotation/zoom controls.
"""

from pathlib import Path
from typing import Optional
import json
import numpy as np

from fairwaycut.core.models import SwingPhase, SwingEvent
from fairwaycut.pose.normalizer import (
    NormalizedPose,
    SKELETON_CONNECTIONS_3D,
    NORMALIZED_JOINT_NAMES,
)

# Phase colors (hex)
PHASE_COLORS_HEX = {
    SwingPhase.IDLE: "#6b7280",
    SwingPhase.ADDRESS: "#3b82f6",
    SwingPhase.BACKSWING: "#8b5cf6",
    SwingPhase.TOP: "#ec4899",
    SwingPhase.DOWNSWING: "#f97316",
    SwingPhase.IMPACT: "#eab308",
    SwingPhase.FOLLOW_THROUGH: "#10b981",
    SwingPhase.FINISH: "#06b6d4",
}


def generate_swing_animation_viewer(
    poses: list[NormalizedPose],
    swing: Optional[SwingEvent] = None,
    output_path: Optional[Path] = None,
    title: str = "Golf Swing 3D",
    fps: float = 30.0,
) -> str:
    """Generate an elegant animated 3D swing viewer (Sketchfab-style).
    
    Args:
        poses: List of NormalizedPose objects
        swing: Optional SwingEvent with metadata
        output_path: Path to save HTML file
        title: Title for the viewer
        fps: Playback frames per second
        
    Returns:
        HTML string or path to saved file
    """
    if not poses:
        raise ValueError("No poses provided")
    
    # Prepare animation data
    frames_data = []
    for pose in poses:
        joints = pose.joints.tolist()
        phase = pose.phase.name if hasattr(pose.phase, 'name') else str(pose.phase)
        color = PHASE_COLORS_HEX.get(pose.phase, "#10b981")
        frames_data.append({
            "joints": joints,
            "phase": phase,
            "color": color,
            "timestamp": pose.timestamp,
            "velocity": float(pose.velocities.max()) if pose.velocities is not None else 0,
        })
    
    # Skeleton connections
    bones = [[int(a), int(b)] for a, b in SKELETON_CONNECTIONS_3D]
    
    # Build wrist trail data
    wrist_trails = {
        "left": [{"x": float(p.joints[7][0]), "y": float(p.joints[7][1]), "z": float(p.joints[7][2]), 
                  "v": float(p.velocities[7]) if p.velocities is not None else 0} for p in poses],
        "right": [{"x": float(p.joints[8][0]), "y": float(p.joints[8][1]), "z": float(p.joints[8][2]),
                   "v": float(p.velocities[8]) if p.velocities is not None else 0} for p in poses],
    }
    
    # Swing info
    swing_info = {
        "impact_time": swing.impact_time if swing else 0,
        "confidence": swing.combined_confidence if swing else 0,
        "swing_id": swing.swing_id if swing else 1,
    }
    
    html = _generate_threejs_viewer(
        frames_data=frames_data,
        bones=bones,
        wrist_trails=wrist_trails,
        swing_info=swing_info,
        title=title,
        fps=fps,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        return str(output_path)
    
    return html


def _generate_threejs_viewer(
    frames_data: list[dict],
    bones: list[list[int]],
    wrist_trails: dict,
    swing_info: dict,
    title: str,
    fps: float,
) -> str:
    """Generate Three.js based viewer HTML."""
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%);
            color: #e2e8f0;
            overflow: hidden;
            height: 100vh;
        }}
        
        #canvas-container {{
            width: 100%;
            height: calc(100vh - 140px);
            position: relative;
        }}
        
        canvas {{
            display: block;
        }}
        
        .header {{
            padding: 16px 24px;
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .title {{
            font-size: 18px;
            font-weight: 600;
            color: #f8fafc;
        }}
        
        .subtitle {{
            font-size: 13px;
            color: #94a3b8;
            margin-top: 2px;
        }}
        
        .stats {{
            display: flex;
            gap: 24px;
        }}
        
        .stat {{
            text-align: right;
        }}
        
        .stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: #10b981;
        }}
        
        .stat-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .controls {{
            padding: 16px 24px;
            background: rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .timeline-container {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 12px;
        }}
        
        .play-btn {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.15s, box-shadow 0.15s;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }}
        
        .play-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }}
        
        .play-btn:active {{
            transform: scale(0.98);
        }}
        
        .play-btn svg {{
            width: 18px;
            height: 18px;
            fill: white;
            margin-left: 2px;
        }}
        
        .play-btn.playing svg {{
            margin-left: 0;
        }}
        
        .timeline {{
            flex: 1;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            cursor: pointer;
            position: relative;
        }}
        
        .timeline-progress {{
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%);
            border-radius: 3px;
            width: 0%;
            transition: width 0.05s linear;
        }}
        
        .timeline-handle {{
            width: 14px;
            height: 14px;
            background: white;
            border-radius: 50%;
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            cursor: grab;
        }}
        
        .timeline-handle:active {{
            cursor: grabbing;
            transform: translate(-50%, -50%) scale(1.1);
        }}
        
        .time-display {{
            font-size: 13px;
            font-family: 'SF Mono', Monaco, monospace;
            color: #94a3b8;
            min-width: 45px;
        }}
        
        .bottom-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .phase-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .phase-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            transition: background 0.2s;
        }}
        
        .phase-name {{
            font-size: 13px;
            font-weight: 500;
            min-width: 120px;
        }}
        
        .speed-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .speed-btn {{
            padding: 6px 12px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        
        .speed-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: #e2e8f0;
        }}
        
        .speed-btn.active {{
            background: rgba(16, 185, 129, 0.2);
            border-color: #10b981;
            color: #10b981;
        }}
        
        .view-hint {{
            position: absolute;
            bottom: 160px;
            left: 24px;
            font-size: 12px;
            color: #64748b;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .view-hint span {{
            opacity: 0.7;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 14px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">{title}</div>
            <div class="subtitle">Swing #{swing_info['swing_id']} • Impact at {swing_info['impact_time']:.2f}s</div>
        </div>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{swing_info['confidence']:.0%}</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(frames_data)}</div>
                <div class="stat-label">Frames</div>
            </div>
        </div>
    </div>
    
    <div id="canvas-container">
        <div class="loading">Loading 3D viewer...</div>
        <div class="view-hint">
            <span>🖱️ Drag to rotate</span>
            <span>🔍 Scroll to zoom</span>
            <span>⇧+Drag to pan</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="timeline-container">
            <button class="play-btn" id="playBtn">
                <svg viewBox="0 0 24 24" id="playIcon"><polygon points="5,3 19,12 5,21"/></svg>
                <svg viewBox="0 0 24 24" id="pauseIcon" style="display:none"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
            </button>
            <div class="time-display" id="currentTime">0.00s</div>
            <div class="timeline" id="timeline">
                <div class="timeline-progress" id="timelineProgress"></div>
                <div class="timeline-handle" id="timelineHandle"></div>
            </div>
            <div class="time-display" id="totalTime">{len(frames_data) / fps:.2f}s</div>
        </div>
        <div class="bottom-controls">
            <div class="phase-indicator">
                <div class="phase-dot" id="phaseDot"></div>
                <div class="phase-name" id="phaseName">IDLE</div>
            </div>
            <div class="speed-controls">
                <span style="font-size: 12px; color: #64748b; margin-right: 4px;">Speed:</span>
                <button class="speed-btn" data-speed="0.25">0.25x</button>
                <button class="speed-btn" data-speed="0.5">0.5x</button>
                <button class="speed-btn active" data-speed="1">1x</button>
                <button class="speed-btn" data-speed="2">2x</button>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Animation data
        const framesData = {json.dumps(frames_data)};
        const bones = {json.dumps(bones)};
        const wristTrails = {json.dumps(wrist_trails)};
        const FPS = {fps};
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let skeleton, joints = [], boneLines = [];
        let trailLines = {{ left: null, right: null }};
        let gridHelper, groundPlane;
        
        // Animation state
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1;
        let lastTime = 0;
        let accumulator = 0;
        
        // DOM elements
        const container = document.getElementById('canvas-container');
        const playBtn = document.getElementById('playBtn');
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');
        const timeline = document.getElementById('timeline');
        const timelineProgress = document.getElementById('timelineProgress');
        const timelineHandle = document.getElementById('timelineHandle');
        const currentTimeEl = document.getElementById('currentTime');
        const phaseDot = document.getElementById('phaseDot');
        const phaseName = document.getElementById('phaseName');
        const speedBtns = document.querySelectorAll('.speed-btn');
        
        function init() {{
            // Scene
            scene = new THREE.Scene();
            
            // Camera
            camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
            camera.position.set(3, 2, 3);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setClearColor(0x000000, 0);
            container.innerHTML = '';
            container.appendChild(renderer.domElement);
            
            // Re-add hints
            const hints = document.createElement('div');
            hints.className = 'view-hint';
            hints.innerHTML = '<span>🖱️ Drag to rotate</span><span>🔍 Scroll to zoom</span><span>⇧+Drag to pan</span>';
            container.appendChild(hints);
            
            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0.8, 0);
            controls.update();
            
            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 5);
            scene.add(directionalLight);
            
            // Ground grid
            gridHelper = new THREE.GridHelper(4, 20, 0x333333, 0x222222);
            gridHelper.position.y = 0;
            scene.add(gridHelper);
            
            // Ground plane (subtle)
            const groundGeo = new THREE.PlaneGeometry(4, 4);
            const groundMat = new THREE.MeshBasicMaterial({{ 
                color: 0x111111, 
                transparent: true, 
                opacity: 0.5,
                side: THREE.DoubleSide
            }});
            groundPlane = new THREE.Mesh(groundGeo, groundMat);
            groundPlane.rotation.x = -Math.PI / 2;
            groundPlane.position.y = -0.01;
            scene.add(groundPlane);
            
            // Create skeleton
            createSkeleton();
            
            // Create wrist trails
            createTrails();
            
            // Update to first frame
            updateFrame(0);
            
            // Start render loop
            animate();
        }}
        
        function createSkeleton() {{
            // Create joints (spheres)
            const jointGeo = new THREE.SphereGeometry(0.03, 16, 16);
            
            for (let i = 0; i < 13; i++) {{
                const jointMat = new THREE.MeshStandardMaterial({{ 
                    color: 0x10b981,
                    emissive: 0x10b981,
                    emissiveIntensity: 0.3
                }});
                const joint = new THREE.Mesh(jointGeo, jointMat);
                joints.push(joint);
                scene.add(joint);
            }}
            
            // Create bones (lines)
            const boneMat = new THREE.LineBasicMaterial({{ 
                color: 0x10b981,
                linewidth: 2
            }});
            
            for (const [a, b] of bones) {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6);
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const line = new THREE.Line(geometry, boneMat.clone());
                boneLines.push({{ line, a, b }});
                scene.add(line);
            }}
        }}
        
        function createTrails() {{
            // Left wrist trail
            const leftPositions = [];
            for (const p of wristTrails.left) {{
                leftPositions.push(p.x, p.y, p.z);
            }}
            const leftGeo = new THREE.BufferGeometry();
            leftGeo.setAttribute('position', new THREE.Float32BufferAttribute(leftPositions, 3));
            const leftMat = new THREE.LineBasicMaterial({{ 
                color: 0xf97316,
                transparent: true,
                opacity: 0.6
            }});
            trailLines.left = new THREE.Line(leftGeo, leftMat);
            scene.add(trailLines.left);
            
            // Right wrist trail
            const rightPositions = [];
            for (const p of wristTrails.right) {{
                rightPositions.push(p.x, p.y, p.z);
            }}
            const rightGeo = new THREE.BufferGeometry();
            rightGeo.setAttribute('position', new THREE.Float32BufferAttribute(rightPositions, 3));
            const rightMat = new THREE.LineBasicMaterial({{ 
                color: 0xf97316,
                transparent: true,
                opacity: 0.6
            }});
            trailLines.right = new THREE.Line(rightGeo, rightMat);
            scene.add(trailLines.right);
        }}
        
        function updateFrame(frameIndex) {{
            if (frameIndex < 0 || frameIndex >= framesData.length) return;
            
            currentFrame = frameIndex;
            const frame = framesData[frameIndex];
            const color = new THREE.Color(frame.color);
            
            // Update joints
            for (let i = 0; i < joints.length; i++) {{
                const j = frame.joints[i];
                // Coordinate mapping: our Y (up) stays Y, our Z (depth) goes to Z
                joints[i].position.set(j[0], j[1], j[2]);
                joints[i].material.color = color;
                joints[i].material.emissive = color;
            }}
            
            // Update bones
            for (const bone of boneLines) {{
                const ja = frame.joints[bone.a];
                const jb = frame.joints[bone.b];
                const positions = bone.line.geometry.attributes.position.array;
                positions[0] = ja[0]; positions[1] = ja[1]; positions[2] = ja[2];
                positions[3] = jb[0]; positions[4] = jb[1]; positions[5] = jb[2];
                bone.line.geometry.attributes.position.needsUpdate = true;
                bone.line.material.color = color;
            }}
            
            // Update UI
            const progress = frameIndex / (framesData.length - 1) * 100;
            timelineProgress.style.width = progress + '%';
            timelineHandle.style.left = progress + '%';
            currentTimeEl.textContent = (frameIndex / FPS).toFixed(2) + 's';
            phaseDot.style.background = frame.color;
            phaseName.textContent = frame.phase;
        }}
        
        function animate(time = 0) {{
            requestAnimationFrame(animate);
            
            if (isPlaying) {{
                const delta = (time - lastTime) / 1000;
                accumulator += delta * playbackSpeed;
                
                const frameDuration = 1 / FPS;
                while (accumulator >= frameDuration) {{
                    accumulator -= frameDuration;
                    currentFrame++;
                    if (currentFrame >= framesData.length) {{
                        currentFrame = 0;
                    }}
                    updateFrame(currentFrame);
                }}
            }}
            
            lastTime = time;
            controls.update();
            renderer.render(scene, camera);
        }}
        
        // Event listeners
        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            playIcon.style.display = isPlaying ? 'none' : 'block';
            pauseIcon.style.display = isPlaying ? 'block' : 'none';
            playBtn.classList.toggle('playing', isPlaying);
        }});
        
        timeline.addEventListener('click', (e) => {{
            const rect = timeline.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const frame = Math.floor(x * framesData.length);
            updateFrame(Math.max(0, Math.min(framesData.length - 1, frame)));
        }});
        
        let isDragging = false;
        timelineHandle.addEventListener('mousedown', () => isDragging = true);
        document.addEventListener('mouseup', () => isDragging = false);
        document.addEventListener('mousemove', (e) => {{
            if (!isDragging) return;
            const rect = timeline.getBoundingClientRect();
            const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const frame = Math.floor(x * framesData.length);
            updateFrame(Math.max(0, Math.min(framesData.length - 1, frame)));
        }});
        
        speedBtns.forEach(btn => {{
            btn.addEventListener('click', () => {{
                speedBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                playbackSpeed = parseFloat(btn.dataset.speed);
            }});
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{
                e.preventDefault();
                playBtn.click();
            }} else if (e.code === 'ArrowLeft') {{
                updateFrame(Math.max(0, currentFrame - 1));
            }} else if (e.code === 'ArrowRight') {{
                updateFrame(Math.min(framesData.length - 1, currentFrame + 1));
            }}
        }});
        
        // Resize handler
        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});
        
        // Initialize
        init();
    </script>
</body>
</html>'''


# Keep the old function name as alias for backwards compatibility
def generate_interactive_swing_viewer(
    poses: list[NormalizedPose],
    swing: Optional[SwingEvent] = None,
    output_path: Optional[Path] = None,
    title: str = "Golf Swing 3D",
) -> str:
    """Generate interactive 3D swing viewer (alias for generate_swing_animation_viewer)."""
    return generate_swing_animation_viewer(poses, swing, output_path, title)
