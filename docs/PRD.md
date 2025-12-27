# Product Requirements Document (PRD)
## Project: Golf Swing Auto-Segmentation CLI
**License:** MIT (Open Source)  
**Audience:** Golfers, power users, and open-source contributors  
**Status:** Initial PRD (Phase 0–1)

---

## 1. Problem Statement

Golfers often record long, continuous videos at the driving range to review their swings. Manually scrubbing through footage to identify impact moments and cut individual swings is time-consuming and inconsistent. Existing solutions either rely on manual editing or focus on advanced coaching features rather than solving the fundamental problem of **automatic swing segmentation**.

This project aims to provide a **local, offline, deterministic command-line tool** that automatically detects and extracts individual golf swings from a continuous video recording. The system will serve as a foundation for future features such as swing analysis, pose estimation, and ball trajectory mapping.

---

## 2. Product Vision

> As a golfer, I want my driving range videos to be automatically segmented into individual swing clips so that I can quickly review, compare, and edit my swings without manual video work.

The product prioritizes:
- Reliability and determinism  
- Transparency and debuggability  
- Local, offline execution  
- Long-term extensibility

---

## 3. Goals and Non-Goals

### Goals
- Automatically detect individual golf swings in a continuous video
- Export each swing as a separate video clip
- Produce deterministic and repeatable results
- Generate structured reports for validation and tuning
- Support future extensibility without re-architecting the core system

### Non-Goals
- No graphical user interface in this phase
- No cloud processing or user accounts
- No real-time or live capture
- No swing coaching or performance scoring
- No mobile application

---

## 4. User Personas

### Golfer (Primary)
- Records driving range sessions on a phone or camera
- Wants fast turnaround for swing review
- Uses third-party video editors or analysis tools

### Power User / Developer (Secondary)
- Comfortable using CLI tools
- Interested in tuning detection logic
- May extend the system with new analysis modules

---

## 5. User Stories

### Core Workflow
1. As a golfer, I want to provide a long driving-range video and receive individual swing clips automatically.
2. As a golfer, I want the system to detect swings without manually marking timestamps.
3. As a golfer, I want clips to be consistently framed around the moment of impact.

### Trust and Control
4. As a user, I want deterministic behavior so that the same input produces the same results.
5. As a user, I want visibility into what the system detected and why.
6. As a user, I want to adjust sensitivity and timing parameters when needed.

### Extensibility
7. As a future user, I want to add swing analysis features without redesigning the system.
8. As a developer, I want a modular architecture where detection, segmentation, and export are decoupled.

---

## 6. System Decomposition (Logical)

### 6.1 Input and Normalization
- Accepts a video file as input
- Normalizes audio and video timebases
- Produces deterministic data streams for analysis

### 6.2 Signal Analysis
- Analyzes audio to detect candidate impact events
- Analyzes video to detect motion patterns consistent with a golf swing
- Outputs time-indexed activity signals

### 6.3 Event Detection
- Combines audio and motion signals
- Applies temporal constraints and validation rules
- Assigns confidence scores to detected events

### 6.4 Segment Construction
- Converts detected events into time-based segments
- Applies configurable pre-impact and post-impact buffers
- Prevents overlapping or duplicate segments

### 6.5 Export and Reporting
- Exports each segment as a standalone video clip
- Generates a structured report describing all detected swings
- Preserves metadata for reprocessing or auditing

---

## 7. Functional Requirements

### Video Ingest
- The system shall accept a single video file as input.
- The system shall support variable frame rates and resolutions.
- The system shall operate without internet connectivity.

### Audio Analysis
- The system shall analyze audio to detect transient impact events.
- The system shall adapt detection thresholds to environmental noise.
- The system shall enforce a minimum time gap between detected events.

### Motion Analysis
- The system shall analyze video motion to detect swing-like activity.
- The system shall use motion data to validate audio-based detections.
- The system shall support configurable regions of interest.

### Swing Event Detection
- The system shall combine multiple signals to identify swing events.
- Each detected event shall include a timestamp and confidence score.
- Detected swings shall be ordered deterministically.

### Segment Definition
- The system shall generate a time window around each detected swing.
- Pre-impact and post-impact durations shall be configurable.
- Segments shall not overlap.

### Export
- The system shall export each swing as an individual video file.
- Audio and video synchronization shall be preserved.
- Output filenames shall be predictable and stable.

### Reporting
- The system shall generate a structured report containing:
  - Swing timestamps
  - Segment boundaries
  - Confidence metrics
  - Parameters used during detection
- Reports shall be reproducible and machine-readable.

---

## 8. Non-Functional Requirements

### Performance
- Processing time shall scale linearly with video duration.
- The system shall run on consumer-grade hardware.

### Determinism
- Identical inputs and parameters shall produce identical outputs.

### Portability
- The system shall run locally on common desktop operating systems.
- No proprietary or cloud-dependent services shall be required.

### Observability
- The system shall support generating debug artifacts.
- Intermediate analysis data shall be inspectable when enabled.

### Extensibility
- The architecture shall allow new analysis modules without modifying core segmentation logic.

---

## 9. Success Metrics (Phase 1)

- ≥90% of actual swings detected in typical range recordings
- Low false-positive rate in noisy environments
- Consistent clip framing across sessions
- Ability to tune parameters for different recording setups
- Clear separation between detection, segmentation, and export components

---

## 10. Future Extensions (Out of Scope)

- Swing phase segmentation
- Clubhead and ball tracking
- Ball launch and trajectory mapping
- Pose estimation and biomechanical metrics
- Batch processing of multiple videos
- Interactive review interfaces

---

## 11. Risks and Open Questions

- Variability in audio quality across recording devices
- High background motion in range environments
- Double-impact events (mat and ball)
- Camera placement variability

---

## 12. Summary

This project focuses on solving a foundational problem: **automatic, reliable swing segmentation**. By emphasizing deterministic signal analysis, transparency, and modular system boundaries, it establishes a solid base for advanced golf analytics in future phases.

---