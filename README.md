# FairwayCut

![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
![CI](https://img.shields.io/badge/CI-pending-yellow)

**Auto-cut golf videos into clean swing clips — open-source, accurate, and batch-friendly.**

FairwayCut is a local, offline command-line tool designed for golfers and developers. It automatically detects and extracts individual golf swings from continuous driving range videos without the need for manual editing or cloud processing.

## Key Features

- **Local & Privacy-Focused**: Runs entirely on your machine. No cloud uploads.
- **Deterministic**: Same input + same settings = same output.
- **Flexible Modes**:
    - `audio`: Fast detection based on impact sounds.
    - `hybrid`: Combines audio with pose estimation (Apple Vision / MediaPipe) for high accuracy.
- **Overlay Visualizations**: Generate demo videos with pose skeletons and audio waveforms.

## Installation

**Prerequisites**:
- Python 3.11 or higher
- [ffmpeg](https://ffmpeg.org/) (required for video processing)

Install via `pip` or `uv` (recommended):

```bash
# Install from source
git clone https://github.com/itspalomo/fairwaycut.git
cd fairwaycut
uv sync
```

### Apple Silicon Acceleration
For hardware-accelerated pose estimation on macOS:

```bash
uv sync --extra apple
```

## Quick Start

1.  **Extract Swings**: The most common use case.
    ```bash
    uv run fairwaycut extract input_video.mov --output-dir ./swings --mode hybrid
    ```

2.  **View Help**:
    ```bash
    uv run fairwaycut --help
    ```

## Command Reference

### `extract`
Detects swings and saves them as individual video files.

```bash
fairwaycut extract <VIDEO_PATH> [OPTIONS]
```
- `--mode`: `audio` (fast), `hybrid` (accurate), `lite`, or `full`.
- `--pre-impact`: Seconds to include before impact (default: 2.5).
- `--post-impact`: Seconds to include after impact (default: 1.0).

### `analyze`
Detects swings and prints a text report without saving videos. Good for testing parameters.

```bash
fairwaycut analyze <VIDEO_PATH>
```

### `demo`
Creates a full-length video with debugging overlays (skeletons, waveforms, impact markers).

```bash
fairwaycut demo <VIDEO_PATH> --output demo.mp4 --skeleton --waveform
```

### `plot`
Generates a matplotlib figure showing audio analysis and detection signals.

```bash
fairwaycut plot <VIDEO_PATH>
```

## Roadmap

- [ ] Batch processing for multiple files
- [ ] Advanced swing phase analysis (Top of Backswing, Address, Finish)
- [ ] Ball flight tracking

## Contributing

We welcome unique contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up your development environment and submit pull requests.

## License

MIT License. See [LICENSE](LICENSE) for details.