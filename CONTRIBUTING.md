# Contributing to FairwayCut

Thank you for your interest in contributing! We want to make it easy for you to get started.

## Development Setup

We use `uv` for dependency management and `ruff` for code styling.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/itspalomo/fairwaycut.git
    cd fairwaycut
    ```

2.  **Install dependencies**:
    ```bash
    # Standard installation
    uv sync

    # With Apple Silicon support (optional)
    uv sync --extra apple
    ```

3.  **Activate virtual environment**:
    ```bash
    source .venv/bin/activate
    ```

## Code Quality

We strictly enforce code style using `ruff`. Please ensure your code passes checks before submitting a PR.

```bash
# Format code
uv run ruff format src

# Check for linting errors
uv run ruff check src --fix
```

## Running Tests

Run the test suite using `pytest`:

```bash
uv run pytest
```

## Branching Strategy

- **main**: Stable production code.
- **Feature Branches**: Create a new branch for your feature or fix (e.g., `feature/new-detector` or `fix/audio-sync`).

## Pull Request Process

1.  Fork the repo and create your branch from `main`.
2.  Add tests for any new functionality.
3.  Ensure all tests pass and code is formatted.
4.  Submit a Pull Request with a clear description of your changes.

## Reporting Issues

If you find a bug or have a feature request, please open an Issue on GitHub with as much detail as possible (logs, reproduction steps, video samples if applicable).
