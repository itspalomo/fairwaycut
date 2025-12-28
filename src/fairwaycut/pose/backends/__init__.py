"""Pose estimation backends with platform-aware selection.

This module provides a factory for creating pose estimation backends,
automatically selecting the best available backend for the current platform:

- macOS/iOS: Apple Vision Framework (hardware-accelerated via Neural Engine)
- Windows/Linux: MediaPipe (CPU-based, cross-platform)

Usage:
    from fairwaycut.pose.backends import create_backend
    
    # Automatically selects best backend for current platform
    backend = create_backend()
    
    # Or explicitly request a specific backend
    backend = create_backend(prefer_native=False)  # Force MediaPipe
    
    # Use the backend
    with backend:
        result = backend.process_video("video.mp4")
"""

import platform
from typing import Optional

from fairwaycut.pose.backends.base import (
    PoseBackend,
    STANDARD_LANDMARK_INDICES,
    GOLF_REQUIRED_LANDMARKS,
)


def get_available_backends() -> list[str]:
    """
    Get list of available backend names on the current platform.
    
    Returns:
        List of backend names that can be instantiated.
    """
    available = []
    
    # Check Apple Vision
    if platform.system() == "Darwin":
        try:
            from fairwaycut.pose.backends.apple_vision import is_available
            if is_available():
                available.append("apple_vision")
        except ImportError:
            pass
    
    # Check MediaPipe
    try:
        from fairwaycut.pose.backends.mediapipe import is_available
        if is_available():
            available.append("mediapipe")
    except ImportError:
        pass
    
    return available


def _filter_kwargs_for_backend(backend_type: str, kwargs: dict) -> dict:
    """Filter kwargs to only include those accepted by the specified backend."""
    apple_vision_params = {"max_frame_size", "min_confidence"}
    mediapipe_params = {
        "max_frame_size",
        "model_complexity",
        "min_detection_confidence",
        "min_tracking_confidence",
        "enable_segmentation",
    }
    
    if backend_type == "apple_vision":
        return {k: v for k, v in kwargs.items() if k in apple_vision_params}
    else:  # mediapipe
        return {k: v for k, v in kwargs.items() if k in mediapipe_params}


def create_backend(
    prefer_native: bool = True,
    backend_name: Optional[str] = None,
    **kwargs,
) -> PoseBackend:
    """
    Create a pose estimation backend, selecting the best one for the platform.
    
    Args:
        prefer_native: If True, prefer platform-native backends (Apple Vision on macOS).
        backend_name: Explicitly request a specific backend ("apple_vision" or "mediapipe").
        **kwargs: Additional arguments passed to the backend constructor.
                  Only compatible arguments are passed to each backend.
    
    Returns:
        Initialized PoseBackend instance.
    
    Raises:
        ImportError: If no suitable backend is available.
        ValueError: If requested backend_name is not available.
    
    Example:
        # Auto-select best backend
        backend = create_backend()
        
        # Force MediaPipe backend
        backend = create_backend(prefer_native=False)
        
        # Explicit backend selection
        backend = create_backend(backend_name="apple_vision")
        
        # Pass arguments to backend (filtered automatically)
        backend = create_backend(max_frame_size=480, min_confidence=0.6)
    """
    system = platform.system()
    
    # If explicit backend requested, try to use it
    if backend_name:
        if backend_name == "apple_vision":
            try:
                from fairwaycut.pose.backends.apple_vision import (
                    AppleVisionBackend,
                    is_available,
                )
                if not is_available():
                    raise ValueError(
                        "Apple Vision backend is not available on this system. "
                        "Requires macOS 11+ and pyobjc-framework-Vision."
                    )
                filtered_kwargs = _filter_kwargs_for_backend("apple_vision", kwargs)
                return AppleVisionBackend(**filtered_kwargs)
            except ImportError as e:
                raise ValueError(
                    f"Apple Vision backend could not be imported: {e}"
                ) from e
        
        elif backend_name == "mediapipe":
            try:
                from fairwaycut.pose.backends.mediapipe import (
                    MediaPipeBackend,
                    is_available,
                )
                if not is_available():
                    raise ValueError(
                        "MediaPipe backend is not available. "
                        "Install with: pip install mediapipe"
                    )
                filtered_kwargs = _filter_kwargs_for_backend("mediapipe", kwargs)
                return MediaPipeBackend(**filtered_kwargs)
            except ImportError as e:
                raise ValueError(
                    f"MediaPipe backend could not be imported: {e}"
                ) from e
        
        else:
            raise ValueError(
                f"Unknown backend: {backend_name}. "
                f"Available: {get_available_backends()}"
            )
    
    # Auto-select based on platform
    if prefer_native and system == "Darwin":
        # Try Apple Vision first on macOS
        try:
            from fairwaycut.pose.backends.apple_vision import (
                AppleVisionBackend,
                is_available,
            )
            if is_available():
                filtered_kwargs = _filter_kwargs_for_backend("apple_vision", kwargs)
                return AppleVisionBackend(**filtered_kwargs)
        except ImportError:
            pass  # Fall through to MediaPipe
    
    # Fall back to MediaPipe
    try:
        from fairwaycut.pose.backends.mediapipe import (
            MediaPipeBackend,
            is_available,
        )
        if is_available():
            filtered_kwargs = _filter_kwargs_for_backend("mediapipe", kwargs)
            return MediaPipeBackend(**filtered_kwargs)
    except ImportError:
        pass
    
    # No backend available
    available = get_available_backends()
    if not available:
        raise ImportError(
            "No pose estimation backend is available. "
            "Install one of:\n"
            "  - MediaPipe: pip install mediapipe\n"
            "  - Apple Vision (macOS only): pip install pyobjc-framework-Vision"
        )
    
    raise ImportError(
        f"Could not initialize any backend. Available backends: {available}"
    )


# Re-export key classes and constants
__all__ = [
    "PoseBackend",
    "create_backend",
    "get_available_backends",
    "STANDARD_LANDMARK_INDICES",
    "GOLF_REQUIRED_LANDMARKS",
]

