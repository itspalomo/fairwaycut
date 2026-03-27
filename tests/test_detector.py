import numpy as np

from fairwaycut.core.config import ProcessingMode
from fairwaycut.core.models import AudioData, DetectionResult, ImpactEvent
from fairwaycut.fusion.detector import (
    _build_merged_pose_windows,
    detect_swings,
)


def _empty_detection_result() -> DetectionResult:
    return DetectionResult(
        events=[],
        parameters={},
        envelope=np.array([], dtype=np.float32),
        envelope_times=np.array([], dtype=np.float32),
        envelope_db=np.array([], dtype=np.float32),
    )


def test_detect_swings_reuses_provided_audio(monkeypatch, tmp_path):
    video_path = tmp_path / "reuse.mp4"
    video_path.write_bytes(b"video")
    audio = AudioData(
        samples=np.ones(8, dtype=np.float32),
        sample_rate=4,
        duration=2.0,
        source_file=str(video_path),
        start_time=10.0,
    )

    monkeypatch.setattr(
        "fairwaycut.fusion.detector.extract_audio_from_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected audio decode")),
    )
    monkeypatch.setattr(
        "fairwaycut.fusion.detector.detect_impacts_adaptive_snr",
        lambda *args, **kwargs: _empty_detection_result(),
    )

    result = detect_swings(
        video_path,
        mode=ProcessingMode.AUDIO,
        start_time=10.0,
        end_time=12.0,
        audio=audio,
    )

    assert result.parameters["analysis_start_time"] == 10.0
    assert result.parameters["analysis_end_time"] == 12.0


def test_build_merged_pose_windows_merges_overlapping_ranges():
    events = [
        ImpactEvent(timestamp=10.0, confidence=1.0, amplitude_db=-3.0),
        ImpactEvent(timestamp=12.0, confidence=1.0, amplitude_db=-3.0),
        ImpactEvent(timestamp=20.0, confidence=1.0, amplitude_db=-3.0),
    ]

    windows = _build_merged_pose_windows(
        events,
        pre_impact_sec=3.0,
        post_impact_sec=2.0,
        video_start=0.0,
        video_end=30.0,
    )

    assert [(window.start_time, window.end_time) for window in windows] == [
        (7.0, 14.0),
        (17.0, 22.0),
    ]
    assert [window.event_indices for window in windows] == [(0, 1), (2,)]
