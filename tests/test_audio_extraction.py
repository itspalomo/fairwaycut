import numpy as np

from fairwaycut.audio.extraction import extract_audio_from_video
from fairwaycut.core.models import AudioData


def test_extract_audio_from_video_uses_ffmpeg_range(monkeypatch, tmp_path):
    video_path = tmp_path / "range.mp4"
    video_path.write_bytes(b"video")

    captured = {}
    samples = np.array([0.1, -0.2], dtype=np.float32)

    monkeypatch.setattr(
        "fairwaycut.audio.extraction._has_audio_stream",
        lambda _: True,
    )

    def fake_run(command, capture_output, check):
        captured["command"] = command
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": samples.tobytes(), "stderr": b""},
        )()

    monkeypatch.setattr("fairwaycut.audio.extraction.subprocess.run", fake_run)

    audio = extract_audio_from_video(
        video_path,
        start_time=12.5,
        end_time=13.0,
        sample_rate=4,
    )

    assert audio.start_time == 12.5
    assert audio.duration == 0.5
    assert np.array_equal(audio.samples, samples)
    assert "-ss" in captured["command"]
    assert "12.500000" in captured["command"]
    assert "-t" in captured["command"]
    assert "0.500000" in captured["command"]


def test_audio_segment_uses_absolute_timestamps():
    audio = AudioData(
        samples=np.arange(20, dtype=np.float32),
        sample_rate=2,
        duration=10.0,
        source_file="test.wav",
        start_time=5.0,
    )

    segment = audio.get_segment(7.0, 10.0)

    assert segment.start_time == 7.0
    assert segment.end_time == 10.0
    assert np.array_equal(segment.samples, np.array([4, 5, 6, 7, 8, 9], dtype=np.float32))
