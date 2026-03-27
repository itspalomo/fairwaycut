import numpy as np

from fairwaycut.core.models import AudioData, DetectionResult, FusionResult, SwingEvent
from fairwaycut.video.extraction import attach_audio_to_video, extract_video_clip
from fairwaycut.video.generator import DemoVideoGenerator, DemoVideoOptions


def test_extract_video_clip_uses_ffmpeg(monkeypatch, tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "clip.mp4"
    captured = {}

    def fake_run(command, capture_output, check):
        captured["command"] = command
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": b"", "stderr": b""},
        )()

    monkeypatch.setattr("fairwaycut.video.extraction.subprocess.run", fake_run)

    extract_video_clip(video_path, output_path, start_time=1.25, end_time=3.5)

    assert captured["command"][:5] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    assert "-ss" in captured["command"]
    assert "1.250000" in captured["command"]
    assert "-t" in captured["command"]
    assert "2.250000" in captured["command"]


def test_attach_audio_to_video_uses_ffmpeg(monkeypatch, tmp_path):
    rendered_video = tmp_path / "rendered.mp4"
    rendered_video.write_bytes(b"video")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source")
    output_path = tmp_path / "muxed.mp4"
    captured = {}

    monkeypatch.setattr(
        "fairwaycut.video.extraction._has_audio_stream",
        lambda _: True,
    )

    def fake_run(command, capture_output, check):
        captured["command"] = command
        output_path.write_bytes(b"muxed")
        return type(
            "CompletedProcess",
            (),
            {"returncode": 0, "stdout": b"", "stderr": b""},
        )()

    monkeypatch.setattr("fairwaycut.video.extraction.subprocess.run", fake_run)

    attach_audio_to_video(
        rendered_video,
        source_video,
        output_path,
        start_time=1.25,
        end_time=3.5,
    )

    assert captured["command"][:7] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
    ]
    assert str(rendered_video) in captured["command"]
    assert "-ss" in captured["command"]
    assert "1.250000" in captured["command"]
    assert "-t" in captured["command"]
    assert "2.250000" in captured["command"]
    assert str(source_video) in captured["command"]
    assert output_path.exists()


def test_generate_swing_clip_attaches_source_audio(monkeypatch, tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "swing_001.mp4"

    monkeypatch.setattr(
        "fairwaycut.video.generator.get_video_info",
        lambda _: type(
            "Info",
            (),
            {
                "width": 4,
                "height": 4,
                "fps": 1.0,
                "total_frames": 2,
                "duration": 2.0,
            },
        )(),
    )

    class FakeCapture:
        def __init__(self, _):
            self.frames = [
                np.zeros((4, 4, 3), dtype=np.uint8),
                np.zeros((4, 4, 3), dtype=np.uint8),
            ]
            self.index = 0

        def set(self, *_args):
            return True

        def isOpened(self):
            return self.index < len(self.frames)

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame.copy()

        def release(self):
            return None

    class FakeWriter:
        def __init__(self, *_args):
            self.frames = []

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            return None

    monkeypatch.setattr("fairwaycut.video.generator.cv2.VideoCapture", FakeCapture)
    monkeypatch.setattr("fairwaycut.video.generator.cv2.VideoWriter", FakeWriter)
    monkeypatch.setattr("fairwaycut.video.generator.cv2.VideoWriter_fourcc", lambda *_: 0)

    attached = {}

    def fake_attach(rendered_video_path, source_video_path, final_output_path, start_time=0.0, end_time=None, audio_codec="aac"):
        attached["rendered"] = rendered_video_path
        attached["source"] = source_video_path
        attached["output"] = final_output_path
        attached["start_time"] = start_time
        attached["end_time"] = end_time
        return final_output_path

    monkeypatch.setattr("fairwaycut.video.generator.attach_audio_to_video", fake_attach)

    generator = DemoVideoGenerator(options=DemoVideoOptions(show_waveform=False))
    audio = AudioData(
        samples=np.zeros(2, dtype=np.float32),
        sample_rate=1,
        duration=2.0,
        source_file=str(video_path),
    )
    swing = SwingEvent(
        swing_id=1,
        impact_time=1.0,
        start_time=0.0,
        end_time=2.0,
    )
    detection = DetectionResult(
        events=[],
        parameters={},
        envelope=np.array([], dtype=np.float32),
        envelope_times=np.array([], dtype=np.float32),
        envelope_db=np.array([], dtype=np.float32),
    )
    fusion_result = FusionResult(
        swings=[swing],
        audio_result=detection,
        pose_result=None,
    )

    clip_path = generator.generate_swing_clip(
        video_path,
        output_path,
        swing,
        fusion_result,
        audio,
    )

    assert clip_path == output_path
    assert attached["source"] == video_path
    assert attached["output"] == output_path
    assert attached["start_time"] == 0.0
    assert attached["end_time"] == 2.0
