from fairwaycut.video.extraction import extract_video_clip


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
