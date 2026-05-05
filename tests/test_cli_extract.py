import numpy as np
from click.testing import CliRunner

from fairwaycut import cli as cli_module
from fairwaycut.core.models import AudioData, DetectionResult, FusionResult, SwingEvent


def test_extract_with_overlays_decodes_audio_once(monkeypatch, tmp_path):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")

    extracted_audio = AudioData(
        samples=np.ones(20, dtype=np.float32),
        sample_rate=10,
        duration=2.0,
        source_file=str(video_path),
    )
    audio_calls = []

    def fake_extract_audio(video, start_time=None, end_time=None, sample_rate=44100):
        audio_calls.append((video, start_time, end_time, sample_rate))
        return extracted_audio

    detection_result = DetectionResult(
        events=[],
        parameters={},
        envelope=np.array([], dtype=np.float32),
        envelope_times=np.array([], dtype=np.float32),
        envelope_db=np.array([], dtype=np.float32),
    )
    fusion_result = FusionResult(
        swings=[
            SwingEvent(
                swing_id=1,
                impact_time=1.0,
                start_time=0.0,
                end_time=2.0,
            )
        ],
        audio_result=detection_result,
        pose_result=None,
        parameters={},
    )

    def fake_detect_swings(*args, **kwargs):
        assert kwargs["audio"] is extracted_audio
        return fusion_result

    def fake_generate_all_swing_clips(video, output_dir, result, audio, options=None, progress_callback=None):
        assert audio is extracted_audio
        return [tmp_path / "swing_001.mp4"]

    monkeypatch.setattr("fairwaycut.audio.extraction.extract_audio_from_video", fake_extract_audio)
    monkeypatch.setattr("fairwaycut.fusion.detector.detect_swings", fake_detect_swings)
    monkeypatch.setattr("fairwaycut.video.generator.generate_all_swing_clips", fake_generate_all_swing_clips)

    runner = CliRunner()
    result = runner.invoke(cli_module.main, ["extract", str(video_path), "--with-overlays", "all"])

    assert result.exit_code == 0
    assert len(audio_calls) == 1
