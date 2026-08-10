from lyric_align.model import AlignedLine, Segment
from lyric_align.qwen_backend import AudioBuffer
import lyric_align.pipeline as pipeline


def _install_pipeline_fakes(monkeypatch, *, aligner_error=False):
    audio = AudioBuffer([0.0] * 100, 10)

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def separate(self, _audio_path, output_path, *, overwrite=False):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return output_path

    class FakeASR:
        kwargs_seen = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).kwargs_seen = kwargs
            FakeAligner.asr_kwargs_seen = kwargs

        def transcribe_segments(self, _audio, *, language):
            return [
                Segment(0.0, 2.0, "甲乙"),
                Segment(2.0, 4.0, "丙丁"),
            ]

        def close(self):
            self.closed = True

    class FakeAligner:
        calls = []
        asr_kwargs_seen = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.attempts = 0

        def load(self):
            return self

        @property
        def active_providers(self):
            return ("CUDAExecutionProvider",)

        def align_chunk(self, _audio, chunk, *, language):
            self.attempts += 1
            self.calls.append(chunk.padding)
            if aligner_error:
                raise ValueError("synthetic alignment anomaly")
            if self.attempts == 1:
                raise ValueError("synthetic first-attempt anomaly")
            return [
                AlignedLine(
                    line.line,
                    0.2 + index,
                    0.8 + index,
                    line.score,
                    True,
                    chars=[
                        {
                            "text": line.line[0],
                            "start": 0.2 + index,
                            "end": 0.8 + index,
                        }
                    ],
                )
                for index, line in enumerate(chunk.lines)
            ]

        def close(self):
            self.closed = True

    monkeypatch.setattr(pipeline, "MelBandSeparator", FakeSeparator)
    monkeypatch.setattr(pipeline, "QwenASRBackend", FakeASR)
    monkeypatch.setattr(pipeline, "QwenForcedAlignerBackend", FakeAligner)
    monkeypatch.setattr(pipeline, "load_audio", lambda _path: audio)
    return FakeAligner


def test_pipeline_retries_with_four_second_padding(tmp_path, monkeypatch):
    aligner_type = _install_pipeline_fakes(monkeypatch)
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"placeholder")

    result = pipeline.run_qwen_pipeline(
        audio_path,
        ["甲乙", "丙丁"],
        project_root=tmp_path,
        device="cuda:0",
        pairing=1,
    )

    assert result.retry_count == 1
    assert result.failures == ()
    assert aligner_type.calls == [1.5, 4.0]
    assert all(line.matched for line in result.aligned)


def test_pipeline_marks_second_alignment_failure_as_unmatched(tmp_path, monkeypatch):
    _install_pipeline_fakes(monkeypatch, aligner_error=True)
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"placeholder")

    result = pipeline.run_qwen_pipeline(
        audio_path,
        ["甲乙", "丙丁"],
        project_root=tmp_path,
        pairing=1,
    )

    assert result.retry_count == 3
    assert len(result.failures) == 2
    assert all(not line.matched for line in result.aligned)
    assert all(line.start is None and line.end is None for line in result.aligned)


def test_pipeline_interpolates_only_when_requested(tmp_path, monkeypatch):
    _install_pipeline_fakes(monkeypatch, aligner_error=True)
    interpolated = []
    monkeypatch.setattr(
        pipeline,
        "interpolate_gaps",
        lambda lines: interpolated.append(lines) or lines,
    )
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"placeholder")

    result = pipeline.run_qwen_pipeline(
        audio_path,
        ["甲乙", "丙丁"],
        project_root=tmp_path,
        pairing=1,
        interpolate=True,
    )

    assert len(result.failures) == 2
    assert interpolated == [result.aligned]
    assert all(not line.matched for line in result.aligned)


def test_coarse_gap_bridge_only_changes_forced_align_planning_copy():
    coarse = [
        AlignedLine("前", 10.0, 12.0, 1.0, True),
        AlignedLine("漏", None, None, 0.1, False),
        AlignedLine("后", 14.0, 16.0, 1.0, True),
    ]

    planned = pipeline._fill_coarse_gaps(coarse, audio_duration=20.0)

    assert coarse[1].start is None and coarse[1].end is None
    assert planned[1].start is not None and planned[1].end is not None
    assert planned[1].start < planned[1].end
    assert planned[1].matched is False


def test_coarse_gap_bridge_does_not_reprocess_inner_gap_lines():
    coarse = [
        AlignedLine("左", 56.0, 64.0, 1.0, True),
        AlignedLine("一", None, None, 0.1, False),
        AlignedLine("二", None, None, 0.1, False),
        AlignedLine("三", None, None, 0.1, False),
        AlignedLine("右", 84.0, 92.0, 1.0, True),
    ]

    planned = pipeline._fill_coarse_gaps(coarse, audio_duration=120.0)

    centers = [
        (planned[index].start + planned[index].end) / 2
        for index in (1, 2, 3)
    ]
    assert centers == sorted(centers)
    assert centers[0] < centers[1] < centers[2]


def test_hubertfa_chunks_respect_line_and_audio_limits():
    coarse = [
        AlignedLine(f"行{i}", i * 8.0, (i + 1) * 8.0, 1.0, True)
        for i in range(36)
    ]

    chunks = pipeline._build_hubertfa_chunks(
        coarse,
        padding=1.5,
        audio_duration=300.0,
    )

    assert len(chunks) == 3
    assert all(10 <= len(chunk.lines) <= 15 for chunk in chunks)
    assert all(chunk.audio_end - chunk.audio_start <= 100.0 for chunk in chunks)
    assert [index for chunk in chunks for index in chunk.line_indices] == list(range(36))


def test_hubertfa_mode_does_not_load_qwen_coarse_forced_aligner(tmp_path, monkeypatch):
    aligner_type = _install_pipeline_fakes(monkeypatch)
    monkeypatch.setattr(pipeline, "HubertFABackend", aligner_type)
    audio_path = tmp_path / "song.flac"
    audio_path.write_bytes(b"placeholder")

    result = pipeline.run_qwen_pipeline(
        audio_path,
        ["甲乙", "丙丁"],
        project_root=tmp_path,
        pairing=1,
        aligner_backend="hubertfa",
    )

    assert result.failures == ()
    assert aligner_type.asr_kwargs_seen["forced_aligner_model"] is None
