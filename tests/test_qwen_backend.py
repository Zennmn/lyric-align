from types import SimpleNamespace

import pytest

from lyric_align.chunking import build_chunks
from lyric_align.model import AlignedLine
from lyric_align.qwen_backend import (
    AudioBuffer,
    QwenASRBackend,
    _character_tokenize,
    _items_to_lines,
    _resolve_model_path,
    normalize_language,
)


def test_language_codes_are_mapped_to_qwen_names():
    assert normalize_language("zh") == "Chinese"
    assert normalize_language("ja") == "Japanese"
    assert normalize_language(None) is None


def test_asr_uses_forced_timestamps_for_segment_bounds():
    class FakeResult:
        text = "そ I need you"
        time_stamps = SimpleNamespace(
            items=[
                SimpleNamespace(text="そ", start_time=0.2, end_time=0.4),
                SimpleNamespace(text="I", start_time=0.4, end_time=0.6),
                SimpleNamespace(text="need", start_time=0.6, end_time=0.9),
                SimpleNamespace(text="you", start_time=0.9, end_time=1.2),
            ]
        )

    class FakeModel:
        def transcribe(self, **kwargs):
            assert kwargs["return_time_stamps"] is True
            return [FakeResult()]

    backend = QwenASRBackend(
        window_seconds=2.0,
        overlap_seconds=0.0,
        forced_aligner_model="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    backend._model = FakeModel()

    segments = backend.transcribe_segments(
        AudioBuffer([0.0] * 20, 10), language="ja"
    )

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(0.2)
    assert segments[0].end == pytest.approx(1.2)
    assert [word.word for word in segments[0].words] == [
        "そ", "I", "need", "you"
    ]


def test_forced_aligner_items_are_mapped_to_cjk_lines_and_global_time():
    coarse = [
        AlignedLine("甲乙", 10.0, 12.0, 0.8, True),
        AlignedLine("丙丁", 12.0, 14.0, 0.7, True),
    ]
    chunk = build_chunks(coarse, min_lines=2, max_lines=4, padding=1.5)[0]
    items = [
        SimpleNamespace(text="甲", start_time=1.7, end_time=1.9),
        SimpleNamespace(text="乙", start_time=1.9, end_time=2.2),
        SimpleNamespace(text="丙", start_time=2.3, end_time=2.6),
        SimpleNamespace(text="丁", start_time=2.6, end_time=2.9),
    ]

    result = _items_to_lines(items, chunk, "Chinese", audio_duration=20.0)

    assert [line.line for line in result] == ["甲乙", "丙丁"]
    assert result[0].chars[0]["start"] == pytest.approx(10.2)
    assert result[0].chars[1]["end"] == pytest.approx(10.7)
    assert result[1].chars[0]["start"] == pytest.approx(10.8)
    assert result[1].chars[1]["end"] == pytest.approx(11.4)


def test_forced_aligner_item_count_mismatch_is_rejected():
    line = AlignedLine("甲乙", 0.0, 1.0, 1.0, True)
    chunk = build_chunks([line], min_lines=2, max_lines=4)[0]
    items = [SimpleNamespace(text="甲", start_time=0.0, end_time=0.2)]

    with pytest.raises(ValueError, match="item count"):
        _items_to_lines(items, chunk, "Chinese", audio_duration=2.0)


def test_japanese_alignment_uses_character_units():
    line = AlignedLine("あいう", 10.0, 12.0, 1.0, True)
    chunk = build_chunks([line], min_lines=2, max_lines=4, padding=1.5)[0]
    items = [
        SimpleNamespace(text="あ", start_time=1.5, end_time=1.8),
        SimpleNamespace(text="い", start_time=1.8, end_time=2.1),
        SimpleNamespace(text="う", start_time=2.1, end_time=2.4),
    ]

    result = _items_to_lines(items, chunk, "Japanese", audio_duration=5.0)

    assert [item["text"] for item in result[0].chars] == ["あ", "い", "う"]


def test_mixed_japanese_tokenizer_groups_latin_words():
    assert _character_tokenize("そして I need you") == [
        "そ", "し", "て", "I", "need", "you"
    ]


def test_mixed_language_items_keep_latin_words_as_ttml_units():
    line = AlignedLine("Friday night 白い", 10.0, 12.0, 1.0, True)
    chunk = build_chunks([line], min_lines=2, max_lines=4, padding=1.5)[0]
    items = [
        SimpleNamespace(text="Friday", start_time=1.0, end_time=1.6),
        SimpleNamespace(text="night", start_time=1.6, end_time=2.0),
        SimpleNamespace(text="白", start_time=2.0, end_time=2.2),
        SimpleNamespace(text="い", start_time=2.2, end_time=2.4),
    ]

    result = _items_to_lines(items, chunk, "Japanese", audio_duration=5.0)

    assert [item["text"] for item in result[0].chars] == [
        "Friday", "night", "白", "い"
    ]
    assert all(item.get("unit") is True for item in result[0].chars)
    assert all(item["end"] >= item["start"] for item in result[0].chars)


def test_forced_aligner_timestamp_outside_chunk_is_rejected():
    line = AlignedLine("甲乙", 1.0, 2.0, 1.0, True)
    chunk = build_chunks([line], min_lines=2, max_lines=4, padding=0.5)[0]
    items = [
        SimpleNamespace(text="甲", start_time=0.0, end_time=1.0),
        SimpleNamespace(text="乙", start_time=1.0, end_time=2.1),
    ]

    with pytest.raises(ValueError, match="exceeds chunk audio"):
        _items_to_lines(items, chunk, "Chinese", audio_duration=2.0)


def test_forced_aligner_result_outside_coarse_window_is_rejected():
    line = AlignedLine("甲乙", 10.0, 12.0, 1.0, True)
    chunk = build_chunks([line], min_lines=2, max_lines=4, padding=1.5)[0]
    items = [
        SimpleNamespace(text="甲", start_time=0.0, end_time=0.1),
        SimpleNamespace(text="乙", start_time=0.1, end_time=0.2),
    ]

    with pytest.raises(ValueError, match="outside the coarse"):
        _items_to_lines(items, chunk, "Chinese", audio_duration=20.0)


def test_model_resolver_prefers_local_qwen_directory(tmp_path):
    model_path = tmp_path / "Qwen3-ASR-1.7B"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    resolved = _resolve_model_path(
        "Qwen/Qwen3-ASR-1.7B-hf", tmp_path, local_files_only=True
    )

    assert resolved == str(model_path.resolve())


def test_model_resolver_offline_failure_is_explicit(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("network disabled")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    with pytest.raises(FileNotFoundError, match="not available locally"):
        _resolve_model_path("Qwen/missing-model", tmp_path, local_files_only=True)

    assert calls[0]["local_files_only"] is True
    assert "local_dir" not in calls[0]
