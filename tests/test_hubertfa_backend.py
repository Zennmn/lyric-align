from types import SimpleNamespace

from lyric_align.chunking import build_chunks
from lyric_align.hubertfa_backend import _line_g2p, _words_to_lines
from lyric_align.model import AlignedLine


def _dictionaries(tmp_path):
    japanese = tmp_path / "japanese.txt"
    english = tmp_path / "english.txt"
    japanese.write_text(
        "ma\tm a\n"
        "ta\tt a\n"
        "i\ti\n"
        "n\tN\n"
        "te\tt e\n",
        encoding="utf-8",
    )
    english.write_text(
        "i'll\tay l\n"
        "need\tn iy d\n"
        "you\ty uw\n",
        encoding="utf-8",
    )
    from lyric_align.hubertfa_backend import _read_dictionary

    return _read_dictionary(japanese), _read_dictionary(english)


def test_hubertfa_mixed_g2p_keeps_japanese_and_english_prefixes(tmp_path):
    import pykakasi

    japanese, english = _dictionaries(tmp_path)
    specs, phonemes, mapping = _line_g2p(
        "また I'll need you",
        0,
        japanese,
        english,
        pykakasi.kakasi(),
    )

    assert [spec.source_text for spec in specs] == ["また", "I'll", "need", "you"]
    assert "ja/m" in phonemes
    assert "en/iy" in phonemes
    assert "ja/cl" not in phonemes
    assert len(phonemes) == len(mapping)


def test_hubertfa_word_intervals_split_japanese_chars_and_keep_english_words():
    coarse = [
        AlignedLine("また I'll", 10.0, 12.0, 1.0, True),
    ]
    chunk = build_chunks(coarse, min_lines=1, max_lines=1, padding=1.5)[0]
    specs = [
        SimpleNamespace(line_index=0, source_text="また", language="ja"),
        SimpleNamespace(line_index=0, source_text="I'll", language="en"),
    ]
    words = [
        SimpleNamespace(start=1.0, end=2.0),
        SimpleNamespace(start=2.0, end=2.5),
    ]

    result = _words_to_lines(words, specs, chunk)

    assert [item["text"] for item in result[0].chars] == ["ま", "た", "I'll"]
    assert result[0].chars[0]["start"] == 9.5
    assert result[0].chars[-1]["end"] == 11.0
    assert all(item["end"] > item["start"] for item in result[0].chars)


def test_hubertfa_rejects_implausibly_long_line_span():
    coarse = [AlignedLine("また", 10.0, 11.0, 1.0, True)]
    chunk = build_chunks(coarse, min_lines=1, max_lines=1, padding=1.5)[0]
    specs = [SimpleNamespace(line_index=0, source_text="また", language="ja")]
    words = [SimpleNamespace(start=0.0, end=20.0)]

    import pytest

    with pytest.raises(ValueError, match="implausibly long"):
        _words_to_lines(words, specs, chunk)


def test_hubertfa_active_providers_are_empty_before_lazy_load(tmp_path):
    from lyric_align.hubertfa_backend import HubertFABackend

    backend = HubertFABackend(
        model_path=tmp_path / "model.onnx",
        source_dir=tmp_path / "source",
    )

    assert backend.active_providers == ()
