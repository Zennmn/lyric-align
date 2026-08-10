import math

import pytest

from lyric_align.chunking import AlignmentChunk, build_chunks
from lyric_align.model import AlignedLine


def line(start, end, *, matched=True):
    return AlignedLine("line", start, end, 1.0 if matched else 0.0, matched)


def test_empty_and_unusable_input_returns_no_chunks():
    assert build_chunks([]) == []
    assert build_chunks(None) == []
    assert build_chunks([line(None, None, matched=False), line(2.0, 1.0)]) == []
    assert build_chunks([line(math.nan, 1.0)]) == []


def test_only_valid_contiguous_lines_are_grouped():
    aligned = [line(0.0, 1.0), line(1.1, 2.0), line(None, None, matched=False),
               line(4.0, 5.0), line(5.1, 6.0)]

    chunks = build_chunks(aligned)

    assert [chunk.line_indices for chunk in chunks] == [(0, 1), (3, 4)]
    assert all(chunk.lines[0] is aligned[chunk.line_indices[0]] for chunk in chunks)


def test_odd_runs_are_balanced_without_an_undersized_tail():
    aligned = [line(float(i), float(i + 1)) for i in range(7)]

    chunks = build_chunks(aligned)

    assert [len(chunk.lines) for chunk in chunks] == [4, 3]
    assert [chunk.line_indices for chunk in chunks] == [(0, 1, 2, 3), (4, 5, 6)]


def test_single_short_run_is_retained():
    chunks = build_chunks([line(3.0, 4.0)])

    assert len(chunks) == 1
    assert chunks[0].line_indices == (0,)


def test_padding_is_clamped_and_can_be_replaced_for_retry():
    chunk = build_chunks([line(0.5, 1.0), line(1.0, 2.0)])[0]

    assert chunk.coarse_start == 0.5
    assert chunk.coarse_end == 2.0
    assert chunk.audio_start == 0.0
    assert chunk.audio_end == 3.5

    retry = chunk.with_padding(4.0)
    assert retry.line_indices == chunk.line_indices
    assert retry.lines == chunk.lines
    assert retry.padding == 4.0
    assert retry.audio_start == 0.0
    assert retry.audio_end == 6.0


def test_chunking_is_repeatable_and_chunks_are_frozen():
    aligned = [line(float(i), float(i + 1)) for i in range(5)]

    first = build_chunks(aligned)
    second = build_chunks(aligned)

    assert first == second
    assert isinstance(first[0], AlignmentChunk)
    with pytest.raises((AttributeError, TypeError)):
        first[0].padding = 4.0
