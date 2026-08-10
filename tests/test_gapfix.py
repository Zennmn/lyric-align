from copy import deepcopy

import pytest

from lyric_align.gapfix import fix_small_gaps
from lyric_align.model import AlignedLine


def make_line(start=0.0, end=2.0, chars=None):
    return AlignedLine("ab", start, end, 0.9, True, chars)


def test_small_gap_is_split_and_text_key_metadata_are_preserved():
    line = make_line(
        chars=[
            {"char": "a", "start": 0.0, "end": 1.0, "confidence": 0.8},
            {"text": "b", "start": 1.05, "end": 2.0, "space_after": False},
        ]
    )

    fixed = fix_small_gaps([line])[0]

    assert fixed.chars[0]["end"] == pytest.approx(1.025)
    assert fixed.chars[1]["start"] == pytest.approx(1.025)
    assert fixed.chars[0]["end"] <= fixed.chars[1]["start"]
    assert fixed.chars[0]["confidence"] == 0.8
    assert fixed.chars[1]["text"] == "b"
    assert fixed.chars[1]["space_after"] is False
    assert fixed.start <= min(char["start"] for char in fixed.chars)
    assert fixed.end >= max(char["end"] for char in fixed.chars)


def test_eighty_millisecond_gap_is_not_changed():
    line = make_line(chars=[
        {"char": "a", "start": 0.0, "end": 1.0},
        {"char": "b", "start": 1.08, "end": 2.0},
    ])
    before = deepcopy(line)

    fixed = fix_small_gaps([line])[0]

    assert fixed == before


def test_none_times_and_empty_chars_are_left_alone():
    lines = [
        make_line(chars=[]),
        make_line(chars=None),
        make_line(chars=[
            {"text": "a", "start": 0.0, "end": 1.0},
            {"text": "b", "start": None, "end": 1.04},
            {"text": "c", "start": 1.05, "end": 2.0},
        ]),
    ]
    before = deepcopy(lines)

    fixed = fix_small_gaps(lines)

    assert fixed == before
    assert fixed[0] is not lines[0]
    assert fixed[1] is not lines[1]


def test_overlapping_children_are_made_ordered_and_parent_is_widened():
    line = make_line(1.0, 1.1, chars=[
        {"text": "a", "start": 0.9, "end": 1.04},
        {"text": "b", "start": 1.02, "end": 1.2},
    ])

    fixed = fix_small_gaps([line])[0]

    assert fixed.chars[0]["start"] <= fixed.chars[0]["end"]
    assert fixed.chars[1]["start"] <= fixed.chars[1]["end"]
    assert fixed.chars[0]["end"] <= fixed.chars[1]["start"]
    assert fixed.start == pytest.approx(0.9)
    assert fixed.end == pytest.approx(1.2)


def test_fix_does_not_mutate_input_and_is_idempotent():
    line = make_line(chars=[
        {"char": "a", "start": 0.0, "end": 1.0, "tag": {"x": 1}},
        {"char": "b", "start": 1.04, "end": 2.0},
    ])
    original = deepcopy(line)

    once = fix_small_gaps([line])
    twice = fix_small_gaps(once)

    assert line == original
    assert line.chars[0]["end"] == 1.0
    assert once == twice
    assert once[0] is not line
    assert once[0].chars is not line.chars
    assert once[0].chars[0] is not line.chars[0]


def test_local_zero_duration_runs_use_parent_and_child_anchors():
    line = make_line(start=0.0, end=4.0, chars=[
        {"text": "a", "start": 0.0, "end": 0.0},
        {"text": "b", "start": 0.0, "end": 0.0},
        {"text": "c", "start": 2.0, "end": 3.0},
        {"text": "d", "start": 3.0, "end": 3.0},
    ])

    fixed = fix_small_gaps([line])[0]

    assert all(char["end"] > char["start"] for char in fixed.chars)
    assert fixed.chars[0]["start"] == pytest.approx(0.0)
    assert fixed.chars[-1]["end"] == pytest.approx(4.0)


def test_all_zero_duration_line_is_not_invented():
    line = make_line(start=0.0, end=4.0, chars=[
        {"text": "a", "start": 0.0, "end": 0.0},
        {"text": "b", "start": 0.0, "end": 0.0},
    ])

    fixed = fix_small_gaps([line])[0]

    assert fixed == line
