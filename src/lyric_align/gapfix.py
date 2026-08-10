"""Repair tiny gaps in per-character timings.

Forced aligners often return a few milliseconds of silence between adjacent
characters even when the intended karaoke sweep is continuous.  The public
function in this module makes a new alignment list and new character records;
the caller's result is never modified in place.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from collections.abc import Mapping
from typing import Optional

from .model import AlignedLine


def fix_small_gaps(
    aligned: list[AlignedLine], max_gap: float = 0.08
) -> list[AlignedLine]:
    """Return a copy of ``aligned`` with tiny adjacent character gaps closed.

    For each adjacent pair whose finite intervals are usable, a gap strictly
    smaller than ``max_gap`` is split at its midpoint.  Thus the previous
    character ends exactly where the next starts, without making either
    interval backwards or overlapping.  Negative gaps (overlapping intervals)
    are handled by the same boundary rule so the resulting usable pair is
    ordered as well.

    Character dictionaries retain every original field, including either a
    ``char`` or ``text`` text key and backend-specific metadata.  Pairs with a
    ``None``/invalid time are left untouched.  A line's finite bounds are
    widened only when necessary to contain valid child times; missing bounds
    remain missing rather than being invented.

    The operation is idempotent: applying it again to its own result produces
    equivalent timings.
    """

    threshold = _validate_max_gap(max_gap)
    if aligned is None:
        return []

    result: list[AlignedLine] = []
    for line in aligned:
        chars = _copy_chars(getattr(line, "chars", None))
        if not isinstance(chars, list) or not chars:
            # ``replace`` still gives callers a new line object, while the
            # explicit early path ensures empty/absent chars are not changed.
            result.append(replace(line, chars=chars))
            continue

        _fill_zero_duration_runs(chars, line.start, line.end)
        _close_adjacent_gaps(chars, threshold)
        start, end = _contain_children(line.start, line.end, chars)
        result.append(replace(line, start=start, end=end, chars=chars))
    return result


def _validate_max_gap(max_gap: float) -> float:
    if isinstance(max_gap, bool):
        raise ValueError("max_gap must be a finite, non-negative number")
    try:
        value = float(max_gap)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("max_gap must be a finite, non-negative number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError("max_gap must be a finite, non-negative number")
    return value


def _copy_chars(chars: object) -> object:
    """Deep-copy char records while ensuring mappings can be edited safely."""

    if not isinstance(chars, (list, tuple)):
        # ``chars`` is documented as a list, but preserving a malformed value
        # is safer than making a gap fixer turn a result into an exception.
        return deepcopy(chars)

    copied = []
    for item in chars:
        if isinstance(item, Mapping):
            copied.append(dict(deepcopy(item)))
        else:
            copied.append(deepcopy(item))
    return copied


def _number(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _interval(item: object) -> Optional[tuple[float, float]]:
    if not isinstance(item, Mapping):
        return None
    start = _number(item.get("start"))
    end = _number(item.get("end"))
    if start is None or end is None or start > end:
        return None
    return start, end


def _close_adjacent_gaps(chars: list[object], max_gap: float) -> None:
    for previous, current in zip(chars, chars[1:]):
        previous_interval = _interval(previous)
        current_interval = _interval(current)
        if previous_interval is None or current_interval is None:
            continue

        previous_start, previous_end = previous_interval
        current_start, current_end = current_interval
        gap = current_start - previous_end
        if gap >= max_gap:
            continue

        # For the normal chronological case, the midpoint lies in the union
        # of the two intervals.  Clamping protects start<=end when an aligner
        # has returned overlapping intervals.
        if previous_start <= current_end:
            boundary = (previous_end + current_start) / 2.0
            boundary = max(previous_start, min(boundary, current_end))
            _set_time(previous, "end", boundary)
            _set_time(current, "start", boundary)
            continue

        # A reversed pair has no boundary that lies inside both original
        # intervals.  Collapse both to one legal point at the later feasible
        # edge.  This is an exceptional malformed-input path; ordinary small
        # gaps use the midpoint branch above.
        boundary = max(previous_start, current_end)
        _set_time(previous, "end", boundary)
        _set_time(current, "start", boundary)
        _set_time(current, "end", boundary)


def _fill_zero_duration_runs(
    chars: list[object], line_start: object, line_end: object
) -> None:
    """Interpolate local zero spans when a line has real timing anchors.

    Qwen's timestamp decoder can assign the same bin to a short run of
    characters while still producing valid positive spans before or after the
    run.  Those local gaps can be filled inside the parent line interval.  A
    line made entirely of zero spans is deliberately left untouched: there is
    no evidence from which to invent its timing.
    """

    positive = []
    for index, item in enumerate(chars):
        interval = _interval(item)
        if interval is not None and interval[1] - interval[0] > 0.001:
            positive.append((index, interval))
    if not positive:
        return

    parent_start = _number(line_start)
    parent_end = _number(line_end)
    cursor = 0
    while cursor < len(chars):
        interval = _interval(chars[cursor])
        if interval is None or interval[1] - interval[0] > 0.001:
            cursor += 1
            continue
        run_start = cursor
        while cursor < len(chars):
            current = _interval(chars[cursor])
            if current is None or current[1] - current[0] > 0.001:
                break
            cursor += 1
        run_end = cursor

        left = next(
            (item[1][1] for item in reversed(positive) if item[0] < run_start),
            parent_start,
        )
        right = next(
            (item[1][0] for item in positive if item[0] >= run_end),
            parent_end,
        )
        if left is None or right is None or right <= left + 0.001:
            continue

        step = (right - left) / (run_end - run_start)
        for offset, index in enumerate(range(run_start, run_end), 1):
            _set_time(chars[index], "start", left + step * (offset - 1))
            _set_time(chars[index], "end", left + step * offset)


def _set_time(item: object, key: str, value: float) -> None:
    if isinstance(item, dict):
        item[key] = value


def _contain_children(
    line_start: object, line_end: object, chars: list[object]
) -> tuple[object, object]:
    intervals = [_interval(item) for item in chars]
    intervals = [interval for interval in intervals if interval is not None]
    if not intervals:
        return line_start, line_end

    child_start = min(interval[0] for interval in intervals)
    child_end = max(interval[1] for interval in intervals)
    start = _number(line_start)
    end = _number(line_end)

    # Missing or invalid line bounds stay missing/invalid.  A known finite
    # bound is widened only when it would exclude a child.  This both keeps
    # normal lines byte-for-byte stable and ensures valid parent bounds contain
    # all usable child intervals.
    if start is not None and start > child_start:
        line_start = child_start
    if end is not None and end < child_end:
        line_end = child_end
    return line_start, line_end


__all__ = ["fix_small_gaps"]
