"""Small, dependency-free chunks for forced alignment.

The coarse lyric matcher works line by line, while a forced aligner is more
useful when it receives a short piece of adjacent lyrics and the corresponding
piece of audio.  This module provides the small data contract between those
two stages without knowing anything about a particular forced-aligner backend.

Only lines with finite, ordered ``start``/``end`` values participate in a
chunk.  A line without usable bounds also breaks continuity: silently joining
the lines on either side would make a forced-aligner result look more certain
than the coarse alignment actually is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Optional

from .model import AlignedLine


DEFAULT_PADDING = 1.5


@dataclass(frozen=True)
class AlignmentChunk:
    """An immutable group of adjacent coarse-alignment lines.

    ``line_indices`` are indices in the original aligned list and ``lines``
    are the corresponding line objects in source order.  The coarse bounds
    cover the lines themselves; the audio bounds are the coarse bounds widened
    by ``padding`` and clamped at zero.

    The dataclass is frozen and its collection fields are tuples, so callers
    cannot accidentally replace chunk membership while a backend is using the
    chunk.  The ``AlignedLine`` objects inside ``lines`` remain the caller's
    objects, as expected for a read-only planning value.
    """

    line_indices: tuple[int, ...]
    lines: tuple[AlignedLine, ...]
    coarse_start: float
    coarse_end: float
    audio_start: float
    audio_end: float
    padding: float

    def with_padding(self, padding: float) -> "AlignmentChunk":
        """Return this chunk with a different audio-side padding.

        This is useful for a bounded retry: a normal chunk can use the default
        ``1.5`` seconds, then an exceptional forced-alignment result can be
        retried with ``chunk.with_padding(4.0)``.  Membership and coarse bounds
        are unchanged.
        """

        pad = _validate_padding(padding)
        audio_start, audio_end = _audio_bounds(
            self.coarse_start, self.coarse_end, pad
        )
        return replace(
            self,
            audio_start=audio_start,
            audio_end=audio_end,
            padding=pad,
        )


def build_chunks(
    aligned: Iterable[AlignedLine] | None,
    min_lines: int = 2,
    max_lines: int = 4,
    padding: float = DEFAULT_PADDING,
) -> list[AlignmentChunk]:
    """Build forced-alignment chunks from coarse line timings.

    Lines with missing, non-finite, reversed, or otherwise non-numeric bounds
    are ignored and break a contiguous run.  Valid runs are partitioned as
    evenly as possible into groups no larger than ``max_lines`` and no smaller
    than ``min_lines`` whenever that is possible.  A run shorter than
    ``min_lines`` is retained as one chunk: a short final run should remain
    visible rather than being dropped.

    ``min_lines`` and ``max_lines`` must be positive integers with
    ``min_lines <= max_lines``.  Padding must be a finite, non-negative number.
    Invalid alignment entries (and an empty or ``None`` input) simply produce
    an empty list; invalid configuration values raise ``ValueError`` so a
    caller cannot accidentally issue a malformed retry.
    """

    _validate_line_limits(min_lines, max_lines)
    pad = _validate_padding(padding)

    if aligned is None:
        return []
    try:
        iterator = iter(aligned)
    except TypeError:
        return []

    chunks: list[AlignmentChunk] = []
    run: list[tuple[int, AlignedLine, float, float]] = []

    def flush() -> None:
        if not run:
            return
        for group in _partition_run(run, min_lines, max_lines):
            chunks.append(_make_chunk(group, pad))
        run.clear()

    for index, line in enumerate(iterator):
        bounds = _line_bounds(line)
        if bounds is None:
            flush()
            continue
        start, end = bounds
        run.append((index, line, start, end))
    flush()
    return chunks


def _validate_line_limits(min_lines: int, max_lines: int) -> None:
    # bool is an int subclass, but accepting True as a line count is almost
    # certainly a caller mistake and makes a typo particularly hard to spot.
    if (isinstance(min_lines, bool) or isinstance(max_lines, bool)
            or not isinstance(min_lines, int)
            or not isinstance(max_lines, int)
            or min_lines < 1
            or max_lines < 1
            or min_lines > max_lines):
        raise ValueError("min_lines and max_lines must be positive integers with min_lines <= max_lines")


def _validate_padding(padding: float) -> float:
    if isinstance(padding, bool):
        raise ValueError("padding must be a finite, non-negative number")
    try:
        value = float(padding)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("padding must be a finite, non-negative number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError("padding must be a finite, non-negative number")
    return value


def _time(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _line_bounds(line: object) -> Optional[tuple[float, float]]:
    if line is None:
        return None
    start = _time(getattr(line, "start", None))
    end = _time(getattr(line, "end", None))
    if start is None or end is None or start > end:
        return None
    return start, end


def _partition_run(
    run: list[tuple[int, AlignedLine, float, float]],
    min_lines: int,
    max_lines: int,
) -> list[list[tuple[int, AlignedLine, float, float]]]:
    """Partition one valid run while keeping the tail from being undersized."""

    count = len(run)
    if count <= max_lines or count < min_lines:
        return [run]

    # The smallest number of max-sized groups leaves the most room for a
    # healthy final group.  With the normal 2..4 limits this gives 2+3 for
    # five lines, 3+2 for five, 4+3 for seven, and 3+3+3 for nine.
    group_count = math.ceil(count / max_lines)
    while group_count > 1 and count < group_count * min_lines:
        group_count -= 1

    base, remainder = divmod(count, group_count)
    sizes = [base + (1 if i < remainder else 0) for i in range(group_count)]

    groups: list[list[tuple[int, AlignedLine, float, float]]] = []
    offset = 0
    for size in sizes:
        groups.append(run[offset:offset + size])
        offset += size
    return groups


def _audio_bounds(
    coarse_start: float, coarse_end: float, padding: float
) -> tuple[float, float]:
    return (
        max(0.0, coarse_start - padding),
        max(0.0, coarse_end + padding),
    )


def _make_chunk(
    group: list[tuple[int, AlignedLine, float, float]], padding: float
) -> AlignmentChunk:
    indices = tuple(item[0] for item in group)
    lines = tuple(item[1] for item in group)
    coarse_start = min(item[2] for item in group)
    coarse_end = max(item[3] for item in group)
    audio_start, audio_end = _audio_bounds(coarse_start, coarse_end, padding)
    return AlignmentChunk(
        line_indices=indices,
        lines=lines,
        coarse_start=coarse_start,
        coarse_end=coarse_end,
        audio_start=audio_start,
        audio_end=audio_end,
        padding=padding,
    )


__all__ = ["AlignmentChunk", "build_chunks"]
