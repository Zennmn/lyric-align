"""End-to-end Mel-Band + Qwen lyric alignment pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable

from .anchor import align, interpolate_gaps
from .chunking import AlignmentChunk, build_chunks
from .gapfix import fix_small_gaps
from .hubertfa_backend import HubertFABackend
from .melband_backend import MelBandSeparator
from .model import AlignedLine
from .qwen_backend import (
    QwenASRBackend,
    QwenForcedAlignerBackend,
    load_audio,
)
from .runtime import RuntimePaths


Log = Callable[[str], None]


def _finite_bounds(line: AlignedLine) -> tuple[float, float] | None:
    try:
        start = float(line.start)
        end = float(line.end)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or start > end:
        return None
    return start, end


def _fill_coarse_gaps(
    aligned: list[AlignedLine], audio_duration: float
) -> list[AlignedLine]:
    """Give unmatched lines bounded search windows for ForcedAligner.

    Fuzzy matching can miss a sung line even when its neighbours are placed
    correctly.  Those lines should not be emitted with invented timestamps,
    but dropping them before forced alignment makes recovery impossible.  This
    helper copies the coarse results and fills only the planning copy with
    narrow, ordered windows between the nearest known lines.  The original
    ``aligned`` list remains honest; a planning line becomes ``matched=True``
    only if ForcedAligner later returns a valid result for it.
    """

    planned = [replace(line) for line in aligned]
    valid = [
        index for index, line in enumerate(aligned)
        if _finite_bounds(line) is not None
    ]
    if not valid or audio_duration <= 0:
        return planned

    gap_start = 0
    while gap_start < len(aligned):
        if _finite_bounds(aligned[gap_start]) is not None:
            gap_start += 1
            continue
        gap_end = gap_start
        while (
            gap_end + 1 < len(aligned)
            and _finite_bounds(aligned[gap_end + 1]) is None
        ):
            gap_end += 1

        left_index = next((i for i in reversed(valid) if i < gap_start), None)
        right_index = next((i for i in valid if i > gap_end), None)
        count = gap_end - gap_start + 1

        if left_index is not None and right_index is not None:
            left = _finite_bounds(aligned[left_index])
            right = _finite_bounds(aligned[right_index])
            assert left is not None and right is not None
            left_anchor = (left[0] + left[1]) / 2.0
            right_anchor = (right[0] + right[1]) / 2.0
            if right_anchor <= left_anchor:
                right_anchor = left_anchor + max(2.0, count * 2.0)
            step = (right_anchor - left_anchor) / (count + 1)
            width = min(2.0, max(0.5, step * 0.75))
            for offset, index in enumerate(range(gap_start, gap_end + 1), 1):
                center = left_anchor + step * offset
                start = max(0.0, center - width / 2.0)
                end = min(audio_duration, center + width / 2.0)
                if end <= start:
                    end = min(audio_duration, start + 0.1)
                planned[index] = replace(
                    planned[index], start=start, end=end
                )
        elif left_index is not None:
            left = _finite_bounds(aligned[left_index])
            assert left is not None
            cursor = max(0.0, left[1])
            for index in range(gap_start, gap_end + 1):
                start = min(audio_duration, cursor)
                end = min(audio_duration, start + 2.0)
                planned[index] = replace(
                    planned[index], start=start, end=end
                )
                cursor = end
        elif right_index is not None:
            right = _finite_bounds(aligned[right_index])
            assert right is not None
            cursor = min(audio_duration, max(0.0, right[0]))
            for index in range(gap_end, gap_start - 1, -1):
                end = cursor
                start = max(0.0, end - 2.0)
                planned[index] = replace(
                    planned[index], start=start, end=end
                )
                cursor = start

        gap_start = gap_end + 1

    return planned


def _usable_refined_line(line: AlignedLine) -> bool:
    """Reject a forced-align line with no trustworthy positive spans."""

    bounds = _finite_bounds(line)
    if bounds is None or bounds[1] - bounds[0] < 0.05:
        return False
    chars = line.chars
    if not chars:
        return False
    positive = 0
    for char in chars:
        try:
            start = float(char["start"])
            end = float(char["end"])
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end - start > 0.01:
            positive += 1
    return positive > 0 and positive / len(chars) >= 0.5


def _single_line_chunk(
    index: int,
    line: AlignedLine,
    padding: float,
    audio_duration: float,
) -> AlignmentChunk | None:
    bounds = _finite_bounds(line)
    if bounds is None or bounds[1] <= bounds[0]:
        return None
    start, end = bounds
    return AlignmentChunk(
        line_indices=(index,),
        lines=(line,),
        coarse_start=start,
        coarse_end=end,
        audio_start=max(0.0, start - padding),
        audio_end=min(audio_duration, end + padding),
        padding=padding,
    )


def _build_hubertfa_chunks(
    aligned: list[AlignedLine],
    *,
    padding: float,
    audio_duration: float,
    min_lines: int = 10,
    max_lines: int = 15,
    max_audio_seconds: float = 100.0,
) -> list[AlignmentChunk]:
    """Build bounded global HubertFA blocks.

    HubertFA benefits from enough lyric context to disambiguate repeated
    chorus lines, but its ONNX decoder can allocate a very large softmax
    buffer for a full song.  Group consecutive coarse lines into 10--15 line
    blocks and split earlier when the padded audio span would exceed the
    tested safe limit on a 16 GB GPU.
    """

    if (
        isinstance(min_lines, bool)
        or isinstance(max_lines, bool)
        or min_lines < 1
        or max_lines < min_lines
    ):
        raise ValueError("HubertFA line limits are invalid")
    try:
        padding = float(padding)
        max_audio_seconds = float(max_audio_seconds)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("HubertFA chunk bounds are invalid") from None
    if (
        not math.isfinite(padding)
        or padding < 0
        or not math.isfinite(max_audio_seconds)
        or max_audio_seconds <= 0
        or audio_duration <= 0
    ):
        raise ValueError("HubertFA chunk bounds are invalid")

    chunks: list[AlignmentChunk] = []
    run: list[tuple[int, AlignedLine, float, float]] = []

    def flush() -> None:
        if not run:
            return
        coarse_start = min(item[2] for item in run)
        coarse_end = max(item[3] for item in run)
        audio_start = max(0.0, coarse_start - padding)
        audio_end = min(audio_duration, coarse_end + padding)
        if audio_end > audio_start:
            chunks.append(
                AlignmentChunk(
                    line_indices=tuple(item[0] for item in run),
                    lines=tuple(item[1] for item in run),
                    coarse_start=coarse_start,
                    coarse_end=coarse_end,
                    audio_start=audio_start,
                    audio_end=audio_end,
                    padding=padding,
                )
            )
        run.clear()

    for index, line in enumerate(aligned):
        bounds = _finite_bounds(line)
        if bounds is None:
            flush()
            continue
        start, end = bounds
        if run:
            candidate_start = min(start, min(item[2] for item in run))
            candidate_end = max(end, max(item[3] for item in run))
            candidate_duration = candidate_end - candidate_start + 2.0 * padding
            if len(run) >= max_lines or candidate_duration > max_audio_seconds:
                flush()
        run.append((index, line, start, end))
    flush()
    return chunks


@dataclass(frozen=True)
class PipelineResult:
    """Artifacts and diagnostics produced by one complete run."""

    aligned: list[AlignedLine]
    vocals_path: Path
    segment_count: int
    chunk_count: int
    retry_count: int
    failures: tuple[str, ...]


def run_qwen_pipeline(
    audio_path: str | Path,
    lyrics: list[str],
    *,
    project_root: str | Path = ".",
    device: str = "cuda:0",
    dtype: str = "bf16",
    melband_model: str = "melband-roformer-kim-vocals",
    melband_overlap: int = 4,
    asr_model: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    aligner_backend: str = "qwen",
    coarse_aligner_model: str | Path | None = "Qwen/Qwen3-ForcedAligner-0.6B",
    hubertfa_model: str | Path | None = None,
    hubertfa_source: str | Path | None = None,
    language: str = "ja",
    asr_window_seconds: float = 20.0,
    asr_overlap_seconds: float = 1.0,
    pairing: int | str = "auto",
    threshold: float | None = None,
    window: int = 4,
    min_lines: int = 2,
    max_lines: int = 4,
    chunk_padding: float = 1.5,
    retry_padding: float = 4.0,
    max_gap: float = 0.08,
    max_inference_batch_size: int = 1,
    max_new_tokens: int = 256,
    local_files_only: bool = False,
    overwrite_vocals: bool = False,
    interpolate: bool = False,
    log: Log | None = None,
) -> PipelineResult:
    """Run the target flow with one model stage resident at a time."""

    if not lyrics:
        raise ValueError("lyrics must contain at least one non-empty line")
    aligner_backend = str(aligner_backend).strip().lower()
    if aligner_backend not in {"qwen", "hubertfa"}:
        raise ValueError("aligner_backend must be 'qwen' or 'hubertfa'")
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    logger = log or (lambda _message: None)
    paths = RuntimePaths.from_root(project_root).configure_environment()
    vocals_path = paths.audio / f"{audio_path.stem}_vocals.wav"
    # HubertFA is the complete alignment backend for this mode.  In
    # particular, do not quietly load Qwen3-ForcedAligner just to manufacture
    # coarse timestamps: that defeats the purpose of selecting HubertFA and
    # keeps an unnecessary 0.6B model resident on the GPU.  Qwen ASR still
    # supplies text windows for lyric matching; without a coarse FA its
    # Segment bounds are the ASR window bounds, which HubertFA refines later.
    coarse_model = coarse_aligner_model if aligner_backend == "qwen" else None
    if aligner_backend == "qwen" and coarse_model is None:
        coarse_model = aligner_model

    logger("stage 1/4: Mel-Band vocal separation")
    separator = MelBandSeparator(
        model=melband_model,
        models_dir=paths.melband_models,
        device=device,
        overlap=melband_overlap,
    )
    separator.separate(
        audio_path,
        vocals_path,
        overwrite=overwrite_vocals,
    )
    logger(f"vocal stem: {vocals_path}")

    # The separator releases CUDA memory before Qwen is loaded.
    audio = load_audio(vocals_path)
    if aligner_backend == "hubertfa":
        logger(
            "stage 2/4: Qwen3-ASR coarse transcription "
            "(Qwen3-ForcedAligner disabled)"
        )
    else:
        logger("stage 2/4: Qwen3-ASR coarse transcription + timestamps")
    asr = QwenASRBackend(
        model=asr_model,
        device=device,
        dtype=dtype,
        window_seconds=asr_window_seconds,
        overlap_seconds=asr_overlap_seconds,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner_model=coarse_model,
        local_files_only=local_files_only,
        model_dir=paths.qwen_models,
    )
    try:
        segments = asr.transcribe_segments(audio, language=language)
    finally:
        asr.close()
    if not segments:
        raise RuntimeError("Qwen3-ASR produced no usable coarse segments")
    logger(f"ASR segments: {len(segments)}")

    logger("stage 3/4: CJK fuzzy coarse matching and chunk planning")
    coarse = align(
        segments,
        lyrics,
        pairing=pairing,
        threshold=threshold,
        window=window,
        karaoke=False,
    )
    planning_coarse = _fill_coarse_gaps(coarse, audio.duration)
    if aligner_backend == "hubertfa":
        # A 10--15-line block gives HubertFA enough context to resolve the
        # repeated Japanese/English chorus.  _build_hubertfa_chunks also caps
        # the padded audio span at 100s, avoiding the full-song ONNX OOM seen
        # on the 259s sample.
        chunks = _build_hubertfa_chunks(
            planning_coarse,
            padding=chunk_padding,
            audio_duration=audio.duration,
            min_lines=max(10, min_lines),
            max_lines=max(15, max_lines),
        )
        logger(f"HubertFA global alignment blocks: {len(chunks)}")
    else:
        chunks = build_chunks(
            planning_coarse,
            min_lines=min_lines,
            max_lines=max_lines,
            padding=chunk_padding,
        )
        logger(f"alignment chunks: {len(chunks)}")

    logger(f"stage 4/4: {aligner_backend} character timestamps")
    refined = list(coarse)
    failures: list[str] = []
    retry_count = 0
    if aligner_backend == "qwen":
        aligner = QwenForcedAlignerBackend(
            model=aligner_model,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            model_dir=paths.qwen_models,
        )
    else:
        model_path = (
            Path(hubertfa_model).expanduser()
            if hubertfa_model is not None
            else paths.models / "hubertfa" / "1218_hfa_model_new_dict" / "model.onnx"
        )
        if not model_path.is_absolute():
            model_path = paths.root / model_path
        source_path = (
            Path(hubertfa_source).expanduser()
            if hubertfa_source is not None
            else paths.root / "third_party" / "HubertFA-v0.0.7"
        )
        if not source_path.is_absolute():
            source_path = paths.root / source_path
        aligner = HubertFABackend(
            model_path=model_path,
            source_dir=source_path,
            device=device,
        )
    try:
        if aligner_backend == "hubertfa":
            aligner.load()
            providers = getattr(aligner, "active_providers", ())
            logger(f"HubertFA ONNX providers: {', '.join(providers)}")
        for chunk in chunks:
            result = None
            chunk_error: Exception | None = None
            try:
                result = aligner.align_chunk(audio, chunk, language=language)
            except Exception as first_error:
                retry_count += 1
                logger(
                    f"chunk {chunk.line_indices}: retrying with "
                    f"padding ±{retry_padding:g}s ({first_error})"
                )
                try:
                    result = aligner.align_chunk(
                        audio,
                        chunk.with_padding(retry_padding),
                        language=language,
                    )
                except Exception as second_error:
                    chunk_error = second_error

            fallback_indices: list[int] = []
            if result is not None:
                if len(result) != len(chunk.line_indices):
                    chunk_error = ValueError(
                        "ForcedAligner returned an unexpected line count"
                    )
                    result = None
                else:
                    for index, line in zip(chunk.line_indices, result):
                        if _usable_refined_line(line):
                            refined[index] = line
                        else:
                            fallback_indices.append(index)
                            logger(
                                f"line {index}: weak chunk alignment; "
                                "trying a single-line window"
                            )
            if result is None:
                fallback_indices = list(chunk.line_indices)

            def try_single_line(index: int) -> tuple[AlignedLine | None, str]:
                nonlocal retry_count
                planning_line = planning_coarse[index]
                normal = _single_line_chunk(
                    index, planning_line, chunk_padding, audio.duration
                )
                if normal is None:
                    return None, "no finite coarse window"
                try:
                    candidate = aligner.align_chunk(
                        audio, normal, language=language
                    )
                    if len(candidate) != 1 or not _usable_refined_line(candidate[0]):
                        raise ValueError("single-line result has no reliable span")
                    return candidate[0], ""
                except Exception as first_error:
                    retry_count += 1
                    logger(
                        f"line {index}: retrying single-line window with "
                        f"padding ±{retry_padding:g}s ({first_error})"
                    )
                    try:
                        candidate = aligner.align_chunk(
                            audio,
                            normal.with_padding(retry_padding),
                            language=language,
                        )
                        if len(candidate) != 1 or not _usable_refined_line(candidate[0]):
                            raise ValueError("single-line retry has no reliable span")
                        return candidate[0], ""
                    except Exception as second_error:
                        return None, str(second_error)

            for index in fallback_indices:
                recovered, error = try_single_line(index)
                if recovered is not None:
                    refined[index] = recovered
                    continue
                coarse_line = coarse[index]
                refined[index] = AlignedLine(
                    line=coarse_line.line,
                    start=None,
                    end=None,
                    score=coarse_line.score,
                    matched=False,
                    chars=None,
                )
                detail = error or str(chunk_error or "forced alignment failed")
                failures.append(f"line {index} in chunk {chunk.line_indices}: {detail}")
                logger(f"line {index}: forced alignment failed; marking unmatched")
    finally:
        aligner.close()

    refined = fix_small_gaps(refined, max_gap=max_gap)
    if interpolate:
        refined = interpolate_gaps(refined)
    return PipelineResult(
        aligned=refined,
        vocals_path=vocals_path,
        segment_count=len(segments),
        chunk_count=len(chunks),
        retry_count=retry_count,
        failures=tuple(failures),
    )


__all__ = ["PipelineResult", "run_qwen_pipeline"]
