"""Qwen3 ASR and ForcedAligner adapters for the GPU pipeline.

The official ``qwen-asr`` package currently imports ``nagisa`` at module import
time.  On Windows, DyNet can fail to load nagisa's bundled model.  CJK karaoke
alignment does not need Japanese word segmentation, so this module supplies a
small character tokenizer fallback in that specific case and keeps the rest of
the Qwen implementation unchanged.
"""

from __future__ import annotations

import gc
import math
import os
import re
import sys
import types
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .chunking import AlignmentChunk
from .model import AlignedLine, Segment, Word


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "en": "English",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


@dataclass(frozen=True)
class AudioBuffer:
    """Mono floating-point audio kept in CPU RAM between GPU stages."""

    samples: Any
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)

    def slice(self, start: float, end: float) -> Any:
        left = max(0, int(math.floor(start * self.sample_rate)))
        right = min(
            len(self.samples),
            int(math.ceil(max(end, start) * self.sample_rate)),
        )
        return self.samples[left:right]


def load_audio(path: str | Path) -> AudioBuffer:
    """Load a FLAC/WAV/other libsndfile-supported file as mono float32."""

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qwen audio processing requires numpy and soundfile."
        ) from exc

    samples, sample_rate = sf.read(
        Path(path), dtype="float32", always_2d=True
    )
    mono = samples.mean(axis=1, dtype=np.float32)
    return AudioBuffer(mono, int(sample_rate))


def normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    return LANGUAGE_NAMES.get(value.lower(), value)


def _dtype(name: str, device: str):
    import torch

    value = name.lower()
    if value in {"auto", "default"}:
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if value in {"bf16", "bfloat16"}:
        if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported by the active CUDA device")
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name!r}")


def _install_nagisa_fallback() -> bool:
    """Make qwen-asr importable when DyNet cannot load nagisa on Windows."""

    try:
        import nagisa  # noqa: F401

        return False
    except (ImportError, OSError, RuntimeError):
        # A failed import can leave a partially initialized module behind.
        sys.modules.pop("nagisa", None)

    module = types.ModuleType("nagisa")

    def tagging(text: str):
        return types.SimpleNamespace(words=_character_tokenize(text))

    module.tagging = tagging  # type: ignore[attr-defined]
    sys.modules["nagisa"] = module
    return True


def _character_tokenize(text: str) -> list[str]:
    """Tokenize mixed Japanese/Latin lyrics for the Qwen aligner.

    Japanese and CJK characters are alignment units on their own, while a
    contiguous ASCII word is kept together.  Passing ``I need you`` as eleven
    separate letters makes the Japanese tokenizer assign a single timestamp
    bin to most of the English phrase; keeping the words together lets the
    acoustic model score the English sequence as words before the adapter
    expands those word spans back to characters for TTML.
    """

    units: list[str] = []
    latin: list[str] = []

    def flush_latin() -> None:
        if latin:
            units.append("".join(latin))
            latin.clear()

    for char in str(text):
        if char.isspace():
            flush_latin()
            continue
        if char.isascii() and (char.isalnum() or char == "'"):
            latin.append(char)
            continue
        flush_latin()
        if char == "'" or unicodedata.category(char).startswith(("L", "N")):
            units.append(char)
    flush_latin()
    return units


def _qwen_classes():
    _install_nagisa_fallback()
    try:
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qwen pipeline requires qwen-asr. Install the pipeline extra."
        ) from exc
    return Qwen3ASRModel, Qwen3ForcedAligner


def _looks_like_model_directory(path: Path) -> bool:
    """Return whether ``path`` looks like a complete local HF model."""

    if not path.is_dir():
        return False
    markers = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return any((path / marker).exists() for marker in markers)


def _resolve_model_path(
    model: str | Path,
    model_dir: Path | None,
    local_files_only: bool,
) -> str:
    """Resolve a model id to a local path while keeping downloads on D:."""

    raw = str(model)
    requested = Path(os.path.expandvars(raw)).expanduser()
    path_like = (
        requested.is_absolute()
        or ":" in raw
        or "\\" in raw
        or raw.startswith((".", "~"))
    )

    if requested.is_dir():
        return str(requested.resolve())
    if path_like:
        raise FileNotFoundError(f"local model directory not found: {requested}")

    repo_id = raw.strip().rstrip("/")
    if not repo_id:
        raise ValueError("model id/path must not be empty")
    repo_name = repo_id.rsplit("/", 1)[-1]

    # Prefer an explicitly provisioned local model under models\\qwen.  The
    # aliases cover the official repo names with and without the ``-hf``
    # suffix, so a user can keep either naming convention on disk.
    if model_dir is not None:
        root = Path(model_dir).expanduser()
        aliases = [repo_name]
        if repo_name.endswith("-hf"):
            aliases.append(repo_name[:-3])
        else:
            aliases.append(f"{repo_name}-hf")
        for alias in dict.fromkeys(aliases):
            candidate = root / alias
            if _looks_like_model_directory(candidate):
                return str(candidate.resolve())

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Qwen model resolution requires huggingface-hub."
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_files_only": bool(local_files_only),
    }
    if model_dir is not None and not local_files_only:
        destination = Path(model_dir).expanduser() / repo_name
        destination.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(destination)

    try:
        resolved = snapshot_download(**kwargs)
    except Exception as exc:
        if local_files_only:
            raise FileNotFoundError(
                f"model {repo_id!r} is not available locally; "
                "disable --local-files-only to download it"
            ) from exc
        raise RuntimeError(f"could not download model {repo_id!r}: {exc}") from exc
    return str(Path(resolved).expanduser().resolve())


class QwenASRBackend:
    """Run Qwen3-ASR on bounded windows with optional forced timestamps."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-ASR-1.7B",
        device: str = "cuda:0",
        dtype: str = "bf16",
        window_seconds: float = 20.0,
        overlap_seconds: float = 1.0,
        max_inference_batch_size: int = 1,
        max_new_tokens: int = 256,
        forced_aligner_model: str | Path | None = None,
        local_files_only: bool = False,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model
        self.device = "cuda:0" if device == "cuda" else device
        self.dtype_name = dtype
        self.window_seconds = float(window_seconds)
        self.overlap_seconds = float(overlap_seconds)
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.forced_aligner_model = forced_aligner_model
        self.local_files_only = bool(local_files_only)
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None

    def load(self) -> "QwenASRBackend":
        if self._model is not None:
            return self
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Qwen ASR requested CUDA, but CUDA is unavailable")
        Qwen3ASRModel, _ = _qwen_classes()
        model_path = _resolve_model_path(
            self.model_name, self.model_dir, self.local_files_only
        )
        aligner_path = None
        if self.forced_aligner_model is not None:
            aligner_path = _resolve_model_path(
                self.forced_aligner_model,
                self.model_dir,
                self.local_files_only,
            )
        kwargs = {
            "dtype": _dtype(self.dtype_name, self.device),
            "device_map": self.device,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
            "local_files_only": self.local_files_only,
        }
        if aligner_path is not None:
            kwargs["forced_aligner"] = aligner_path
            kwargs["forced_aligner_kwargs"] = {
                "dtype": _dtype(self.dtype_name, self.device),
                "device_map": self.device,
                "local_files_only": self.local_files_only,
            }
        self._model = Qwen3ASRModel.from_pretrained(model_path, **kwargs)
        # The package's Japanese tokenizer delegates to nagisa, which is both
        # fragile on native Windows and groups Japanese text more coarsely than
        # the lyric pipeline needs.  Keep Japanese as characters and Latin as
        # words for the optional coarse timestamp pass too.
        if aligner_path is not None:
            forced_aligner = getattr(self._model, "forced_aligner", None)
            processor = getattr(forced_aligner, "aligner_processor", None)
            if processor is not None:
                processor.tokenize_japanese = _character_tokenize
        return self

    def transcribe_segments(
        self,
        audio: AudioBuffer | str | Path,
        *,
        language: str | None = None,
    ) -> list[Segment]:
        self.load()
        if not isinstance(audio, AudioBuffer):
            audio = load_audio(audio)
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.overlap_seconds < 0 or self.overlap_seconds >= self.window_seconds:
            raise ValueError("overlap_seconds must be >= 0 and less than window_seconds")

        step = self.window_seconds - self.overlap_seconds
        language_name = normalize_language(language)
        segments: list[Segment] = []
        start = 0.0
        while start < audio.duration or not segments:
            end = min(audio.duration, start + self.window_seconds)
            if end <= start:
                break
            result = self._model.transcribe(
                audio=(audio.slice(start, end), audio.sample_rate),
                language=language_name,
                return_time_stamps=self.forced_aligner_model is not None,
            )[0]
            text = str(getattr(result, "text", "") or "").strip()
            if text:
                words: list[Word] = []
                timestamps = getattr(result, "time_stamps", None)
                items = getattr(timestamps, "items", timestamps or [])
                for item in items:
                    item_start = _finite(getattr(item, "start_time", None))
                    item_end = _finite(getattr(item, "end_time", None))
                    item_text = str(getattr(item, "text", "") or "").strip()
                    if (
                        item_start is None
                        or item_end is None
                        or item_end < item_start
                        or not item_text
                    ):
                        continue
                    words.append(
                        Word(
                            start=start + item_start,
                            end=start + item_end,
                            word=item_text,
                        )
                    )
                if words:
                    segment_start = min(word.start for word in words)
                    segment_end = max(word.end for word in words)
                else:
                    segment_start, segment_end = start, end
                candidate = Segment(segment_start, segment_end, text, words)
                if not _is_duplicate_window(segments, candidate):
                    segments.append(candidate)
            if end >= audio.duration:
                break
            start += step
        return segments

    def close(self) -> None:
        self._model = None
        _release_cuda()


class QwenForcedAlignerBackend:
    """Run Qwen3-ForcedAligner on one bounded AlignmentChunk."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "cuda:0",
        dtype: str = "bf16",
        local_files_only: bool = False,
        model_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model
        self.device = "cuda:0" if device == "cuda" else device
        self.dtype_name = dtype
        self.local_files_only = bool(local_files_only)
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: Any = None

    def load(self) -> "QwenForcedAlignerBackend":
        if self._model is not None:
            return self
        import torch

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "ForcedAligner requested CUDA, but CUDA is unavailable"
            )
        _, Qwen3ForcedAligner = _qwen_classes()
        model_path = _resolve_model_path(
            self.model_name, self.model_dir, self.local_files_only
        )
        kwargs = {
            "dtype": _dtype(self.dtype_name, self.device),
            "device_map": self.device,
            "local_files_only": self.local_files_only,
        }
        self._model = Qwen3ForcedAligner.from_pretrained(model_path, **kwargs)
        processor = getattr(self._model, "aligner_processor", None)
        if processor is not None:
            processor.tokenize_japanese = _character_tokenize
        return self

    def align_chunk(
        self,
        audio: AudioBuffer,
        chunk: AlignmentChunk,
        *,
        language: str | None,
    ) -> list[AlignedLine]:
        self.load()
        language_name = normalize_language(language) or "Japanese"
        audio_slice = audio.slice(chunk.audio_start, chunk.audio_end)
        if len(audio_slice) == 0:
            raise ValueError("forced-align chunk contains no audio")
        text = "\n".join(line.line for line in chunk.lines)
        result = self._model.align(
            audio=(audio_slice, audio.sample_rate),
            text=text,
            language=language_name,
        )[0]
        items = list(result)
        chunk_duration = len(audio_slice) / float(audio.sample_rate)
        return _items_to_lines(
            items,
            chunk,
            language_name,
            chunk_duration,
        )

    def close(self) -> None:
        self._model = None
        _release_cuda()


def _is_duplicate_window(
    previous: list[Segment], candidate: Segment
) -> bool:
    if not previous or candidate.start >= previous[-1].end:
        return False
    left = _normalize_text(previous[-1].text)
    right = _normalize_text(candidate.text)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _is_kept(char: str) -> bool:
    return char == "'" or unicodedata.category(char).startswith(("L", "N"))


def _units_for_line(text: str, language: str) -> list[str]:
    if language == "Japanese":
        return _character_tokenize(text)

    if language == "Chinese":
        units: list[str] = []
        word: list[str] = []
        for char in text:
            if 0x3400 <= ord(char) <= 0x9FFF or 0xF900 <= ord(char) <= 0xFAFF:
                if word:
                    units.append("".join(word))
                    word.clear()
                units.append(char)
            elif _is_kept(char):
                word.append(char)
            elif word:
                units.append("".join(word))
                word.clear()
        if word:
            units.append("".join(word))
        return units

    return [part for part in re.split(r"\s+", text.strip()) if part]


def _unit_timing(
    unit: str, start: float, end: float
) -> dict[str, float | str | bool]:
    """Keep one model unit as one output timing unit.

    Japanese units are one character; ASCII units are one word.  The explicit
    marker lets TTML/eLRC preserve that distinction instead of assuming every
    record corresponds to one source character.
    """

    return {
        "text": unit,
        "start": start,
        "end": end,
        "unit": True,
    }


def _items_to_lines(
    items: Iterable[Any],
    chunk: AlignmentChunk,
    language: str,
    audio_duration: float,
) -> list[AlignedLine]:
    normalized = []
    previous_start: float | None = None
    for item in items:
        start = _finite(getattr(item, "start_time", None))
        end = _finite(getattr(item, "end_time", None))
        text = str(getattr(item, "text", "") or "")
        if start is None or end is None or start < 0 or end < start:
            raise ValueError("ForcedAligner returned invalid timestamps")
        if end > audio_duration + 0.05:
            raise ValueError("ForcedAligner timestamp exceeds chunk audio")
        if previous_start is not None and start + 0.05 < previous_start:
            raise ValueError("ForcedAligner timestamps are not chronological")
        previous_start = start
        normalized.append((text, start, end))
    if not normalized:
        raise ValueError("ForcedAligner returned no timestamp items")

    observed_start = min(item[1] for item in normalized)
    observed_end = max(item[2] for item in normalized)
    coarse_start = max(0.0, chunk.coarse_start - chunk.audio_start)
    coarse_end = min(
        audio_duration,
        max(coarse_start, chunk.coarse_end - chunk.audio_start),
    )
    if observed_end + 0.05 < coarse_start or observed_start - 0.05 > coarse_end:
        raise ValueError(
            "ForcedAligner result lies completely outside the coarse time window"
        )

    expected_counts = [
        len(_units_for_line(line.line, language)) for line in chunk.lines
    ]
    if sum(expected_counts) != len(normalized):
        raise ValueError(
            "ForcedAligner item count does not match lyric units: "
            f"expected {sum(expected_counts)}, got {len(normalized)}"
        )

    output: list[AlignedLine] = []
    cursor = 0
    offset = chunk.audio_start
    for line, count in zip(chunk.lines, expected_counts):
        line_items = normalized[cursor:cursor + count]
        cursor += count
        chars: list[dict[str, float | str | bool]] = []
        for unit, (_observed, start, end) in zip(
            _units_for_line(line.line, language), line_items
        ):
            chars.append(
                _unit_timing(unit, offset + start, offset + end)
            )
        output.append(
            AlignedLine(
                line=line.line,
                start=min(item["start"] for item in chars),
                end=max(item["end"] for item in chars),
                score=line.score,
                matched=True,
                chars=chars,
            )
        )
    return output


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - cleanup must not mask failures
        pass


__all__ = [
    "AudioBuffer",
    "QwenASRBackend",
    "QwenForcedAlignerBackend",
    "load_audio",
    "normalize_language",
]
