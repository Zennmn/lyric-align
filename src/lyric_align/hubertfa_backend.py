"""HubertFA singing-voice forced-alignment adapter.

HubertFA is intentionally kept behind a lazy optional backend.  Its ONNX model
does not accept written lyrics directly: Japanese and English text must first
be converted to the model's language-prefixed phoneme vocabulary.  The adapter
keeps that representation private and exposes the same ``align_chunk`` contract
as the Qwen backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

from .chunking import AlignmentChunk
from .model import AlignedLine
from .qwen_backend import AudioBuffer


_DLL_HANDLES: list[Any] = []
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
_SILENT_PHONEMES = {"", "SP", "AP", "EP", "CL", "cl", "GS", "pau", "Pau"}


@dataclass(frozen=True)
class _WordSpec:
    """One HubertFA dictionary word and its source lyric span."""

    line_index: int
    source_text: str
    language: str
    phonemes: tuple[str, ...]


def _read_dictionary(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"HubertFA dictionary not found: {path}")
    result: dict[str, tuple[str, ...]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or "\t" not in raw:
            continue
        word, phones = raw.split("\t", 1)
        result[word.strip()] = tuple(part for part in phones.split() if part)
    return result


def _romanize_keys(value: str, keys: set[str]) -> list[str]:
    """Split a pykakasi Hepburn string into dictionary mora keys."""

    normalized = unicodedata.normalize("NFKD", value).encode(
        "ascii", "ignore"
    ).decode("ascii").lower()
    normalized = normalized.replace("'", "")
    ordered = sorted(keys, key=len, reverse=True)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        match = next(
            (key for key in ordered if normalized.startswith(key, index)),
            None,
        )
        if match is not None:
            tokens.append(match)
            index += len(match)
            continue

        # Pykakasi writes a doubled consonant for Japanese sokuon (っ).
        # HubertFA's Japanese dictionary represents that closure as ``cl``.
        if (
            index + 1 < len(normalized)
            and normalized[index] == normalized[index + 1]
            and normalized[index] not in "aeiou"
            and "cl" in keys
        ):
            tokens.append("cl")
            index += 1
            continue
        raise ValueError(
            f"cannot tokenize Japanese reading {value!r} near "
            f"{normalized[index:]!r}"
        )
    return tokens


def _is_japanese_text(text: str) -> bool:
    return bool(_JAPANESE.search(text))


def _add_word(
    specs: list[_WordSpec],
    phonemes: list[str],
    phoneme_to_word: list[int],
    *,
    line_index: int,
    source_text: str,
    language: str,
    phones: tuple[str, ...],
) -> None:
    if not phones:
        raise ValueError(f"empty HubertFA pronunciation for {source_text!r}")
    if phonemes[-1] != "SP":
        phonemes.append("SP")
        phoneme_to_word.append(-1)
    word_index = len(specs)
    specs.append(_WordSpec(line_index, source_text, language, phones))
    for phone in phones:
        if phone in _SILENT_PHONEMES:
            if phonemes[-1] != "SP":
                phonemes.append("SP")
                phoneme_to_word.append(-1)
        else:
            phonemes.append(f"{language}/{phone}")
            phoneme_to_word.append(word_index)
    if phonemes[-1] != "SP":
        phonemes.append("SP")
        phoneme_to_word.append(-1)


def _line_g2p(
    text: str,
    line_index: int,
    japanese: dict[str, tuple[str, ...]],
    english: dict[str, tuple[str, ...]],
    kakasi: Any,
) -> tuple[list[_WordSpec], list[str], list[int]]:
    """Build a mixed Japanese/English HubertFA phoneme sequence."""

    specs: list[_WordSpec] = []
    phonemes = ["SP"]
    phoneme_to_word = [-1]
    for item in kakasi.convert(text):
        original = str(item.get("orig", ""))
        if not original.strip():
            continue
        reading = str(item.get("hepburn", original)).strip()
        if _is_japanese_text(original):
            source_text = "".join(
                char for char in original if not char.isspace()
            )
            if not source_text:
                continue
            keys = _romanize_keys(reading, set(japanese))
            phones: list[str] = []
            for key in keys:
                try:
                    phones.extend(japanese[key])
                except KeyError as exc:
                    raise ValueError(
                        f"Japanese dictionary has no entry {key!r} "
                        f"for {source_text!r}"
                    ) from exc
            _add_word(
                specs,
                phonemes,
                phoneme_to_word,
                line_index=line_index,
                source_text=source_text,
                language="ja",
                phones=tuple(phones),
            )
            continue

        for word in _LATIN_WORD.findall(original):
            key = word.lower()
            if key not in english:
                raise ValueError(
                    f"English HubertFA dictionary has no entry for {word!r}"
                )
            _add_word(
                specs,
                phonemes,
                phoneme_to_word,
                line_index=line_index,
                source_text=word,
                language="en",
                phones=english[key],
            )
    if len(specs) == 0:
        raise ValueError(f"HubertFA G2P produced no words for {text!r}")
    return specs, phonemes, phoneme_to_word


def _split_word_time(
    text: str, start: float, end: float
) -> list[dict[str, float | str | bool]]:
    """Split a Japanese dictionary word into character-level output units."""

    chars = [char for char in text if not char.isspace()]
    if not chars:
        return []
    duration = max(0.0001, end - start)
    result = []
    for index, char in enumerate(chars):
        left = start + duration * index / len(chars)
        right = start + duration * (index + 1) / len(chars)
        result.append(
            {"text": char, "start": left, "end": max(left + 0.0001, right), "unit": True}
        )
    return result


def _words_to_lines(
    words: list[Any],
    specs: list[_WordSpec],
    chunk: AlignmentChunk,
) -> list[AlignedLine]:
    if len(words) != len(specs):
        raise ValueError(
            "HubertFA returned an unexpected word count: "
            f"expected {len(specs)}, got {len(words)}"
        )

    lines: list[list[dict[str, float | str | bool]]] = [
        [] for _ in chunk.lines
    ]
    offset = chunk.audio_start
    for word, spec in zip(words, specs):
        start = float(getattr(word, "start")) + offset
        end = float(getattr(word, "end")) + offset
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("HubertFA returned an invalid word interval")
        if spec.language == "ja":
            units = _split_word_time(spec.source_text, start, end)
        else:
            units = [{"text": spec.source_text, "start": start, "end": end, "unit": True}]
        lines[spec.line_index].extend(units)

    output: list[AlignedLine] = []
    for line, chars in zip(chunk.lines, lines):
        if not chars:
            raise ValueError(f"HubertFA returned no words for line {line.line!r}")
        line_start = min(float(item["start"]) for item in chars)
        line_end = max(float(item["end"]) for item in chars)
        try:
            coarse_start = float(line.start)
            coarse_end = float(line.end)
        except (TypeError, ValueError, OverflowError):
            coarse_start = coarse_end = math.nan
        if math.isfinite(coarse_start) and math.isfinite(coarse_end):
            coarse_span = max(0.1, coarse_end - coarse_start)
            max_allowed = max(12.0, coarse_span * 4.0 + 4.0)
            if line_end - line_start > max_allowed:
                raise ValueError(
                    f"HubertFA returned an implausibly long span for {line.line!r}: "
                    f"{line_end - line_start:.3f}s vs coarse {coarse_span:.3f}s"
                )
        output.append(
            AlignedLine(
                line=line.line,
                start=line_start,
                end=line_end,
                score=line.score,
                matched=True,
                chars=chars,
            )
        )
    return output


def _add_torch_cuda_dlls() -> None:
    """Make PyTorch's bundled CUDA/cuDNN DLLs visible to ONNX Runtime."""

    if os.name != "nt":
        return
    try:
        import torch
    except ImportError:
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib.exists():
        return
    torch_lib_text = str(torch_lib)
    if torch_lib_text not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = torch_lib_text + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _DLL_HANDLES.append(add_dll_directory(torch_lib_text))


class HubertFABackend:
    """Run the HubertFA v0.0.7 ONNX singing aligner on one chunk."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        source_dir: str | Path,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.device = "cuda:0" if device == "cuda" else device
        self._inference: Any = None
        self._session: Any = None
        self._japanese: dict[str, tuple[str, ...]] | None = None
        self._english: dict[str, tuple[str, ...]] | None = None
        self._kakasi: Any = None

    def load(self) -> "HubertFABackend":
        if self._inference is not None:
            return self
        if not self.model_path.exists():
            raise FileNotFoundError(f"HubertFA model not found: {self.model_path}")
        if not self.source_dir.exists():
            raise FileNotFoundError(f"HubertFA source not found: {self.source_dir}")

        _add_torch_cuda_dlls()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "HubertFA requires onnxruntime-gpu; install the hubertfa extra"
            ) from exc
        if str(self.source_dir) not in sys.path:
            sys.path.insert(0, str(self.source_dir))
        inference_module = importlib.import_module("onnx_infer")
        InferenceOnnx = getattr(inference_module, "InferenceOnnx")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(
            str(self.model_path), options, providers=providers
        )
        active = session.get_providers()
        if self.device.startswith("cuda") and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                "HubertFA requested CUDA, but ONNX Runtime fell back to "
                f"{active}. Check the PyTorch CUDA DLL directory and cuDNN runtime."
            )

        inference = InferenceOnnx(self.model_path)
        inference.load_config()
        inference.init_decoder()
        inference.model = session
        model_dir = self.model_path.parent
        self._japanese = _read_dictionary(model_dir / "japanese_dict_full.txt")
        self._english = _read_dictionary(model_dir / "ds_cmudict-07b.txt")
        try:
            import pykakasi
        except ImportError as exc:
            raise ImportError(
                "HubertFA Japanese G2P requires pykakasi; install the hubertfa extra"
            ) from exc
        self._kakasi = pykakasi.kakasi()
        self._session = session
        self._inference = inference
        return self

    def align_chunk(
        self,
        audio: AudioBuffer,
        chunk: AlignmentChunk,
        *,
        language: str | None,
    ) -> list[AlignedLine]:
        del language  # HubertFA uses per-word ja/en prefixes from the mixed G2P.
        self.load()
        assert self._inference is not None
        assert self._japanese is not None and self._english is not None
        samples = audio.slice(chunk.audio_start, chunk.audio_end)
        if len(samples) == 0:
            raise ValueError("HubertFA chunk contains no audio")

        import numpy as np

        samples = np.asarray(samples, dtype=np.float32)
        target_rate = int(self._inference.mel_cfg["sample_rate"])
        if audio.sample_rate != target_rate:
            import librosa

            samples = librosa.resample(
                samples,
                orig_sr=audio.sample_rate,
                target_sr=target_rate,
            ).astype(np.float32, copy=False)
        duration = len(samples) / float(target_rate)

        specs: list[_WordSpec] = []
        phonemes = ["SP"]
        phoneme_to_word = [-1]
        for line_index, line in enumerate(chunk.lines):
            line_specs, line_phonemes, line_mapping = _line_g2p(
                line.line,
                line_index,
                self._japanese,
                self._english,
                self._kakasi,
            )
            base = len(specs)
            specs.extend(line_specs)
            for phone, word_index in zip(line_phonemes[1:], line_mapping[1:]):
                phonemes.append(phone)
                phoneme_to_word.append(
                    -1 if word_index < 0 else base + word_index
                )
        words, _ = self._inference._infer(
            samples,
            0,
            [spec.source_text for spec in specs],
            phonemes,
            phoneme_to_word,
            duration,
            [],
        )
        return _words_to_lines(list(words), specs, chunk)

    @property
    def active_providers(self) -> tuple[str, ...]:
        """Return the ONNX Runtime providers active for this backend."""

        if self._session is None:
            return ()
        return tuple(self._session.get_providers())

    def close(self) -> None:
        self._inference = None
        self._session = None
        self._japanese = None
        self._english = None
        self._kakasi = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - cleanup must not mask failures
            pass


__all__ = ["HubertFABackend"]
