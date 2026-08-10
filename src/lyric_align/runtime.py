"""Runtime paths and optional accelerator helpers for the heavy pipeline."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """All mutable caches and generated assets owned by one project root."""

    root: Path
    models: Path
    melband_models: Path
    qwen_models: Path
    cache: Path
    huggingface: Path
    modelscope: Path
    torch: Path
    pip: Path
    uv: Path
    audio: Path
    outputs: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "RuntimePaths":
        root = Path(root).expanduser().resolve()
        models = root / "models"
        cache = root / ".cache"
        return cls(
            root=root,
            models=models,
            melband_models=models / "melband",
            qwen_models=models / "qwen",
            cache=cache,
            huggingface=cache / "huggingface",
            modelscope=cache / "modelscope",
            torch=cache / "torch",
            pip=cache / "pip",
            uv=cache / "uv",
            audio=root / "audio_cache",
            outputs=root / "outputs",
        )

    def ensure(self) -> "RuntimePaths":
        for path in (
            self.models,
            self.melband_models,
            self.qwen_models,
            self.cache,
            self.huggingface,
            self.modelscope,
            self.torch,
            self.pip,
            self.uv,
            self.audio,
            self.outputs,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def configure_environment(self) -> "RuntimePaths":
        """Point model/download caches at this root for the current process."""

        self.ensure()
        os.environ["HF_HOME"] = str(self.huggingface)
        os.environ["HF_HUB_CACHE"] = str(self.huggingface / "hub")
        os.environ["HF_DATASETS_CACHE"] = str(self.huggingface / "datasets")
        os.environ["MODELSCOPE_CACHE"] = str(self.modelscope)
        os.environ["TORCH_HOME"] = str(self.torch)
        os.environ["PIP_CACHE_DIR"] = str(self.pip)
        os.environ["UV_CACHE_DIR"] = str(self.uv)
        os.environ["MELBAND_ROFORMER_MODELS_PATH"] = str(self.melband_models)

        ffmpeg = find_ffmpeg()
        if ffmpeg:
            parent = str(Path(ffmpeg).parent)
            current_path = os.environ.get("PATH", "")
            if parent not in current_path.split(os.pathsep):
                os.environ["PATH"] = parent + os.pathsep + current_path
        return self


def find_ffmpeg() -> str | None:
    """Find a system ffmpeg or the D-drive imageio-ffmpeg binary."""

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # pragma: no cover - optional dependency/platform path
        return None
    return candidate if Path(candidate).exists() else None


__all__ = ["RuntimePaths", "find_ffmpeg"]
