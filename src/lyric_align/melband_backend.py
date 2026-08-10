"""Mel-Band RoFormer vocal separation backend.

The dependency is optional and imported only when the Qwen pipeline is used.
The wrapper intentionally follows the public API of ``melband-roformer-infer``
0.1.x rather than invoking its CLI, so the model can be released before Qwen
loads and the whole pipeline keeps a bounded GPU footprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MelBandSeparator:
    """Separate a vocal stem with a CUDA-resident Mel-Band RoFormer."""

    def __init__(
        self,
        *,
        model: str = "melband-roformer-kim-vocals",
        models_dir: str | Path | None = None,
        device: str = "cuda:0",
        overlap: int = 4,
    ) -> None:
        self.model_name = model
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        self.device_name = device
        self.overlap = int(overlap)
        if self.overlap < 1:
            raise ValueError("overlap must be a positive integer")
        self._model: Any = None

    def separate(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Write and return a cached ``vocals.wav`` path."""

        audio_path = Path(audio_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"audio not found: {audio_path}")
        if output_path.exists() and not overwrite:
            return output_path

        torch, sf, torchaudio, ConfigDict, yaml = self._imports()
        if self.device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Mel-Band separation, but the active "
                "PyTorch build cannot see a CUDA device."
            )

        from mel_band_roformer import ensure_model_assets, get_model_from_config
        from mel_band_roformer.inference import SafeLoaderWithTuple
        from mel_band_roformer.utils import demix_track

        model_path, config_path = ensure_model_assets(
            self.model_name,
            models_dir=self.models_dir,
        )
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = ConfigDict(yaml.load(handle, Loader=SafeLoaderWithTuple))
        if not hasattr(config, "inference"):
            raise RuntimeError("Mel-Band config has no inference section")
        config.inference.num_overlap = self.overlap

        model = get_model_from_config("mel_band_roformer", config)
        if model is None:
            raise RuntimeError("Could not construct Mel-Band RoFormer from config")
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        device = torch.device(self.device_name)
        model.to(device).eval()
        self._model = model

        try:
            mix, sample_rate = sf.read(
                audio_path,
                dtype="float32",
                always_2d=True,
            )
            mix_tensor = torch.from_numpy(mix.T.copy())
            target_rate = int(getattr(config.model, "sample_rate", sample_rate))
            if sample_rate != target_rate:
                mix_tensor = torchaudio.functional.resample(
                    mix_tensor, sample_rate, target_rate
                )
                sample_rate = target_rate

            # The Kim checkpoint is stereo.  Keep the output shape stable even
            # when the input is a mono FLAC/WAV.
            if mix_tensor.shape[0] == 1:
                mix_tensor = mix_tensor.repeat(2, 1)
            elif mix_tensor.shape[0] > 2:
                mix_tensor = mix_tensor.mean(dim=0, keepdim=True).repeat(2, 1)

            result, _ = demix_track(
                config,
                model,
                mix_tensor,
                device,
                None,
            )
            instruments = list(config.training.instruments)
            target = getattr(config.training, "target_instrument", None)
            target = target or (instruments[0] if instruments else "vocals")
            if target not in result:
                raise RuntimeError(
                    f"Mel-Band output has no {target!r} stem; available: "
                    f"{sorted(result)}"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            vocals = result[target].T
            sf.write(output_path, vocals, sample_rate, subtype="FLOAT")
            return output_path
        finally:
            self.close()

    def close(self) -> None:
        """Release the model and its CUDA allocations."""

        self._model = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - cleanup must not mask failures
            pass

    @staticmethod
    def _imports():
        try:
            import torch
            import torchaudio
            import yaml
            import soundfile as sf
            from ml_collections import ConfigDict
        except ImportError as exc:  # pragma: no cover - exercised by optional envs
            raise ImportError(
                "Mel-Band separation requires torch, torchaudio, soundfile, "
                "pyyaml and ml-collections. Install the pipeline extra."
            ) from exc
        return torch, sf, torchaudio, ConfigDict, yaml


__all__ = ["MelBandSeparator"]
