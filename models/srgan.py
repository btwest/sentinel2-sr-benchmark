"""
OpenSR-SRGAN wrapper — trained on SEN2NAIP cross-sensor dataset.

4x super-resolution (10m → 2.5m equivalent).
Input:  (4, H, W) float32 [0, 1] — bands B02, B03, B04, B08
Output: (4, 4H, 4W) float32 [0, 1]
"""

from pathlib import Path
import numpy as np
import torch

from .base import SRModel

DEFAULT_CKPT = Path("logs/sentinel2-sr-benchmark/2026-03-20_18-35-20/epoch=32-step=113800.ckpt")


class SRGANModel(SRModel):
    name = "srgan"
    _scale = 4

    def __init__(self, ckpt_path: str | Path | None = None):
        try:
            from opensr_srgan.model import SRGAN_model
            from omegaconf import OmegaConf
        except ImportError as e:
            raise ImportError("opensr_srgan not installed: pip install opensr-srgan") from e

        path = Path(ckpt_path or DEFAULT_CKPT)
        if not path.exists():
            raise FileNotFoundError(f"SRGAN checkpoint not found: {path}")

        config = OmegaConf.load("training/config.yaml")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SRGAN_model.load_from_checkpoint(
            str(path), config=config
        ).to(self._device).eval()

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, lr: np.ndarray) -> np.ndarray:
        """
        Args:
            lr: float32 (4, H, W) in [0, 1] — B02, B03, B04, B08
        Returns:
            sr: float32 (4, 4H, 4W) in [0, 1]
        """
        if lr.ndim != 3 or lr.shape[0] != 4:
            raise ValueError(f"SRGANModel expects (4, H, W), got {lr.shape}")

        t = torch.from_numpy(lr).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._model.generator(t)
        return out.squeeze(0).cpu().numpy().astype(np.float32)
