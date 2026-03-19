"""Classical interpolation baselines: Bicubic and Lanczos."""

import numpy as np
from PIL import Image

from .base import SRModel


class BicubicModel(SRModel):
    name = "bicubic"

    def __init__(self, scale: int = 2):
        self._scale = scale

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, lr: np.ndarray) -> np.ndarray:
        return _pil_resize(lr, self._scale, Image.BICUBIC)


class LanczosModel(SRModel):
    name = "lanczos"

    def __init__(self, scale: int = 2):
        self._scale = scale

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, lr: np.ndarray) -> np.ndarray:
        return _pil_resize(lr, self._scale, Image.LANCZOS)


def _pil_resize(arr: np.ndarray, scale: int, resample) -> np.ndarray:
    """
    Resize a float32 array using PIL. Handles both (H, W) and (C, H, W).
    Input/output values in [0, 1].
    """
    if arr.ndim == 2:
        h, w = arr.shape
        img = Image.fromarray((arr * 65535).astype(np.uint16), mode="I;16")
        img = img.resize((w * scale, h * scale), resample=resample)
        return np.array(img).astype(np.float32) / 65535.0
    elif arr.ndim == 3:
        return np.stack([_pil_resize(arr[c], scale, resample) for c in range(arr.shape[0])])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")
