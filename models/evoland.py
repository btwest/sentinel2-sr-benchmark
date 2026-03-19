"""
EVOLAND super-resolution wrapper — direct ONNX inference.

Model: s2v2x2_spatrad  (2× scale, 10m → 5m)
Input:  (1, 4, H, W) float32, bands B02/B03/B04/B08, raw reflectance [0, 10000]
Output: (1, 4, 2H, 2W) float32, same units

Tile padding
------------
The model requires a 66-output-pixel (= 33-input-pixel) margin on each side
of every tile to suppress border artifacts.  The caller is expected to pad
tiles before calling upscale() and crop the margin from the result.

This wrapper handles that automatically: feed it an un-padded (4, H, W) tile
and it returns (4, 2H, 2W), having added and stripped the margin internally
(via reflect padding so it works even on edge tiles).
"""

import sys
from pathlib import Path

import numpy as np

from .base import SRModel

# Location of the ONNX model bundled with the package
_PKG_MODELS = (
    Path(sys.prefix) / "lib" /
    f"python{sys.version_info.major}.{sys.version_info.minor}" /
    "site-packages" / "sentinel2_superresolution" / "models"
)

_ONNX_PATH  = _PKG_MODELS / "s2v2x2_spatrad.onnx"
_MARGIN_OUT = 66    # output pixels to strip (66 px at 5m = 330m)
_MARGIN_IN  = 33    # = _MARGIN_OUT / 2  (input pixels to pad)


class EvolandModel(SRModel):
    name   = "evoland"
    _scale = 2

    def __init__(self, onnx_path: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime not installed: pip install onnxruntime") from exc

        path = str(onnx_path or _ONNX_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(f"EVOLAND ONNX model not found: {path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 4
        so.use_deterministic_compute = True

        self._session = ort.InferenceSession(
            path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._ro = ort.RunOptions()
        self._ro.add_run_config_entry("log_severity_level", "3")

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, lr: np.ndarray) -> np.ndarray:
        """
        Args:
            lr: float32 (4, H, W) in [0, 1]  — bands B02, B03, B04, B08
        Returns:
            sr: float32 (4, 2H, 2W) in [0, 1]
        """
        if lr.ndim != 3 or lr.shape[0] != 4:
            raise ValueError(f"EvolandModel expects (4, H, W), got {lr.shape}")

        # Scale [0,1] → raw reflectance [0,10000]
        refl = (lr * 10000.0).astype(np.float32)   # (4, H, W)

        # Reflect-pad the input tile by MARGIN_IN pixels on each side
        padded = np.pad(
            refl,
            ((0, 0), (_MARGIN_IN, _MARGIN_IN), (_MARGIN_IN, _MARGIN_IN)),
            mode="reflect",
        )                                            # (4, H+2*33, W+2*33)

        inp = padded[np.newaxis]                     # (1, 4, H+66, W+66)
        out = self._session.run(
            None, {"input": inp}, run_options=self._ro
        )[0]                                         # (1, 4, 2*(H+66), 2*(W+66))

        sr_padded = out[0]                           # (4, 2H+132, 2W+132)

        # Crop the margin back out
        sr = sr_padded[
            :,
            _MARGIN_OUT : sr_padded.shape[1] - _MARGIN_OUT,
            _MARGIN_OUT : sr_padded.shape[2] - _MARGIN_OUT,
        ]                                            # (4, 2H, 2W)

        return np.clip(sr / 10000.0, 0.0, 1.0).astype(np.float32)
