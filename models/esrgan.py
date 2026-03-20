"""
Real-ESRGAN x2plus wrapper.

A general-purpose GAN-based super-resolution model trained on natural images
with synthetic degradation. Unlike EVOLAND which was trained on Sentinel-2/VENµS
pairs, Real-ESRGAN has no satellite-specific knowledge — it applies learned
natural image priors to enhance any RGB input.

Input:  (3, H, W) float32 in [0, 1]  — RGB (B04, B03, B02)
Output: (3, 2H, 2W) float32 in [0, 1]

Weights: weights/RealESRGAN_x2plus.pth (~64MB)
"""

from pathlib import Path

import numpy as np
import torch

from .base import SRModel

WEIGHTS_PATH = Path("weights/RealESRGAN_x2plus.pth")


class RealESRGANModel(SRModel):
    name   = "esrgan"
    _scale = 2

    def __init__(self, weights_path: str | None = None):
        path = Path(weights_path or WEIGHTS_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN weights not found: {path}\n"
                "Download with:\n"
                "  wget -O weights/RealESRGAN_x2plus.pth "
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            )

        try:
            # basicsr references a torchvision internal module removed in 0.17+
            import sys, types, torchvision.transforms.functional as _F
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                _m = types.ModuleType("torchvision.transforms.functional_tensor")
                _m.rgb_to_grayscale = _F.rgb_to_grayscale
                sys.modules["torchvision.transforms.functional_tensor"] = _m

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise ImportError("pip install realesrgan") from exc

        arch = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23, num_grow_ch=32,
            scale=2,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._upsampler = RealESRGANer(
            scale=2,
            model_path=str(path),
            model=arch,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            half=torch.cuda.is_available(),  # fp16 on GPU for speed
            device=device,
        )

    @property
    def scale(self) -> int:
        return self._scale

    def upscale(self, lr: np.ndarray) -> np.ndarray:
        """
        Args:
            lr: float32 (3, H, W) in [0, 1]  — RGB
        Returns:
            sr: float32 (3, 2H, 2W) in [0, 1]
        """
        if lr.ndim == 3 and lr.shape[0] == 4:
            lr = lr[:3]  # drop NIR if passed 4-band input

        if lr.ndim != 3 or lr.shape[0] != 3:
            raise ValueError(f"RealESRGANModel expects (3, H, W), got {lr.shape}")

        # RealESRGANer expects uint8 HWC BGR
        rgb_hwc = (lr.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        bgr_hwc = rgb_hwc[:, :, ::-1]

        sr_bgr, _ = self._upsampler.enhance(bgr_hwc, outscale=2)

        sr_rgb = sr_bgr[:, :, ::-1]
        sr = sr_rgb.astype(np.float32) / 255.0
        return sr.transpose(2, 0, 1)  # HWC -> CHW
