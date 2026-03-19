"""
Quantitative SR evaluation metrics.

All functions accept float32 numpy arrays with values in [0, 1].
Arrays can be (H, W) for single-band or (C, H, W) for multi-band.
LPIPS requires a GPU-capable torch install; it gracefully falls back
to returning None if unavailable.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB). Higher is better.
    Computed in [0, 1] range (data_range=1.0).
    """
    return float(peak_signal_noise_ratio(hr, sr, data_range=1.0))


def ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    Structural Similarity Index. Higher is better, max 1.0.
    For multi-band arrays (C, H, W), averaged across channels.
    """
    if sr.ndim == 2:
        return float(structural_similarity(hr, sr, data_range=1.0))
    elif sr.ndim == 3:
        scores = [
            structural_similarity(hr[c], sr[c], data_range=1.0)
            for c in range(sr.shape[0])
        ]
        return float(np.mean(scores))
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {sr.shape}")


def lpips(sr: np.ndarray, hr: np.ndarray, net: str = "alex") -> float | None:
    """
    Learned Perceptual Image Patch Similarity. Lower is better.
    Requires the 'lpips' package and torch. Returns None if unavailable.

    Expects RGB input (3, H, W) in [0, 1]. If single-band, duplicates
    the channel to form a grayscale-as-RGB input.
    """
    try:
        import torch
        import lpips as lpips_lib

        loss_fn = lpips_lib.LPIPS(net=net, verbose=False)

        def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr])
            elif arr.shape[0] == 1:
                arr = np.concatenate([arr, arr, arr], axis=0)
            # LPIPS expects [-1, 1]
            t = torch.from_numpy(arr * 2.0 - 1.0).float().unsqueeze(0)
            return t

        with torch.no_grad():
            score = loss_fn(_to_tensor(sr), _to_tensor(hr))
        return float(score.item())

    except ImportError:
        return None


def evaluate_all(sr: np.ndarray, hr: np.ndarray) -> dict:
    """Run all metrics and return a dict of results."""
    results = {
        "psnr": psnr(sr, hr),
        "ssim": ssim(sr, hr),
        "lpips": lpips(sr, hr),
    }
    return results
