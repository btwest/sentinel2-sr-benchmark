"""
Benchmark runner — evaluates all registered SR models on a data split.

Usage:
    python eval/benchmark.py --split test --max-patches 500
    python eval/benchmark.py --split test --models bicubic,lanczos

Outputs:
    eval/results/results.json   -- per-model aggregate metrics
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.metrics import evaluate_all
from models.classical import BicubicModel, LanczosModel

RESULTS_DIR = Path("eval/results")
MANIFEST_PATH = Path("data/patches/manifest.json")


def get_models(names: list[str] | None):
    all_models = [
        BicubicModel(scale=2),
        LanczosModel(scale=2),
    ]

    try:
        from models.evoland import EvolandModel
        all_models.append(EvolandModel())
    except (ImportError, FileNotFoundError):
        pass

    try:
        from models.esrgan import RealESRGANModel
        all_models.append(RealESRGANModel())
    except (ImportError, FileNotFoundError):
        pass

    if names:
        return [m for m in all_models if m.name in names]
    return all_models


def iter_patches(manifest_entry: list[dict], max_patches: int | None):
    """Yield (lr, hr) float32 numpy pairs from .npz scene files."""
    count = 0
    for scene in manifest_entry:
        data = np.load(scene["npz"])
        lr_all = data["lr"]  # [N, C, H, W]
        hr_all = data["hr"]  # [N, C, H, W]
        for i in range(lr_all.shape[0]):
            yield lr_all[i], hr_all[i]
            count += 1
            if max_patches and count >= max_patches:
                return


def run_benchmark(split: str, model_names: list[str] | None, max_patches: int | None) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text())
    scenes = manifest.get(split, [])
    if not scenes:
        print(f"No scenes in split '{split}'.")
        return {}

    total = min(sum(s["n_patches"] for s in scenes), max_patches or 999999)
    models = get_models(model_names)
    print(f"\nModels: {[m.name for m in models]}")
    print(f"Split: {split} | Patches: {total}\n")

    results = {}
    for model in models:
        scores = {"psnr": [], "ssim": [], "lpips": []}
        t0 = time.time()

        for lr, hr in tqdm(iter_patches(scenes, max_patches), total=total, desc=model.name):
            sr = model.upscale(lr)
            h, w = hr.shape[-2], hr.shape[-1]
            sr = sr[..., :h, :w]
            m = evaluate_all(sr, hr)
            scores["psnr"].append(m["psnr"])
            scores["ssim"].append(m["ssim"])
            if m["lpips"] is not None:
                scores["lpips"].append(m["lpips"])

        elapsed = time.time() - t0
        results[model.name] = {
            "psnr_mean": round(float(np.mean(scores["psnr"])), 4),
            "psnr_std":  round(float(np.std(scores["psnr"])), 4),
            "ssim_mean": round(float(np.mean(scores["ssim"])), 4),
            "ssim_std":  round(float(np.std(scores["ssim"])), 4),
            "lpips_mean": round(float(np.mean(scores["lpips"])), 4) if scores["lpips"] else None,
            "n_patches": len(scores["psnr"]),
            "elapsed_s": round(elapsed, 1),
        }
        r = results[model.name]
        lpips_str = f"{r['lpips_mean']:.4f}" if r["lpips_mean"] is not None else "N/A"
        print(f"  {model.name:<12} PSNR={r['psnr_mean']:.2f}dB  SSIM={r['ssim_mean']:.4f}  LPIPS={lpips_str}  ({elapsed:.1f}s)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--models", help="Comma-separated model names (default: all)")
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")] if args.models else None
    results = run_benchmark(args.split, model_names, args.max_patches)

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
