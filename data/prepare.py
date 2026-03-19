"""
Prepare SEN2VENuS patches for benchmarking and training.

Actual format (discovered from data):
  Each site directory contains .pt files per acquisition date:
    <SITE>_<TILE>_<DATE>_10m_b2b3b4b8.pt  -- S2 LR  [N, 4, 128, 128] int16
    <SITE>_<TILE>_<DATE>_05m_b2b3b4b8.pt  -- VENuS HR [N, 4, 256, 256] int16
  Plus index.csv listing all acquisitions and patch counts.

  Scale factor: 2x (128px at 10m -> 256px at 5m)
  Values: surface reflectance int16, range roughly [-500, 10000]
  Normalization: clamp [0, 10000], divide by 10000 -> float32 [0, 1]

This script:
  1. Reads index.csv from each site
  2. Loads paired 10m/05m tensors
  3. Normalizes to float32 [0, 1]
  4. Saves per-scene .npz files (lr + hr arrays) to data/patches/scenes/
  5. Writes data/patches/manifest.json with train/val/test split by scene

Usage:
    python data/prepare.py
    python data/prepare.py --raw data/raw --out data/patches --splits 0.8,0.1,0.1 --seed 42
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

RAW_DIR = Path("data/raw")
PATCHES_DIR = Path("data/patches")
REFLECTANCE_MAX = 10000.0  # S2 L2A surface reflectance scale


def normalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert int16 reflectance tensor to float32 [0, 1] numpy array."""
    arr = tensor.numpy().astype(np.float32)
    arr = np.clip(arr, 0, REFLECTANCE_MAX) / REFLECTANCE_MAX
    return arr


def find_scenes(raw_dir: Path) -> list[dict]:
    """Walk raw_dir and return list of scene dicts from each site's index.csv."""
    scenes = []
    for site_dir in sorted(raw_dir.iterdir()):
        index_path = site_dir / "index.csv"
        if not index_path.exists():
            continue
        with open(index_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                lr_file = site_dir / row["tensor_10m_b2b3b4b8"]
                hr_file = site_dir / row["tensor_05m_b2b3b4b8"]
                if not lr_file.exists() or not hr_file.exists():
                    continue
                scenes.append({
                    "site": row["vns_site"],
                    "date": row["date"],
                    "n_patches": int(row["nb_patches"]),
                    "lr_path": str(lr_file),
                    "hr_path": str(hr_file),
                })
    return scenes


def process_scene(scene: dict, out_dir: Path) -> str:
    """Load, normalize and save a scene's patches as .npz. Returns output path."""
    stem = f"{scene['site']}_{scene['date']}"
    out_path = out_dir / f"{stem}.npz"

    if out_path.exists():
        return str(out_path)

    lr = normalize(torch.load(scene["lr_path"], weights_only=True))
    hr = normalize(torch.load(scene["hr_path"], weights_only=True))
    np.savez_compressed(out_path, lr=lr, hr=hr)
    return str(out_path)


def split_scenes(
    scenes: list[dict], ratios: tuple, seed: int
) -> dict[str, list]:
    random.seed(seed)
    shuffled = scenes.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW_DIR))
    parser.add_argument("--out", default=str(PATCHES_DIR))
    parser.add_argument("--splits", default="0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    out_dir = Path(args.out)
    scenes_dir = out_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    ratios = tuple(float(x) for x in args.splits.split(","))
    assert abs(sum(ratios) - 1.0) < 1e-6, "splits must sum to 1"

    print("Scanning for scenes...")
    scenes = find_scenes(raw_dir)
    total_patches = sum(s["n_patches"] for s in scenes)
    print(f"Found {len(scenes)} scenes, {total_patches} total patches\n")

    splits = split_scenes(scenes, ratios, args.seed)
    manifest = {}

    for split, scene_list in splits.items():
        print(f"Processing {split} ({len(scene_list)} scenes)...")
        manifest[split] = []
        for scene in scene_list:
            npz_path = process_scene(scene, scenes_dir)
            manifest[split].append({
                "site": scene["site"],
                "date": scene["date"],
                "n_patches": scene["n_patches"],
                "npz": npz_path,
            })
            print(f"  {scene['site']} {scene['date']} — {scene['n_patches']} patches")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")
    print(f"Total patches: {total_patches}")


if __name__ == "__main__":
    main()
