"""
Apply SR models to existing Sentinel-2 TCI renders from the parent project.

Reads each *_cog.tif from ../sentinel-2/renders/ and processes it in
512×512 tiles to stay within memory limits (full scenes are ~500MB).

Output: sr_renders/<method>/*_cog.tif — same footprint, 2x pixel resolution.

Usage:
    python sr/process.py                      # all scenes, all models
    python sr/process.py --methods bicubic    # one model
    python sr/process.py --scene <filename>   # one scene
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal
from rasterio.transform import Affine
from rasterio.windows import Window

gdal.UseExceptions()

RENDERS_DIR = Path("../sentinel-2/renders")
SR_DIR      = Path("sr_renders")
TILE_SIZE   = 512   # pixels — fits comfortably in RAM per chunk


def get_models(names: list[str] | None):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.classical import BicubicModel, LanczosModel
    all_models = [BicubicModel(scale=2), LanczosModel(scale=2)]

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


def upscale_cog(src_path: Path, out_path: Path, model, scale: int = 2) -> None:
    """
    Tile-by-tile SR upscale. Reads TILE_SIZE chunks, applies model,
    writes output incrementally — constant ~50MB RAM usage regardless of scene size.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")

    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        transform = src.transform
        profile = src.profile.copy()
        # Use only RGB bands (1, 2, 3) — drop alpha if present
        n_bands = min(src.count, 3)

    h2, w2 = h * scale, w * scale
    new_transform = Affine(
        transform.a / scale, transform.b, transform.c,
        transform.d, transform.e / scale, transform.f,
    )

    profile.update({
        "driver": "GTiff", "count": n_bands,
        "height": h2, "width": w2,
        "transform": new_transform,
        "compress": "deflate",
        "tiled": True, "blockxsize": 512, "blockysize": 512,
    })

    with rasterio.open(src_path) as src, rasterio.open(tmp_path, "w", **profile) as dst:
        n_tiles = ((h + TILE_SIZE - 1) // TILE_SIZE) * ((w + TILE_SIZE - 1) // TILE_SIZE)
        done = 0
        for row_off in range(0, h, TILE_SIZE):
            for col_off in range(0, w, TILE_SIZE):
                tile_h = min(TILE_SIZE, h - row_off)
                tile_w = min(TILE_SIZE, w - col_off)
                win_in = Window(col_off, row_off, tile_w, tile_h)

                chunk = src.read(list(range(1, n_bands + 1)), window=win_in)  # (C, H, W) uint8
                lr = chunk.astype(np.float32) / 255.0
                sr = model.upscale(lr)                                         # (C, H*2, W*2)
                sr_uint8 = (np.clip(sr, 0, 1) * 255).astype(np.uint8)

                win_out = Window(col_off * scale, row_off * scale,
                                 sr_uint8.shape[2], sr_uint8.shape[1])
                dst.write(sr_uint8, window=win_out)

                done += 1
                if done % 20 == 0 or done == n_tiles:
                    print(f"    {done}/{n_tiles} tiles", end="\r", flush=True)

    print()  # newline after progress

    # Convert to COG
    gdal.Warp(
        str(out_path), str(tmp_path),
        format="COG",
        creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=AVERAGE"],
    )
    tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", help="Comma-separated model names")
    parser.add_argument("--scene",   help="Process only this filename")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.methods.split(",")] if args.methods else None
    models = get_models(model_names)
    scenes = sorted(RENDERS_DIR.glob("*_cog.tif"))
    if args.scene:
        scenes = [p for p in scenes if p.name == args.scene]

    print(f"Scenes: {len(scenes)}  |  Models: {[m.name for m in models]}\n")

    for scene_path in scenes:
        print(f"{scene_path.name}")
        for model in models:
            out_path = SR_DIR / model.name / scene_path.name
            if out_path.exists():
                print(f"  {model.name}: already exists, skipping")
                continue
            print(f"  {model.name}:")
            upscale_cog(scene_path, out_path, model)
            print(f"  {model.name}: done -> {out_path}")

    print("\nFinished.")


if __name__ == "__main__":
    main()
