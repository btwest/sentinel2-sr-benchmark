"""
Run EVOLAND super-resolution on raw Sentinel-2 bands for any scene.

Reads B02, B03, B04, B08 JP2 files (10m), processes in 512x512 tiles,
produces a 5m-equivalent TCI COG at sr_renders/evoland/<scene>_evoland_cog.tif.

The output TCI uses bands B04/B03/B02 (RGB) from the SR result — same band
ordering as the parent project's TCI renders so it slots into the viewer.

Usage:
    python sr/process_evoland.py --scene S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546
    python sr/process_evoland.py --scene S2A_MSIL2A_20260319T104041_N0512_R008_T32ULV_20260319T173915
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

sys.path.insert(0, str(Path(__file__).parent.parent))

DOWNLOADS_BASE = Path("../sentinel-2/downloads")
SR_DIR         = Path("sr_renders/evoland")
TILE_SIZE      = 512   # pixels per tile (constant ~100 MB RAM)
SCALE          = 2


def parse_scene(scene_name: str) -> tuple[str, str, str]:
    """Return (tile_id, date_str, year_month) parsed from scene name."""
    parts      = scene_name.split("_")
    date_str   = parts[2][:15]                                                # 20240827T081609
    tile_id    = next(p for p in parts if p.startswith("T") and len(p) == 6)  # T36RXV
    year_month = date_str[:6]                                                  # 202408
    return tile_id, date_str, year_month


def load_band_window(path: Path, window: Window) -> np.ndarray:
    """Read one band window as float32 in [0, 1]."""
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", required=True,
        help="Full scene name without .SAFE"
    )
    args = parser.parse_args()

    scene_name = args.scene.replace(".SAFE", "")
    tile_id, date_str, year_month = parse_scene(scene_name)

    bands_dir  = DOWNLOADS_BASE / f"{year_month[:4]}-{year_month[4:6]}" / scene_name
    band_files = {
        band: bands_dir / f"{tile_id}_{date_str}_{band}_10m.jp2"
        for band in ["B02", "B03", "B04", "B08"]
    }

    from models.evoland import EvolandModel

    # Verify all band files exist
    for band, path in band_files.items():
        if not path.exists():
            print(f"Missing {band}: {path}")
            sys.exit(1)

    SR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SR_DIR / f"{scene_name}_evoland_cog.tif"
    tmp_path = out_path.with_suffix(".tmp.tif")

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    print(f"Scene:  {scene_name}")
    print(f"Tile:   {tile_id}  Date: {date_str}")
    print(f"Output: {out_path}")
    print()

    print("Loading EVOLAND model...")
    model = EvolandModel()
    print("  Model ready.")

    # Get scene dimensions and transform from B02
    with rasterio.open(band_files["B02"]) as ref:
        h, w        = ref.height, ref.width
        transform   = ref.transform
        crs         = ref.crs
        profile     = ref.profile.copy()

    h2, w2 = h * SCALE, w * SCALE
    new_transform = Affine(
        transform.a / SCALE, transform.b, transform.c,
        transform.d, transform.e / SCALE, transform.f,
    )

    profile.update({
        "driver":    "GTiff",
        "count":     3,          # RGB output
        "height":    h2,
        "width":     w2,
        "dtype":     "uint8",
        "transform": new_transform,
        "crs":       crs,
        "compress":  "deflate",
        "tiled":     True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    n_tiles = ((h + TILE_SIZE - 1) // TILE_SIZE) * ((w + TILE_SIZE - 1) // TILE_SIZE)
    done    = 0

    print(f"Scene: {h}x{w} px  ->  SR output: {h2}x{w2} px")
    print(f"Tiles: {n_tiles}  (writing to {tmp_path.name})")

    with rasterio.open(tmp_path, "w", **profile) as dst:
        for row_off in range(0, h, TILE_SIZE):
            for col_off in range(0, w, TILE_SIZE):
                tile_h = min(TILE_SIZE, h - row_off)
                tile_w = min(TILE_SIZE, w - col_off)
                win_in = Window(col_off, row_off, tile_w, tile_h)

                # Stack 4 bands: B02, B03, B04, B08 (model expects this order)
                lr = np.stack([
                    load_band_window(band_files["B02"], win_in),
                    load_band_window(band_files["B03"], win_in),
                    load_band_window(band_files["B04"], win_in),
                    load_band_window(band_files["B08"], win_in),
                ], axis=0)   # (4, H, W) float32 [0,1]

                sr = model.upscale(lr)   # (4, 2H, 2W) float32 [0,1]

                # Build uint8 TCI: R=B04(idx2), G=B03(idx1), B=B02(idx0)
                # ESA-equivalent brightness stretch: white point 0.4 reflectance
                tci_sr = np.stack([sr[2], sr[1], sr[0]], axis=0)  # (3, 2H, 2W)
                tci_u8 = (np.clip(tci_sr / 0.4, 0.0, 1.0) * 255).astype(np.uint8)

                win_out = Window(col_off * SCALE, row_off * SCALE,
                                 tci_u8.shape[2], tci_u8.shape[1])
                dst.write(tci_u8, window=win_out)

                done += 1
                if done % 10 == 0 or done == n_tiles:
                    print(f"  {done}/{n_tiles} tiles", end="\r", flush=True)

    print(f"\nConverting to COG -> {out_path.name}")
    gdal.Warp(
        str(out_path), str(tmp_path),
        format="COG",
        creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=AVERAGE"],
    )
    tmp_path.unlink()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
