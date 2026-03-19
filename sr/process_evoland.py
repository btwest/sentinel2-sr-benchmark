"""
Run EVOLAND super-resolution on raw Sentinel-2 bands for the Aug 27 2024 scene.

Reads B02, B03, B04, B08 JP2 files (10m), processes in 512x512 tiles,
produces a 5m-equivalent TCI COG at sr_renders/evoland/<scene>_cog.tif.

The output TCI uses bands B04/B03/B02 (RGB) from the SR result — same band
ordering as the parent project's TCI renders so it slots into the viewer.

Usage:
    python sr/process_evoland.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal
from rasterio.transform import Affine
from rasterio.windows import Window

gdal.UseExceptions()

sys.path.insert(0, str(Path(__file__).parent.parent))

SCENE_NAME = "S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546"
BANDS_DIR  = Path("../sentinel-2/downloads/2024-08") / SCENE_NAME
SR_DIR     = Path("sr_renders/evoland")
TILE_SIZE  = 512   # pixels per tile (constant ~100 MB RAM)
SCALE      = 2

# Band filenames
TILE_DATE  = "T36RXV_20240827T081609"
BAND_FILES = {
    "B02": BANDS_DIR / f"{TILE_DATE}_B02_10m.jp2",
    "B03": BANDS_DIR / f"{TILE_DATE}_B03_10m.jp2",
    "B04": BANDS_DIR / f"{TILE_DATE}_B04_10m.jp2",
    "B08": BANDS_DIR / f"{TILE_DATE}_B08_10m.jp2",
}


def load_band_window(path: Path, window: Window) -> np.ndarray:
    """Read one band window as float32 in [0, 1]."""
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
    return np.clip(data / 10000.0, 0.0, 1.0)


def main() -> None:
    from models.evoland import EvolandModel

    # Verify all band files exist
    for band, path in BAND_FILES.items():
        if not path.exists():
            print(f"Missing {band}: {path}")
            sys.exit(1)

    SR_DIR.mkdir(parents=True, exist_ok=True)
    out_name  = f"S2_20240827T081609_evoland_cog.tif"
    out_path  = SR_DIR / out_name
    tmp_path  = out_path.with_suffix(".tmp.tif")

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    print("Loading EVOLAND model...")
    model = EvolandModel()
    print("  Model ready.")

    # Get scene dimensions and transform from B02
    with rasterio.open(BAND_FILES["B02"]) as ref:
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

    print(f"Scene: {h}x{w} px  →  SR output: {h2}x{w2} px")
    print(f"Tiles: {n_tiles}  (writing to {tmp_path.name})")

    with rasterio.open(tmp_path, "w", **profile) as dst:
        for row_off in range(0, h, TILE_SIZE):
            for col_off in range(0, w, TILE_SIZE):
                tile_h = min(TILE_SIZE, h - row_off)
                tile_w = min(TILE_SIZE, w - col_off)
                win_in = Window(col_off, row_off, tile_w, tile_h)

                # Stack 4 bands: B02, B03, B04, B08 (model expects this order)
                lr = np.stack([
                    load_band_window(BAND_FILES["B02"], win_in),
                    load_band_window(BAND_FILES["B03"], win_in),
                    load_band_window(BAND_FILES["B04"], win_in),
                    load_band_window(BAND_FILES["B08"], win_in),
                ], axis=0)   # (4, H, W) float32 [0,1]

                sr = model.upscale(lr)   # (4, 2H, 2W) float32 [0,1]

                # Build uint8 TCI: R=B04(idx2), G=B03(idx1), B=B02(idx0)
                # Apply ESA-equivalent brightness stretch: map 0-30% reflectance
                # to 0-255 (same scaling ESA uses for their TCI renders), so the
                # EVOLAND output matches the visual brightness of the original TCI.
                tci_sr = np.stack([sr[2], sr[1], sr[0]], axis=0)  # (3, 2H, 2W)
                tci_u8 = (np.clip(tci_sr / 0.4, 0.0, 1.0) * 255).astype(np.uint8)

                win_out = Window(col_off * SCALE, row_off * SCALE,
                                 tci_u8.shape[2], tci_u8.shape[1])
                dst.write(tci_u8, window=win_out)

                done += 1
                if done % 10 == 0 or done == n_tiles:
                    print(f"  {done}/{n_tiles} tiles", end="\r", flush=True)

    print(f"\nConverting to COG → {out_path.name}")
    gdal.Warp(
        str(out_path), str(tmp_path),
        format="COG",
        creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=AVERAGE"],
    )
    tmp_path.unlink()
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
