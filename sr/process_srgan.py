"""
Run trained OpenSR-SRGAN on the Aug 27 2024 Gaza scene.

Reads B02, B03, B04, B08 JP2 files (10m), processes in 256x256 tiles
with 32px overlap padding to eliminate tile boundary artifacts,
produces a 2.5m-equivalent TCI COG at sr_renders/srgan/<scene>_cog.tif.

Usage:
    PYTHONPATH=. python sr/process_srgan.py
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
SR_DIR     = Path("sr_renders/srgan")
TILE_SIZE  = 256   # core tile size in input pixels
MARGIN     = 32    # overlap padding in input pixels (= 128px in output)
SCALE      = 4

TILE_DATE  = "T36RXV_20240827T081609"
BAND_FILES = {
    "B02": BANDS_DIR / f"{TILE_DATE}_B02_10m.jp2",
    "B03": BANDS_DIR / f"{TILE_DATE}_B03_10m.jp2",
    "B04": BANDS_DIR / f"{TILE_DATE}_B04_10m.jp2",
    "B08": BANDS_DIR / f"{TILE_DATE}_B08_10m.jp2",
}


def load_band_window(path: Path, col_off: int, row_off: int,
                     width: int, height: int, src_w: int, src_h: int) -> np.ndarray:
    """Read a band window clamped to scene bounds, zero-pad if needed."""
    col_off_c = max(0, col_off)
    row_off_c = max(0, row_off)
    col_end_c = min(src_w, col_off + width)
    row_end_c = min(src_h, row_off + height)

    win = Window(col_off_c, row_off_c, col_end_c - col_off_c, row_end_c - row_off_c)
    with rasterio.open(path) as src:
        patch = src.read(1, window=win).astype(np.float32)

    # Embed in zero-padded array of full requested size
    out = np.zeros((height, width), dtype=np.float32)
    dst_r = row_off_c - row_off
    dst_c = col_off_c - col_off
    out[dst_r:dst_r + patch.shape[0], dst_c:dst_c + patch.shape[1]] = patch
    return np.clip(out / 10000.0, 0.0, 1.0)


def main() -> None:
    from models.srgan import SRGANModel

    for band, path in BAND_FILES.items():
        if not path.exists():
            print(f"Missing {band}: {path}")
            sys.exit(1)

    SR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SR_DIR / "S2_20240827T081609_srgan_cog.tif"
    tmp_path = out_path.with_suffix(".tmp.tif")

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    print("Loading SRGAN model...")
    model = SRGANModel()
    print("  Model ready.")

    with rasterio.open(BAND_FILES["B02"]) as ref:
        h, w      = ref.height, ref.width
        transform = ref.transform
        crs       = ref.crs
        profile   = ref.profile.copy()

    h2, w2 = h * SCALE, w * SCALE
    new_transform = Affine(
        transform.a / SCALE, transform.b, transform.c,
        transform.d, transform.e / SCALE, transform.f,
    )

    profile.update({
        "driver":     "GTiff",
        "count":      3,
        "height":     h2,
        "width":      w2,
        "dtype":      "uint8",
        "transform":  new_transform,
        "crs":        crs,
        "compress":   "deflate",
        "tiled":      True,
        "blockxsize": 512,
        "blockysize": 512,
    })

    n_tiles = ((h + TILE_SIZE - 1) // TILE_SIZE) * ((w + TILE_SIZE - 1) // TILE_SIZE)
    done = 0
    margin_out = MARGIN * SCALE   # 128px margin in output space

    print(f"Scene: {h}x{w} px  →  SR output: {h2}x{w2} px  ({n_tiles} tiles)")

    with rasterio.open(tmp_path, "w", **profile) as dst:
        for row_off in range(0, h, TILE_SIZE):
            for col_off in range(0, w, TILE_SIZE):
                tile_h = min(TILE_SIZE, h - row_off)
                tile_w = min(TILE_SIZE, w - col_off)

                # Padded input window (clamped to scene bounds with zero-padding)
                pad_col = col_off - MARGIN
                pad_row = row_off - MARGIN
                pad_w   = tile_w + 2 * MARGIN
                pad_h   = tile_h + 2 * MARGIN

                # SEN2NAIP band order: B04, B03, B02, B08 (R, G, B, NIR)
                lr = np.stack([
                    load_band_window(BAND_FILES["B04"], pad_col, pad_row, pad_w, pad_h, w, h),
                    load_band_window(BAND_FILES["B03"], pad_col, pad_row, pad_w, pad_h, w, h),
                    load_band_window(BAND_FILES["B02"], pad_col, pad_row, pad_w, pad_h, w, h),
                    load_band_window(BAND_FILES["B08"], pad_col, pad_row, pad_w, pad_h, w, h),
                ], axis=0)   # (4, tile_h+2*MARGIN, tile_w+2*MARGIN)

                sr = model.upscale(lr)   # (4, 4*(tile_h+2*MARGIN), 4*(tile_w+2*MARGIN))

                # Strip the margin from all four sides
                sr_core = sr[
                    :,
                    margin_out : sr.shape[1] - margin_out,
                    margin_out : sr.shape[2] - margin_out,
                ]   # (4, 4*tile_h, 4*tile_w)

                # TCI: model outputs (R, G, B, NIR) — indices 0,1,2 = R,G,B
                tci_sr = np.stack([sr_core[0], sr_core[1], sr_core[2]], axis=0)
                tci_u8 = (np.clip(tci_sr / 0.4, 0.0, 1.0) * 255).astype(np.uint8)

                win_out = Window(col_off * SCALE, row_off * SCALE,
                                 tci_u8.shape[2], tci_u8.shape[1])
                dst.write(tci_u8, window=win_out)

                done += 1
                if done % 20 == 0 or done == n_tiles:
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
