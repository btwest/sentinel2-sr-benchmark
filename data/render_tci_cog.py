"""
Convert a downloaded Sentinel-2 TCI JP2 to a Cloud-Optimised GeoTIFF
and deposit it in ../sentinel-2/renders/ so process.py picks it up.

Usage:
    python data/render_tci_cog.py --scene S2A_MSIL2A_20260319T104041_N0512_R008_T32ULV_20260319T173915
    python data/render_tci_cog.py --scene S2C_MSIL2A_20260318T064631_N0512_R020_T40RDQ_20260318T123223
"""

import argparse
import sys
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()

DOWNLOADS_BASE = Path("../sentinel-2/downloads")
RENDERS_DIR    = Path("../sentinel-2/renders")


def parse_scene(scene_name: str) -> tuple[str, str, str]:
    parts      = scene_name.split("_")
    date_str   = parts[2][:15]
    tile_id    = next(p for p in parts if p.startswith("T") and len(p) == 6)
    year_month = date_str[:6]
    return tile_id, date_str, year_month


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Full scene name without .SAFE")
    args = parser.parse_args()

    scene_name = args.scene.replace(".SAFE", "")
    tile_id, date_str, year_month = parse_scene(scene_name)

    src_path = (
        DOWNLOADS_BASE
        / f"{year_month[:4]}-{year_month[4:6]}"
        / scene_name
        / f"{tile_id}_{date_str}_TCI_10m.jp2"
    )

    if not src_path.exists():
        print(f"Source not found: {src_path}")
        sys.exit(1)

    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RENDERS_DIR / f"{scene_name}_cog.tif"

    if out_path.exists():
        print(f"Already exists: {out_path}")
        sys.exit(0)

    print(f"Source:  {src_path}")
    print(f"Output:  {out_path}")

    # Open JP2, keep bands 1-3 (TCI is already RGB uint8)
    ds = gdal.Open(str(src_path))
    if ds is None:
        print("Failed to open source file")
        sys.exit(1)

    print(f"  Size: {ds.RasterXSize} x {ds.RasterYSize}  bands: {ds.RasterCount}")

    gdal.Warp(
        str(out_path),
        ds,
        format="COG",
        outputType=gdal.GDT_Byte,
        creationOptions=[
            "COMPRESS=DEFLATE",
            "OVERVIEW_RESAMPLING=AVERAGE",
            "BLOCKSIZE=512",
        ],
    )
    ds = None
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
