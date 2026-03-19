"""
Export SR outputs as Cloud-Optimized GeoTIFFs for web tile serving.

For each patch in a scene, this script:
  1. Reads the patch bounding box from the scene's .gpkg file
  2. Runs all SR models on the LR patch (+ saves the HR ground truth)
  3. Writes each output as a georeferenced EPSG:3857 COG

Output structure:
  data/cogs/<site>_<date>/<method>/patch_<index>.tif
    methods: lr (bilinear upscaled for display), bicubic, lanczos, hr (ground truth)

Usage:
    python data/export_cogs.py --scene FGMANAUS_2020-07-31 --max-patches 10
    python data/export_cogs.py  # all scenes, all patches
"""

import argparse
import json
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from osgeo import gdal

gdal.UseExceptions()

from models.classical import BicubicModel, LanczosModel

PATCHES_DIR = Path("data/patches")
COGS_DIR = Path("data/cogs")
RAW_DIR = Path("data/raw")
MANIFEST_PATH = PATCHES_DIR / "manifest.json"

METHODS = {
    "bicubic": BicubicModel(scale=2),
    "lanczos": LanczosModel(scale=2),
}

# RGB display bands: B4=Red, B3=Green, B2=Blue (indices 2, 1, 0 in b2b3b4b8 order)
RGB_BANDS = [2, 1, 0]


def get_patch_bounds(gpkg_path: str, index: int) -> tuple:
    """Return (minx, miny, maxx, maxy) in UTM for patch at given index."""
    with fiona.open(gpkg_path) as f:
        feat = list(f)[index]
        coords = feat["geometry"]["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return min(xs), min(ys), max(xs), max(ys)


def get_utm_crs(gpkg_path: str) -> CRS:
    with fiona.open(gpkg_path) as f:
        return CRS.from_user_input(f.crs)


def array_to_cog(arr: np.ndarray, bounds: tuple, src_crs: CRS, out_path: Path) -> None:
    """
    Write a float32 (C, H, W) array as an EPSG:3857 COG.
    Applies per-band 2–98% percentile stretch for display — satellite imagery
    reflectance values are often clustered in a narrow low range and need
    contrast stretching to be visually useful.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.tif")

    c, h, w = arr.shape
    stretched = np.empty_like(arr)
    for i in range(c):
        lo = np.percentile(arr[i], 2)
        hi = np.percentile(arr[i], 98)
        stretched[i] = np.clip((arr[i] - lo) / (hi - lo + 1e-8), 0, 1)
    uint8 = (stretched * 255).astype(np.uint8)

    transform = from_bounds(*bounds, width=w, height=h)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": c,
        "height": h,
        "width": w,
        "crs": src_crs,
        "transform": transform,
    }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(uint8)

    gdal.Warp(
        str(out_path), str(tmp_path),
        dstSRS="EPSG:3857",
        format="COG",
        resampleAlg="bilinear",
        creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=AVERAGE"],
    )
    tmp_path.unlink()


def find_gpkg(site: str, date: str) -> str | None:
    """Find the .gpkg for a given site+date in data/raw."""
    for gpkg in RAW_DIR.rglob(f"*{date}*patches.gpkg"):
        return str(gpkg)
    return None


def export_scene(scene: dict, max_patches: int | None) -> None:
    site, date = scene["site"], scene["date"]
    scene_key = f"{site}_{date}"
    print(f"\n[{scene_key}]")

    gpkg_path = find_gpkg(site, date)
    if not gpkg_path:
        print(f"  no .gpkg found, skipping")
        return

    utm_crs = get_utm_crs(gpkg_path)
    data = np.load(scene["npz"])
    lr_all = data["lr"]  # [N, 4, 128, 128]
    hr_all = data["hr"]  # [N, 4, 256, 256]

    n = min(lr_all.shape[0], max_patches or 999999)

    for i in range(n):
        bounds = get_patch_bounds(gpkg_path, i)
        lr = lr_all[i]  # [4, 128, 128]
        hr = hr_all[i]  # [4, 256, 256]

        # LR displayed at HR resolution (bilinear upscale for visual comparison only)
        lr_display = BicubicModel(scale=2).upscale(lr)

        outputs = {
            "lr":     lr_display[RGB_BANDS],
            "hr":     hr[RGB_BANDS],
            "bicubic": METHODS["bicubic"].upscale(lr)[RGB_BANDS],
            "lanczos": METHODS["lanczos"].upscale(lr)[RGB_BANDS],
        }

        for method, arr in outputs.items():
            out_path = COGS_DIR / scene_key / method / f"patch_{i:04d}.tif"
            if not out_path.exists():
                array_to_cog(arr, bounds, utm_crs, out_path)

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} patches exported")

    # Write patch index JSON for the viewer (include WGS84 bounds for Leaflet)
    from rasterio.warp import transform_bounds
    index = []
    for i in range(n):
        bounds = get_patch_bounds(gpkg_path, i)
        wgs84 = transform_bounds(utm_crs, "EPSG:4326", *bounds)
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        index.append({
            "patch": i,
            "bounds_utm": bounds,
            "bounds_wgs84": list(wgs84),
            "center_utm": [cx, cy],
        })

    index_path = COGS_DIR / scene_key / "index.json"
    index_path.write_text(json.dumps({"scene": scene_key, "crs": str(utm_crs), "patches": index}, indent=2))
    print(f"  index.json written ({n} patches)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", help="Scene key e.g. FGMANAUS_2020-07-31 (default: all)")
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    all_scenes = []
    for split_scenes in manifest.values():
        all_scenes.extend(split_scenes)

    if args.scene:
        all_scenes = [s for s in all_scenes if f"{s['site']}_{s['date']}" == args.scene]
        if not all_scenes:
            print(f"Scene '{args.scene}' not found in manifest.")
            return

    for scene in all_scenes:
        export_scene(scene, args.max_patches)

    print("\nDone. COGs written to data/cogs/")


if __name__ == "__main__":
    main()
