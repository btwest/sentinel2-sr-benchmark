import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
from osgeo import gdal


def get_nodata_mask(scene_dir, reference_tif):
    detfoo_files = list(Path(scene_dir).glob("MSK_DETFOO_B04.jp2"))
    if not detfoo_files:
        raise FileNotFoundError(f"No DETFOO mask in {scene_dir}")
    
    with rasterio.open(detfoo_files[0]) as src:
        mask = src.read(1)
    
    alpha = np.where(mask > 0, 255, 0).astype(np.uint8)
    
    with rasterio.open(reference_tif) as ref:
        target_shape = (ref.height, ref.width)
    
    if alpha.shape != target_shape:
        alpha = np.array(Image.fromarray(alpha).resize(
            (target_shape[1], target_shape[0]), Image.NEAREST
        ))
    
    return alpha

def make_transparent(tif_path, out_path, scene_dir, mask_reference=None):
    reference = mask_reference if mask_reference else tif_path
    alpha = get_nodata_mask(scene_dir, reference)
    
    with rasterio.open(tif_path) as src:
        data = src.read()
        meta = src.meta.copy()
    
    if alpha.shape != (data.shape[1], data.shape[2]):
        alpha = np.array(Image.fromarray(alpha).resize(
            (data.shape[2], data.shape[1]), Image.NEAREST
        ))
    
    rgba = np.vstack([data, alpha[np.newaxis, :, :]])
    meta.update({"count": 4, "dtype": "uint8"})
    
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(rgba)
    
    print(f"Transparent tif saved: {out_path}")

def convert_to_cog(transparent_path, cog_path):
    gdal.Translate(
        cog_path,
        transparent_path,
        format="COG",
        creationOptions=["COMPRESS=DEFLATE", "OVERVIEW_RESAMPLING=AVERAGE"]
    )
    print(f"COG saved: {cog_path}")

def reproject_to_webmercator(jp2_path, out_path):
    gdal.Warp(out_path, jp2_path, dstSRS="EPSG:3857", format="GTiff")
    print(f"Reprojected: {out_path}")

def find_all_scenes(downloads_dir="./downloads"):
    scenes = []
    for scene_dir in Path(downloads_dir).iterdir():
        if not scene_dir.is_dir():
            continue
        tci_files = list(scene_dir.glob("*_TCI_10m.jp2"))
        if tci_files:
            scenes.append((scene_dir, tci_files[0]))
    return sorted(scenes, key=lambda x: x[0].name)

if __name__ == "__main__":
    scenes = find_all_scenes()
    print(f"Found {len(scenes)} scenes\n")
    
    for scene_dir, tci_path in scenes:
        print(f"Processing: {scene_dir.name}")
        try:
            scene_name = scene_dir.name
            reprojected_path = f"./renders/{scene_name}_3857.tif"
            transparent_path = f"./renders/{scene_name}_transparent.tif"
            cog_path = f"./renders/{scene_name}_cog.tif"

            if Path(cog_path).exists():
                print(f"  Already rendered, skipping")
                continue

            reproject_to_webmercator(str(tci_path), reprojected_path)
            make_transparent(reprojected_path, transparent_path, scene_dir, str(tci_path))
            convert_to_cog(transparent_path, cog_path)

            Path(reprojected_path).unlink()
            Path(transparent_path).unlink()

        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
        print()