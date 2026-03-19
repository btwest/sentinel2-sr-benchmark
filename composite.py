import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import glob
import os

RENDERS_DIR = "./renders"
OUTPUT_PATH = "./renders/composite_median.tif"
BLOCK_ROWS = 256  # Process this many rows at a time, tune down if still OOMing

scenes = sorted(glob.glob(os.path.join(RENDERS_DIR, "*_cog.tif")))
print(f"Found {len(scenes)} COG scenes")

# Reference grid from first scene
with rasterio.open(scenes[0]) as ref:
    profile = ref.profile.copy()
    crs = ref.crs
    transform = ref.transform
    width = ref.width
    height = ref.height
    count = ref.count

print(f"Reference grid: {width}x{height}, {count} bands")

profile.update(
    driver="GTiff",
    count=count,
    dtype="uint8",
    compress="deflate",
    tiled=True,
    blockxsize=512,
    blockysize=512,
    interleave="pixel",
)

with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
    for row_off in range(0, height, BLOCK_ROWS):
        row_count = min(BLOCK_ROWS, height - row_off)
        window = rasterio.windows.Window(0, row_off, width, row_count)

        block_stack = []
        for path in scenes:
            with rasterio.open(path) as src:
                with WarpedVRT(src, crs=crs, transform=transform,
                               width=width, height=height,
                               resampling=Resampling.bilinear) as vrt:
                    data = vrt.read(window=window).astype(np.float32)
                    data[data == 0] = np.nan
                    block_stack.append(data)

        block_stack = np.array(block_stack)
        median = np.nanmedian(block_stack, axis=0)
        median = np.nan_to_num(median, nan=0).astype(np.uint8)
        dst.write(median, window=window)

        pct = (row_off + row_count) / height * 100
        print(f"  Progress: {pct:.1f}%")

print(f"Composite saved: {OUTPUT_PATH}")