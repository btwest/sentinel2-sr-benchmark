"""
Pre-extract SEN2NAIP taco file to individual .npz files for fast training.
Run once: python data/extract_sen2naip.py
"""
import sys
from pathlib import Path
import numpy as np
import rasterio
import tacoreader
from tqdm import tqdm

TACO_FILE = "data/sen2naip/sen2naipv2-crosssensor.taco"
OUT_DIR   = Path("data/sen2naip/extracted")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = tacoreader.load(TACO_FILE)
print(f"Extracting {len(ds)} samples to {OUT_DIR}...")

for i in tqdm(range(len(ds))):
    out_path = OUT_DIR / f"{i:05d}.npz"
    if out_path.exists():
        continue
    sample = ds.read(i)
    lr_path = sample.read(0)
    hr_path = sample.read(1)
    with rasterio.open(lr_path) as src:
        lr = src.read()   # (4, 130, 130) uint16
    with rasterio.open(hr_path) as src:
        hr = src.read()   # (4, 520, 520) uint16
    np.savez_compressed(out_path, lr=lr, hr=hr)

print(f"Done. {len(list(OUT_DIR.glob('*.npz')))} files in {OUT_DIR}")
