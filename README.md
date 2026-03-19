# sentinel2-sr-benchmark

Benchmarking super-resolution techniques on Sentinel-2 satellite imagery, with a focus on Battle Damage Assessment (BDA) applications over the Gaza Strip.

Fork of [sentinel-2-bda](https://github.com/btwest/sentinel-2-bda).

---

## What this does

Compares three super-resolution methods on Sentinel-2 L2A imagery (10m GSD → 5m GSD):

| Method | Type | PSNR | SSIM |
|--------|------|------|------|
| Bicubic | Classical interpolation | 41.18 dB | 0.9662 |
| Lanczos | Classical interpolation | 41.60 dB | 0.9696 |
| EVOLAND | Neural (CARN, ONNX) | 40.83 dB | 0.9657 |

Benchmark scores are computed against real **VENµS 5m ground truth** from the [SEN2VENµS dataset](https://zenodo.org/record/6514159) — a paired Sentinel-2 / VENµS dataset curated for exactly this purpose. VENµS was a joint ESA/ISA mission whose spectral bands were deliberately designed to match Sentinel-2, making it the standard ground truth for Sentinel-2 super-resolution evaluation.

The interactive viewer shows EVOLAND's output on the **Aug 27, 2024 Gaza scene** — the peak damage scene from the parent BDA project, chosen because it has clear skies and high structural complexity (rubble, destroyed buildings, displaced earth) where SR resolution improvements are most visible.

---

## Why Lanczos beats EVOLAND on PSNR

EVOLAND was trained on Theia-processed L2A products; this benchmark uses ESA-processed data. The slight pipeline difference causes distribution shift at inference time, hurting pixel-level metrics. Classical interpolation makes conservative smooth predictions that minimize average error. EVOLAND generates sharper, more detailed output — which is perceptually better for imagery analysis but penalized by PSNR/SSIM, which reward smoothness. A perceptual metric (LPIPS) would likely reverse the ranking.

---

## Stack

- **Python** — data pipeline, SR processing, benchmarking
- **EVOLAND** (`sentinel2_superresolution`) — CARN-based ONNX model, 2× SR, trained on SEN2VENµS
- **onnxruntime** — CPU inference
- **rasterio / GDAL** — geospatial I/O, COG export
- **TiTiler** — FastAPI COG tile server
- **Leaflet** — interactive map with swipe compare

---

## Project structure

```
models/
  base.py           # Abstract SRModel interface
  classical.py      # Bicubic and Lanczos (PIL-based)
  evoland.py        # EVOLAND ONNX wrapper

sr/
  process.py        # Tile-based SR on TCI renders (bicubic/lanczos)
  process_evoland.py # SR on raw B02/B03/B04/B08 bands (EVOLAND)
  download_bands.py # Downloads raw bands for a scene from Copernicus OData

eval/
  metrics.py        # PSNR, SSIM, LPIPS
  benchmark.py      # Runs all models on SEN2VENuS test patches
  results/
    results.json    # Benchmark scores

data/
  download_sen2venus.py  # Downloads SEN2VENuS .7z archives
  prepare.py             # Converts .pt tensors to .npz patches + manifest

server.py           # FastAPI server (TiTiler + scene/results endpoints)
map.html            # Leaflet viewer with swipe SR compare
```

---

## Running locally

```bash
# 1. Create virtualenv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/Evoland-Land-Monitoring-Evolution/sentinel2_superresolution.git

# 2. Expects parent project renders at ../sentinel-2/renders/
# and raw bands at ../sentinel-2/downloads/

# 3. Start the server
uvicorn server:app --host 0.0.0.0 --port 8002

# 4. Open http://localhost:8002
```

---

## Benchmark

```bash
# Download and prepare SEN2VENuS test data (requires ~500MB)
python data/download_sen2venus.py
python data/prepare.py

# Run benchmark (all models)
PYTHONPATH=. python eval/benchmark.py --split test
```

---

## Data

- **Sentinel-2 imagery**: Copernicus Open Access Hub, MGRS tile 36RXV, Aug 27 2024
- **SR benchmark**: [SEN2VENµS](https://zenodo.org/record/6514159), FGMANAUS site (Amazon, Brazil)
- Raw band downloads require a Copernicus Data Space account (`.env` with `COPERNICUS_USER` / `COPERNICUS_PASSWORD`)
