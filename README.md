# Sentinel-2 Super-Resolution Benchmark

Benchmarks and visual comparison of super-resolution techniques applied to Sentinel-2 L2A satellite imagery over the Gaza Strip (Aug 27, 2024 — peak damage scene).

Compares classical interpolation, a satellite-domain neural SR model, and a general-purpose GAN across quantitative metrics and side-by-side visual inspection.

---

## What it does

- Applies four SR methods to a Sentinel-2 10m scene, producing 5m-equivalent output
- Evaluates each method quantitatively against real 5m VENµS ground truth (SEN2VENµS dataset)
- Serves results through an interactive Leaflet viewer with a swipe-to-compare interface

---

## SR Methods

| Method | Type | Input | PSNR | SSIM |
|--------|------|-------|------|------|
| Bicubic | Classical interpolation | TCI (RGB) | 41.18 dB | 0.9662 |
| Lanczos | Classical interpolation | TCI (RGB) | 41.60 dB | 0.9696 |
| EVOLAND (`s2v2x2_spatrad`) | CNN trained on Sentinel-2/VENµS pairs | Raw bands B02/B03/B04/B08 | 40.83 dB | 0.9657 |
| Real-ESRGAN x2plus | GAN trained on natural images | TCI (RGB) | 45.32 dB | 0.9552 |

Benchmark scores computed on 60 test patches from the SEN2VENµS dataset (Amazon rainforest site). See [Benchmark Notes](#benchmark-notes).

---

## Models

**Bicubic / Lanczos** — classical pixel interpolation using fixed convolution kernels. No learning, no parameters. Fast and deterministic. Lanczos has slightly better high-frequency preservation than bicubic.

**EVOLAND** ([sentinel2_superresolution](https://framagit.org/jmichel-otb/sentinel2_superresolution)) — a CARN (Cascading Residual Network) trained specifically on paired Sentinel-2 (10m) / VENµS (5m) acquisitions. Takes raw reflectance bands B02/B03/B04/B08 as input, outputs 2x resolution. Developed by CESBIO/CNES. Runs via ONNX (`s2v2x2_spatrad.onnx`).

**Real-ESRGAN x2plus** — a residual-in-residual dense block GAN trained on natural images with synthetic degradation. Not satellite-specific — applies learned natural image priors to enhance any RGB input. Higher PSNR than satellite-specific models on this benchmark but lower SSIM, reflecting the GAN tradeoff: synthesizes sharp perceptual detail that may not match ground truth pixel-for-pixel. Runs on GPU via PyTorch (RTX 3080, ~7 min for full scene).

---

## Benchmark Notes

- Ground truth: VENµS 5m imagery from the [SEN2VENµS dataset](https://zenodo.org/record/6514159) (Zenodo record 6514159)
- VENµS was a joint ESA/ISA mission designed as Sentinel-2's high-resolution companion, with matched spectral bands and coordinated acquisition schedules
- Test set: 60 patches of 128x128px from the FGMANAUS (Amazon rainforest) site
- EVOLAND scores should be interpreted cautiously as its training data overlap with SEN2VENµS is not fully documented
- Benchmark terrain (Amazon rainforest) differs from displayed imagery (Gaza, arid urban/coastal) — scores are valid for relative method comparison but are not terrain-matched validation

---

## Setup

```bash
git clone git@github.com:btwest/sentinel2-sr-benchmark.git
cd sentinel2-sr-benchmark
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires the parent project [`sentinel-2-bda`](https://github.com/btwest/sentinel-2-bda) cloned as a sibling directory (`../sentinel-2/`) for the Gaza TCI renders.

Download Real-ESRGAN weights:
```bash
mkdir weights
wget -O weights/RealESRGAN_x2plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
```

---

## Running

**Start the viewer:**
```bash
uvicorn server:app --host 0.0.0.0 --port 8002
# open http://localhost:8002
```

**Run SR on the Aug 27 scene:**
```bash
# Classical methods (bicubic, lanczos)
PYTHONPATH=. python3 sr/process.py --scene S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546_cog.tif

# EVOLAND (requires raw bands)
PYTHONPATH=. python3 sr/process_evoland.py

# Real-ESRGAN (GPU recommended)
PYTHONPATH=. python3 sr/process.py --methods esrgan \
  --scene S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546_cog.tif
```

**Run benchmark:**
```bash
python3 data/download_sen2venus.py
python3 data/prepare.py
PYTHONPATH=. python3 eval/benchmark.py --split test
```

---

## Stack

- **Inference**: ONNX Runtime (EVOLAND), PyTorch + CUDA (Real-ESRGAN)
- **Geospatial**: GDAL, Rasterio, Cloud-Optimized GeoTIFF
- **Tile server**: TiTiler (FastAPI + COG)
- **Viewer**: Leaflet, custom swipe-compare with containerPointToLayerPoint clip alignment
- **Metrics**: scikit-image (PSNR, SSIM)
- **Data**: Copernicus OData API, Zenodo (SEN2VENµS)
