import json
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from titiler.core.factory import TilerFactory
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers

app = FastAPI(title="Sentinel-2 SR Benchmark")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cog = TilerFactory()
app.include_router(cog.router, prefix="/cog")
add_exception_handlers(app, DEFAULT_STATUS_CODES)

RENDERS_DIR  = Path("../sentinel-2/renders")
SR_DIR       = Path("./sr_renders")
RESULTS_PATH = Path("eval/results/results.json")

# ── Location registry ─────────────────────────────────────────────────────────
# Each entry defines one viewable location.
# sr_overrides: use these exact SR_DIR-relative paths instead of the default
# naming convention (needed for Gaza whose EVOLAND/SRGAN were built before the
# full-scene-name convention was adopted).
_LOCATIONS_CFG = [
    {
        "id":     "gaza",
        "name":   "Gaza Strip",
        "date":   "Aug 27 2024",
        "center": [31.38, 34.40],
        "zoom":   13,
        "tci":    "S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546_cog.tif",
        "sr_overrides": {
            "evoland": "evoland/S2_20240827T081609_evoland_cog.tif",
            "srgan":   "srgan/S2_20240827T081609_srgan_cog.tif",
        },
    },
    {
        "id":     "ramstein",
        "name":   "Ramstein Air Base",
        "date":   "Mar 19 2026",
        "center": [49.44, 7.60],
        "zoom":   13,
        "tci":    "S2A_MSIL2A_20260319T104041_N0512_R008_T32ULV_20260319T173915_cog.tif",
    },
    {
        "id":     "bandar_abbas",
        "name":   "Port of Bandar Abbas",
        "date":   "Mar 18 2026",
        "center": [27.12, 56.18],
        "zoom":   13,
        "tci":    "S2C_MSIL2A_20260318T064631_N0512_R020_T40RDQ_20260318T123223_cog.tif",
    },
]

_SR_METHODS = ["bicubic", "lanczos", "esrgan", "evoland", "srgan"]


def _resolve_sr(tci_name: str, method: str, overrides: dict) -> Path | None:
    """Return the SR output Path for a method, or None if the file doesn't exist."""
    if method in overrides:
        p = SR_DIR / overrides[method]
    elif method == "evoland":
        stem = tci_name.replace("_cog.tif", "")
        p = SR_DIR / "evoland" / f"{stem}_evoland_cog.tif"
    elif method == "srgan":
        stem = tci_name.replace("_cog.tif", "")
        p = SR_DIR / "srgan" / f"{stem}_srgan_cog.tif"
    else:
        p = SR_DIR / method / tci_name
    return p if p.exists() else None


# ── API endpoints ─────────────────────────────────────────────────────────────

def parse_date(filename: str) -> str:
    match = re.search(r"_(\d{8})T\d{6}_", filename)
    return match.group(1) if match else "00000000"


def scene_entry(path: Path) -> dict:
    date_str = parse_date(path.name)
    return {
        "name": path.name,
        "path": str(path),
        "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
    }


@app.get("/locations")
def list_locations():
    """
    Structured location data for the scene selector.
    Returns resolved absolute paths for TCI and each SR method,
    plus geographic bounds derived from the raster.
    """
    import rasterio
    from rasterio.warp import transform_bounds as _rtb

    out = []
    for cfg in _LOCATIONS_CFG:
        tci_path = RENDERS_DIR / cfg["tci"]
        if not tci_path.exists():
            continue

        # Geographic bounds from the raster (WGS84)
        bounds = None
        try:
            with rasterio.open(tci_path) as src:
                w, s, e, n = _rtb(src.crs, "EPSG:4326", *src.bounds)
            bounds = [[round(s, 4), round(w, 4)], [round(n, 4), round(e, 4)]]
        except Exception:
            pass

        # Resolve SR paths — only include methods whose files exist
        overrides = cfg.get("sr_overrides", {})
        methods = {}
        for m in _SR_METHODS:
            p = _resolve_sr(cfg["tci"], m, overrides)
            if p:
                methods[m] = str(p.resolve())

        out.append({
            "id":       cfg["id"],
            "name":     cfg["name"],
            "date":     cfg["date"],
            "center":   cfg["center"],
            "zoom":     cfg["zoom"],
            "bounds":   bounds,
            "tci_path": str(tci_path.resolve()),
            "methods":  methods,
        })

    return out


@app.get("/scenes")
def list_scenes():
    """Original TCI renders — kept for backward compatibility."""
    cogs = sorted(RENDERS_DIR.glob("*_cog.tif"))
    return {"scenes": [scene_entry(p) for p in cogs]}


@app.get("/sr/scenes")
def list_sr_scenes():
    """SR renders grouped by method — kept for backward compatibility."""
    if not SR_DIR.exists():
        return {"methods": {}}
    methods = {}
    for method_dir in sorted(SR_DIR.iterdir()):
        if not method_dir.is_dir():
            continue
        cogs = sorted(method_dir.glob("*_cog.tif"))
        if cogs:
            methods[method_dir.name] = [scene_entry(p) for p in cogs]
    return {"methods": methods}


@app.get("/results")
def get_results():
    """Quantitative benchmark results from SEN2VENuS evaluation."""
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text())


@app.get("/")
def serve_map():
    return FileResponse("map.html")
