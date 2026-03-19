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

RENDERS_DIR = Path("../sentinel-2/renders")   # parent project renders
SR_DIR      = Path("./sr_renders")            # SR outputs written by sr/process.py
RESULTS_PATH = Path("eval/results/results.json")


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


@app.get("/scenes")
def list_scenes():
    """Original TCI renders from parent project."""
    cogs = sorted(RENDERS_DIR.glob("*_cog.tif"))
    return {"scenes": [scene_entry(p) for p in cogs]}


@app.get("/sr/scenes")
def list_sr_scenes():
    """
    SR-processed renders, grouped by method.
    Returns { method: [scene, ...] } for each available SR method.
    """
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
