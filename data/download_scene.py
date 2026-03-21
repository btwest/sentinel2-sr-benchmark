"""
Download TCI + raw bands (B02, B03, B04, B08) for a Sentinel-2 L2A scene.

Usage:
    python data/download_scene.py --scene S2A_MSIL2A_20260319T104041_N0512_R008_T32ULV_20260319T173915
    python data/download_scene.py --scene S2C_MSIL2A_20260318T064631_N0512_R020_T40RDQ_20260318T123223
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE      = "https://download.dataspace.copernicus.eu/odata/v1"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"


def get_token() -> str:
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type":  "password",
            "username":    os.getenv("CDSE_USERNAME"),
            "password":    os.getenv("CDSE_PASSWORD"),
            "client_id":   "cdse-public",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_product_id(token: str, product_name: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{CATALOGUE}/Products?$filter=Name eq '{product_name}.SAFE'&$top=1"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    results = resp.json()["value"]
    if not results:
        raise RuntimeError(f"Product not found: {product_name}")
    return results[0]["Id"]


def get_granule_name(token: str, product_id: str, product_name: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE}/Products({product_id})/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes"
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["result"][0]["Id"]


def download_file(token: str, url: str, out_path: Path) -> None:
    if out_path.exists():
        print(f"  already exists: {out_path.name}")
        return
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Full scene name (without .SAFE)")
    args = parser.parse_args()

    scene_name = args.scene.replace(".SAFE", "")

    # Parse date and tile from scene name
    # e.g. S2A_MSIL2A_20260319T104041_N0512_R008_T32ULV_20260319T173915
    parts     = scene_name.split("_")
    date_str  = parts[2][:15]                              # 20260319T104041
    tile_id   = next(p for p in parts if p.startswith("T") and len(p) == 6)  # T32ULV
    year_month = date_str[:6]                              # 202603

    # Store alongside Gaza scenes in parent project downloads
    out_dir = Path("../sentinel-2/downloads") / f"{year_month[:4]}-{year_month[4:6]}" / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scene:  {scene_name}")
    print(f"Tile:   {tile_id}  Date: {date_str}")
    print(f"Output: {out_dir}")
    print()

    print("Getting Copernicus token...")
    token = get_token()

    print("Looking up product ID...")
    product_id = get_product_id(token, scene_name)
    print(f"  Product ID: {product_id}")

    print("Getting granule name...")
    granule_name = get_granule_name(token, product_id, scene_name)
    print(f"  Granule: {granule_name}")

    def dl(filename, res):
        url = (
            f"{BASE}/Products({product_id})"
            f"/Nodes({scene_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes({res})/Nodes({filename})/$value"
        )
        print(f"Downloading {filename}...")
        download_file(token, url, out_dir / filename)

    # TCI (true color image) — for quick visual verification and bicubic/lanczos/esrgan
    dl(f"{tile_id}_{date_str}_TCI_10m.jp2", "R10m")

    # Raw bands — for EVOLAND and SRGAN
    for band in ["B02", "B03", "B04", "B08"]:
        dl(f"{tile_id}_{date_str}_{band}_10m.jp2", "R10m")

    print(f"\nDone. Files written to {out_dir}")
    print(f"\nTo verify visually, open the TCI in QGIS or run:")
    print(f"  python -c \"import rasterio; src=rasterio.open('{out_dir}/{tile_id}_{date_str}_TCI_10m.jp2'); print(src.meta)\"")


if __name__ == "__main__":
    main()
