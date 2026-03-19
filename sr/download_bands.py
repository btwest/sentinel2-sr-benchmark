"""
Download the visible bands (B02, B03, B04) for a specific scene
that already exists in the parent project's downloads directory.

These are needed for EVOLAND which requires raw reflectance input.

Usage:
    python sr/download_bands.py
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE      = "https://download.dataspace.copernicus.eu/odata/v1"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"

TARGET_SCENE = "S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546"
DOWNLOADS_DIR = Path("../sentinel-2/downloads/2024-08")


def get_token() -> str:
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "username": os.getenv("CDSE_USERNAME"),
            "password": os.getenv("CDSE_PASSWORD"),
            "client_id": "cdse-public",
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
    scene_dir = DOWNLOADS_DIR / TARGET_SCENE
    if not scene_dir.exists():
        print(f"Scene directory not found: {scene_dir}")
        sys.exit(1)

    print(f"Scene: {TARGET_SCENE}")
    print("Getting Copernicus token...")
    token = get_token()

    print("Looking up product ID...")
    product_id = get_product_id(token, TARGET_SCENE)
    print(f"  Product ID: {product_id}")

    print("Getting granule name...")
    granule_name = get_granule_name(token, product_id, TARGET_SCENE)
    print(f"  Granule: {granule_name}")

    # Parse tile + date from product name
    # S2B_MSIL2A_20240827T081609_N0511_R121_T36RXV_20240827T113546
    parts = TARGET_SCENE.split("_")
    date_str = parts[2][:15]    # 20240827T081609
    tile_id  = [p for p in parts if p.startswith("T")][0]  # T36RXV

    bands = ["B02", "B03", "B04"]
    for band in bands:
        filename = f"{tile_id}_{date_str}_{band}_10m.jp2"
        url = (
            f"{BASE}/Products({product_id})"
            f"/Nodes({TARGET_SCENE}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
            f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({filename})/$value"
        )
        print(f"Downloading {filename}...")
        download_file(token, url, scene_dir / filename)

    print(f"\nDone. Bands written to {scene_dir}")


if __name__ == "__main__":
    main()
