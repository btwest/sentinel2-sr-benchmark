import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

BASE = "https://download.dataspace.copernicus.eu/odata/v1"
CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1"

def get_token():
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "username": os.getenv("CDSE_USERNAME"),
            "password": os.getenv("CDSE_PASSWORD"),
            "client_id": "cdse-public",
        }
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def search_scenes(token, tile_id, start_date, end_date, max_cloud=20):
    headers = {"Authorization": f"Bearer {token}"}
    url = (
        f"{CATALOGUE}/Products"
        f"?$filter=Collection/Name eq 'SENTINEL-2'"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'tileId'"
        f" and att/OData.CSC.StringAttribute/Value eq '{tile_id}')"
        f" and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover'"
        f" and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
        f" and ContentDate/Start gt '{start_date}'"
        f" and ContentDate/Start lt '{end_date}'"
        f"&$expand=Attributes"
        f"&$orderby=ContentDate/Start asc&$top=1000"
    )
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["value"]

def group_by_month(scenes):
    monthly = defaultdict(list)
    for scene in scenes:
        month = scene["ContentDate"]["Start"][:7]  # "2024-03"
        monthly[month].append(scene)
    return monthly

def get_granule_name(token, product_id, product_name):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{BASE}/Products({product_id})/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["result"][0]["Id"]

def download_file(token, url, out_path):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    print()

def download_scene(product, month_str, out_dir="./downloads"):
    token = get_token()

    product_id = product["Id"]
    product_name = product["Name"].replace(".SAFE", "")
    scene_dir = Path(out_dir) / month_str / product_name

    if scene_dir.exists():
        print(f"  Already downloaded, skipping")
        return scene_dir

    scene_dir.mkdir(parents=True, exist_ok=True)

    granule_name = get_granule_name(token, product_id, product_name)
    date_str = product_name.split("_")[2][:15]
    tile_id = [p for p in product_name.split("_") if p.startswith("T")][0]

    # TCI
    tci_filename = f"{tile_id}_{date_str}_TCI_10m.jp2"
    print(f"  Downloading TCI: {tci_filename}")
    download_file(token,
        f"{BASE}/Products({product_id})"
        f"/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({tci_filename})/$value",
        scene_dir / tci_filename
    )

    # B08 (NIR) - 10m
    b08_filename = f"{tile_id}_{date_str}_B08_10m.jp2"
    print(f"  Downloading B08: {b08_filename}")
    download_file(token,
        f"{BASE}/Products({product_id})"
        f"/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
        f"/Nodes(IMG_DATA)/Nodes(R10m)/Nodes({b08_filename})/$value",
        scene_dir / b08_filename
    )

    # B12 (SWIR) - 20m
    b12_filename = f"{tile_id}_{date_str}_B12_20m.jp2"
    print(f"  Downloading B12: {b12_filename}")
    download_file(token,
        f"{BASE}/Products({product_id})"
        f"/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
        f"/Nodes(IMG_DATA)/Nodes(R20m)/Nodes({b12_filename})/$value",
        scene_dir / b12_filename
    )

    # DETFOO mask
    print(f"  Downloading DETFOO mask")
    download_file(token,
        f"{BASE}/Products({product_id})"
        f"/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
        f"/Nodes(QI_DATA)/Nodes(MSK_DETFOO_B04.jp2)/$value",
        scene_dir / "MSK_DETFOO_B04.jp2"
    )

    # SCL (Scene Classification Layer) for cloud masking - 20m
    scl_filename = f"{tile_id}_{date_str}_SCL_20m.jp2"
    print(f"  Downloading SCL: {scl_filename}")
    download_file(token,
        f"{BASE}/Products({product_id})"
        f"/Nodes({product_name}.SAFE)/Nodes(GRANULE)/Nodes({granule_name})"
        f"/Nodes(IMG_DATA)/Nodes(R20m)/Nodes({scl_filename})/$value",
        scene_dir / scl_filename
    )

    print(f"  Saved to: {scene_dir}")
    return scene_dir

if __name__ == "__main__":
    tile_id = input("MGRS tile ID (e.g. 36RXV): ").strip().upper()
    start_date = input("Start date (YYYY-MM-DD): ").strip()
    end_date = input("End date (YYYY-MM-DD): ").strip()
    max_cloud = float(input("Max cloud cover % (e.g. 20): ").strip())

    token = get_token()
    print("\nSearching...")
    scenes = search_scenes(token, tile_id, f"{start_date}T00:00:00.000Z", f"{end_date}T23:59:59.000Z", max_cloud)
    print(f"Found {len(scenes)} scenes matching criteria")

    monthly = group_by_month(scenes)
    total = sum(len(v) for v in monthly.values())
    print(f"\n{len(monthly)} months, {total} scenes total:\n")
    for month, month_scenes in sorted(monthly.items()):
        print(f"  {month}: {len(month_scenes)} scenes")

    confirm = input(f"\nDownload all {total} scenes? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        exit()

    for month, month_scenes in sorted(monthly.items()):
        print(f"\n--- {month} ({len(month_scenes)} scenes) ---")
        for i, scene in enumerate(month_scenes, 1):
            print(f"[{i}/{len(month_scenes)}] {scene['Name']}")
            download_scene(scene, month)
            print()

    print("All done.")