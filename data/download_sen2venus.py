"""
Download SEN2VENuS patches from Zenodo.

Dataset: https://zenodo.org/records/6514159
~133k paired patches across 29 sites.
Each site is a separate .7z archive.

Usage:
    python data/download_sen2venus.py --list
    python data/download_sen2venus.py --sites ESTUAMAR --out data/raw
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "6514159"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def fetch_file_list() -> list[dict]:
    resp = requests.get(ZENODO_API, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "site": f["key"].replace(".7z", ""),
            "filename": f["key"],
            "url": f["links"]["self"],
            "size": f["size"],
            "checksum": f.get("checksum", ""),
        }
        for f in data["files"]
        if f["key"].endswith(".7z")
    ]


def download_file(url: str, dest: Path, size: int, checksum: str) -> None:
    if dest.exists() and dest.stat().st_size == size:
        print(f"  already downloaded: {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name} ({size / 1e9:.2f} GB)...")

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f, tqdm(
            total=size, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    if checksum:
        algo, expected = checksum.split(":")
        actual = hashlib.new(algo, dest.read_bytes()).hexdigest()
        if actual != expected:
            dest.unlink()
            raise RuntimeError(f"Checksum mismatch for {dest.name}")


def extract_7z(archive: Path, out_dir: Path) -> None:
    print(f"  extracting {archive.name}...")
    result = subprocess.run(
        ["7z", "x", str(archive), f"-o{out_dir}", "-y"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"7z extraction failed:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEN2VENuS patches from Zenodo")
    parser.add_argument("--sites", help="Comma-separated site codes, e.g. ESTUAMAR,JAM2018")
    parser.add_argument("--list", action="store_true", help="List all available sites and exit")
    parser.add_argument("--out", default="data/raw", help="Output directory (default: data/raw)")
    parser.add_argument("--keep-archive", action="store_true", help="Keep .7z after extraction")
    args = parser.parse_args()

    print("Fetching Zenodo record metadata...")
    files = fetch_file_list()
    site_map = {f["site"]: f for f in files}

    if args.list:
        print(f"\nAvailable sites ({len(site_map)}):")
        for code, info in sorted(site_map.items(), key=lambda x: x[1]["size"]):
            print(f"  {code:<14}  {info['size'] / 1e9:.2f} GB")
        sys.exit(0)

    if not args.sites:
        parser.error("Specify --sites <codes> or --list to see options")

    requested = [s.strip().upper() for s in args.sites.split(",")]
    missing = [s for s in requested if s not in site_map]
    if missing:
        print(f"Unknown site codes: {missing}. Run --list to see valid codes.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for code in requested:
        info = site_map[code]
        archive_path = out_dir / info["filename"]
        print(f"\n[{code}]")
        download_file(info["url"], archive_path, info["size"], info["checksum"])
        extract_7z(archive_path, out_dir)
        if not args.keep_archive:
            archive_path.unlink()
            print(f"  removed {archive_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
