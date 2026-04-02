"""
Download the nuScenes mini dataset for demos.

The mini split contains 10 scenes (~4GB) with full sensor data:
    - 6 cameras (360° coverage)
    - 1 LiDAR (32-beam)
    - 5 RADAR
    - Full 3D annotations
    - HD maps

NOTE: nuScenes requires free registration at https://www.nuscenes.org/
This script provides instructions and attempts automatic download.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


NUSCENES_MINI_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
NUSCENES_MINI_SIZE = "~4GB"


def check_existing(dataroot: Path) -> bool:
    """Check if nuScenes mini is already downloaded."""
    expected_dirs = ["maps", "samples", "sweeps", "v1.0-mini"]
    if all((dataroot / d).exists() for d in expected_dirs):
        print(f"✓ nuScenes mini already exists at {dataroot}")
        return True
    return False


def download_with_wget(url: str, output: str):
    """Download using wget."""
    cmd = ["wget", "-c", "--no-check-certificate", url, "-O", output]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def extract_tgz(filepath: str, dest: str):
    """Extract tar.gz file."""
    cmd = ["tar", "-xzf", filepath, "-C", dest]
    print(f"Extracting to {dest}...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download nuScenes mini dataset")
    parser.add_argument(
        "--dataroot", type=str, default="./data/nuscenes",
        help="Where to store the dataset",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip download, only verify")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)

    print("=" * 60)
    print("  nuScenes Mini Dataset Downloader")
    print(f"  Target: {dataroot.resolve()}")
    print(f"  Size: {NUSCENES_MINI_SIZE}")
    print("=" * 60)

    # Check if already exists
    if check_existing(dataroot):
        print("\nDataset ready! You can run demos now.")
        return

    if args.skip_download:
        print("\n⚠ Dataset not found. Download manually from:")
        print(f"  {NUSCENES_MINI_URL}")
        print(f"\nThen extract to: {dataroot}")
        sys.exit(1)

    # Create directory
    dataroot.mkdir(parents=True, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  nuScenes requires free registration to download.       ║
║                                                         ║
║  Option 1: Register at https://www.nuscenes.org/        ║
║            Download v1.0-mini.tgz manually              ║
║            Extract to {str(dataroot):<36s}    ║
║                                                         ║
║  Option 2: If you have the AWS CLI configured with      ║
║            nuScenes credentials, this script will       ║
║            attempt automatic download.                  ║
╚══════════════════════════════════════════════════════════╝
""")

    # Attempt download
    tgz_path = str(dataroot / "v1.0-mini.tgz")

    try:
        print("Attempting download...")
        download_with_wget(NUSCENES_MINI_URL, tgz_path)
        print("\nDownload complete! Extracting...")
        extract_tgz(tgz_path, str(dataroot))
        os.remove(tgz_path)
        print(f"✓ Dataset extracted to {dataroot}")

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠ Automatic download failed (likely needs registration).")
        print("\nManual steps:")
        print("  1. Go to https://www.nuscenes.org/nuscenes#download")
        print("  2. Register for a free account")
        print("  3. Download 'Mini' split (v1.0-mini.tgz)")
        print(f"  4. Extract to: {dataroot.resolve()}")
        print("\nExpected structure:")
        print(f"  {dataroot}/")
        print(f"    ├── maps/")
        print(f"    ├── samples/")
        print(f"    │   ├── CAM_FRONT/")
        print(f"    │   ├── LIDAR_TOP/")
        print(f"    │   └── ...")
        print(f"    ├── sweeps/")
        print(f"    └── v1.0-mini/")
        print(f"        ├── sample.json")
        print(f"        ├── scene.json")
        print(f"        └── ...")

    # Verify
    if check_existing(dataroot):
        print("\n✓ Dataset ready!")
        # Quick stats
        try:
            from nuscenes.nuscenes import NuScenes
            nusc = NuScenes(version="v1.0-mini", dataroot=str(dataroot), verbose=False)
            print(f"  Scenes: {len(nusc.scene)}")
            print(f"  Samples: {len(nusc.sample)}")
            print(f"  Annotations: {len(nusc.sample_annotation)}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
