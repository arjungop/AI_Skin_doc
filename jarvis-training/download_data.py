#!/usr/bin/env python3
"""
Dataset Downloader for Skin Disease Classification
Downloads ALL 6 datasets from Kaggle for maximum training data.

Datasets:
  1. ISIC 2019          (~25k images, 8 classes)
  2. HAM10000           (~10k images, 7 classes)
  3. DermNet            (~20k images, 23 classes)
  4. PAD-UFES-20        (~2.3k images, 6 classes)
  5. Fitzpatrick17k     (~17k images, 114 raw classes)
  6. Massive 2 Balanced (~262k images, 34 classes)  [LARGE ~50GB]

Usage:
  python download_data.py                    # Download all datasets
  python download_data.py --skip-massive     # Skip the 50GB Massive dataset
  python download_data.py --datasets-dir /path/to/dir

Prerequisites:
  - Kaggle API configured: ~/.kaggle/kaggle.json
  - pip install kaggle
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


# ============================================================================
# DATASET REGISTRY
# ============================================================================
DATASETS = {
    "isic_2019": {
        "slug": "andrewmvd/isic-2019",
        "description": "ISIC 2019 Challenge (~25k images, 8 skin cancer classes)",
        "size_hint": "~5 GB",
        "priority": 1,
    },
    "ham10000": {
        "slug": "kmader/skin-cancer-mnist-ham10000",
        "description": "HAM10000 (~10k dermoscopic images, 7 classes)",
        "size_hint": "~3 GB",
        "priority": 2,
    },
    "dermnet": {
        "slug": "shubhamgoel27/dermnet",
        "description": "DermNet (~20k images, 23 disease categories)",
        "size_hint": "~2 GB",
        "priority": 3,
    },
    "pad_ufes_20": {
        "slug": "mahdavi1202/skin-cancer",
        "description": "PAD-UFES-20 (~2.3k images, 6 classes)",
        "size_hint": "~2 GB",
        "priority": 4,
    },
    "fitzpatrick17k": {
        "slug": "mmaximillian/fitzpatrick17k-images",
        "description": "Fitzpatrick17k (~17k images, diverse skin tones)",
        "size_hint": "~5 GB",
        "priority": 5,
    },
    "massive_balanced": {
        "slug": "muhammadabdulsami/massive-skin-disease-balanced-dataset",
        "description": "Massive Balanced Dataset (~262k images, 34 classes)",
        "size_hint": "~50 GB",
        "priority": 6,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def check_kaggle_api():
    """Verify Kaggle API is configured and accessible."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("=" * 60)
        print("ERROR: Kaggle API not configured!")
        print("=" * 60)
        print()
        print("To fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Download kaggle.json")
        print("  4. Place it at: ~/.kaggle/kaggle.json")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("On Jarvis Labs:")
        print("  - Upload kaggle.json to the instance")
        print("  - mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/")
        print("  - chmod 600 ~/.kaggle/kaggle.json")
        return False

    # Verify permissions
    mode = oct(kaggle_json.stat().st_mode)[-3:]
    if mode != "600":
        print(f"Fixing kaggle.json permissions (currently {mode})...")
        kaggle_json.chmod(0o600)

    # Test kaggle CLI
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print(f"  Kaggle CLI: {result.stdout.strip()}")
            return True
        else:
            print(f"  Kaggle CLI error: {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
        return True
    except subprocess.TimeoutExpired:
        print("  Kaggle CLI test timed out, proceeding anyway...")
        return True


def download_kaggle_dataset(slug: str, dest_dir: Path, name: str) -> bool:
    """Download and extract a Kaggle dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (has files)
    existing_files = list(dest_dir.rglob("*"))
    existing_images = [f for f in existing_files
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".csv"}]
    if len(existing_images) > 10:
        print(f"  [SKIP] {name} already downloaded ({len(existing_images)} files)")
        return True

    print(f"  Downloading {name}...")
    print(f"    Slug: {slug}")
    print(f"    Dest: {dest_dir}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(dest_dir), "--unzip"],
            capture_output=False,  # Show download progress
            timeout=7200,  # 2 hour timeout for large datasets
        )
        if result.returncode != 0:
            print(f"  [FAIL] Download failed for {name}")
            return False

        # Verify something was downloaded
        post_files = list(dest_dir.rglob("*"))
        if len(post_files) == 0:
            print(f"  [FAIL] No files found after download for {name}")
            return False

        print(f"  [OK] {name}: {len(post_files)} files")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Download timed out for {name}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error downloading {name}: {e}")
        return False


def download_fitzpatrick_csv(dest_dir: Path) -> bool:
    """Download the Fitzpatrick17k label CSV from GitHub."""
    csv_path = dest_dir / "fitzpatrick17k.csv"
    if csv_path.exists():
        print("  [SKIP] Fitzpatrick17k CSV already exists")
        return True

    url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/master/fitzpatrick17k.csv"
    print(f"  Downloading Fitzpatrick17k label CSV...")

    try:
        import requests
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        csv_path.write_text(resp.text)
        print(f"  [OK] Fitzpatrick17k CSV saved ({len(resp.text)} bytes)")
        return True
    except Exception as e:
        print(f"  [WARN] Could not download Fitzpatrick17k CSV: {e}")
        print(f"         Training will still work without Fitzpatrick17k data.")
        return False


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Download skin disease datasets from Kaggle"
    )
    parser.add_argument(
        "--datasets-dir", type=str, default="datasets",
        help="Directory to store downloaded datasets (default: datasets/)"
    )
    parser.add_argument(
        "--skip-massive", action="store_true",
        help="Skip the Massive Balanced dataset (~50GB) to save space/time"
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    datasets_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 65)
    print("  SKIN DISEASE DATASET DOWNLOADER")
    print("  Target: All datasets for 20-class classification")
    print("=" * 65)
    print(f"  Download directory: {datasets_dir}")
    print()

    # Step 1: Check Kaggle API
    print("[1/3] Checking Kaggle API...")
    if not check_kaggle_api():
        print()
        print("Cannot proceed without Kaggle API. See instructions above.")
        sys.exit(1)
    print()

    # Step 2: Download all datasets
    print("[2/3] Downloading datasets...")
    print()

    results = {}
    sorted_datasets = sorted(DATASETS.items(), key=lambda x: x[1]["priority"])

    for name, info in sorted_datasets:
        if name == "massive_balanced" and args.skip_massive:
            print(f"  [SKIP] {info['description']} (--skip-massive flag)")
            results[name] = "skipped"
            continue

        print(f"  {'=' * 55}")
        print(f"  {info['description']}")
        print(f"  Size: {info['size_hint']}")
        print(f"  {'=' * 55}")

        dest = datasets_dir / name
        ok = download_kaggle_dataset(info["slug"], dest, name)
        results[name] = "ok" if ok else "failed"

        # Download Fitzpatrick CSV separately
        if name == "fitzpatrick17k" and ok:
            download_fitzpatrick_csv(dest)

        print()
        time.sleep(2)  # Small delay between downloads

    # Step 3: Summary
    print("[3/3] Download Summary")
    print("=" * 65)
    total_ok = sum(1 for v in results.values() if v == "ok")
    total = len(results)

    for name, status in results.items():
        icon = {"ok": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]"}[status]
        print(f"  {icon:>8}  {name}")

    print()
    print(f"  {total_ok}/{total} datasets downloaded successfully")
    print()

    if total_ok >= 3:
        print("  Next step: python prepare_data.py")
        print()
    else:
        print("  WARNING: Less than 3 datasets available.")
        print("  Training quality may be reduced.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
