#!/usr/bin/env python3
"""
Simple Dataset Downloader - Run with nohup or screen
Usage: nohup python scripts/download_datasets_simple.py &
"""

import os
import subprocess
from pathlib import Path

# Create datasets folder
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("🚀 DOWNLOADING SKIN DISEASE DATASETS")
print("=" * 50)


# ============================================================================
# 1. PAD-UFES-20 (from Kaggle)
# ============================================================================
print("\n📥 [1] PAD-UFES-20 (Kaggle)")
print("-" * 40)

# First, make sure kaggle is installed and configured
# Run: pip install kaggle
# Then: mkdir -p ~/.kaggle && echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json

try:
    import kaggle
    pad_dir = DATASETS_DIR / "pad_ufes_20"
    if not pad_dir.exists():
        print("Downloading PAD-UFES-20...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "mahdavi1202/skin-cancer",
            "-p", str(pad_dir),
            "--unzip"
        ], check=True)
        print("✅ PAD-UFES-20 downloaded!")
    else:
        print("✅ Already exists")
except ImportError:
    print("⚠️ Install kaggle: pip install kaggle")
except Exception as e:
    print(f"⚠️ Error: {e}")


# ============================================================================
# 2. ISIC 2019 (from Kaggle) 
# ============================================================================
print("\n📥 [2] ISIC 2019 (Kaggle)")
print("-" * 40)

try:
    isic_dir = DATASETS_DIR / "isic_2019"
    if not isic_dir.exists():
        print("Downloading ISIC 2019...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "andrewmvd/isic-2019",
            "-p", str(isic_dir),
            "--unzip"
        ], check=True)
        print("✅ ISIC 2019 downloaded!")
    else:
        print("✅ Already exists")
except Exception as e:
    print(f"⚠️ Error: {e}")


# ============================================================================
# 3. HAM10000 (from Kaggle)
# ============================================================================
print("\n📥 [3] HAM10000 (Kaggle)")
print("-" * 40)

try:
    ham_dir = DATASETS_DIR / "ham10000"
    if not ham_dir.exists():
        print("Downloading HAM10000...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "kmader/skin-cancer-mnist-ham10000",
            "-p", str(ham_dir),
            "--unzip"
        ], check=True)
        print("✅ HAM10000 downloaded!")
    else:
        print("✅ Already exists")
except Exception as e:
    print(f"⚠️ Error: {e}")


# ============================================================================
# 4. DermNet (from Kaggle)
# ============================================================================
print("\n📥 [4] DermNet (Kaggle)")
print("-" * 40)

try:
    dermnet_dir = DATASETS_DIR / "dermnet"
    if not dermnet_dir.exists():
        print("Downloading DermNet...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "shubhamgoel27/dermnet",
            "-p", str(dermnet_dir),
            "--unzip"
        ], check=True)
        print("✅ DermNet downloaded!")
    else:
        print("✅ Already exists")
except Exception as e:
    print(f"⚠️ Error: {e}")


# ============================================================================
# 5. Fitzpatrick17k (GitHub + download images)
# ============================================================================
print("\n📥 [5] Fitzpatrick17k")
print("-" * 40)

import urllib.request
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

fitz_dir = DATASETS_DIR / "fitzpatrick17k"
fitz_dir.mkdir(exist_ok=True)
images_dir = fitz_dir / "images"
images_dir.mkdir(exist_ok=True)

csv_url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
csv_path = fitz_dir / "fitzpatrick17k.csv"

# Download CSV
if not csv_path.exists():
    print("Downloading Fitzpatrick17k metadata...")
    urllib.request.urlretrieve(csv_url, csv_path)
    print("✅ Metadata downloaded")

# Download images
if csv_path.exists():
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    def download_image(row):
        md5 = row.get('md5hash', '')
        url = row.get('url', '')
        if not md5 or not url:
            return False
        
        img_path = images_dir / f"{md5}.jpg"
        if img_path.exists():
            return True
        
        try:
            urllib.request.urlretrieve(url, img_path)
            return True
        except:
            return False
    
    existing = len(list(images_dir.glob("*.jpg")))
    if existing < len(rows) * 0.9:  # Download if <90% exists
        print(f"Downloading images ({existing}/{len(rows)} exist)...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(download_image, rows), total=len(rows)))
        success = sum(results)
        print(f"✅ Downloaded {success}/{len(rows)} images")
    else:
        print(f"✅ Already have {existing} images")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 50)
print("📊 DOWNLOAD SUMMARY")
print("=" * 50)

for d in DATASETS_DIR.iterdir():
    if d.is_dir():
        count = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
        print(f"  {d.name}: {count} images")

print("\n✅ ALL DONE!")
print("Next: python scripts/prepare_unified_v3.py")
