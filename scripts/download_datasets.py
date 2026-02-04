#!/usr/bin/env python3
"""
Comprehensive Skin Disease Dataset Downloader
Downloads and organizes datasets for training on A100 40GB

Target: 15+ diseases with sufficient samples per class

Datasets:
1. ISIC 2019 Challenge (~25k dermoscopic images, 8 classes)
2. HAM10000 (~10k dermoscopic images, 7 classes)  
3. DermNet (Already present - 23 classes)
4. Fitzpatrick17k (~17k clinical images, 114 labels)
5. PAD-UFES-20 (~2.3k, 6 classes - diverse skin tones)
6. SD-198 (Skin Disease 198 - academic)
7. SCIN Dataset (Google - diverse skin conditions)

Note: Some datasets require manual download due to licensing.
This script handles what can be automated and provides instructions for the rest.
"""

import os
import sys
import subprocess
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Optional
import urllib.request
import json

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
DOWNLOAD_DIR = DATASETS_DIR / "downloads"

# Create directories
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: str, cwd: Optional[Path] = None) -> bool:
    """Run a shell command and return success status."""
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False
    return True

def check_kaggle_api():
    """Check if Kaggle API is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n" + "="*60)
        print("[WARNING]  KAGGLE API NOT CONFIGURED")
        print("="*60)
        print("""
To download datasets from Kaggle, you need to:

1. Go to https://www.kaggle.com/settings
2. Click 'Create New API Token' 
3. This downloads kaggle.json
4. Move it to ~/.kaggle/kaggle.json
5. Run: chmod 600 ~/.kaggle/kaggle.json

After setup, run this script again.
""")
        return False
    return True

def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "kaggle", "gdown", "tqdm", "requests"], check=True)
    print("  [OK] Dependencies installed")

def download_kaggle_dataset(dataset_slug: str, target_dir: Path, dataset_name: str):
    """Download a dataset from Kaggle."""
    print(f"\n📥 Downloading {dataset_name} from Kaggle...")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if any(target_dir.iterdir()):
        print(f"  [SKIP]  {dataset_name} already exists, skipping...")
        return True
    
    cmd = f"kaggle datasets download -d {dataset_slug} -p {target_dir} --unzip"
    if run_cmd(cmd):
        print(f"  [OK] {dataset_name} downloaded successfully")
        return True
    return False

def download_isic_2019():
    """Download ISIC 2019 Challenge dataset."""
    print("\n" + "="*60)
    print("📥 ISIC 2019 CHALLENGE DATASET")
    print("="*60)
    
    isic_dir = DATASETS_DIR / "isic_data" / "isic_2019"
    
    if (isic_dir / "ISIC_2019_Training_Input").exists():
        print("  [SKIP]  ISIC 2019 already exists")
        return True
    
    # ISIC 2019 is available on Kaggle
    return download_kaggle_dataset(
        "andrewmvd/isic-2019",
        isic_dir,
        "ISIC 2019"
    )

def download_ham10000():
    """Download HAM10000 dataset."""
    print("\n" + "="*60)
    print("📥 HAM10000 DATASET")
    print("="*60)
    
    ham_dir = DATASETS_DIR / "ham10000"
    
    if ham_dir.exists() and any(ham_dir.iterdir()):
        print("  [SKIP]  HAM10000 already exists")
        return True
    
    return download_kaggle_dataset(
        "kmader/skin-cancer-mnist-ham10000",
        ham_dir,
        "HAM10000"
    )

def download_fitzpatrick17k():
    """Download Fitzpatrick17k dataset."""
    print("\n" + "="*60)
    print("📥 FITZPATRICK17K DATASET")
    print("="*60)
    
    fitz_dir = DATASETS_DIR / "data"
    csv_path = fitz_dir / "fitzpatrick17k.csv"
    img_dir = fitz_dir / "finalfitz17k"
    
    if csv_path.exists() and img_dir.exists():
        print("  [SKIP]  Fitzpatrick17k already exists")
        return True
    
    fitz_dir.mkdir(parents=True, exist_ok=True)
    
    print("""
  [WARNING]  Fitzpatrick17k requires manual download:
  
  1. Go to: https://github.com/mattgroh/fitzpatrick17k
  2. Follow instructions to download the dataset
  3. Place CSV at: datasets/data/fitzpatrick17k.csv
  4. Place images at: datasets/data/finalfitz17k/
  
  Alternatively, use the Kaggle version:
  kaggle datasets download -d mmaximillian/fitzpatrick17k-images
    """)
    
    # Try Kaggle version
    return download_kaggle_dataset(
        "mmaximillian/fitzpatrick17k-images",
        fitz_dir,
        "Fitzpatrick17k"
    )

def download_pad_ufes():
    """Download PAD-UFES-20 dataset (diverse skin tones)."""
    print("\n" + "="*60)
    print("📥 PAD-UFES-20 DATASET (Diverse Skin Tones)")
    print("="*60)
    
    pad_dir = DATASETS_DIR / "pad_ufes_20"
    
    if pad_dir.exists() and any(pad_dir.iterdir()):
        print("  [SKIP]  PAD-UFES-20 already exists")
        return True
    
    return download_kaggle_dataset(
        "mahdavi1202/pad-ufes-20",
        pad_dir,
        "PAD-UFES-20"
    )

def download_dermnet():
    """Download DermNet dataset."""
    print("\n" + "="*60)
    print("📥 DERMNET DATASET")
    print("="*60)
    
    dermnet_dir = DATASETS_DIR / "dermnet_main"
    
    if dermnet_dir.exists() and (dermnet_dir / "train").exists():
        print("  [SKIP]  DermNet already exists")
        return True
    
    return download_kaggle_dataset(
        "shubhamgoel27/dermnet",
        dermnet_dir,
        "DermNet"
    )

def download_skin_cancer_isic():
    """Download additional ISIC skin cancer images."""
    print("\n" + "="*60)
    print("📥 SKIN CANCER ISIC (Extended)")
    print("="*60)
    
    isic_ext_dir = DATASETS_DIR / "skin_cancer_isic_extended"
    
    if isic_ext_dir.exists() and any(isic_ext_dir.iterdir()):
        print("  [SKIP]  Skin Cancer ISIC Extended already exists")
        return True
    
    return download_kaggle_dataset(
        "nodoubttome/skin-cancer9-classesisic",
        isic_ext_dir,
        "Skin Cancer ISIC 9 Classes"
    )

def download_massive_skin_dataset():
    """Download Massive Skin Disease Balanced Dataset (262K images)."""
    print("\n" + "="*60)
    print("📥 MASSIVE SKIN DISEASE BALANCED DATASET (262K+ images)")
    print("="*60)
    
    massive_dir = DATASETS_DIR / "massive_skin_disease"
    
    if massive_dir.exists() and any(massive_dir.iterdir()):
        print("  [SKIP]  Massive Skin Disease dataset already exists")
        return True
    
    print("""
  [WARNING]  This is a LARGE dataset (~50GB+)
  
  Options:
  1. Download from Kaggle (requires Kaggle API):
     kaggle datasets download -d kylegraupe/skin-disease-balanced-dataset
     
  2. Or download subset for specific diseases you need
    """)
    
    return download_kaggle_dataset(
        "kylegraupe/skin-disease-balanced-dataset",
        massive_dir,
        "Massive Skin Disease Balanced"
    )

def download_scin_dataset():
    """Download Google SCIN dataset."""
    print("\n" + "="*60)
    print("📥 GOOGLE SCIN DATASET (Diverse Conditions)")
    print("="*60)
    
    scin_dir = DATASETS_DIR / "scin"
    
    print("""
  📋 SCIN Dataset requires manual download:
  
  1. Go to: https://github.com/google-research/scin-dataset
  2. Follow the download instructions
  3. Accept the data use agreement
  4. Place files in: datasets/scin/
  
  This dataset provides diverse skin conditions and skin tones.
    """)
    
    return False  # Manual download required

def create_dataset_status_report():
    """Create a status report of all datasets."""
    print("\n" + "="*60)
    print("📊 DATASET STATUS REPORT")
    print("="*60)
    
    datasets = {
        "ISIC 2019": DATASETS_DIR / "isic_data" / "isic_2019",
        "HAM10000": DATASETS_DIR / "ham10000",
        "DermNet": DATASETS_DIR / "dermnet_main",
        "Fitzpatrick17k": DATASETS_DIR / "data" / "finalfitz17k",
        "PAD-UFES-20": DATASETS_DIR / "pad_ufes_20",
        "Skin Cancer ISIC Ext": DATASETS_DIR / "skin_cancer_isic_extended",
        "Massive Skin Disease": DATASETS_DIR / "massive_skin_disease",
        "SCIN": DATASETS_DIR / "scin",
        "Diverse Derm": DATASETS_DIR / "diverse_derm",
        "Skin Disease Dataset": DATASETS_DIR / "skin_disease",
    }
    
    status_report = {}
    total_images = 0
    
    for name, path in datasets.items():
        if path.exists():
            # Count images
            count = 0
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                count += len(list(path.rglob(ext)))
            
            status = "[OK] Ready" if count > 0 else "[WARNING] Empty"
            status_report[name] = {"path": str(path), "images": count, "status": status}
            total_images += count
            print(f"  {status} {name}: {count:,} images")
        else:
            status_report[name] = {"path": str(path), "images": 0, "status": "[ERROR] Missing"}
            print(f"  [ERROR] {name}: Not downloaded")
    
    print(f"\n  📈 Total images available: {total_images:,}")
    
    # Save report
    report_path = DATASETS_DIR / "dataset_status.json"
    with open(report_path, "w") as f:
        json.dump(status_report, f, indent=2)
    print(f"\n  📄 Report saved to: {report_path}")
    
    return status_report

def print_a100_training_recommendations():
    """Print recommendations for A100 40GB training."""
    print("\n" + "="*60)
    print(" A100 40GB TRAINING RECOMMENDATIONS")
    print("="*60)
    print("""
With A100 40GB, you can use:

┌─────────────────────────────────────────────────────────┐
│ RECOMMENDED CONFIGURATION                               │
├─────────────────────────────────────────────────────────┤
│ Model:          Swin-Base or ConvNeXt-Large            │
│ Batch Size:     64-128 (with mixed precision)           │
│ Image Size:     384x384 or 448x448                      │
│ Optimizer:      AdamW with cosine LR                    │
│ Epochs:         50-100                                  │
│ Mixed Precision: [OK] Enable (bf16/fp16)                   │
│ Gradient Accum:  2-4 steps if needed                    │
└─────────────────────────────────────────────────────────┘

Expected Training Time:
- 100K images @ 384x384: ~4-6 hours
- 250K images @ 384x384: ~10-15 hours

Target Metrics:
- Category Accuracy: >90%
- Disease Accuracy: >85% (15+ diseases)
- Sensitivity for cancers: >95%
    """)

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     SKIN DISEASE DATASET DOWNLOADER                          ║
║     Target: 15+ diseases for A100 40GB training              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Install dependencies
    install_dependencies()
    
    # Check Kaggle API
    has_kaggle = check_kaggle_api()
    
    if has_kaggle:
        # Download datasets
        download_isic_2019()
        download_ham10000()
        download_dermnet()
        download_fitzpatrick17k()
        download_pad_ufes()
        download_skin_cancer_isic()
        
        # Large dataset - confirm before downloading
        print("\n" + "="*60)
        print("[WARNING]  OPTIONAL: Download Massive Skin Disease Dataset (50GB+)?")
        print("="*60)
        response = input("Download massive dataset? [y/N]: ").strip().lower()
        if response == 'y':
            download_massive_skin_dataset()
    
    # Always show manual download instructions
    download_scin_dataset()
    
    # Create status report
    create_dataset_status_report()
    
    # Show A100 recommendations
    print_a100_training_recommendations()
    
    print("\n[OK] Dataset setup complete!")
    print("   Next step: Run 'python scripts/prepare_unified_dataset.py' to unify datasets")

if __name__ == "__main__":
    main()
