#!/usr/bin/env python3
"""
Hierarchical Dataset Preparation (Server-Compatible Version)
Supports datasets downloaded via server_setup.sh

Sources (after running server_setup.sh):
1. ISIC 2019 (~25k) - from datasets/isic_2019/
2. HAM10000 (~10k) - from datasets/ham10000/
3. DermNet (~19k) - from datasets/dermnet/
4. Fitzpatrick17k (~16.5k) - from datasets/fitzpatrick/
5. PAD-UFES-20 (~2.3k) - from datasets/pad_ufes/
6. 20 Skin Diseases (~5k) - from datasets/skin20/
7. Massive Skin Disease (~262k) - from datasets/massive/ (34 classes, balanced)

Legacy paths (local development):
- ISIC 2018, isic_data, dermnet_main, etc.
"""
import os
import csv
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import argparse

# Configuration
DATASETS_DIR = Path("datasets")
UNIFIED_DIR = Path("data/unified_train")
VAL_DIR = Path("data/unified_val")
VAL_SPLIT = 0.1  # 10% for validation

# --- Hierarchical Structure ---
HIERARCHY = {
    "cancer": ["melanoma", "bcc", "scc", "ak"],
    "benign": ["nevus", "seborrheic_keratosis", "angioma", "wart"],
    "inflammatory": ["eczema", "psoriasis", "lichen_planus", "urticaria"],
    "infectious": ["impetigo", "herpes", "candida", "scabies", "tinea"],
    "pigmentary": ["vitiligo", "melasma", "hyperpigmentation"],
}

# Disease to Category mapping
DISEASE_TO_CATEGORY = {}
for cat, diseases in HIERARCHY.items():
    for d in diseases:
        DISEASE_TO_CATEGORY[d] = cat

MAIN_DISEASES = set([d for cat in HIERARCHY.values() for d in cat])

# ============================================================================
# KEYWORD MAPPING for folder-based datasets
# ============================================================================
KEYWORD_TO_DISEASE = [
    # Cancer
    ("melanoma", "melanoma"), ("malignant melanoma", "melanoma"),
    ("basal cell", "bcc"), ("bcc", "bcc"),
    ("squamous cell", "scc"), ("scc", "scc"), ("keratoacanthoma", "scc"),
    ("actinic", "ak"), ("actinic keratosis", "ak"),
    # Benign
    ("nevus", "nevus"), ("nevi", "nevus"), ("mole", "nevus"),
    ("seborrheic keratosis", "seborrheic_keratosis"), ("seborrheic", "seborrheic_keratosis"),
    ("angioma", "angioma"), ("hemangioma", "angioma"), ("vascular", "angioma"),
    ("cherry angioma", "angioma"), ("port wine", "angioma"),
    ("wart", "wart"), ("verruca", "wart"), ("papilloma", "wart"),
    # Inflammatory
    ("eczema", "eczema"), ("dermatitis", "eczema"), ("atopic", "eczema"),
    ("psoriasis", "psoriasis"), ("plaque psoriasis", "psoriasis"),
    ("lichen planus", "lichen_planus"), ("lichen", "lichen_planus"),
    ("urticaria", "urticaria"), ("hives", "urticaria"),
    # Infectious
    ("impetigo", "impetigo"),
    ("herpes", "herpes"), ("shingles", "herpes"), ("zoster", "herpes"),
    ("candida", "candida"), ("candidiasis", "candida"), ("fungal", "candida"),
    ("scabies", "scabies"), ("mite", "scabies"),
    ("tinea", "tinea"), ("ringworm", "tinea"), ("dermatophyte", "tinea"),
    # Pigmentary
    ("vitiligo", "vitiligo"),
    ("melasma", "melasma"),
    ("hyperpigmentation", "hyperpigmentation"), ("hypopigmentation", "hyperpigmentation"),
]

# ============================================================================
# DATASET PROCESSORS
# ============================================================================

class DatasetProcessor:
    """Base class for dataset processing."""
    
    def __init__(self):
        self.counts = Counter()
        self.train_files = []
        self.val_files = []
    
    def get_disease_from_text(self, text: str) -> str:
        """Map text to disease using keywords."""
        text_lower = text.lower()
        for keyword, disease in KEYWORD_TO_DISEASE:
            if keyword in text_lower:
                return disease
        return None
    
    def add_file(self, src: Path, disease: str, prefix: str):
        """Add file to train or val set."""
        if not src.exists() or disease not in MAIN_DISEASES:
            return
        
        import random
        is_val = random.random() < VAL_SPLIT
        
        entry = {
            'src': src,
            'disease': disease,
            'prefix': prefix,
        }
        
        if is_val:
            self.val_files.append(entry)
        else:
            self.train_files.append(entry)
        
        self.counts[disease] += 1


def process_isic_csv(processor: DatasetProcessor, csv_path: Path, img_dir: Path, prefix: str):
    """Process ISIC-style CSV dataset."""
    if not csv_path.exists():
        print(f"  Skipping {prefix}: CSV not found at {csv_path}")
        return
    
    # ISIC column mappings
    cols = {
        'MEL': 'melanoma', 'BCC': 'bcc', 'SCC': 'scc', 'AK': 'ak', 'AKIEC': 'ak',
        'NV': 'nevus', 'BKL': 'seborrheic_keratosis', 'VASC': 'angioma', 'DF': 'angioma'
    }
    
    print(f"  Processing {prefix}...")
    with open(csv_path) as f:
        for row in tqdm(csv.DictReader(f)):
            for col, disease in cols.items():
                try:
                    if float(row.get(col, 0)) == 1.0:
                        img_name = row.get('image', row.get('image_id', ''))
                        
                        # Try different extensions and paths
                        for ext in ['.jpg', '.jpeg', '.png', '']:
                            img_path = img_dir / f"{img_name}{ext}"
                            if img_path.exists():
                                processor.add_file(img_path, disease, prefix)
                                break
                        break
                except (ValueError, KeyError):
                    continue


def process_ham10000(processor: DatasetProcessor, base_dir: Path):
    """Process HAM10000 dataset."""
    csv_path = base_dir / "HAM10000_metadata.csv"
    
    if not csv_path.exists():
        # Try alternate path
        csv_path = base_dir / "hmnist_28_28_RGB.csv"
        if not csv_path.exists():
            print(f"  Skipping HAM10000: metadata not found")
            return
    
    print(f"  Processing HAM10000...")
    
    # Find image directories
    img_dirs = [
        base_dir / "HAM10000_images_part_1",
        base_dir / "HAM10000_images_part_2",
        base_dir,
    ]
    
    dx_mapping = {
        'mel': 'melanoma', 'bcc': 'bcc', 'akiec': 'ak',
        'nv': 'nevus', 'bkl': 'seborrheic_keratosis', 'vasc': 'angioma', 'df': 'angioma'
    }
    
    with open(csv_path) as f:
        for row in tqdm(csv.DictReader(f)):
            dx = row.get('dx', '').lower()
            disease = dx_mapping.get(dx)
            if not disease:
                continue
            
            img_name = row.get('image_id', '')
            for img_dir in img_dirs:
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = img_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        processor.add_file(img_path, disease, "ham")
                        break


def process_folder_dataset(processor: DatasetProcessor, base_dir: Path, prefix: str):
    """Process folder-organized dataset."""
    if not base_dir.exists():
        print(f"  Skipping {prefix}: not found at {base_dir}")
        return
    
    print(f"  Processing {prefix}...")
    
    for root, dirs, files in os.walk(base_dir):
        folder_name = Path(root).name.lower()
        
        # Try to match folder name to disease
        disease = processor.get_disease_from_text(folder_name)
        if not disease:
            # Try parent folder
            disease = processor.get_disease_from_text(Path(root).parent.name.lower())
        
        if disease:
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
                    processor.add_file(Path(root) / f, disease, prefix)


def process_skin20(processor: DatasetProcessor, base_dir: Path):
    """Process 20 Skin Diseases dataset with specific folder mapping."""
    if not base_dir.exists():
        print(f"  Skipping skin20: not found")
        return
    
    print(f"  Processing skin20...")
    
    # Specific mappings for skin20 dataset folders
    folder_mapping = {
        'actinic keratosis': 'ak',
        'basal cell carcinoma': 'bcc',
        'melanoma': 'melanoma',
        'squamous cell carcinoma': 'scc',
        'nevus': 'nevus',
        'benign keratosis': 'seborrheic_keratosis',
        'vascular lesion': 'angioma',
        'warts': 'wart',
        'eczema': 'eczema',
        'psoriasis': 'psoriasis',
        'lichen planus': 'lichen_planus',
        'urticaria': 'urticaria',
        'scabies': 'scabies',
        'herpes': 'herpes',
        'vitiligo': 'vitiligo',
        'melasma': 'melasma',
    }
    
    for train_or_test in ['train', 'test', 'Train', 'Test']:
        train_dir = base_dir / train_or_test
        if not train_dir.exists():
            continue
        
        for folder in train_dir.iterdir():
            if not folder.is_dir():
                continue
            
            folder_lower = folder.name.lower()
            disease = folder_mapping.get(folder_lower)
            
            if not disease:
                disease = processor.get_disease_from_text(folder_lower)
            
            if disease:
                for img in folder.iterdir():
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        processor.add_file(img, disease, "skin20")


def process_pad_ufes(processor: DatasetProcessor, base_dir: Path):
    """Process PAD-UFES-20 dataset."""
    csv_path = base_dir / "metadata.csv"
    img_dir = base_dir / "images"
    
    # Try alternate structure
    if not csv_path.exists():
        for f in base_dir.rglob("*.csv"):
            csv_path = f
            img_dir = f.parent / "images" if (f.parent / "images").exists() else f.parent
            break
    
    if not csv_path.exists():
        print(f"  Skipping PAD-UFES: metadata not found")
        return
    
    print(f"  Processing PAD-UFES...")
    
    # PAD-UFES diagnosis mapping
    dx_mapping = {
        'mel': 'melanoma', 'bcc': 'bcc', 'scc': 'scc', 'ack': 'ak',
        'nev': 'nevus', 'sek': 'seborrheic_keratosis',
    }
    
    with open(csv_path) as f:
        for row in tqdm(csv.DictReader(f)):
            dx = row.get('diagnostic', row.get('diagnosis', '')).lower()[:3]
            disease = dx_mapping.get(dx)
            if disease:
                img_name = row.get('img_id', row.get('image_id', ''))
                for ext in ['.png', '.jpg', '.jpeg']:
                    img_path = img_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        processor.add_file(img_path, disease, "pad")
                        break


def create_symlinks(processor: DatasetProcessor, output_dir: Path, files: list, name: str):
    """Create symlinks for processed files."""
    print(f"  Creating {name} set ({len(files)} files)...")
    
    for entry in tqdm(files):
        src = entry['src']
        disease = entry['disease']
        prefix = entry['prefix']
        
        dest_dir = output_dir / disease
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest = dest_dir / f"{prefix}_{src.name}"
        
        try:
            if dest.exists():
                dest.unlink()
            os.symlink(src.resolve(), dest)
        except Exception as e:
            pass


def save_hierarchy(output_dir: Path):
    """Save hierarchy metadata."""
    meta = {
        "hierarchy": HIERARCHY,
        "categories": list(HIERARCHY.keys()),
        "diseases": list(MAIN_DISEASES),
        "disease_to_category": DISEASE_TO_CATEGORY,
    }
    
    with open(output_dir / "hierarchy.json", 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare unified skin disease dataset")
    parser.add_argument("--datasets-dir", type=str, default="datasets", help="Datasets directory")
    parser.add_argument("--output-dir", type=str, default="data/unified_train", help="Output directory")
    parser.add_argument("--val-dir", type=str, default="data/unified_val", help="Validation directory")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()
    
    global DATASETS_DIR, UNIFIED_DIR, VAL_DIR, VAL_SPLIT
    DATASETS_DIR = Path(args.datasets_dir)
    UNIFIED_DIR = Path(args.output_dir)
    VAL_DIR = Path(args.val_dir)
    VAL_SPLIT = args.val_split
    
    print("="*60)
    print("SKIN DISEASE DATASET PREPARATION")
    print("="*60)
    print(f"Datasets dir: {DATASETS_DIR}")
    print(f"Output dir: {UNIFIED_DIR}")
    print(f"Val dir: {VAL_DIR}")
    print(f"Val split: {VAL_SPLIT}")
    print()
    
    # Clean output directories
    print("Cleaning output directories...")
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    if VAL_DIR.exists():
        shutil.rmtree(VAL_DIR)
    
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process datasets
    processor = DatasetProcessor()
    
    print("\nProcessing datasets...")
    
    # SERVER PATHS (primary, from server_setup.sh)
    process_isic_csv(processor, 
                    DATASETS_DIR / "isic_2019/ISIC_2019_Training_GroundTruth.csv",
                    DATASETS_DIR / "isic_2019/ISIC_2019_Training_Input",
                    "isic19")
    
    process_ham10000(processor, DATASETS_DIR / "ham10000")
    process_folder_dataset(processor, DATASETS_DIR / "dermnet/train", "dermnet")
    process_skin20(processor, DATASETS_DIR / "skin20")
    process_pad_ufes(processor, DATASETS_DIR / "pad_ufes")
    
    # LEGACY LOCAL PATHS (for backward compatibility)
    process_isic_csv(processor,
                    DATASETS_DIR / "isic_data/isic_2019/ISIC_2019_Training_GroundTruth.csv",
                    DATASETS_DIR / "isic_data/isic_2019/ISIC_2019_Training_Input",
                    "isic19_local")
    
    process_isic_csv(processor,
                    DATASETS_DIR / "ISIC2018_Task3_Training_Input/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv",
                    DATASETS_DIR / "ISIC2018_Task3_Training_Input",
                    "isic18")
    
    process_folder_dataset(processor, DATASETS_DIR / "dermnet_main/train", "dermnet_local")
    process_folder_dataset(processor, DATASETS_DIR / "diverse_derm", "diverse")
    
    # Fitzpatrick (check multiple paths)
    fitz_paths = [
        (DATASETS_DIR / "fitzpatrick", DATASETS_DIR / "fitzpatrick"),
        (DATASETS_DIR / "data/fitzpatrick17k.csv", DATASETS_DIR / "data/finalfitz17k"),
    ]
    for csv_path, img_dir in fitz_paths:
        if csv_path.exists():
            # Process Fitzpatrick specially
            process_folder_dataset(processor, img_dir, "fitz")
            break
    
    # MASSIVE SKIN DISEASE DATASET (262K images, 34 classes, balanced)
    # This is our largest dataset with excellent coverage
    massive_dir = DATASETS_DIR / "massive"
    if massive_dir.exists():
        print("  Processing Massive dataset (this may take a while)...")
        process_folder_dataset(processor, massive_dir, "massive")
    
    # Create symlinks
    print("\nCreating datasets...")
    create_symlinks(processor, UNIFIED_DIR, processor.train_files, "train")
    create_symlinks(processor, VAL_DIR, processor.val_files, "val")
    
    # Save hierarchy
    save_hierarchy(UNIFIED_DIR)
    save_hierarchy(VAL_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {len(processor.train_files) + len(processor.val_files)}")
    print(f"Train samples: {len(processor.train_files)}")
    print(f"Val samples: {len(processor.val_files)}")
    
    print("\nPer-disease counts:")
    for disease in sorted(MAIN_DISEASES):
        count = processor.counts.get(disease, 0)
        status = "✅" if count >= 500 else "⚠️" if count >= 100 else "❌"
        print(f"  {status} {disease}: {count}")
    
    print(f"\nFiles saved to:")
    print(f"  Train: {UNIFIED_DIR}")
    print(f"  Val: {VAL_DIR}")


if __name__ == "__main__":
    main()
