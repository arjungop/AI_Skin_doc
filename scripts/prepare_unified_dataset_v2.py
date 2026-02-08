#!/usr/bin/env python3
"""
Unified dataset preparation v2 - Intelligent Stratified Splitting
Supports: Train (80%), Val (10%), Test (10%)
"""
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random
import json
from collections import defaultdict

DATASETS_DIR = Path("datasets")
OUTPUT_TRAIN = Path("data/unified_train")
OUTPUT_VAL = Path("data/unified_val")
OUTPUT_TEST = Path("data/unified_test")

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Disease label mapping
DISEASE_MAP = {
    # ISIC 2019
    'MEL': 'melanoma',
    'NV': 'nevus',
    'BCC': 'bcc',
    'AK': 'ak',
    'BKL': 'seborrheic_keratosis',
    'DF': 'dermatofibroma',
    'VASC': 'angioma',
    'SCC': 'scc',
    
    # HAM10000
    'mel': 'melanoma',
    'nv': 'nevus',
    'bcc': 'bcc',
    'akiec': 'ak',
    'bkl': 'seborrheic_keratosis',
    'df': 'dermatofibroma',
    'vasc': 'angioma',
    
    # DermNet - normalize folder names
    'Acne and Rosacea Photos': 'acne',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': 'ak',
    'Atopic Dermatitis Photos': 'eczema',
    'Bullous Disease Photos': 'bullous',
    'Cellulitis Impetigo and other Bacterial Infections': 'impetigo',
    'Eczema Photos': 'eczema',
    'Exanthems and Drug Eruptions': 'drug_eruption',
    'Hair Loss Photos Alopecia and other Hair Diseases': 'alopecia',
    'Herpes HPV and other STDs Photos': 'herpes',
    'Light Diseases and Disorders of Pigmentation': 'hyperpigmentation',
    'Lupus and other Connective Tissue diseases': 'lupus',
    'Melanoma Skin Cancer Nevi and Moles': 'melanoma',
    'Nail Fungus and other Nail Disease': 'nail_fungus',
    'Poison Ivy Photos and other Contact Dermatitis': 'dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases': 'psoriasis',
    'Scabies Lyme Disease and other Infestations and Bites': 'scabies',
    'Seborrheic Keratoses and other Benign Tumors': 'seborrheic_keratosis',
    'Systemic Disease': 'systemic',
    'Tinea Ringworm Candidiasis and other Fungal Infections': 'candida',
    'Urticaria Hives': 'urticaria',
    'Vascular Tumors': 'angioma',
    'Vasculitis Photos': 'vasculitis',
    'Warts Molluscum and other Viral Infections': 'viral',
    
    # Massive 2 Dataset Mapping
    'Acne And Rosacea Photos': 'acne',
    'Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions': 'ak',
    'Atopic Dermatitis Photos': 'eczema',
    'Ba  Cellulitis': 'impetigo',
    'Ba Impetigo': 'impetigo',
    'Benign': 'benign',
    'Bullous Disease Photos': 'bullous',
    'Cellulitis Impetigo And Other Bacterial Infections': 'impetigo',
    'Eczema Photos': 'eczema',
    'Exanthems And Drug Eruptions': 'drug_eruption',
    'Fu Athlete Foot': 'candida',
    'Fu Nail Fungus': 'candida',
    'Fu Ringworm': 'candida',
    'Hair Loss Photos Alopecia And Other Hair Diseases': 'alopecia',
    'Heathy': 'healthy',
    'Herpes Hpv And Other Stds Photos': 'herpes',
    'Light Diseases And Disorders Of Pigmentation': 'hyperpigmentation',
    'Lupus And Other Connective Tissue Diseases': 'lupus',
    'Malignant': 'malignant',
    'Melanoma Skin Cancer Nevi And Moles': 'melanoma',
    'Nail Fungus And Other Nail Disease': 'candida',
    'Pa Cutaneous Larva Migrans': 'scabies',
    'Poison Ivy Photos And Other Contact Dermatitis': 'dermatitis',
    'Psoriasis Pictures Lichen Planus And Related Diseases': 'psoriasis',
    'Rashes': 'rash',
    'Scabies Lyme Disease And Other Infestations And Bites': 'scabies',
    'Seborrheic Keratoses And Other Benign Tumors': 'seborrheic_keratosis',
    'Systemic Disease': 'systemic',
    'Tinea Ringworm Candidiasis And Other Fungal Infections': 'candida',
    'Urticaria Hives': 'urticaria',
    'Vascular Tumors': 'angioma',
    'Vasculitis Photos': 'vasculitis',
    'Vi Chickenpox': 'herpes',
    'Vi Shingles': 'herpes',
    'Warts Molluscum And Other Viral Infections': 'wart',
}

def process_massive2():
    """Process the Massive 2 Dataset (262k images)"""
    print("  Processing Massive 2 dataset...")
    massive_paths = [
        Path("massive_skin_disease/balanced_dataset/balanced_dataset"),
        Path("massive_skin_disease"),
        Path("massive 2/balanced_dataset/balanced_dataset"),
        DATASETS_DIR / "massive_skin_disease",
        DATASETS_DIR / "massive 2" / "balanced_dataset" / "balanced_dataset",
    ]
    
    source_dir = None
    for p in massive_paths:
        if p.exists() and any(p.iterdir()):
            source_dir = p
            break
        
    if not source_dir:
        print("  Skipping Massive 2: Directory not found")
        return []

    # Check for subdirectories if the top-level was found
    if source_dir.name == "massive_skin_disease" and (source_dir / "balanced_dataset").exists():
        source_dir = source_dir / "balanced_dataset" / "balanced_dataset"
        
    samples = []
    for disease_folder in source_dir.iterdir():
        if not disease_folder.is_dir(): continue
        label = DISEASE_MAP.get(disease_folder.name)
        if label:
            for img_path in disease_folder.glob("*.jpg"):
                samples.append((img_path, label))
    print(f"    Found {len(samples)} images from Massive 2")
    return samples

def process_isic2019():
    """Process ISIC 2019"""
    print("  Processing ISIC 2019...")
    
    # Try common cluster paths
    csv_paths = [
        DATASETS_DIR / "isic_data" / "isic_2019" / "ISIC_2019_Training_GroundTruth.csv",
        Path("isic_data/isic_2019/ISIC_2019_Training_GroundTruth.csv"),
        Path("isic_data/ISIC_2019_Training_GroundTruth.csv")
    ]
    
    csv_path = next((p for p in csv_paths if p.exists()), None)
    if not csv_path:
        print("    ISIC 2019 CSV not found.")
        return []

    images_dirs = [
        csv_path.parent / "ISIC_2019_Training_Input",
        csv_path.parent / "images"
    ]
    images_dir = next((d for d in images_dirs if d.exists()), None)
    if not images_dir:
        print("    ISIC 2019 Image directory not found.")
        return []

    df = pd.read_csv(csv_path)
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        img_name = row['image']
        # Find active class
        disease = None
        for col in df.columns:
            if col != 'image' and row[col] == 1.0:
                disease = DISEASE_MAP.get(col, col.lower())
                break
        
        if disease:
            img_path = images_dir / f"{img_name}.jpg"
            if img_path.exists(): 
                samples.append((img_path, disease))
    print(f"    Found {len(samples)} images")
    return samples

def process_ham10000():
    """Process HAM10000"""
    print("  Processing HAM10000...")
    csv_path = DATASETS_DIR / "ham10000" / "HAM10000_metadata.csv"
    img_dir1 = DATASETS_DIR / "ham10000" / "HAM10000_images_part_1"
    img_dir2 = DATASETS_DIR / "ham10000" / "HAM10000_images_part_2"
    if not csv_path.exists(): return []
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        disease = DISEASE_MAP.get(row['dx'], row['dx'])
        img_path = img_dir1 / f"{img_id}.jpg"
        if not img_path.exists(): img_path = img_dir2 / f"{img_id}.jpg"
        if img_path.exists(): samples.append((img_path, disease))
    print(f"    Found {len(samples)} images")
    return samples

def process_dermnet():
    """Process DermNet"""
    print("  Processing DermNet...")
    dermnet_dir = DATASETS_DIR / "dermnet_main"
    if not dermnet_dir.exists(): return []
    samples = []
    for split in ['train', 'test']:
        split_dir = dermnet_dir / split
        if not split_dir.exists(): continue
        for folder in split_dir.iterdir():
            if not folder.is_dir(): continue
            disease = DISEASE_MAP.get(folder.name, folder.name.lower())
            for img_path in list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
                samples.append((img_path, disease))
    print(f"    Found {len(samples)} images")
    return samples

def main():
    print("=" * 60)
    print("STRATIFIED SKIN DISEASE DATASET PREPARATION")
    print("=" * 60)
    
    # 1. Collect all samples
    all_data = []
    all_data.extend(process_isic2019())
    all_data.extend(process_ham10000())
    all_data.extend(process_dermnet())
    all_data.extend(process_massive2())
    
    if not all_data:
        print("❌ No images found! Check paths.")
        return

    # 2. Organize by disease for stratified split
    disease_groups = defaultdict(list)
    for img_path, disease in all_data:
        disease_groups[disease].append(img_path)

    # 3. Clean and Create directories
    for path in [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST]:
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    # 4. Perform Intelligent Split
    train_final, val_final, test_final = [], [], []
    
    print("\nSplitting samples per class...")
    for disease, images in tqdm(disease_groups.items()):
        random.shuffle(images)
        n = len(images)
        n_val = int(n * VAL_SPLIT)
        n_test = int(n * TEST_SPLIT)
        
        # Ensure at least 1 image for val/test if possible
        if n > 5 and n_val == 0: n_val = 1
        if n > 5 and n_test == 0: n_test = 1
        
        test_final.extend([(img, disease) for img in images[:n_test]])
        val_final.extend([(img, disease) for img in images[n_test : n_test + n_val]])
        train_final.extend([(img, disease) for img in images[n_test + n_val:]])

    # 5. Copy Files (using Fast Hardlinks)
    def copy_files(samples, target_dir, name):
        print(f"  Organizing {name} set ({len(samples)} files)...")
        for img_path, disease in tqdm(samples, leave=False):
            dest_dir = target_dir / disease
            dest_dir.mkdir(exist_ok=True)
            dest_file = dest_dir / img_path.name
            
            try:
                # Try hardlink (instant, 0 extra space)
                if dest_file.exists():
                    dest_file.unlink()
                os.link(img_path, dest_file)
            except (OSError, AttributeError):
                # Fallback to copy if hardlink fails (e.g., across partitions)
                shutil.copy2(img_path, dest_file)

    copy_files(train_final, OUTPUT_TRAIN, "TRAIN")
    copy_files(val_final, OUTPUT_VAL, "VAL")
    copy_files(test_final, OUTPUT_TEST, "TEST")

    # 6. Save Hierarchy
    hierarchy = {
        "categories": ["cancer", "benign", "inflammatory", "infectious", "pigmentary"],
        "diseases": sorted(list(disease_groups.keys())),
        "disease_to_category": {
            d: ("cancer" if d in ["melanoma", "bcc", "scc", "ak"] else
                "benign" if d in ["nevus", "seborrheic_keratosis", "angioma", "wart", "benign"] else
                "infectious" if d in ["impetigo", "herpes", "candida", "scabies", "viral", "bacterial", "fungal"] else
                "pigmentary" if d in ["vitiligo", "melasma", "hyperpigmentation"] else
                "inflammatory") for d in disease_groups.keys()
        }
    }
    for path in [OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST]:
        with open(path / "hierarchy.json", "w") as f:
            json.dump(hierarchy, f, indent=2)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Split {len(all_data)} images")
    print(f"  Train: {len(train_final)} | Val: {len(val_final)} | Test: {len(test_final)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
