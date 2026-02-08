#!/usr/bin/env python3
"""
Unified dataset preparation v2 - Fixed paths
"""
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import random
import json

DATASETS_DIR = Path("datasets")
OUTPUT_TRAIN = Path("data/unified_train")
OUTPUT_VAL = Path("data/unified_val")
VAL_SPLIT = 0.1

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
    
    # Try multiple possible paths for Massive 2
    massive_paths = [
        Path("massive 2/balanced_dataset/balanced_dataset"),
        Path("massive_2/balanced_dataset/balanced_dataset"),
        DATASETS_DIR / "massive 2" / "balanced_dataset" / "balanced_dataset",
        Path("/Users/gopal/Skin-Doc/massive 2/balanced_dataset/balanced_dataset")
    ]
    
    source_dir = None
    for p in massive_paths:
        if p.exists():
            source_dir = p
            break
            
    if not source_dir:
        print("  Skipping Massive 2: Directory not found")
        return []
        
    samples = []
    for disease_folder in source_dir.iterdir():
        if not disease_folder.is_dir():
            continue
            
        label = DISEASE_MAP.get(disease_folder.name)
        if not label:
            # Try fuzzy match or skip
            continue
            
        for img_path in disease_folder.glob("*.jpg"):
            samples.append((img_path, label))
            
    print(f"    Found {len(samples)} images from Massive 2")
    return samples

def process_isic2019():
    """Process ISIC 2019 with FIXED CSV path"""
    print("  Processing ISIC 2019...")
    
    csv_path = DATASETS_DIR / "isic_data" / "isic_2019" / "ISIC_2019_Training_GroundTruth.csv"
    images_dir = DATASETS_DIR / "isic_data" / "isic_2019" / "ISIC_2019_Training_Input"
    
    if not csv_path.exists():
        print(f"  Skipping ISIC 2019: CSV not found at {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['image']
        
        # Find actual disease column
        disease_cols = [col for col in df.columns if col != 'image']
        disease = None
        for col in disease_cols:
            if row[col] == 1.0:
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
    images_dir1 = DATASETS_DIR / "ham10000" / "HAM10000_images_part_1"
    images_dir2 = DATASETS_DIR / "ham10000" / "HAM10000_images_part_2"
    
    if not csv_path.exists():
        print(f"  Skipping HAM10000: CSV not found")
        return []
    
    df = pd.read_csv(csv_path)
    samples = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['image_id']
        disease = DISEASE_MAP.get(row['dx'], row['dx'])
        
        img_path = images_dir1 / f"{img_id}.jpg"
        if not img_path.exists():
            img_path = images_dir2 / f"{img_id}.jpg"
        
        if img_path.exists():
            samples.append((img_path, disease))
    
    print(f"    Found {len(samples)} images")
    return samples

def process_dermnet():
    """Process DermNet with FIXED path"""
    print("  Processing DermNet...")
    
    dermnet_dir = DATASETS_DIR / "dermnet_main"
    
    if not dermnet_dir.exists():
        print(f"  Skipping DermNet: not found at {dermnet_dir}")
        return []
    
    samples = []
    
    for split in ['train', 'test']:
        split_dir = dermnet_dir / split
        if not split_dir.exists():
            continue
            
        for disease_folder in split_dir.iterdir():
            if not disease_folder.is_dir():
                continue
                
            disease = DISEASE_MAP.get(disease_folder.name, disease_folder.name.lower())
            
            for img_path in disease_folder.glob("*.jpg"):
                samples.append((img_path, disease))
            for img_path in disease_folder.glob("*.png"):
                samples.append((img_path, disease))
    
    print(f"    Found {len(samples)} images")
    return samples

def main():
    print("=" * 60)
    print("SKIN DISEASE DATASET PREPARATION - FIXED PATHS")
    print("=" * 60)
    print(f"Datasets dir: {DATASETS_DIR}")
    print(f"Output dir: {OUTPUT_TRAIN}")
    print(f"Val dir: {OUTPUT_VAL}")
    print(f"Val split: {VAL_SPLIT}")
    print()
    
    # Clean output directories
    print("Cleaning output directories...")
    shutil.rmtree(OUTPUT_TRAIN, ignore_errors=True)
    shutil.rmtree(OUTPUT_VAL, ignore_errors=True)
    OUTPUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUTPUT_VAL.mkdir(parents=True, exist_ok=True)
    print()
    
    # Process all datasets
    print("Processing datasets...")
    all_samples = []
    
    all_samples.extend(process_isic2019())
    all_samples.extend(process_ham10000())
    all_samples.extend(process_dermnet())
    all_samples.extend(process_massive2())
    
    print()
    
    # Shuffle and split
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print("Creating datasets...")
    
    # Copy train samples
    print(f"  Creating train set ({len(train_samples)} files)...")
    for img_path, disease in tqdm(train_samples):
        disease_dir = OUTPUT_TRAIN / disease
        disease_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, disease_dir / img_path.name)
    
    # Copy val samples
    print(f"  Creating val set ({len(val_samples)} files)...")
    for img_path, disease in tqdm(val_samples):
        disease_dir = OUTPUT_VAL / disease
        disease_dir.mkdir(exist_ok=True)
        shutil.copy2(img_path, disease_dir / img_path.name)
    
    # Summary
    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print()
    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print()
    
    # Per-disease counts
    disease_counts = {}
    for _, disease in all_samples:
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    print("Per-disease counts:")
    for disease in sorted(disease_counts.keys()):
        count = disease_counts[disease]
        status = "✅" if count >= 500 else "⚠️" if count >= 100 else "❌"
        print(f"  {status} {disease}: {count}")
    
    print()
    print("Files saved to:")
    print(f"  Train: {OUTPUT_TRAIN}")
    print(f"  Val: {OUTPUT_VAL}")

    # Save hierarchy.json
    hierarchy = {
        "categories": ["cancer", "benign", "inflammatory", "infectious", "pigmentary"],
        "diseases": [
            "melanoma", "bcc", "scc", "ak", 
            "nevus", "seborrheic_keratosis", "angioma", "wart", "benign",
            "eczema", "psoriasis", "lichen_planus", "urticaria", "acne", "alopecia", "rosacea",
            "impetigo", "herpes", "candida", "scabies", "viral", "bacterial", "fungal",
            "vitiligo", "melasma", "hyperpigmentation"
        ],
        "disease_to_category": {
            "melanoma": "cancer", "bcc": "cancer", "scc": "cancer", "ak": "cancer",
            "nevus": "benign", "seborrheic_keratosis": "benign", "angioma": "benign", "wart": "benign", "benign": "benign",
            "eczema": "inflammatory", "psoriasis": "inflammatory", "lichen_planus": "inflammatory", "urticaria": "inflammatory", 
            "acne": "inflammatory", "alopecia": "inflammatory", "rosacea": "inflammatory",
            "impetigo": "infectious", "herpes": "infectious", "candida": "infectious", "scabies": "infectious", 
            "viral": "infectious", "bacterial": "infectious", "fungal": "infectious",
            "vitiligo": "pigmentary", "melasma": "pigmentary", "hyperpigmentation": "pigmentary"
        }
    }
    
    with open(OUTPUT_TRAIN / "hierarchy.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    with open(OUTPUT_VAL / "hierarchy.json", "w") as f:
        json.dump(hierarchy, f, indent=2)
    print("  Created hierarchy.json in output directories.")

if __name__ == "__main__":
    main()
