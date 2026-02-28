#!/usr/bin/env python3
"""
Unified Dataset Preparation for 20-Class Skin Disease Classification

Processes ALL downloaded datasets, applies label unification, selects the
top-N classes by sample count, and creates stratified train/val/test splits.

Input:  datasets/ (raw Kaggle downloads)
Output: data/train/, data/val/, data/test/ (class subfolders with symlinks)
        data/class_info.json (class metadata)

Usage:
  python prepare_data.py                         # Default: top 20 classes
  python prepare_data.py --top-n 25              # Top 25 classes
  python prepare_data.py --datasets-dir /path    # Custom datasets location
  python prepare_data.py --verify                # Verify image integrity
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas")
    sys.exit(1)


# ============================================================================
# UNIFIED DISEASE LABEL MAP
# Maps raw labels from ALL datasets → canonical disease names
# ============================================================================
DISEASE_MAP = {
    # ── ISIC 2019 (CSV column names) ──
    "MEL": "melanoma",
    "NV": "nevus",
    "BCC": "bcc",
    "AK": "ak",
    "BKL": "seborrheic_keratosis",
    "DF": "dermatofibroma",
    "VASC": "angioma",
    "SCC": "scc",

    # ── HAM10000 (dx column values) ──
    "mel": "melanoma",
    "nv": "nevus",
    "bcc": "bcc",
    "akiec": "ak",
    "bkl": "seborrheic_keratosis",
    "df": "dermatofibroma",
    "vasc": "angioma",

    # ── PAD-UFES-20 (folder names) ──
    "ACK": "ak",
    "BCC": "bcc",  # already mapped above, harmless duplicate
    "MEL": "melanoma",
    "NEV": "nevus",
    "SCC": "scc",
    "SEK": "seborrheic_keratosis",

    # ── ISIC case variants ──
    "Bcc": "bcc",
    "Mel": "melanoma",
    "Nv": "nevus",
    "Ak": "ak",
    "Scc": "scc",

    # ── DermNet (folder names) ──
    "Acne and Rosacea Photos": "acne",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "ak",
    "Atopic Dermatitis Photos": "eczema",
    "Bullous Disease Photos": "bullous",
    "Cellulitis Impetigo and other Bacterial Infections": "impetigo",
    "Eczema Photos": "eczema",
    "Exanthems and Drug Eruptions": "drug_eruption",
    "Hair Loss Photos Alopecia and other Hair Diseases": "alopecia",
    "Herpes HPV and other STDs Photos": "herpes",
    "Light Diseases and Disorders of Pigmentation": "hyperpigmentation",
    "Lupus and other Connective Tissue diseases": "lupus",
    "Melanoma Skin Cancer Nevi and Moles": "melanoma",
    "Nail Fungus and other Nail Disease": "nail_fungus",
    "Poison Ivy Photos and other Contact Dermatitis": "dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases": "psoriasis",
    "Scabies Lyme Disease and other Infestations and Bites": "scabies",
    "Seborrheic Keratoses and other Benign Tumors": "seborrheic_keratosis",
    "Systemic Disease": "systemic",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "fungal",
    "Urticaria Hives": "urticaria",
    "Vascular Tumors": "angioma",
    "Vasculitis Photos": "vasculitis",
    "Warts Molluscum and other Viral Infections": "wart",

    # ── Massive 2 Balanced Dataset (folder names) ──
    "Acne And Rosacea Photos": "acne",
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions": "ak",
    "Atopic Dermatitis Photos": "eczema",  # duplicate key, same value
    "Ba  Cellulitis": "impetigo",
    "Ba Cellulitis": "impetigo",
    "Ba Impetigo": "impetigo",
    "Benign": "benign",
    "Bullous Disease Photos": "bullous",  # duplicate key
    "Cellulitis Impetigo And Other Bacterial Infections": "impetigo",
    "Eczema Photos": "eczema",  # duplicate key
    "Exanthems And Drug Eruptions": "drug_eruption",
    "Fu Athlete Foot": "fungal",
    "Fu Nail Fungus": "fungal",
    "Fu Ringworm": "fungal",
    "Hair Loss Photos Alopecia And Other Hair Diseases": "alopecia",
    "Heathy": "healthy",
    "Healthy": "healthy",
    "Herpes Hpv And Other Stds Photos": "herpes",
    "Light Diseases And Disorders Of Pigmentation": "hyperpigmentation",
    "Lupus And Other Connective Tissue Diseases": "lupus",
    "Malignant": "malignant",
    "Melanoma Skin Cancer Nevi And Moles": "melanoma",
    "Nail Fungus And Other Nail Disease": "fungal",
    "Pa Cutaneous Larva Migrans": "scabies",
    "Poison Ivy Photos And Other Contact Dermatitis": "dermatitis",
    "Psoriasis Pictures Lichen Planus And Related Diseases": "psoriasis",
    "Rashes": "rash",
    "Scabies Lyme Disease And Other Infestations And Bites": "scabies",
    "Seborrheic Keratoses And Other Benign Tumors": "seborrheic_keratosis",
    "Systemic Disease": "systemic",  # duplicate key
    "Tinea Ringworm Candidiasis And Other Fungal Infections": "fungal",
    "Urticaria Hives": "urticaria",  # duplicate key
    "Vascular Tumors": "angioma",  # duplicate key
    "Vasculitis Photos": "vasculitis",  # duplicate key
    "Vi Chickenpox": "viral",
    "Vi Shingles": "viral",
    "Warts Molluscum And Other Viral Infections": "wart",

    # ── Fitzpatrick17k (label column, lowercase) ──
    "basal cell carcinoma": "bcc",
    "squamous cell carcinoma": "scc",
    "melanoma": "melanoma",
    "nevus": "nevus",
    "psoriasis": "psoriasis",
    "eczema": "eczema",
    "seborrheic keratosis": "seborrheic_keratosis",
    "actinic keratosis": "ak",
    "benign keratosis-like lesions": "seborrheic_keratosis",
    "dermatofibroma": "dermatofibroma",
    "vascular lesions": "angioma",
    "atopic dermatitis": "eczema",
    "contact dermatitis": "dermatitis",
    "urticaria": "urticaria",
    "impetigo": "impetigo",
    "herpes simplex": "herpes",
    "herpes zoster": "herpes",
    "tinea": "fungal",
    "tinea pedis": "fungal",
    "tinea corporis": "fungal",
    "folliculitis": "acne",
    "rosacea": "acne",
    "scabies": "scabies",
    "alopecia areata": "alopecia",
    "vitiligo": "hyperpigmentation",
    "lupus erythematosus": "lupus",
    "drug eruption": "drug_eruption",
    "bullous pemphigoid": "bullous",
    "wart": "wart",
    "verruca": "wart",
    "molluscum contagiosum": "wart",
}


# Category hierarchy for each unified disease label
DISEASE_CATEGORY = {
    "melanoma": "cancer", "bcc": "cancer", "scc": "cancer", "ak": "cancer",
    "nevus": "benign", "seborrheic_keratosis": "benign", "angioma": "benign",
    "dermatofibroma": "benign", "benign": "benign", "malignant": "cancer",
    "eczema": "inflammatory", "psoriasis": "inflammatory", "acne": "inflammatory",
    "dermatitis": "inflammatory", "urticaria": "inflammatory", "bullous": "inflammatory",
    "drug_eruption": "inflammatory", "systemic": "inflammatory", "vasculitis": "inflammatory",
    "rash": "inflammatory",
    "impetigo": "infectious", "herpes": "infectious", "fungal": "infectious",
    "scabies": "infectious", "wart": "infectious", "nail_fungus": "infectious",
    "viral": "infectious",
    "hyperpigmentation": "pigmentary", "lupus": "autoimmune",
    "alopecia": "other", "healthy": "other",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================================
# DATASET PROCESSORS
# ============================================================================
def find_dir(base: Path, *candidates) -> Path | None:
    """Search for a directory under base, trying multiple names."""
    for candidate in candidates:
        p = base / candidate
        if p.is_dir():
            return p
    # Recursive search one level deep
    for child in base.iterdir():
        if child.is_dir():
            for candidate in candidates:
                p = child / candidate
                if p.is_dir():
                    return p
    return None


def find_file(base: Path, *candidates) -> Path | None:
    """Search for a file under base, trying multiple names."""
    for candidate in candidates:
        p = base / candidate
        if p.is_file():
            return p
    # Recursive search
    for candidate in candidates:
        matches = list(base.rglob(candidate))
        if matches:
            return matches[0]
    return None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def process_isic2019(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process ISIC 2019 dataset."""
    base = datasets_dir / "isic_2019"
    if not base.exists():
        print("    [SKIP] ISIC 2019 not found")
        return []

    csv_path = find_file(
        base,
        "ISIC_2019_Training_GroundTruth.csv",
        "ISIC_2019_Training_GroundTruth_v2.csv",
    )
    if csv_path is None:
        print("    [SKIP] ISIC 2019 CSV not found")
        return []

    images_dir = find_dir(base, "ISIC_2019_Training_Input", "images", "train")
    if images_dir is None:
        # Images might be directly in base
        images_dir = base

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        img_name = str(row["image"])
        disease = None
        for col in df.columns:
            if col == "image":
                continue
            try:
                if float(row[col]) == 1.0:
                    disease = DISEASE_MAP.get(col, col.lower())
                    break
            except (ValueError, TypeError):
                if str(row[col]).strip() == "1":
                    disease = DISEASE_MAP.get(col, col.lower())
                    break

        if disease is None:
            continue

        # Find image file
        for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
            img_path = images_dir / f"{img_name}{ext}"
            if img_path.exists():
                samples.append((img_path.resolve(), disease))
                break

    print(f"    ISIC 2019: {len(samples)} images")
    return samples


def process_ham10000(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process HAM10000 dataset."""
    base = datasets_dir / "ham10000"
    if not base.exists():
        print("    [SKIP] HAM10000 not found")
        return []

    csv_path = find_file(base, "HAM10000_metadata.csv", "HAM10000_metadata")
    if csv_path is None:
        print("    [SKIP] HAM10000 metadata not found")
        return []

    # Images can be in part_1/part_2 or directly
    image_dirs = []
    for name in ["HAM10000_images_part_1", "HAM10000_images_part_2",
                  "ham10000_images_part_1", "ham10000_images_part_2",
                  "images", "."]:
        d = base / name
        if d.is_dir():
            image_dirs.append(d)
    if not image_dirs:
        image_dirs = [base]

    df = pd.read_csv(csv_path)
    samples = []

    for _, row in df.iterrows():
        img_id = str(row["image_id"])
        raw_label = str(row["dx"])
        disease = DISEASE_MAP.get(raw_label, raw_label.lower())

        # Search for image across all directories
        found = False
        for img_dir in image_dirs:
            for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                img_path = img_dir / f"{img_id}{ext}"
                if img_path.exists():
                    samples.append((img_path.resolve(), disease))
                    found = True
                    break
            if found:
                break

    print(f"    HAM10000: {len(samples)} images")
    return samples


def process_dermnet(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process DermNet dataset (folder-based)."""
    base = datasets_dir / "dermnet"
    if not base.exists():
        print("    [SKIP] DermNet not found")
        return []

    samples = []
    for split_name in ["train", "test", "Train", "Test"]:
        split_dir = base / split_name
        if not split_dir.is_dir():
            # Check one level deeper
            for child in base.iterdir():
                if child.is_dir() and (child / split_name).is_dir():
                    split_dir = child / split_name
                    break
            else:
                continue

        for category_dir in split_dir.iterdir():
            if not category_dir.is_dir():
                continue
            disease = DISEASE_MAP.get(category_dir.name, category_dir.name.lower())
            for img_path in category_dir.iterdir():
                if is_image(img_path):
                    samples.append((img_path.resolve(), disease))

    # Also check if images are directly in class folders (no train/test split)
    if not samples:
        for category_dir in base.iterdir():
            if not category_dir.is_dir():
                continue
            if category_dir.name.lower() in {"train", "test"}:
                continue
            disease = DISEASE_MAP.get(category_dir.name, category_dir.name.lower())
            for img_path in category_dir.iterdir():
                if is_image(img_path):
                    samples.append((img_path.resolve(), disease))

    print(f"    DermNet: {len(samples)} images")
    return samples


def process_pad_ufes(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process PAD-UFES-20 dataset."""
    base = datasets_dir / "pad_ufes_20"
    if not base.exists():
        print("    [SKIP] PAD-UFES-20 not found")
        return []

    samples = []

    # Try folder-based: class subfolders (ACK, BCC, MEL, NEV, SCC, SEK)
    pad_labels = {"ACK": "ak", "BCC": "bcc", "MEL": "melanoma",
                  "NEV": "nevus", "SCC": "scc", "SEK": "seborrheic_keratosis"}

    # Search recursively for class folders
    for folder in base.rglob("*"):
        if folder.is_dir() and folder.name in pad_labels:
            disease = pad_labels[folder.name]
            for img_path in folder.iterdir():
                if is_image(img_path):
                    samples.append((img_path.resolve(), disease))

    # If no folder structure, try CSV-based
    if not samples:
        csv_path = find_file(base, "metadata.csv", "pad-ufes-20.csv",
                             "PAD-UFES-20.csv", "labels.csv")
        if csv_path is not None:
            try:
                df = pd.read_csv(csv_path)
                label_col = None
                for col in ["diagnostic", "label", "dx", "class"]:
                    if col in df.columns:
                        label_col = col
                        break
                img_col = None
                for col in ["img_id", "image_id", "image", "filename"]:
                    if col in df.columns:
                        img_col = col
                        break

                if label_col and img_col:
                    for _, row in df.iterrows():
                        raw = str(row[label_col]).strip()
                        disease = DISEASE_MAP.get(raw, raw.lower())
                        img_name = str(row[img_col])
                        for img_dir in [base / "imgs", base / "images", base]:
                            for ext in [".png", ".jpg", ".jpeg"]:
                                p = img_dir / f"{img_name}{ext}"
                                if p.exists():
                                    samples.append((p.resolve(), disease))
                                    break
            except Exception as e:
                print(f"    [WARN] PAD-UFES CSV error: {e}")

    print(f"    PAD-UFES-20: {len(samples)} images")
    return samples


def process_fitzpatrick17k(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process Fitzpatrick17k dataset."""
    base = datasets_dir / "fitzpatrick17k"
    if not base.exists():
        print("    [SKIP] Fitzpatrick17k not found")
        return []

    # Find CSV
    csv_path = find_file(base, "fitzpatrick17k.csv", "labels.csv")
    if csv_path is None:
        print("    [SKIP] Fitzpatrick17k CSV not found")
        return []

    # Find images directory
    images_dir = find_dir(base, "images", "finalfitz17k", "imgs")
    if images_dir is None:
        images_dir = base  # Images might be in same dir as CSV

    df = pd.read_csv(csv_path)
    samples = []

    # Determine column names
    label_col = "label" if "label" in df.columns else "diagnostic"
    hash_col = "md5hash" if "md5hash" in df.columns else None
    # Fallback: try other possible image ID columns
    if hash_col is None:
        for alt in ["hasher", "image_id", "image", "filename"]:
            if alt in df.columns:
                hash_col = alt
                break

    if label_col not in df.columns:
        print("    [SKIP] Fitzpatrick17k: no label column found")
        return []

    if hash_col is None:
        print("    [SKIP] Fitzpatrick17k: no image identifier column found")
        print(f"           Columns: {list(df.columns)}")
        return []

    for _, row in df.iterrows():
        raw_label = str(row[label_col]).strip().lower()
        disease = DISEASE_MAP.get(raw_label, raw_label)

        # Find image
        if hash_col and hash_col in df.columns:
            img_name = str(row[hash_col])
        else:
            continue

        found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = images_dir / f"{img_name}{ext}"
            if img_path.exists():
                samples.append((img_path.resolve(), disease))
                found = True
                break

        if not found:
            # Search recursively (slower but more robust)
            for ext in [".jpg", ".jpeg", ".png"]:
                matches = list(base.rglob(f"{img_name}{ext}"))
                if matches:
                    samples.append((matches[0].resolve(), disease))
                    break

    print(f"    Fitzpatrick17k: {len(samples)} images")
    return samples


def process_massive2(datasets_dir: Path) -> list[tuple[Path, str]]:
    """Process Massive Balanced Skin Disease dataset (262k images)."""
    base = datasets_dir / "massive_balanced"
    if not base.exists():
        print("    [SKIP] Massive Balanced not found")
        return []

    # Navigate to the actual image folders (may be nested)
    source_dir = base
    for _ in range(4):  # Max depth
        candidates = ["balanced_dataset", "Balanced_dataset", "skin-disease-balanced-dataset"]
        found = False
        for name in candidates:
            p = source_dir / name
            if p.is_dir():
                source_dir = p
                found = True
                break
        if not found:
            # Check if current dir has class folders directly
            has_class_folders = any(
                child.is_dir() and child.name in DISEASE_MAP
                for child in source_dir.iterdir()
            )
            if has_class_folders:
                break
            # Try going into the first subdirectory
            subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                source_dir = subdirs[0]
            else:
                break

    samples = []
    for folder in source_dir.iterdir():
        if not folder.is_dir():
            continue

        # Try exact match, then case-insensitive
        disease = DISEASE_MAP.get(folder.name)
        if disease is None:
            # Try with space normalization
            normalized = " ".join(folder.name.split())
            disease = DISEASE_MAP.get(normalized)
        if disease is None:
            # Skip unknown folders
            continue

        for img_path in folder.iterdir():
            if is_image(img_path):
                samples.append((img_path.resolve(), disease))

    print(f"    Massive Balanced: {len(samples)} images")
    return samples


# ============================================================================
# DATASET SPLITTING & OUTPUT
# ============================================================================
def create_splits(
    all_samples: list[tuple[Path, str]],
    output_dir: Path,
    top_n: int = 20,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
):
    """Create stratified train/val/test splits for top-N classes."""
    random.seed(seed)

    # Group by disease
    disease_groups: dict[str, list[Path]] = defaultdict(list)
    for img_path, disease in all_samples:
        disease_groups[disease].append(img_path)

    # Sort by count, take top N
    sorted_diseases = sorted(disease_groups.items(), key=lambda x: -len(x[1]))

    print(f"\n  All unified classes ({len(sorted_diseases)}):")
    for i, (disease, images) in enumerate(sorted_diseases):
        marker = " *" if i < top_n else ""
        print(f"    {i+1:3d}. {disease:30s} {len(images):>8,} images{marker}")

    selected = sorted_diseases[:top_n]
    selected_names = [name for name, _ in selected]

    print(f"\n  Selected top {top_n} classes (marked with *)")

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Stratified split
    stats = {"train": Counter(), "val": Counter(), "test": Counter()}
    total_linked = 0
    link_errors = 0

    for disease, images in tqdm(selected, desc="  Splitting"):
        random.shuffle(images)
        n = len(images)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        splits = {
            "test": images[:n_test],
            "val": images[n_test:n_test + n_val],
            "train": images[n_test + n_val:],
        }

        for split_name, split_images in splits.items():
            split_dir = output_dir / split_name / disease
            split_dir.mkdir(parents=True, exist_ok=True)
            stats[split_name][disease] = len(split_images)

            for img_path in split_images:
                dst = split_dir / img_path.name
                # Handle duplicate filenames
                if dst.exists():
                    stem = dst.stem
                    suffix = dst.suffix
                    counter = 1
                    while dst.exists():
                        dst = split_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                try:
                    os.symlink(str(img_path), str(dst))
                    total_linked += 1
                except OSError:
                    try:
                        os.link(str(img_path), str(dst))
                        total_linked += 1
                    except OSError:
                        try:
                            shutil.copy2(str(img_path), str(dst))
                            total_linked += 1
                        except Exception:
                            link_errors += 1

    # Save class info
    class_info = {
        "num_classes": top_n,
        "classes": selected_names,
        "class_to_idx": {name: i for i, name in enumerate(selected_names)},
        "idx_to_class": {i: name for i, name in enumerate(selected_names)},
        "category_map": {
            name: DISEASE_CATEGORY.get(name, "other") for name in selected_names
        },
        "cancer_classes": [
            name for name in selected_names
            if DISEASE_CATEGORY.get(name) == "cancer"
        ],
        "stats": {
            "train": dict(stats["train"]),
            "val": dict(stats["val"]),
            "test": dict(stats["test"]),
            "total_train": sum(stats["train"].values()),
            "total_val": sum(stats["val"].values()),
            "total_test": sum(stats["test"].values()),
        },
    }

    info_path = output_dir / "class_info.json"
    with open(info_path, "w") as f:
        json.dump(class_info, f, indent=2)

    return class_info, total_linked, link_errors


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prepare unified skin disease dataset"
    )
    parser.add_argument(
        "--datasets-dir", type=str, default="datasets",
        help="Directory with raw Kaggle downloads"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory for prepared splits"
    )
    parser.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top classes to select (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify image integrity (slower but catches corrupt images)"
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print()
    print("=" * 65)
    print("  UNIFIED DATASET PREPARATION")
    print(f"  Source: {datasets_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Top-N:  {args.top_n} classes")
    print("=" * 65)
    print()

    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}")
        print("Run 'python download_data.py' first.")
        sys.exit(1)

    # Step 1: Process all datasets
    print("[1/3] Processing datasets...")
    all_samples = []
    all_samples.extend(process_isic2019(datasets_dir))
    all_samples.extend(process_ham10000(datasets_dir))
    all_samples.extend(process_dermnet(datasets_dir))
    all_samples.extend(process_pad_ufes(datasets_dir))
    all_samples.extend(process_fitzpatrick17k(datasets_dir))
    all_samples.extend(process_massive2(datasets_dir))

    if not all_samples:
        print("\nERROR: No images found across any dataset!")
        print("Check that datasets were downloaded correctly.")
        sys.exit(1)

    print(f"\n  Total raw samples: {len(all_samples):,}")

    # Step 2: Optional verification
    if args.verify:
        print("\n[1.5/3] Verifying image integrity...")
        from PIL import Image
        valid = []
        bad = 0
        for img_path, disease in tqdm(all_samples, desc="  Verifying"):
            try:
                img = Image.open(img_path)
                img.verify()
                valid.append((img_path, disease))
            except Exception:
                bad += 1
        print(f"  Valid: {len(valid)}, Corrupt: {bad}")
        all_samples = valid

    # Step 3: Create splits
    print(f"\n[2/3] Creating top-{args.top_n} stratified splits...")
    class_info, total_linked, link_errors = create_splits(
        all_samples, output_dir, top_n=args.top_n, seed=args.seed
    )

    # Step 4: Summary
    print()
    print("[3/3] Summary")
    print("=" * 65)
    print(f"  Classes:    {class_info['num_classes']}")
    print(f"  Train:      {class_info['stats']['total_train']:>8,} images")
    print(f"  Val:        {class_info['stats']['total_val']:>8,} images")
    print(f"  Test:       {class_info['stats']['total_test']:>8,} images")
    total = (class_info['stats']['total_train'] +
             class_info['stats']['total_val'] +
             class_info['stats']['total_test'])
    print(f"  Total:      {total:>8,} images")
    print(f"  Link errs:  {link_errors}")
    print(f"  Class info: {output_dir / 'class_info.json'}")
    print()

    cancer = class_info.get("cancer_classes", [])
    if cancer:
        print(f"  Cancer classes: {', '.join(cancer)}")
    print()
    print("  Next step: python train.py")
    print()


if __name__ == "__main__":
    main()
