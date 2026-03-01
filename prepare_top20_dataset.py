#!/usr/bin/env python3
"""
Prepare top 20 classes from unified dataset for optimal training
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict

# Dataset paths
BASE_DIR = Path("/dist_home/suryansh/arjungop/AI_Skin_doc/data")
UNIFIED_TRAIN = BASE_DIR / "unified_train"
UNIFIED_VAL = BASE_DIR / "unified_val"
UNIFIED_TEST = BASE_DIR / "unified_test"

# Output paths for top 20
TOP20_TRAIN = BASE_DIR / "top20_train"
TOP20_VAL = BASE_DIR / "top20_val"
TOP20_TEST = BASE_DIR / "top20_test"

def count_images(directory):
    """Count images in directory"""
    count = 0
    if not directory.exists():
        return 0
    for file in directory.iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            count += 1
    return count

def get_class_counts(base_dir):
    """Get image counts for all classes"""
    class_counts = {}
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return class_counts
    
    for class_dir in sorted(base_dir.iterdir()):
        if class_dir.is_dir():
            count = count_images(class_dir)
            if count > 0:
                class_counts[class_dir.name] = count
    
    return class_counts

def create_symlinks(src_classes, src_base, dst_base):
    """Create symbolic links for selected classes"""
    os.makedirs(dst_base, exist_ok=True)
    
    for class_name in src_classes:
        src_path = src_base / class_name
        dst_path = dst_base / class_name
        
        if dst_path.exists():
            if dst_path.is_symlink():
                dst_path.unlink()
            else:
                shutil.rmtree(dst_path)
        
        if src_path.exists():
            os.symlink(src_path, dst_path)
            count = count_images(src_path)
            print(f"  ✓ {class_name}: {count:,} images")

# Main execution
print("\n" + "="*80)
print("📊 ANALYZING UNIFIED DATASET")
print("="*80)

# Get class counts from training set
class_counts = get_class_counts(UNIFIED_TRAIN)

if not class_counts:
    print("❌ No classes found in unified_train!")
    exit(1)

# Sort and get top 20
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
top20 = sorted_classes[:20]

print(f"\n✅ Found {len(class_counts)} classes in unified dataset")
print(f"\n🏆 TOP 20 CLASSES BY SAMPLE COUNT:")
print("-"*80)
print(f"{'RANK':<6} {'CLASS NAME':<40} {'SAMPLES':>12}")
print("-"*80)

top20_names = []
total_samples = 0

for rank, (class_name, count) in enumerate(top20, 1):
    top20_names.append(class_name)
    total_samples += count
    print(f"{rank:<6} {class_name:<40} {count:>12,}")

print("-"*80)
print(f"{'TOTAL':<6} {'':<40} {total_samples:>12,}")
print("="*80)

# Create filtered datasets using symlinks
print(f"\n📁 CREATING TOP 20 FILTERED DATASETS...")
print("-"*80)

print("\n🔗 Creating training set...")
create_symlinks(top20_names, UNIFIED_TRAIN, TOP20_TRAIN)

print("\n🔗 Creating validation set...")
create_symlinks(top20_names, UNIFIED_VAL, TOP20_VAL)

print("\n🔗 Creating test set...")
create_symlinks(top20_names, UNIFIED_TEST, TOP20_TEST)

print("\n" + "="*80)
print("✅ TOP 20 DATASET READY!")
print("="*80)
print(f"Training:   {TOP20_TRAIN}")
print(f"Validation: {TOP20_VAL}")
print(f"Test:       {TOP20_TEST}")
print(f"\nClasses: {len(top20_names)}")
print(f"Total training samples: {total_samples:,}")
print("="*80)

# Save class mapping
import json
class_mapping = {
    "top20_classes": top20_names,
    "class_counts": dict(top20),
    "total_classes": len(top20_names),
    "total_training_samples": total_samples
}

with open(BASE_DIR / "top20_mapping.json", "w") as f:
    json.dump(class_mapping, f, indent=2)

print(f"\n💾 Class mapping saved to: {BASE_DIR / 'top20_mapping.json'}\n")
