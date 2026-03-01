#!/usr/bin/env python3
"""
Analyze dataset statistics and predict performance metrics
"""
import os
from pathlib import Path
import json
import math

def count_images(directory):
    """Count images in directory"""
    count = 0
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in exts):
                count += 1
    return count

def estimate_class_accuracy(train_samples, class_difficulty="medium"):
    """
    Estimate per-class accuracy based on training samples
    
    Difficulty levels:
    - easy: Well-defined visual features (90-97%)
    - medium: Moderate visual complexity (80-92%)
    - hard: High visual similarity to other classes (70-85%)
    """
    if class_difficulty == "easy":
        base_acc = 93
        if train_samples < 500:
            return max(70, base_acc - 15)
        elif train_samples < 2000:
            return max(85, base_acc - 8)
        else:
            return min(97, base_acc + 2)
    elif class_difficulty == "hard":
        base_acc = 77
        if train_samples < 500:
            return max(60, base_acc - 12)
        elif train_samples < 2000:
            return max(70, base_acc - 5)
        else:
            return min(85, base_acc + 3)
    else:  # medium
        base_acc = 86
        if train_samples < 500:
            return max(65, base_acc - 15)
        elif train_samples < 2000:
            return max(78, base_acc - 7)
        else:
            return min(92, base_acc + 2)

# Class difficulty mapping based on visual distinctiveness
CLASS_DIFFICULTY = {
    "melanoma": "hard",  # Can look like other pigmented lesions
    "bcc": "medium",  # Basal cell carcinoma
    "scc": "hard",  # Squamous cell carcinoma - visually similar to other lesions
    "ak": "medium",  # Actinic keratosis
    "eczema": "medium",  # Can be confused with other inflammatory conditions
    "psoriasis": "medium",
    "contact_dermatitis": "hard",  # Looks similar to eczema
    "poison_ivy_photos_and_other_contact_dermatitis": "hard",
    "acne": "easy",  # Distinctive pustules/comedones
    "acne_and_rosacea_photos": "medium",
    "urticaria": "easy",  # Distinctive hives
    "vi_chickenpox": "easy",  # Very distinctive vesicular rash
    "vi_shingles": "easy",  # Distinctive dermatomal pattern
    "herpes_hpv_and_other_stds_photos": "easy",
    "fu_athlete_foot": "medium",
    "fu_ringworm": "medium",
    "fu_nail_fungus": "medium",
    "nail_fungus_and_other_nail_disease": "medium",
    "candida": "medium",
    "scabies": "medium",
    "pa_cutaneous_larva_migrans": "easy",  # Distinctive track-like lesions
    "ba_impetigo": "easy",  # Honey-crusted lesions
    "ba__cellulitis": "medium",
    "angioma": "easy",  # Cherry red papules
    "heathy": "easy",  # Normal skin
    "benign": "medium",
    "warts": "easy",
    "nail_disease": "medium",
    "pigmentation_disorder": "hard",
    "light_diseases_and_disorders_of_pigmentation": "hard",
    "lupus_and_other_connective_tissue_diseases": "hard",
    "systemic_disease": "hard",
    "exanthems_and_drug_eruptions": "hard",
    "drug_eruption": "hard",
    "bullous_disease_photos": "medium",
    "dermatofibroma": "medium",
    "hair_loss_photos_alopecia_and_other_hair_diseases": "medium",
}

# Collect data
data_dir = Path("/dist_home/suryansh/arjungop/AI_Skin_doc/data")
splits = ["main_train", "main_val", "main_test"]
classes_data = {}

print("\n🔍 Scanning dataset directories...")

for split in splits:
    split_dir = data_dir / split
    if split_dir.exists():
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                count = count_images(class_dir)
                
                if class_name not in classes_data:
                    classes_data[class_name] = {"train": 0, "val": 0, "test": 0, "total": 0}
                
                split_key = split.replace("main_", "")
                classes_data[class_name][split_key] = count
                classes_data[class_name]["total"] += count

# Print dataset statistics
print("\n" + "="*120)
print("📊 DATASET STATISTICS - 29 Classes (DermNet Main)")
print("="*120)
print(f"{'CLASS NAME':<45} {'TRAIN':>8} {'VAL':>8} {'TEST':>8} {'TOTAL':>9}")
print("-"*120)

total_train = 0
total_val = 0
total_test = 0

for class_name in sorted(classes_data.keys()):
    data = classes_data[class_name]
    print(f"{class_name:<45} {data['train']:>8,} {data['val']:>8,} {data['test']:>8,} {data['total']:>9,}")
    total_train += data["train"]
    total_val += data["val"]
    total_test += data["test"]

print("-"*120)
print(f"{'TOTAL':<45} {total_train:>8,} {total_val:>8,} {total_test:>8,} {total_train+total_val+total_test:>9,}")
print("="*120)

# Predict performance metrics
print("\n" + "="*120)
print("🎯 EXPECTED PERFORMANCE METRICS (ConvNeXt Base, 50 epochs)")
print("="*120)
print(f"{'CLASS NAME':<45} {'SAMPLES':>8} {'ACC %':>7} {'FN/100':>8} {'DIFF':>8}")
print("-"*120)

performance_data = {}
total_samples = sum(d["train"] for d in classes_data.values())

for class_name in sorted(classes_data.keys()):
    train_samples = classes_data[class_name]["train"]
    
    # Get difficulty
    difficulty = CLASS_DIFFICULTY.get(class_name, "medium")
    
    # Estimate accuracy
    est_acc = estimate_class_accuracy(train_samples, difficulty)
    
    # Estimate false negatives per 100 samples (misses)
    # FN% = 100 - Sensitivity, assuming balanced precision/recall
    fn_rate = 100 - est_acc
    fn_per_100 = round(fn_rate)
    
    performance_data[class_name] = {
        "samples": train_samples,
        "accuracy": est_acc,
        "fn_per_100": fn_per_100,
        "difficulty": difficulty
    }
    
    print(f"{class_name:<45} {train_samples:>8,} {est_acc:>6.1f}% {fn_per_100:>8} {difficulty:>8}")

print("-"*120)

# Average metrics
avg_acc = sum(p["accuracy"] for p in performance_data.values()) / len(performance_data)
weighted_acc = sum(p["accuracy"] * classes_data[c]["train"] for c, p in performance_data.items()) / total_train

print(f"\n{'AVERAGE ACCURACY (unweighted):':<60} {avg_acc:.1f}%")
print(f"{'WEIGHTED ACCURACY (by samples):':<60} {weighted_acc:.1f}%")
print("="*120)

# Class imbalance analysis
print("\n" + "="*120)
print("⚠️  CLASS IMBALANCE ANALYSIS")
print("="*120)

samples_list = [(name, data["train"]) for name, data in classes_data.items()]
samples_list.sort(key=lambda x: x[1])

print(f"\n📉 SMALLEST CLASSES (Most challenging due to limited data):")
print(f"{'CLASS':<45} {'SAMPLES':>10} {'RISK':>10}")
print("-"*70)
for name, count in samples_list[:5]:
    risk = "HIGH" if count < 500 else "MEDIUM"
    print(f"{name:<45} {count:>10,} {risk:>10}")

print(f"\n📈 LARGEST CLASSES (Better performance expected):")
print(f"{'CLASS':<45} {'SAMPLES':>10}")
print("-"*70)
for name, count in samples_list[-5:]:
    print(f"{name:<45} {count:>10,}")

print("\n" + "="*120)

# Save detailed results
output = {
    "dataset_stats": classes_data,
    "performance_predictions": performance_data,
    "summary": {
        "total_classes": len(classes_data),
        "total_train_images": total_train,
        "total_val_images": total_val,
        "total_test_images": total_test,
        "average_accuracy": round(avg_acc, 2),
        "weighted_accuracy": round(weighted_acc, 2),
    }
}

with open("dataset_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ Complete analysis saved to: dataset_analysis.json")
print("\n📝 KEY INSIGHTS:")
print("   • Classes with <500 samples may have lower accuracy (65-75%)")
print("   • Classes with >5000 samples expected accuracy: 85-92%")
print("   • Hard classes (melanoma, SCC, contact dermatitis): 70-85% accuracy")
print("   • Easy classes (chickenpox, urticaria, angioma): 90-95% accuracy")
print("   • False negatives: Critical for cancer detection (melanoma, BCC, SCC)")
print("\n")
