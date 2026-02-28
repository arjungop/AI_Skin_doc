#!/usr/bin/env python3
"""
Evaluation Script — 20-Class Skin Disease Classification

Loads the best trained model, evaluates on the test set, and generates:
  - Overall accuracy & balanced accuracy
  - Per-class precision, recall, F1
  - Confusion matrix heatmap
  - Cancer sensitivity analysis
  - Optional Test-Time Augmentation (TTA) for accuracy boost

Usage:
  python evaluate.py                               # Standard evaluation
  python evaluate.py --tta                          # With TTA (+1-2% accuracy)
  python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

try:
    import timm
except ImportError:
    print("ERROR: timm not installed.  Run: pip install timm>=1.0.0")
    sys.exit(1)

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        balanced_accuracy_score,
    )
except ImportError:
    print("ERROR: scikit-learn required.  Run: pip install scikit-learn")
    sys.exit(1)


# ============================================================================
# CONSTANTS
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================================
# DATASET  (same as train.py but standalone)
# ============================================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, img_size: int = 384):
        self.root = Path(root_dir)
        self.transform = transform
        self.img_size = img_size

        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        self.targets = []
        for cls in self.classes:
            idx = self.class_to_idx[cls]
            for p in (self.root / cls).iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(p), idx))
                    self.targets.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.img_size, self.img_size), "black")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================================
# TRANSFORMS
# ============================================================================
def get_eval_transform(img_size: int):
    resize_to = int(img_size * 1.143)
    return T.Compose([
        T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms(img_size: int) -> list:
    """5 augmented views for Test-Time Augmentation."""
    resize_to = int(img_size * 1.143)
    base_pre = [
        T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
    ]
    base_post = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    views = [
        # 1. Standard center crop
        T.Compose(base_pre + base_post),
        # 2. Horizontal flip
        T.Compose(base_pre + [T.RandomHorizontalFlip(p=1.0)] + base_post),
        # 3. Vertical flip
        T.Compose(base_pre + [T.RandomVerticalFlip(p=1.0)] + base_post),
        # 4. Slight rotation +15
        T.Compose([
            T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.RandomRotation((15, 15)),
        ] + base_post),
        # 5. Slight rotation -15
        T.Compose([
            T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.RandomRotation((-15, -15)),
        ] + base_post),
    ]
    return views


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    classes = ckpt.get("classes", [])
    num_classes = len(classes)

    if num_classes == 0:
        raise ValueError("Checkpoint does not contain class information.")

    backbone = config.get("backbone", "convnext_large")
    drop_rate = config.get("drop_rate", 0.4)
    drop_path_rate = config.get("drop_path_rate", 0.3)

    # Create model
    candidates = [backbone, "convnext_large.fb_in22k_ft_in1k_384",
                  "convnext_large.fb_in22k_ft_in1k", "convnext_large"]
    model = None
    for name in dict.fromkeys(candidates):  # deduplicate preserving order
        try:
            model = timm.create_model(
                name, pretrained=False, num_classes=num_classes,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
            )
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Could not create model for backbone: {backbone}")

    # Load weights — prefer EMA if available
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  Loaded EMA weights (best generalisation)")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded model weights")
    else:
        raise ValueError("Checkpoint has no model weights.")

    model = model.to(device)
    model.eval()

    return model, classes, config


# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate_standard(model, loader, device, num_classes):
    """Standard single-pass evaluation."""
    all_preds = []
    all_targets = []

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.numpy().tolist())

    return np.array(all_preds), np.array(all_targets)


@torch.no_grad()
def evaluate_tta(model, dataset, tta_transforms, device, batch_size=32, num_workers=4):
    """Test-Time Augmentation: average predictions over multiple views."""
    num_samples = len(dataset)
    num_classes = None
    all_probs = None

    for view_idx, transform in enumerate(tta_transforms):
        # Create dataset copy with new transform
        ds_view = SkinDataset(
            str(dataset.root), transform=transform, img_size=dataset.img_size
        )
        loader = DataLoader(
            ds_view, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        view_probs = []
        for images, _ in tqdm(loader, desc=f"TTA view {view_idx+1}/{len(tta_transforms)}"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            view_probs.append(probs)

        view_probs = np.concatenate(view_probs, axis=0)

        if all_probs is None:
            all_probs = view_probs
        else:
            all_probs += view_probs

    # Average
    all_probs /= len(tta_transforms)
    all_preds = all_probs.argmax(axis=1)
    all_targets = np.array(dataset.targets)

    return all_preds, all_targets


# ============================================================================
# CONFUSION MATRIX PLOT
# ============================================================================
def plot_confusion_matrix(cm, classes, output_path):
    """Save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("  matplotlib/seaborn not available — skipping confusion matrix plot")
        return

    fig_size = max(10, len(classes) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Normalise rows (recall-based)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained skin disease model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory with test/ subfolder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (5 views)")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory for evaluation results")
    parser.add_argument("--img-size", type=int, default=384)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Load model ──
    print()
    print("=" * 65)
    print("  SKIN DISEASE MODEL EVALUATION")
    print("=" * 65)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Test data  : {args.data_dir}/test")
    print(f"  TTA        : {'Enabled (5 views)' if args.tta else 'Disabled'}")
    print(f"  Device     : {device}")
    print()

    model, classes, config = load_model(args.checkpoint, device)
    num_classes = len(classes)
    print(f"  Classes    : {num_classes}")
    print()

    # ── Load class info ──
    info_path = Path(args.data_dir) / "class_info.json"
    cancer_classes = []
    if info_path.exists():
        with open(info_path) as f:
            class_info = json.load(f)
        cancer_classes = class_info.get("cancer_classes", [])

    # ── Dataset ──
    test_dir = Path(args.data_dir) / "test"
    if not test_dir.is_dir():
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    # ── Evaluate ──
    if args.tta:
        transforms_list = get_tta_transforms(args.img_size)
        test_ds = SkinDataset(str(test_dir), transform=transforms_list[0], img_size=args.img_size)
        all_preds, all_targets = evaluate_tta(
            model, test_ds, transforms_list, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
    else:
        transform = get_eval_transform(args.img_size)
        test_ds = SkinDataset(str(test_dir), transform=transform, img_size=args.img_size)
        loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        all_preds, all_targets = evaluate_standard(model, loader, device, num_classes)

    # ── Metrics ──
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)

    overall_acc = 100.0 * (all_preds == all_targets).mean()
    balanced_acc = 100.0 * balanced_accuracy_score(all_targets, all_preds)

    print(f"\n  Overall Accuracy   : {overall_acc:.2f}%")
    print(f"  Balanced Accuracy  : {balanced_acc:.2f}%")
    print(f"  Total Test Samples : {len(all_targets)}")
    print()

    # Per-class report
    report = classification_report(
        all_targets, all_preds,
        target_names=classes,
        digits=3,
        zero_division=0,
    )
    print("  Per-class Classification Report:")
    print("  " + "-" * 60)
    for line in report.split("\n"):
        print(f"  {line}")

    # Cancer sensitivity analysis
    if cancer_classes:
        print()
        print("  Cancer Sensitivity Analysis:")
        print("  " + "-" * 40)
        for cls_name in cancer_classes:
            if cls_name in classes:
                cls_idx = classes.index(cls_name)
                mask = all_targets == cls_idx
                if mask.sum() > 0:
                    cls_correct = (all_preds[mask] == cls_idx).sum()
                    sensitivity = 100.0 * cls_correct / mask.sum()
                    fn = mask.sum() - cls_correct
                    print(f"    {cls_name:25s}  Sensitivity: {sensitivity:.1f}%  "
                          f"({cls_correct}/{mask.sum()})  FN: {fn}")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, classes, output_dir / "confusion_matrix.png")

    # Save detailed results
    results = {
        "overall_accuracy": round(overall_acc, 3),
        "balanced_accuracy": round(balanced_acc, 3),
        "total_samples": int(len(all_targets)),
        "num_classes": num_classes,
        "tta_enabled": args.tta,
        "checkpoint": args.checkpoint,
        "per_class": {},
    }

    for i, cls_name in enumerate(classes):
        mask = all_targets == i
        n = int(mask.sum())
        correct = int((all_preds[mask] == i).sum()) if n > 0 else 0
        results["per_class"][cls_name] = {
            "samples": n,
            "correct": correct,
            "accuracy": round(100.0 * correct / max(n, 1), 2),
        }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Save confusion matrix as CSV
    cm_path = output_dir / "confusion_matrix.csv"
    import csv
    with open(cm_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + classes)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + row.tolist())
    print(f"  Confusion matrix CSV: {cm_path}")

    print()
    print("=" * 65)
    print(f"  Evaluation complete. Results in: {output_dir}/")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
