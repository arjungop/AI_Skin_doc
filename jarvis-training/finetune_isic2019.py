#!/usr/bin/env python3
"""
Fast ISIC 2019 Fine-tuning — fits in ~45 min on RTX A6000 48GB
===============================================================
Downloads ISIC 2019 from public S3 (no auth needed), then fine-tunes
the last ConvNeXt stage + head only for maximum speed.

Speed design:
  • img_size  = 224  (4× fewer pixels than 384; ConvNeXt is fully conv)
  • batch     = 128  (fills 48 GB VRAM)
  • epochs    = 8
  • AMP fp16  mixed precision
  • Freeze stem + stages 0-2; unfreeze stage 3 + head only
  • No EMA    (saves ~10% time)

Estimated wall time on A6000:
  Download ISIC 2019 (~9 GB S3):  ~15 min
  Extract:                          ~5 min
  Training 8 × 25k images:        ~20 min
  ─────────────────────────────────────────
  Total:                           ~40 min  ← comfortably < 1 hour

ISIC 2019 → our 20 class mapping:
  MEL  → melanoma
  NV   → nevus
  BCC  → melanoma   (basal cell carcinoma, closest malignant class)
  AK   → ak
  BKL  → seborrheic_keratosis
  DF   → dermatitis  (dermatofibroma, benign)
  VASC → angioma
  SCC  → ak         (squamous cell carcinoma, pre/malignant)

Output:
  checkpoints/finetuned_isic2019.pth   — drop-in for best_model.pth
"""

import os
import sys
import csv
import json
import shutil
import zipfile
import logging
import argparse
import warnings
import traceback
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import timm
except ImportError:
    print("ERROR: pip install timm>=1.0.0")
    sys.exit(1)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TARGET_CLASSES = sorted([
    "fungal", "impetigo", "nevus", "eczema", "scabies",
    "viral", "melanoma", "seborrheic_keratosis", "ak",
    "angioma", "psoriasis", "lupus", "wart", "dermatitis",
    "systemic", "hyperpigmentation", "vasculitis", "bullous",
    "drug_eruption", "alopecia",
])  # 20 classes, MUST stay sorted (matches original training)

# ISIC 2019 one-hot column → our class
ISIC2019_MAP = {
    "MEL":  "melanoma",             # Melanoma
    "NV":   "nevus",                # Melanocytic nevi (moles)
    "BCC":  "melanoma",             # Basal cell carcinoma (malignant cancer)
    "AK":   "ak",                   # Actinic keratosis
    "BKL":  "seborrheic_keratosis", # Benign keratosis (SK / solar lentigo)
    "DF":   "dermatitis",           # Dermatofibroma
    "VASC": "angioma",              # Vascular lesions
    "SCC":  "ak",                   # Squamous cell carcinoma
}

ISIC2019_TRAIN_ZIP = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/"
    "ISIC_2019_Training_Input.zip"
)
ISIC2019_GT_CSV = (
    "https://isic-challenge-data.s3.amazonaws.com/2019/"
    "ISIC_2019_Training_GroundTruth.csv"
)


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class Cfg:
    # model
    backbone:   str   = "convnext_large.fb_in22k_ft_in1k_384"
    img_size:   int   = 224   # smaller = faster; ConvNeXt fully-conv, works fine
    # training
    epochs:     int   = 8
    batch_size: int   = 128
    lr:         float = 3e-4   # higher LR ok — head is random init'd
    head_lr:    float = 3e-4
    body_lr:    float = 3e-5   # lower for unfrozen body layers
    min_lr:     float = 1e-6
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    grad_clip:  float = 1.0
    # data
    val_split:  float = 0.10   # 10% holdout
    num_workers: int  = 8
    # paths
    data_dir:   str   = "finetune_data/isic2019"
    output_dir: str   = "checkpoints"
    output_name: str  = "finetuned_isic2019.pth"
    checkpoint: str   = ""     # filled at runtime
    hf_repo:    str   = "arjg/skin-doc-model"
    hf_filename: str  = "best_model.pth"


# ============================================================================
# DOWNLOAD
# ============================================================================
def _download_url(url: str, dest: Path, desc: str = ""):
    """
    Download using the fastest available tool:
      1. aria2c  -x 16  (16 parallel connections — fastest)
      2. wget    (single connection but faster than urllib)
      3. curl    (fallback)
      4. urllib  (last resort)
    """
    import subprocess, shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info("  Already downloaded: %s", dest.name)
        return
    log.info("  Downloading %s → %s", desc or url, dest)

    if shutil.which("aria2c"):
        log.info("  Using aria2c (16 parallel connections) …")
        subprocess.run([
            "aria2c", "-x", "16", "-s", "16", "-k", "1M",
            "--file-allocation=none",
            "--console-log-level=warn",
            "-o", dest.name, "-d", str(dest.parent), url,
        ], check=True)
    elif shutil.which("wget"):
        log.info("  Using wget …")
        subprocess.run([
            "wget", "-q", "--show-progress", "-c", url, "-O", str(dest),
        ], check=True)
    elif shutil.which("curl"):
        log.info("  Using curl …")
        subprocess.run([
            "curl", "-L", "-C", "-", "--progress-bar", url, "-o", str(dest),
        ], check=True)
    else:
        log.info("  Using urllib (no wget/aria2c found) …")
        with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False)
            while True:
                chunk = r.read(131_072)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()


def download_isic2019(data_dir: Path) -> tuple[Path, Path]:
    """
    Download ISIC 2019 training images + ground-truth CSV.
    Returns (images_dir, gt_csv_path).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ISIC_2019_Training_Input.zip"
    gt_path  = data_dir / "ISIC_2019_Training_GroundTruth.csv"
    images_dir = data_dir / "ISIC_2019_Training_Input"

    # 1. Ground truth CSV (tiny, always fetch)
    _download_url(ISIC2019_GT_CSV, gt_path, "ISIC 2019 labels")

    # 2. Images ZIP (~9 GB) — skip if already extracted
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 1000:
        _download_url(ISIC2019_TRAIN_ZIP, zip_path, "ISIC 2019 images (9 GB)")
        log.info("  Extracting %s …", zip_path.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        log.info("  Extracted → %s", images_dir)
        # Remove zip to save space
        zip_path.unlink(missing_ok=True)
    else:
        log.info("  Images already extracted at %s (%d jpegs)",
                 images_dir, len(list(images_dir.glob("*.jpg"))))

    return images_dir, gt_path


# ============================================================================
# DATASET
# ============================================================================
class ISICDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: T.Compose,
    ):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            # Return a black image on corruption
            img = Image.new("RGB", (224, 224), 0)
            return self.transform(img), label


def build_samples(
    images_dir: Path,
    gt_csv: Path,
    val_split: float = 0.10,
    rng: np.random.Generator | None = None,
) -> tuple[list, list, list[str]]:
    """
    Parse ISIC 2019 ground-truth CSV and build (path, label) sample lists.
    Returns (train_samples, val_samples, class_names).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Build image_id → path index
    log.info("  Indexing images in %s …", images_dir)
    path_index: dict[str, Path] = {}
    for p in images_dir.glob("*.jpg"):
        path_index[p.stem] = p
    log.info("  Found %d images", len(path_index))

    # ISIC 2019 csv header:
    # image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK
    all_samples: list[tuple[Path, int]] = []
    skipped = 0
    with open(gt_csv, newline="") as f:
        reader = csv.DictReader(f)
        isic_cols = [c for c in reader.fieldnames if c != "image"]
        for row in reader:
            # Find the '1' column (one-hot)
            our_cls = None
            for col in isic_cols:
                if row.get(col, "0").strip() == "1":
                    our_cls = ISIC2019_MAP.get(col)
                    break
            if our_cls is None or our_cls not in TARGET_CLASSES:
                skipped += 1
                continue
            img_id = row["image"].strip()
            path   = path_index.get(img_id)
            if path is None:
                skipped += 1
                continue
            label = TARGET_CLASSES.index(our_cls)
            all_samples.append((path, label))

    log.info("  Loaded %d samples, %d skipped (unmapped/missing)", len(all_samples), skipped)

    # Class distribution
    counts = Counter(label for _, label in all_samples)
    log.info("  Class distribution:")
    for cls_idx, cnt in sorted(counts.items()):
        log.info("    %-28s %d", TARGET_CLASSES[cls_idx], cnt)

    # Shuffle + split
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_split))
    val_idx   = indices[:n_val]
    train_idx = indices[n_val:]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]
    log.info("  Train: %d | Val: %d", len(train_samples), len(val_samples))
    return train_samples, val_samples, TARGET_CLASSES


def make_weighted_sampler(samples: list) -> WeightedRandomSampler:
    labels = [lbl for _, lbl in samples]
    counts = Counter(labels)
    total  = len(labels)
    weights = [total / counts[lbl] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ============================================================================
# MODEL
# ============================================================================
def load_pretrained_model(cfg: Cfg, device: torch.device) -> tuple[nn.Module, list[str]]:
    """Load original best_model.pth, return (model, classes)."""
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    classes  = sorted(ckpt.get("classes", TARGET_CLASSES))
    n_cls    = len(classes)
    backbone = ckpt.get("config", {}).get("backbone", cfg.backbone)

    log.info("  Backbone: %s | Classes: %d", backbone, n_cls)
    model = timm.create_model(backbone, pretrained=False, num_classes=n_cls,
                               drop_rate=0.3, drop_path_rate=0.2)

    # Prefer EMA weights
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state, strict=False)
    return model.to(device), classes


def freeze_for_fast_finetuning(model: nn.Module) -> None:
    """
    Freeze everything except the last ConvNeXt stage (stage 3) and the head.
    This limits trainable params to ~30% of the model, halving compute.
    """
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze ConvNeXt stage 3 (the last feature stage)
    # timm ConvNeXt: model.stages[3]
    if hasattr(model, "stages"):
        for p in model.stages[3].parameters():
            p.requires_grad = True
        log.info("  Unfrozen: stages[3]")

    # Unfreeze final norm + head
    for name in ["norm_pre", "head"]:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True
            log.info("  Unfrozen: %s", name)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable params: %s / %s (%.1f%%)",
             f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total)


# ============================================================================
# TRANSFORMS
# ============================================================================
def get_transforms(img_size: int) -> tuple[T.Compose, T.Compose]:
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(30),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.14), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler,
    cfg: Cfg,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    bar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]", leave=True)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels, label_smoothing=0.1)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(imgs)
        n          += len(imgs)
        bar.set_postfix(loss=f"{total_loss/n:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / n


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    n          = 0
    for imgs, labels in tqdm(loader, desc="  Val", leave=False):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(imgs)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(imgs)
    return total_loss / n, correct / n


def make_scheduler(optimizer, cfg: Cfg, steps_per_epoch: int):
    """One-cycle LR: linear warmup then cosine decay."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps  = cfg.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================================
# CHECKPOINT SAVE
# ============================================================================
def save_checkpoint(model: nn.Module, classes: list[str], cfg: Cfg, epoch: int, val_acc: float):
    out_dir  = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg.output_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict":   model.state_dict(),  # no EMA — same weights for compat
        "classes":          classes,
        "config": {
            "backbone":       cfg.backbone,
            "img_size":       cfg.img_size,
            "drop_rate":      0.3,
            "drop_path_rate": 0.2,
        },
        "epoch":            epoch,
        "val_acc":          val_acc,
        "dataset":          "ISIC 2019",
    }, out_path)
    log.info("  Saved → %s  (val_acc=%.2f%%)", out_path, val_acc * 100)
    return out_path


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="", help="Path to best_model.pth")
    p.add_argument("--epochs",      type=int,   default=8)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--img-size",    type=int,   default=224)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--data-dir",    default="finetune_data/isic2019")
    p.add_argument("--output-dir",  default="checkpoints")
    p.add_argument("--output-name", default="finetuned_isic2019.pth")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = Cfg(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        img_size   = args.img_size,
        lr         = args.lr,
        data_dir   = args.data_dir,
        output_dir = args.output_dir,
        output_name= args.output_name,
    )

    # ── Device ──────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.error("CUDA GPU required. Exiting.")
        sys.exit(1)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("GPU: %s (%.0f GB)", gpu_name, gpu_mem)

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    if args.checkpoint:
        cfg.checkpoint = args.checkpoint
    else:
        # Look in standard locations
        candidates = [
            Path("../backend/ml/weights/best_model.pth"),
            Path("../backend/ml/weights/best_model_original.pth"),
            Path("checkpoints/best_model.pth"),
        ]
        for c in candidates:
            if c.exists():
                cfg.checkpoint = str(c)
                break
        if not cfg.checkpoint:
            # Try HuggingFace
            log.info("Downloading checkpoint from HuggingFace …")
            try:
                from huggingface_hub import hf_hub_download
                cfg.checkpoint = hf_hub_download(
                    repo_id=cfg.hf_repo, filename=cfg.hf_filename,
                    local_dir="checkpoints",
                )
            except Exception as e:
                log.error("Could not find checkpoint: %s", e)
                log.error("Pass --checkpoint /path/to/best_model.pth")
                sys.exit(1)

    # ── Download ISIC 2019 ───────────────────────────────────────────────────
    data_dir   = Path(cfg.data_dir)
    log.info("=== Downloading ISIC 2019 ===")
    images_dir, gt_csv = download_isic2019(data_dir)

    # ── Build datasets ───────────────────────────────────────────────────────
    log.info("=== Building datasets ===")
    train_samples, val_samples, classes = build_samples(images_dir, gt_csv, cfg.val_split)

    train_tf, val_tf = get_transforms(cfg.img_size)
    train_ds = ISICDataset(train_samples, train_tf)
    val_ds   = ISICDataset(val_samples,   val_tf)

    sampler  = make_weighted_sampler(train_samples)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=cfg.num_workers > 0)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size * 2, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=cfg.num_workers > 0)

    log.info("Train batches/epoch: %d | Val batches: %d", len(train_dl), len(val_dl))

    # ── Load model + freeze ──────────────────────────────────────────────────
    log.info("=== Loading model ===")
    model, classes = load_pretrained_model(cfg, device)
    freeze_for_fast_finetuning(model)

    # ── Optimizer (two param groups: head vs body) ───────────────────────────
    head_params = []
    body_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name or "norm_pre" in name:
            head_params.append(p)
        else:
            body_params.append(p)

    optimizer = optim.AdamW([
        {"params": head_params, "lr": cfg.head_lr},
        {"params": body_params, "lr": cfg.body_lr},
    ], weight_decay=cfg.weight_decay)

    scaler    = torch.cuda.amp.GradScaler()
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch=len(train_dl))

    # ── Training loop ────────────────────────────────────────────────────────
    log.info("=== Training (%d epochs, img_size=%d, batch=%d) ===",
             cfg.epochs, cfg.img_size, cfg.batch_size)

    best_acc  = 0.0
    best_path = None
    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_dl, optimizer, scaler, scheduler,
                                 cfg, device, epoch)
        val_loss, val_acc = val_epoch(model, val_dl, device)
        log.info("Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.2f%%",
                 epoch + 1, cfg.epochs, train_loss, val_loss, val_acc * 100)

        if val_acc > best_acc:
            best_acc  = val_acc
            best_path = save_checkpoint(model, classes, cfg, epoch, val_acc)

    log.info("=== Training complete ===")
    log.info("Best val accuracy: %.2f%%", best_acc * 100)
    log.info("Best checkpoint:   %s", best_path)
    print(f"\n{'='*60}")
    print(f"DONE! Best val accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint: {best_path}")
    print(f"\nTo use this model:")
    print(f"  scp jarvis:{Path(best_path).resolve()} ~/Skin-Doc/backend/ml/weights/best_model.pth")
    print(f"{'='*60}\n")

    return best_path


if __name__ == "__main__":
    main()
