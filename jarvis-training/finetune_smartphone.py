#!/usr/bin/env python3
"""
Fine-tuning Script — Domain Adaptation for Smartphone Photos
=============================================================
Fixes the domain gap between clinical training data and real-world smartphone
photos by fine-tuning the last 2 ConvNeXt stages + head on smartphone-sourced
skin disease datasets.

Datasets used (auto-downloaded):
  • PAD-UFES-20   — 2,298 images, Brazilian clinics, smartphones
  • Fitzpatrick17k — 16,577 images, diverse skin tones, clinical cameras

Strategy (Option B):
  • FREEZE  : stem, stage 0, stage 1  (keep low-level texture features)
  • UNFREEZE: stage 2, stage 3, head  (adapt high-level semantics)

Checkpoint:
  The existing best_model.pth is the REQUIRED starting point.
  It is auto-downloaded from HuggingFace (arjg/skin-doc-model) if not found
  locally. You do NOT need to re-upload anything — just run this script.

Usage:
  # On Jarvis Labs (RTX 6000 Ada, ~35 min):
  python finetune_smartphone.py

  # Custom paths:
  python finetune_smartphone.py --checkpoint /path/to/best_model.pth
  python finetune_smartphone.py --epochs 20 --batch-size 64 --lr 5e-5

Output:
  checkpoints/finetuned_smartphone.pth  — drop-in replacement for best_model.pth
"""

import os
import sys
import csv
import json
import shutil
import zipfile
import hashlib
import logging
import argparse
import warnings
import traceback
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Our 20 target classes ────────────────────────────────────────────────────
TARGET_CLASSES = sorted([
    "fungal", "impetigo", "nevus", "eczema", "scabies",
    "viral", "melanoma", "seborrheic_keratosis", "ak",
    "angioma", "psoriasis", "lupus", "wart", "dermatitis",
    "systemic", "hyperpigmentation", "vasculitis", "bullous",
    "drug_eruption", "alopecia",
])
# NOTE: must be sorted — same order the original model was trained with.

# ── External dataset → our class mappings ───────────────────────────────────
# PAD-UFES-20 labels → our class names
PADUFES_MAP = {
    "MEL": "melanoma",              # Melanoma      → melanoma
    "NEV": "nevus",                 # Nevus          → nevus
    "ACK": "ak",                    # Actinic Keratosis → ak
    "SEK": "seborrheic_keratosis",  # Seborrheic Keratosis
    "BCC": "melanoma",              # Basal Cell Carcinoma — closest cancer class
    "SCC": "ak",                    # Squamous Cell Carcinoma — pre/cancer
}

# Fitzpatrick17k labels → our class names (maps ~30 of 114 classes, rest skipped)
FITZPATRICK_MAP = {
    "eczema": "eczema",
    "atopic-dermatitis": "eczema",
    "contact-dermatitis": "dermatitis",
    "psoriasis": "psoriasis",
    "melanoma": "melanoma",
    "nevus": "nevus",
    "seborrheic-keratosis": "seborrheic_keratosis",
    "actinic-keratosis": "ak",
    "tinea": "fungal",
    "tinea-pedis": "fungal",
    "tinea-corporis": "fungal",
    "tinea-versicolor": "fungal",
    "candidiasis": "fungal",
    "wart": "wart",
    "condylomata": "wart",
    "verruca": "wart",
    "impetigo": "impetigo",
    "folliculitis": "impetigo",
    "hyperpigmentation": "hyperpigmentation",
    "post-inflammatory-hyperpigmentation": "hyperpigmentation",
    "melasma": "hyperpigmentation",
    "alopecia": "alopecia",
    "alopecia-areata": "alopecia",
    "drug-eruption": "drug_eruption",
    "fixed-drug-eruption": "drug_eruption",
    "lupus": "lupus",
    "cutaneous-lupus": "lupus",
    "vasculitis": "vasculitis",
    "bullous-pemphigoid": "bullous",
    "pemphigus": "bullous",
    "viral-exanthem": "viral",
    "herpes": "viral",
    "scabies": "scabies",
    "angioma": "angioma",
    "cherry-angioma": "angioma",
    "systemic-lupus": "systemic",
}


# ============================================================================
# CONFIG
# ============================================================================
@dataclass
class FinetuneConfig:
    backbone: str  = "convnext_large.fb_in22k_ft_in1k_384"
    img_size: int  = 384
    epochs:   int  = 15
    batch_size: int = 32
    lr:       float = 5e-5     # lower LR for fine-tuning
    min_lr:   float = 1e-7
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    drop_rate: float = 0.4
    drop_path_rate: float = 0.3
    num_workers: int = 4
    # Unfreeze these ConvNeXt stage indices (0-indexed, 4 stages total)
    unfreeze_stages: tuple = (2, 3)  # last 2 stages + head
    checkpoint: str = ""
    hf_repo: str = "arjg/skin-doc-model"
    hf_filename: str = "best_model.pth"
    data_dir: str = "finetune_data"
    output_dir: str = "checkpoints"
    output_name: str = "finetuned_smartphone.pth"


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================
def download_with_progress(url: str, dest: Path, desc: str = ""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("  Already downloaded: %s", dest.name)
        return
    log.info("  Downloading %s → %s", desc or url, dest)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False)
        while True:
            chunk = r.read(65536)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))
        bar.close()


def download_padufes(data_dir: Path) -> Path:
    """
    Download PAD-UFES-20 from Kaggle using the kaggle CLI.
    Requires: pip install kaggle  +  ~/.kaggle/kaggle.json credentials.
    Dataset: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
    Images are split across imgs_part_1/, imgs_part_2/, imgs_part_3/.
    """
    import subprocess
    out_dir = data_dir / "pad_ufes_20"
    if (out_dir / "metadata.csv").exists():
        log.info("PAD-UFES-20 already present at %s", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        log.error("="*60)
        log.error("Kaggle credentials not found at ~/.kaggle/kaggle.json")
        log.error("")
        log.error("  1. Go to https://www.kaggle.com/settings")
        log.error("  2. Click 'Create New Token' -> downloads kaggle.json")
        log.error("  3. On this machine run:")
        log.error("       mkdir -p ~/.kaggle")
        log.error("       mv /path/to/kaggle.json ~/.kaggle/kaggle.json")
        log.error("       chmod 600 ~/.kaggle/kaggle.json")
        log.error("  4. Re-run this script.")
        log.error("="*60)
        sys.exit(1)

    log.info("Downloading PAD-UFES-20 from Kaggle (mahdavi1202/skin-cancer) ...")
    log.info("This may take several minutes.")
    result = subprocess.run(
        ["kaggle", "datasets", "download",
         "-d", "mahdavi1202/skin-cancer",
         "--unzip",
         "-p", str(out_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("kaggle download failed (exit code %d)", result.returncode)
        log.error("Try running manually: kaggle datasets download -d mahdavi1202/skin-cancer --unzip -p %s", out_dir)
        sys.exit(1)

    # Flatten if Kaggle nested into a subfolder (e.g. out_dir/skin-cancer/...)
    for sub in sorted(out_dir.iterdir()):
        if sub.is_dir() and (sub / "metadata.csv").exists():
            log.info("Flattening subfolder %s -> %s", sub, out_dir)
            for f in sub.iterdir():
                dest = out_dir / f.name
                if not dest.exists():
                    f.rename(dest)
            sub.rmdir()
            break

    if not (out_dir / "metadata.csv").exists():
        log.error("metadata.csv not found after download. Check: %s", out_dir)
        sys.exit(1)

    log.info("PAD-UFES-20 downloaded successfully to %s", out_dir)
    return out_dir


def download_fitzpatrick(data_dir: Path) -> Path:
    """
    Download Fitzpatrick17k dataset via HuggingFace datasets library.
    Falls back to direct CSV + image download if datasets not installed.
    """
    out_dir = data_dir / "fitzpatrick17k"
    if (out_dir / "done.flag").exists():
        log.info("Fitzpatrick17k already present.")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
        log.info("Downloading Fitzpatrick17k via HuggingFace datasets...")
        ds = load_dataset("nateraw/fitzpatrick17k", split="train")

        images_dir = out_dir / "images"
        images_dir.mkdir(exist_ok=True)
        meta_rows = []

        for i, item in enumerate(tqdm(ds, desc="Fitzpatrick17k", leave=False)):
            try:
                label = item.get("label", item.get("three_partition_label", ""))
                img   = item.get("image", None)
                if img is None or not label:
                    continue
                img_path = images_dir / f"{i:06d}.jpg"
                img.save(img_path, "JPEG", quality=90)
                meta_rows.append({"filename": img_path.name, "label": str(label).lower()})
            except Exception:
                continue

        with open(out_dir / "metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "label"])
            w.writeheader()
            w.writerows(meta_rows)

        (out_dir / "done.flag").write_text("ok")
        log.info("  Fitzpatrick17k: %d images saved", len(meta_rows))

    except Exception as e:
        log.warning("Could not load Fitzpatrick17k via datasets: %s", e)
        log.warning("Install with: pip install datasets")
        log.warning("Skipping Fitzpatrick17k — continuing with PAD-UFES-20 only")

    return out_dir


def load_checkpoint_from_hf(cfg: FinetuneConfig, device: torch.device) -> Path:
    """Download best_model.pth from HuggingFace if not present."""
    cp = Path(cfg.checkpoint) if cfg.checkpoint else None

    # Check local paths in order
    candidates = [
        cp,
        Path("../backend/ml/weights/best_model.pth"),
        Path("checkpoints/best_model.pth"),
        Path("best_model.pth"),
    ]
    for p in candidates:
        if p and p.exists():
            log.info("Using local checkpoint: %s", p)
            return p

    # Download from HuggingFace
    log.info("Checkpoint not found locally — downloading from HuggingFace: %s/%s",
             cfg.hf_repo, cfg.hf_filename)
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        local = hf_hub_download(
            repo_id=cfg.hf_repo,
            filename=cfg.hf_filename,
            local_dir=Path(cfg.output_dir),
        )
        log.info("Downloaded to: %s", local)
        return Path(local)
    except Exception as e:
        log.error("="*60)
        log.error("Cannot download checkpoint from HuggingFace: %s", e)
        log.error("")
        log.error("The HuggingFace repo is private. Copy best_model.pth to Jarvis:")
        log.error("  On your LOCAL machine run:")
        log.error("    scp backend/ml/weights/best_model.pth root@<JARVIS_IP>:/root/AI_Skin_doc/jarvis-training/checkpoints/best_model.pth")
        log.error("  Then re-run: bash run_finetune.sh")
        log.error("="*60)
        sys.exit(1)


# ============================================================================
# DATASET
# ============================================================================
class SmartphoneDataset(Dataset):
    """
    Mixed dataset from PAD-UFES-20 + Fitzpatrick17k.
    Only keeps samples whose label maps to one of our 20 target classes.
    """

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label


def build_samples(data_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[str, int]]:
    """Parse PAD-UFES-20 and Fitzpatrick17k metadata into (path, class_idx) pairs."""
    samples = []
    skipped = 0

    # ── PAD-UFES-20 ──
    padufes_dir = data_dir / "pad_ufes_20"
    meta_path = padufes_dir / "metadata.csv"
    if meta_path.exists():
        # Images are split across imgs_part_1/, imgs_part_2/, imgs_part_3/
        # Kaggle sometimes nests an extra subfolder inside each part
        img_search_dirs = [
            padufes_dir / "imgs_part_1" / "imgs_part_1",
            padufes_dir / "imgs_part_2" / "imgs_part_2",
            padufes_dir / "imgs_part_3" / "imgs_part_3",
            padufes_dir / "imgs_part_1",
            padufes_dir / "imgs_part_2",
            padufes_dir / "imgs_part_3",
            padufes_dir / "padufes",
            padufes_dir / "images",
            padufes_dir,
        ]
        # Build a filename→path index for fast lookup
        img_index: dict[str, Path] = {}
        for search_dir in img_search_dirs:
            if search_dir.is_dir():
                for p in search_dir.iterdir():
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                        img_index[p.name] = p

        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_label = row.get("diagnostic", "").strip().upper()
                our_class = PADUFES_MAP.get(raw_label)
                if not our_class or our_class not in class_to_idx:
                    skipped += 1
                    continue
                img_id = row.get("img_id", "")
                img_path = img_index.get(img_id)
                if img_path is None:
                    skipped += 1
                    continue
                samples.append((str(img_path), class_to_idx[our_class]))
        log.info("PAD-UFES-20: %d usable samples", len(samples))

    fitz_start = len(samples)

    # ── Fitzpatrick17k ──
    fitz_dir = data_dir / "fitzpatrick17k"
    fitz_meta = fitz_dir / "metadata.csv"
    if fitz_meta.exists():
        with open(fitz_meta) as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get("label", "").strip().lower()
                our_class = FITZPATRICK_MAP.get(raw)
                if not our_class or our_class not in class_to_idx:
                    skipped += 1
                    continue
                img_path = fitz_dir / "images" / row["filename"]
                if not img_path.exists():
                    skipped += 1
                    continue
                samples.append((str(img_path), class_to_idx[our_class]))
        log.info("Fitzpatrick17k: %d usable samples", len(samples) - fitz_start)

    log.info("Total combined: %d samples (%d skipped/unmapped)", len(samples), skipped)
    if not samples:
        log.error("No samples found! Check data_dir: %s", data_dir)
        sys.exit(1)
    return samples


def get_train_transform(img_size: int) -> T.Compose:
    resize_to = int(img_size * 1.25)
    return T.Compose([
        T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0),
                            interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        # Aggressive colour augmentation to handle smartphone lighting variation
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2),
    ])


def get_val_transform(img_size: int) -> T.Compose:
    resize_to = int(img_size * 1.143)
    return T.Compose([
        T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# MODEL
# ============================================================================
def load_model_for_finetuning(ckpt_path: Path, cfg: FinetuneConfig,
                               device: torch.device):
    """
    Load the trained checkpoint, freeze early stages, unfreeze last 2 + head.
    """
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    classes = ckpt.get("classes", [])
    if not classes:
        raise ValueError("Checkpoint has no 'classes' key.")
    classes = sorted(classes)  # must match training order

    config   = ckpt.get("config", {})
    backbone = config.get("backbone", cfg.backbone)
    drop_r   = config.get("drop_rate", cfg.drop_rate)
    drop_p   = config.get("drop_path_rate", cfg.drop_path_rate)

    model = timm.create_model(
        backbone, pretrained=False,
        num_classes=len(classes),
        drop_rate=drop_r,
        drop_path_rate=drop_p,
    )

    # Load EMA weights (best accuracy)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    if state is None:
        raise ValueError("No weights in checkpoint.")
    model.load_state_dict(state)
    log.info("  Loaded %s weights", "EMA" if "ema_state_dict" in ckpt else "model")
    log.info("  Classes: %d  |  Backbone: %s", len(classes), backbone)

    # ── Freeze ──────────────────────────────────────────────────────────────
    # ConvNeXt-Large structure: model.stem, model.stages[0..3], model.head
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze specified stages and head
    unfreeze_modules = [model.head]
    for stage_idx in cfg.unfreeze_stages:
        unfreeze_modules.append(model.stages[stage_idx])

    for module in unfreeze_modules:
        for p in module.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "  Unfrozen stages: %s + head  |  Trainable params: %s / %s (%.1f%%)",
        list(cfg.unfreeze_stages),
        f"{trainable:,}", f"{total:,}",
        100 * trainable / total,
    )

    return model.to(device), classes


# ============================================================================
# EMA
# ============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v.float() * (1 - self.decay)

    def state_dict(self):
        return {k: v.half() if v.dtype == torch.float32 else v
                for k, v in self.shadow.items()}


# ============================================================================
# TRAINING
# ============================================================================
@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    bar = tqdm(loader, desc="Validating", leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            logits = model(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += len(labels)
        bar.set_postfix(acc=f"{100*correct/max(1,total):.1f}%")
    return 100.0 * correct / max(1, total)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    total_loss = correct = total = 0
    bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)
    for imgs, labels in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
        bar.set_postfix(loss=f"{loss.item():.3f}",
                        acc=f"{100*correct/max(1,total):.1f}%")
    return total_loss / max(1, total), 100.0 * correct / max(1, total)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Smartphone fine-tuning for skin model")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to best_model.pth (auto-downloaded if not set)")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--data-dir",   type=str,   default="finetune_data")
    parser.add_argument("--output-dir", type=str,   default="checkpoints")
    parser.add_argument("--output-name",type=str,   default="finetuned_smartphone.pth")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (data already in data-dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints/resume.pth if it exists")
    parser.add_argument("--num-workers", type=int,  default=4,
                        help="DataLoader workers (auto-set to 0 on MPS)")
    parser.add_argument("--unfreeze-stages", type=int, nargs="+", default=[2, 3],
                        choices=[0, 1, 2, 3],
                        help="ConvNeXt stage indices to unfreeze (0-3). Default: 2 3")
    args = parser.parse_args()

    cfg = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        checkpoint=args.checkpoint,
        num_workers=args.num_workers,
        unfreeze_stages=tuple(args.unfreeze_stages),
    )

    # ── Device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Device: CUDA (%s)", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Device: CPU (will be slow)")

    # ── Data ──
    data_dir = Path(cfg.data_dir)
    if not args.skip_download:
        log.info("=" * 60)
        log.info("DOWNLOADING DATASETS")
        log.info("=" * 60)
        download_padufes(data_dir)
        download_fitzpatrick(data_dir)
    else:
        log.info("Skipping download (--skip-download set)")

    # ── Checkpoint ──
    log.info("=" * 60)
    log.info("LOADING MODEL")
    log.info("=" * 60)
    ckpt_path = load_checkpoint_from_hf(cfg, device)
    model, classes = load_model_for_finetuning(ckpt_path, cfg, device)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes  = len(classes)

    # ── Samples ──
    log.info("=" * 60)
    log.info("BUILDING DATASET")
    log.info("=" * 60)
    all_samples = build_samples(data_dir, class_to_idx)

    # 90/10 train/val split
    import random
    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]
    log.info("Train: %d  |  Val: %d", len(train_samples), len(val_samples))

    # Weighted sampler to balance classes
    label_counts: dict[int, int] = defaultdict(int)
    for _, lbl in train_samples:
        label_counts[lbl] += 1
    weights = [1.0 / max(1, label_counts[lbl]) for _, lbl in train_samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ds = SmartphoneDataset(train_samples, get_train_transform(cfg.img_size))
    val_ds   = SmartphoneDataset(val_samples,   get_val_transform(cfg.img_size))

    pin = device.type == "cuda"  # pin_memory only benefits CUDA, hurts MPS
    nw  = cfg.num_workers if device.type == "cuda" else 0  # MPS: workers cause overhead
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=nw, pin_memory=pin,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    # ── Optimizer — only trainable params ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    # Cosine schedule with linear warmup
    total_steps  = cfg.epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.min_lr / cfg.lr + 0.5 * (1 - cfg.min_lr / cfg.lr) * (
            1 + np.cos(np.pi * progress)
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler(enabled=device.type == "cuda")
    ema       = EMA(model, cfg.ema_decay)

    # ── Train ──
    log.info("=" * 60)
    log.info("FINE-TUNING  (epochs=%d  lr=%.1e  batch=%d)",
             cfg.epochs, cfg.lr, cfg.batch_size)
    log.info("Unfreezing stages %s + head", list(cfg.unfreeze_stages))
    log.info("=" * 60)

    best_val_acc = 0.0
    start_epoch  = 0
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path   = out_dir / cfg.output_name
    resume_path = out_dir / "resume.pth"

    if args.resume and resume_path.exists():
        log.info("=" * 60)
        log.info("RESUMING from %s", resume_path)
        resume_ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        ema.shadow = {k: v.float() for k, v in resume_ckpt["ema_state_dict"].items()}
        start_epoch  = resume_ckpt["epoch"] + 1
        best_val_acc = resume_ckpt["best_val_acc"]
        log.info("  Continuing from epoch %d/%d  (best val so far: %.2f%%)",
                 start_epoch, cfg.epochs, best_val_acc)
        log.info("=" * 60)
    elif args.resume:
        log.warning("--resume set but %s not found — starting fresh", resume_path)

    for epoch in range(start_epoch, cfg.epochs):
        loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg.epochs
        )
        ema.update(model)
        scheduler.step()

        # Validate with EMA weights
        # Temporarily apply EMA to model for validation
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(
            {k: v.to(device) for k, v in ema.state_dict().items()}, strict=False
        )
        val_acc = evaluate(model, val_loader, device)
        model.load_state_dict(original_state)  # restore

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %2d/%d  |  loss %.4f  train %.1f%%  val %.1f%%  lr %.2e",
            epoch + 1, cfg.epochs, loss, train_acc, val_acc, lr_now,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save in the same checkpoint format as the original training
            ckpt_orig = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            save_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "classes": classes,   # sorted — same as original
                "config": ckpt_orig.get("config", {}),
                "best_val_acc": best_val_acc,
                "finetune_info": {
                    "base_checkpoint": str(ckpt_path),
                    "unfrozen_stages": list(cfg.unfreeze_stages),
                    "finetune_epochs": cfg.epochs,
                    "finetune_lr": cfg.lr,
                    "val_acc": val_acc,
                },
            }
            torch.save(save_ckpt, best_path)
            log.info("  ✅ Saved best checkpoint → %s  (val %.2f%%)", best_path, best_val_acc)

        # Always save resume checkpoint after every epoch (crash recovery)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.shadow,
            "best_val_acc": best_val_acc,
            "classes": classes,
        }, resume_path)

    log.info("=" * 60)
    log.info("DONE  |  Best val acc: %.2f%%", best_val_acc)
    log.info("Output: %s", best_path)
    log.info("")
    log.info("To use the new checkpoint, copy it to backend/ml/weights/:")
    log.info("  cp %s ../backend/ml/weights/best_model.pth", best_path)
    log.info("  # or set LESION_MODEL_WEIGHTS=%s", best_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
