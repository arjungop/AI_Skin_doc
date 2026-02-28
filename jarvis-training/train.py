#!/usr/bin/env python3
"""
Production Training Script — 20-Class Skin Disease Classification
Optimized for NVIDIA A6000 (48 GB VRAM) on Jarvis Labs

Architecture : ConvNeXt-Large (timm, ImageNet-22k → 1k pretrained, 384×384)
Techniques   : FocalLoss, EMA, Mixup/CutMix, Gradient Checkpointing,
               OneCycleLR, WeightedRandomSampler, bfloat16 AMP,
               Cancer-class boosted loss, Signal-safe checkpointing

Usage:
  python train.py                               # Train with defaults
  python train.py --epochs 60 --batch-size 32   # Custom settings
  python train.py --resume                       # Auto-resume from latest checkpoint
"""

import os
import sys
import gc
import copy
import json
import math
import time
import signal
import logging
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

try:
    import timm
except ImportError:
    print("ERROR: timm not installed.  Run: pip install timm>=1.0.0")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# GLOBAL TRAINING STATE (for signal handling)
# ============================================================================
class _TrainingState:
    should_stop: bool = False

TRAINING_STATE = _TrainingState()


def _signal_handler(signum, frame):
    TRAINING_STATE.should_stop = True
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] {sig_name} received — will save checkpoint and exit gracefully.")


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class TrainConfig:
    # ── Model ──
    backbone: str = "convnext_large.fb_in22k_ft_in1k_384"
    img_size: int = 384
    drop_rate: float = 0.4
    drop_path_rate: float = 0.3
    gradient_checkpointing: bool = True

    # ── Training ──
    epochs: int = 50
    batch_size: int = 48
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_pct: float = 0.10        # fraction of total steps for warmup
    label_smoothing: float = 0.1
    grad_accum_steps: int = 1
    grad_clip: float = 1.0

    # ── Focal Loss ──
    focal_gamma: float = 2.0
    cancer_weight_boost: float = 2.0   # extra weight for cancer classes

    # ── Mixup / CutMix ──
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5             # probability of applying mix per batch

    # ── EMA ──
    use_ema: bool = True
    ema_decay: float = 0.9998

    # ── Early Stopping ──
    patience: int = 15

    # ── System ──
    num_workers: int = 6              # A6000 → 7 CPUs, leave 1 for main
    seed: int = 42
    amp_dtype: str = "bfloat16"       # A6000 natively supports bf16

    # ── Paths ──
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    tensorboard_dir: str = "runs"

    # ── Derived (set at runtime) ──
    resume: bool = False

    def to_dict(self):
        return asdict(self)


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"train_{ts}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ============================================================================
# SEED & DEVICE
# ============================================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # NOTE: we keep cudnn.benchmark = True for speed; set deterministic only
    # if exact reproducibility is needed at the cost of ~10% speed.
    torch.backends.cudnn.benchmark = True


def get_device(logger: logging.Logger) -> torch.device:
    if not torch.cuda.is_available():
        logger.warning("No GPU detected — training on CPU will be very slow!")
        return torch.device("cpu")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU : {gpu_name}  |  VRAM : {vram_gb:.1f} GB")

    # A6000/Ampere optimisations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 enabled for matmul and cuDNN")

    return device


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(
            img_size, scale=(0.7, 1.0), ratio=(0.85, 1.15),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08),
        T.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=8),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size: int) -> T.Compose:
    resize_to = int(img_size * 1.143)        # 384 → 439
    return T.Compose([
        T.Resize(resize_to, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================================
# DATASET
# ============================================================================
class SkinDataset(Dataset):
    """Folder-based dataset with robust image loading."""

    def __init__(self, root_dir: str, transform=None, img_size: int = 384):
        super().__init__()
        self.root = Path(root_dir)
        self.transform = transform
        self.img_size = img_size

        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        # Discover classes from subdirectories
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if len(self.classes) == 0:
            raise ValueError(f"No class folders found in {self.root}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect samples
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            idx = self.class_to_idx[cls_name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(p), idx))
                    self.targets.append(idx)

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Retry with a random valid sample (max 3 attempts)
            for _ in range(3):
                alt_idx = np.random.randint(len(self))
                alt_path, alt_label = self.samples[alt_idx]
                try:
                    image = Image.open(alt_path).convert("RGB")
                    label = alt_label
                    break
                except Exception:
                    continue
            else:
                image = Image.new("RGB", (self.img_size, self.img_size), "black")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_counts(self) -> Counter:
        return Counter(self.targets)

    def get_weighted_sampler(
        self, cancer_classes: list[str] | None = None, cancer_boost: float = 2.0
    ) -> WeightedRandomSampler:
        """Create WeightedRandomSampler for class balance + cancer boost."""
        counts = self.get_class_counts()
        sample_weights = []
        for _, label in self.samples:
            w = 1.0 / max(counts[label], 1)
            if cancer_classes:
                cls_name = self.classes[label]
                if cls_name in cancer_classes:
                    w *= cancer_boost
            sample_weights.append(w)
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def get_class_weights(
        self, cancer_classes: list[str] | None = None, cancer_boost: float = 2.0
    ) -> torch.Tensor:
        """Inverse-frequency class weights, boosted for cancer."""
        counts = self.get_class_counts()
        weights = []
        for i in range(len(self.classes)):
            w = 1.0 / max(counts.get(i, 1), 1)
            if cancer_classes and self.classes[i] in cancer_classes:
                w *= cancer_boost
            weights.append(w)
        wt = torch.tensor(weights, dtype=torch.float32)
        wt = wt / wt.sum() * len(self.classes)   # normalise
        return wt


# ============================================================================
# FOCAL LOSS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with optional class weights and label smoothing.

    Focuses learning on hard, misclassified examples — critical for reducing
    false negatives in cancer classes.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ============================================================================
# MODEL EMA (Exponential Moving Average)
# ============================================================================
class ModelEMA:
    """Maintains an exponentially-smoothed copy of the model parameters.

    The EMA model typically generalises better than the raw trained model
    and should be used for final evaluation and deployment.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_v, model_v in zip(
            self.module.state_dict().values(),
            model.state_dict().values(),
        ):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)


# ============================================================================
# MIXUP / CUTMIX
# ============================================================================
class MixupCutmix:
    """Batch-level Mixup + CutMix with configurable probability.

    Returns (mixed_images, targets_a, targets_b, lam).
    When no mixing is applied lam == 1.0 and targets_a == targets_b == targets.
    Compatible with ANY loss function (including FocalLoss).
    """

    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def __call__(self, images: torch.Tensor, targets: torch.Tensor):
        if torch.rand(1).item() > self.prob:
            return images, targets, targets, 1.0

        # 50/50 between mixup and cutmix
        if torch.rand(1).item() < 0.5:
            return self._mixup(images, targets)
        else:
            return self._cutmix(images, targets)

    def _mixup(self, images: torch.Tensor, targets: torch.Tensor):
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[idx]
        return mixed, targets, targets[idx], lam

    def _cutmix(self, images: torch.Tensor, targets: torch.Tensor):
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        idx = torch.randperm(images.size(0), device=images.device)
        B, C, H, W = images.shape

        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y2 = min(cy + cut_h // 2, H)

        images_clone = images.clone()
        images_clone[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

        # Adjust lam to actual area ratio
        actual_lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
        return images_clone, targets, targets[idx], actual_lam


# ============================================================================
# MODEL CREATION
# ============================================================================
def create_model(
    num_classes: int,
    cfg: TrainConfig,
    logger: logging.Logger,
) -> nn.Module:
    """Create ConvNeXt-Large via timm with best available pretrained weights."""

    candidates = [
        cfg.backbone,
        "convnext_large.fb_in22k_ft_in1k_384",
        "convnext_large.fb_in22k_ft_in1k",
        "convnext_large",
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    model = None
    used_name = None

    for name in unique:
        try:
            model = timm.create_model(
                name,
                pretrained=True,
                num_classes=num_classes,
                drop_rate=cfg.drop_rate,
                drop_path_rate=cfg.drop_path_rate,
            )
            used_name = name
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(
            f"Could not create model. Tried: {unique}\n"
            f"Run: python -c \"import timm; print(timm.list_models('convnext_large*'))\" "
            f"to see available models."
        )

    # Gradient checkpointing
    if cfg.gradient_checkpointing:
        try:
            model.set_grad_checkpointing(enable=True)
            logger.info("Gradient checkpointing: ENABLED")
        except AttributeError:
            logger.warning("Gradient checkpointing not supported by this model")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model      : {used_name}")
    logger.info(f"Parameters : {total_params:,} total  |  {trainable:,} trainable")

    return model


# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
def preflight_checks(cfg: TrainConfig, logger: logging.Logger):
    """Validate everything before training starts."""
    errors = []

    # Paths
    data_dir = Path(cfg.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    info_path = data_dir / "class_info.json"

    if not train_dir.is_dir():
        errors.append(f"Training directory not found: {train_dir}")
    if not val_dir.is_dir():
        errors.append(f"Validation directory not found: {val_dir}")
    if not info_path.is_file():
        errors.append(f"class_info.json not found: {info_path}")

    # CUDA
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU — training will be extremely slow.")

    # VRAM estimate
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        if cfg.batch_size > 48 and vram < 45:
            logger.warning(
                f"Batch size {cfg.batch_size} may OOM with {vram:.0f}GB VRAM. "
                f"Consider reducing to 32-48 or increasing grad_accum_steps."
            )

    # timm
    try:
        import timm as _t
        logger.info(f"timm version: {_t.__version__}")
    except Exception:
        errors.append("timm library not available")

    if errors:
        for e in errors:
            logger.error(f"PREFLIGHT FAIL: {e}")
        raise RuntimeError("Pre-flight checks failed. See errors above.")

    logger.info("Pre-flight checks: ALL PASSED")


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: optim.Optimizer,
    scheduler,
    best_val_acc: float,
    cfg: TrainConfig,
    classes: list[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "config": cfg.to_dict(),
        "classes": classes,
    }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: ModelEMA | None,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[int, float]:
    """Returns (start_epoch, best_val_acc)."""
    if not path.exists():
        return 0, 0.0

    logger.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not restore optimizer state: {e}")
    if "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            logger.warning(f"Could not restore scheduler state: {e}")
    if ema is not None and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1
    best_acc = ckpt.get("best_val_acc", 0.0)
    logger.info(f"Resumed from epoch {start_epoch}, best_val_acc={best_acc:.2f}%")
    return start_epoch, best_acc


def cleanup_old_checkpoints(ckpt_dir: Path, keep: int = 3):
    """Keep only the last N epoch checkpoints (plus best_model.pth)."""
    epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    for old in epoch_ckpts[:-keep]:
        try:
            old.unlink()
        except Exception:
            pass


# ============================================================================
# TRAINING EPOCH
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    ema: ModelEMA | None,
    mixup_fn: MixupCutmix | None,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter | None,
) -> dict:
    model.train()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"

    total_loss = 0.0
    correct = 0
    total_samples = 0
    nan_count = 0

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [TRAIN]", ncols=110)

    for step, (images, targets) in enumerate(pbar):
        if TRAINING_STATE.should_stop:
            logger.info("Graceful stop requested mid-epoch.")
            return None

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixup / CutMix
        mixed = False
        if mixup_fn is not None:
            images, targets_a, targets_b, lam = mixup_fn(images, targets)
            mixed = lam < 1.0
        else:
            targets_a = targets_b = targets
            lam = 1.0

        # Forward
        try:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                if mixed:
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
                else:
                    loss = criterion(logits, targets)
                loss = loss / cfg.grad_accum_steps
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM in forward pass — clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad(set_to_none=True)
            continue

        # Check NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count > 20:
                logger.error("Too many NaN losses (>20). Stopping training.")
                return None
            optimizer.zero_grad(set_to_none=True)
            continue

        # Backward  (bf16 does NOT need GradScaler)
        loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        # Metrics
        total_loss += loss.item() * cfg.grad_accum_steps
        if not mixed:
            preds = logits.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total_samples += targets.size(0)

        # Progress bar
        avg_loss = total_loss / (step + 1)
        acc_str = f"{100*correct/total_samples:.1f}%" if total_samples > 0 else "N/A"
        lr_str = f"{scheduler.get_last_lr()[0]:.2e}"
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=acc_str, lr=lr_str)

        # TensorBoard (per 100 steps)
        if writer and step % 100 == 0:
            global_step = epoch * len(loader) + step
            writer.add_scalar("Train/StepLoss", loss.item() * cfg.grad_accum_steps, global_step)

    metrics = {"train_loss": total_loss / max(len(loader), 1)}
    if total_samples > 0:
        metrics["train_acc"] = 100.0 * correct / total_samples
    return metrics


# ============================================================================
# VALIDATION EPOCH
# ============================================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    epoch: int,
    num_classes: int,
) -> dict:
    model.eval()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bfloat16" else torch.float16
    use_amp = device.type == "cuda"

    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = Counter()
    per_class_total = Counter()

    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [ VAL ]", ncols=110):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        # Per-class accuracy
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            per_class_total[int(t)] += 1
            if p == t:
                per_class_correct[int(t)] += 1

    # Per-class sensitivity (recall)
    per_class_acc = {}
    low_sensitivity = []
    for cls_id in range(num_classes):
        n = per_class_total.get(cls_id, 0)
        c = per_class_correct.get(cls_id, 0)
        acc = 100.0 * c / n if n > 0 else 0.0
        per_class_acc[cls_id] = acc
        if acc < 70.0 and n > 0:
            low_sensitivity.append((cls_id, acc, n))

    return {
        "val_loss": total_loss / max(len(loader), 1),
        "val_acc": 100.0 * correct / max(total, 1),
        "per_class_acc": per_class_acc,
        "low_sensitivity": low_sensitivity,
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train(cfg: TrainConfig):
    logger = setup_logging(cfg.log_dir)

    logger.info("=" * 65)
    logger.info("  SKIN DISEASE TRAINING — A6000 OPTIMISED")
    logger.info("=" * 65)
    logger.info(f"Config: {json.dumps(cfg.to_dict(), indent=2)}")

    # ── Seed & device ──
    set_seed(cfg.seed)
    device = get_device(logger)

    # ── Pre-flight ──
    preflight_checks(cfg, logger)

    # ── Load class info ──
    info_path = Path(cfg.data_dir) / "class_info.json"
    with open(info_path) as f:
        class_info = json.load(f)
    classes = class_info["classes"]
    num_classes = len(classes)
    cancer_classes = class_info.get("cancer_classes", [])
    logger.info(f"Classes: {num_classes}  |  Cancer: {cancer_classes}")

    # ── Datasets ──
    train_ds = SkinDataset(
        str(Path(cfg.data_dir) / "train"),
        transform=get_train_transforms(cfg.img_size),
        img_size=cfg.img_size,
    )
    val_ds = SkinDataset(
        str(Path(cfg.data_dir) / "val"),
        transform=get_val_transforms(cfg.img_size),
        img_size=cfg.img_size,
    )
    logger.info(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # Verify classes match
    if sorted(train_ds.classes) != sorted(classes):
        logger.warning(
            f"Class mismatch between class_info.json ({len(classes)}) "
            f"and train dir ({len(train_ds.classes)}). Using train dir classes."
        )
        classes = train_ds.classes
        num_classes = len(classes)
        cancer_classes = [c for c in cancer_classes if c in classes]

    # ── DataLoaders ──
    sampler = train_ds.get_weighted_sampler(cancer_classes, cfg.cancer_weight_boost)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,         # val has no gradients → 2× batch
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    logger.info(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ──
    model = create_model(num_classes, cfg, logger)
    model = model.to(device)

    # ── EMA ──
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    if ema:
        logger.info(f"EMA enabled (decay={cfg.ema_decay})")

    # ── Loss ──
    class_weights = train_ds.get_class_weights(cancer_classes, cfg.cancer_weight_boost)
    class_weights = class_weights.to(device)
    criterion = FocalLoss(
        gamma=cfg.focal_gamma,
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )
    logger.info(f"Loss: FocalLoss(gamma={cfg.focal_gamma}, label_smoothing={cfg.label_smoothing})")

    # ── Optimiser ──
    # No weight decay on bias and LayerNorm
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr)

    # ── Scheduler ──
    total_steps = len(train_loader) * cfg.epochs
    # OneCycleLR: initial_lr = max_lr / div_factor, final_lr = initial_lr / final_div_factor
    # With div_factor=25 (default): initial_lr = lr/25, we want final_lr = min_lr
    # So final_div_factor = (lr / 25) / min_lr
    div_factor = 25.0
    final_div_factor = max((cfg.lr / div_factor) / cfg.min_lr, 1.0)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.warmup_pct,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    logger.info(f"Scheduler: OneCycleLR  max_lr={cfg.lr}  warmup={cfg.warmup_pct*100:.0f}%")

    # ── Mixup / CutMix ──
    mixup_fn = MixupCutmix(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        prob=cfg.mix_prob,
    ) if cfg.mix_prob > 0 else None
    if mixup_fn:
        logger.info(f"Mixup/CutMix: prob={cfg.mix_prob}")

    # ── Resume ──
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val_acc = 0.0

    if cfg.resume:
        latest = ckpt_dir / "checkpoint_latest.pth"
        if latest.exists():
            start_epoch, best_val_acc = load_checkpoint(
                latest, model, ema, optimizer, scheduler, device, logger
            )
        else:
            logger.info("No checkpoint found — starting fresh.")

    # ── TensorBoard ──
    tb_dir = Path(cfg.tensorboard_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(str(tb_dir))
    logger.info(f"TensorBoard: {tb_dir}")

    # Save config
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    with open(ckpt_dir / "class_mapping.json", "w") as f:
        json.dump({"classes": classes, "class_to_idx": {c: i for i, c in enumerate(classes)}}, f, indent=2)

    # ── Training loop ──
    logger.info("=" * 65)
    logger.info("  STARTING TRAINING")
    logger.info("=" * 65)

    patience_counter = 0
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        if TRAINING_STATE.should_stop:
            break

        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            ema, mixup_fn, device, cfg, epoch, logger, writer,
        )
        if train_metrics is None:     # Preemption or fatal error
            save_checkpoint(
                ckpt_dir / "checkpoint_latest.pth",
                epoch, model, ema, optimizer, scheduler, best_val_acc, cfg, classes,
            )
            break

        # Validate — use EMA model if available
        eval_model = ema.module if ema else model
        val_metrics = validate(eval_model, val_loader, criterion, device, cfg, epoch, num_classes)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining_epochs = cfg.epochs - epoch - 1
        eta_sec = (elapsed / (epoch - start_epoch + 1)) * remaining_epochs if epoch > start_epoch else 0

        # Log
        train_acc_str = f"{train_metrics.get('train_acc', 0):.1f}%"
        logger.info(
            f"Epoch {epoch+1:>3}/{cfg.epochs} | "
            f"Train Loss {train_metrics['train_loss']:.4f} Acc {train_acc_str} | "
            f"Val Loss {val_metrics['val_loss']:.4f} Acc {val_metrics['val_acc']:.2f}% | "
            f"Time {epoch_time:.0f}s | ETA {eta_sec/60:.0f}min"
        )

        # Log low-sensitivity classes
        if val_metrics["low_sensitivity"]:
            for cls_id, acc, n in val_metrics["low_sensitivity"]:
                cls_name = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"
                logger.warning(f"  Low sensitivity: {cls_name} = {acc:.1f}% ({n} samples)")

        # TensorBoard
        writer.add_scalar("Train/Loss", train_metrics["train_loss"], epoch)
        writer.add_scalar("Val/Loss", val_metrics["val_loss"], epoch)
        writer.add_scalar("Val/Accuracy", val_metrics["val_acc"], epoch)
        if "train_acc" in train_metrics:
            writer.add_scalar("Train/Accuracy", train_metrics["train_acc"], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Best model
        is_best = val_metrics["val_acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["val_acc"]
            patience_counter = 0
            logger.info(f"  ** New best: {best_val_acc:.2f}% **")
            save_checkpoint(
                ckpt_dir / "best_model.pth",
                epoch, model, ema, optimizer, scheduler, best_val_acc, cfg, classes,
            )
        else:
            patience_counter += 1

        # Save latest + periodic
        save_checkpoint(
            ckpt_dir / "checkpoint_latest.pth",
            epoch, model, ema, optimizer, scheduler, best_val_acc, cfg, classes,
        )
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch+1}.pth",
                epoch, model, ema, optimizer, scheduler, best_val_acc, cfg, classes,
            )
        cleanup_old_checkpoints(ckpt_dir, keep=3)

        # Early stopping
        if patience_counter >= cfg.patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={cfg.patience})")
            break

    # ── Done ──
    total_time = time.time() - start_time
    writer.close()

    logger.info("=" * 65)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 65)
    logger.info(f"  Best validation accuracy : {best_val_acc:.2f}%")
    logger.info(f"  Total training time       : {total_time/3600:.2f} hours")
    logger.info(f"  Checkpoints saved to      : {ckpt_dir}")
    logger.info(f"  TensorBoard logs          : {tb_dir}")
    logger.info("")
    logger.info("  Next: python evaluate.py")
    logger.info("=" * 65)

    return best_val_acc


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train ConvNeXt-Large for 20-class skin disease classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument("--backbone", type=str, default="convnext_large.fb_in22k_ft_in1k_384")
    g.add_argument("--img-size", type=int, default=384)
    g.add_argument("--drop-rate", type=float, default=0.4)
    g.add_argument("--drop-path-rate", type=float, default=0.3)
    g.add_argument("--no-grad-checkpoint", action="store_true",
                   help="Disable gradient checkpointing")

    # Training
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch-size", type=int, default=48)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--warmup-pct", type=float, default=0.10)
    g.add_argument("--label-smoothing", type=float, default=0.1)
    g.add_argument("--grad-accum", type=int, default=1)
    g.add_argument("--grad-clip", type=float, default=1.0)

    # Loss
    g = parser.add_argument_group("Loss")
    g.add_argument("--focal-gamma", type=float, default=2.0)
    g.add_argument("--cancer-boost", type=float, default=2.0)

    # Regularisation
    g = parser.add_argument_group("Regularisation")
    g.add_argument("--mixup-alpha", type=float, default=0.2)
    g.add_argument("--cutmix-alpha", type=float, default=1.0)
    g.add_argument("--mix-prob", type=float, default=0.5)
    g.add_argument("--no-ema", action="store_true")
    g.add_argument("--ema-decay", type=float, default=0.9998)

    # System
    g = parser.add_argument_group("System")
    g.add_argument("--num-workers", type=int, default=6)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    g.add_argument("--patience", type=int, default=15)

    # Paths
    g = parser.add_argument_group("Paths")
    g.add_argument("--data-dir", type=str, default="data")
    g.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    g.add_argument("--log-dir", type=str, default="logs")
    g.add_argument("--tensorboard-dir", type=str, default="runs")
    g.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    cfg = TrainConfig(
        backbone=args.backbone,
        img_size=args.img_size,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        gradient_checkpointing=not args.no_grad_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_pct=args.warmup_pct,
        label_smoothing=args.label_smoothing,
        grad_accum_steps=args.grad_accum,
        grad_clip=args.grad_clip,
        focal_gamma=args.focal_gamma,
        cancer_weight_boost=args.cancer_boost,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        use_ema=not args.no_ema,
        ema_decay=args.ema_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        amp_dtype=args.amp_dtype,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        tensorboard_dir=args.tensorboard_dir,
        resume=args.resume,
    )

    try:
        best_acc = train(cfg)
        print(f"\nTraining finished.  Best accuracy: {best_acc:.2f}%")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
