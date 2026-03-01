#!/usr/bin/env python3
"""
BULLETPROOF ConvNeXt Training - 25 Classes, 93%+ Accuracy Target
Author: AI Assistant
Features:
- Focal Loss + Label Smoothing
- 3x Cancer Weight (minimize false negatives)
- Weighted Sampling for imbalance
- Mixed Precision (bf16/fp16 auto-detect)
- Gradient Accumulation
- Per-class recall tracking
- Early stopping on recall (not accuracy)
- Resume from checkpoint
- Comprehensive logging
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import InterpolationMode

from PIL import Image
import numpy as np
from tqdm import tqdm
import random


# ============================================================================
# 25 OPTIMIZED CLASSES (balanced, clinically important)
# ============================================================================
TARGET_CLASSES = [
    # CANCER (CRITICAL - 3x weight for false negative penalty)
    "melanoma",           # Most dangerous skin cancer
    "bcc",                # Basal cell carcinoma
    "scc",                # Squamous cell carcinoma  
    "ak",                 # Actinic keratosis (pre-cancerous)
    
    # BENIGN (must differentiate from cancer)
    "nevus",              # Common mole
    "seborrheic_keratosis",
    "angioma",            # Vascular lesion
    
    # INFLAMMATORY (very common)
    "eczema",
    "psoriasis",
    "acne",
    "dermatitis",
    "urticaria",          # Hives
    "bullous",            # Blistering diseases
    
    # INFECTIOUS (treatable, need early detection)
    "candida",            # Fungal
    "herpes",             # Viral
    "scabies",            # Parasitic
    "impetigo",           # Bacterial
    "nail_fungus",
    "chickenpox",
    "shingles",
    "wart",
    
    # OTHER COMMON
    "alopecia",           # Hair loss
    "hyperpigmentation",
    "lupus",              # Autoimmune
    "healthy",            # Normal skin
]

CANCER_CLASSES = {"melanoma", "bcc", "scc", "ak"}
CANCER_WEIGHT = 3.0


# ============================================================================
# LOGGING
# ============================================================================
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, 
                 smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # Log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Focal weight
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma
        
        # Cross entropy with smoothing
        loss = -smooth_targets * log_probs * focal_weight
        
        # Apply class weights
        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)
        
        loss = loss.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ============================================================================
# MODEL - ConvNeXt with Strong Head
# ============================================================================
class ConvNeXtClassifier(nn.Module):
    VARIANTS = {
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, 768),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, 1024),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, 1536),
    }
    
    def __init__(self, num_classes: int = 25, variant: str = "convnext_large", 
                 dropout: float = 0.5, pretrained: bool = True):
        super().__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        model_fn, weights, in_features = self.VARIANTS[variant]
        self.backbone = model_fn(weights=weights if pretrained else None)
        self.backbone.classifier = nn.Identity()
        
        # Strong classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])
        return self.classifier(features)


# ============================================================================
# DATASET
# ============================================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, classes: List[str] = None):
        self.root = Path(root_dir)
        self.transform = transform
        self.classes = classes or TARGET_CLASSES
        
        self.samples = []
        self.class_counts = Counter()
        missing_classes = []
        
        for idx, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                missing_classes.append(cls)
                continue
            
            for img in cls_dir.iterdir():
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img, idx))
                    self.class_counts[idx] += 1
        
        if missing_classes:
            print(f"⚠️ Missing classes: {missing_classes}")
        
        print(f"📁 {self.root.name}: {len(self.samples):,} images, {len(self.classes) - len(missing_classes)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_weights(self) -> torch.Tensor:
        counts = [max(self.class_counts.get(i, 1), 1) for i in range(len(self.classes))]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        
        # Apply cancer weight multiplier
        for i, cls in enumerate(self.classes):
            if cls in CANCER_CLASSES:
                weights[i] *= CANCER_WEIGHT
        
        weights = weights / weights.sum() * len(self.classes)
        return weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        sample_weights = []
        for _, label in self.samples:
            w = 1.0 / max(self.class_counts[label], 1)
            if self.classes[label] in CANCER_CLASSES:
                w *= CANCER_WEIGHT
            sample_weights.append(w)
        
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ============================================================================
# TRANSFORMS
# ============================================================================
def get_transforms(size: int = 384, is_training: bool = True):
    if is_training:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return T.Compose([
        T.Resize(int(size * 1.1), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(preds: List[int], targets: List[int], classes: List[str]) -> Dict:
    preds_t = torch.tensor(preds)
    targets_t = torch.tensor(targets)
    
    accuracy = (preds_t == targets_t).float().mean().item() * 100
    
    # Per-class recall
    recalls = {}
    for i, cls in enumerate(classes):
        mask = targets_t == i
        if mask.sum() > 0:
            recalls[cls] = (preds_t[mask] == i).float().mean().item() * 100
    
    mean_recall = np.mean(list(recalls.values())) if recalls else 0
    
    # Cancer recall (critical metric)
    cancer_indices = [i for i, c in enumerate(classes) if c in CANCER_CLASSES]
    cancer_mask = torch.tensor([t in cancer_indices for t in targets])
    cancer_recall = 0
    if cancer_mask.sum() > 0:
        cancer_recall = (preds_t[cancer_mask] == targets_t[cancer_mask]).float().mean().item() * 100
    
    # Worst classes
    worst = sorted(recalls.items(), key=lambda x: x[1])[:3]
    
    return {
        'accuracy': accuracy,
        'mean_recall': mean_recall,
        'cancer_recall': cancer_recall,
        'per_class_recall': recalls,
        'worst_classes': worst
    }


# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect GPU type
        self.gpu_type = self._detect_gpu()
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _detect_gpu(self) -> str:
        if not torch.cuda.is_available():
            self.logger.warning("No GPU available, using CPU")
            return "CPU"
        
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.info(f"GPU: {name} ({mem:.1f} GB)")
        
        if "A100" in name.upper():
            return "A100"
        elif "6000" in name or "A6000" in name.upper():
            return "RTX6000"
        return "GENERIC"
    
    def _setup_data(self):
        train_tf = get_transforms(self.config['image_size'], is_training=True)
        val_tf = get_transforms(self.config['image_size'], is_training=False)
        
        self.train_ds = SkinDataset(self.config['data_dir'], train_tf, TARGET_CLASSES)
        self.val_ds = SkinDataset(self.config['val_dir'], val_tf, TARGET_CLASSES)
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            sampler=self.train_ds.get_sampler() if self.config.get('weighted_sampling', False) else None,
            shuffle=False if self.config.get('weighted_sampling', False) else True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
        
        self.class_weights = self.train_ds.get_class_weights().to(self.device)
        self.num_classes = len(TARGET_CLASSES)
        
        self.logger.info(f"Train: {len(self.train_ds):,} | Val: {len(self.val_ds):,}")
    
    def _setup_model(self):
        self.model = ConvNeXtClassifier(
            num_classes=self.num_classes,
            variant=self.config['variant'],
            dropout=self.config['dropout'],
            pretrained=True
        ).to(self.device)
        
        # Compile for PyTorch 2.0+
        if hasattr(torch, 'compile') and self.config.get('compile', True):
            self.logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model: {self.config['variant']} | Params: {params:,} | Trainable: {trainable:,}")
        
        # Load checkpoint if resuming
        if self.config.get('resume'):
            self._load_checkpoint(self.config['resume'])
    
    def _setup_training(self):
        self.criterion = FocalLossWithSmoothing(
            alpha=self.class_weights,
            gamma=2.0,
            smoothing=0.1
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        total_steps = len(self.train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision - bf16 for A100, fp16 for others
        self.amp_dtype = torch.bfloat16 if self.gpu_type == "A100" else torch.float16
        self.scaler = GradScaler()
        
        self.best_recall = 0
        self.best_accuracy = 0
        self.patience_counter = 0
        self.start_epoch = 0
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        save_dir = Path(self.config['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_recall': self.best_recall,
            'best_accuracy': self.best_accuracy,
            'metrics': metrics,
            'config': self.config,
            'classes': TARGET_CLASSES,
        }
        
        # Save latest
        torch.save(checkpoint, save_dir / "latest.pth")
        
        if is_best:
            torch.save(checkpoint, save_dir / "best_model.pth")
            self.logger.info(f"💾 Saved best model (recall: {metrics['mean_recall']:.2f}%, acc: {metrics['accuracy']:.2f}%)")
            
            # Also save class info
            with open(save_dir / "classes.json", 'w') as f:
                json.dump({'classes': TARGET_CLASSES, 'cancer_classes': list(CANCER_CLASSES)}, f, indent=2)
    
    def _load_checkpoint(self, path: str):
        self.logger.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_recall = ckpt.get('best_recall', 0)
        self.best_accuracy = ckpt.get('best_accuracy', 0)
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.logger.info(f"Resumed from epoch {self.start_epoch}, best recall: {self.best_recall:.2f}%")
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict]:
        self.model.train()
        total_loss = 0
        all_preds, all_targets = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with autocast(dtype=self.amp_dtype):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            
            # Gradient accumulation
            loss = loss / self.config['grad_accum']
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['grad_accum'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            
            total_loss += loss.item() * self.config['grad_accum']
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config['grad_accum']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        metrics = compute_metrics(all_preds, all_targets, TARGET_CLASSES)
        return total_loss / len(self.train_loader), metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        self.model.eval()
        all_preds, all_targets = [], []
        
        for images, targets in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with autocast(dtype=self.amp_dtype):
                logits = self.model(images)
            
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
        
        metrics = compute_metrics(all_preds, all_targets, TARGET_CLASSES)
        
        # Check for improvement (prioritize recall over accuracy)
        is_best = False
        if metrics['mean_recall'] > self.best_recall:
            self.best_recall = metrics['mean_recall']
            self.best_accuracy = metrics['accuracy']
            self.patience_counter = 0
            is_best = True
        elif metrics['mean_recall'] == self.best_recall and metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = metrics['accuracy']
            is_best = True
        else:
            self.patience_counter += 1
        
        self._save_checkpoint(epoch, metrics, is_best)
        
        return metrics
    
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("🚀 BULLETPROOF TRAINING - 25 Classes, 93%+ Target")
        self.logger.info("=" * 60)
        self.logger.info(f"Target classes: {len(TARGET_CLASSES)}")
        self.logger.info(f"Cancer classes (3x weight): {CANCER_CLASSES}")
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            self.logger.info("")
            self.logger.info(f"📊 Epoch {epoch+1} Results:")
            self.logger.info(f"   Train: Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.1f}%, Recall={train_metrics['mean_recall']:.1f}%")
            self.logger.info(f"   Val:   Acc={val_metrics['accuracy']:.1f}%, Recall={val_metrics['mean_recall']:.1f}%, Cancer={val_metrics['cancer_recall']:.1f}%")
            self.logger.info(f"   Worst: {val_metrics['worst_classes']}")
            self.logger.info(f"   Best:  Recall={self.best_recall:.1f}%, Acc={self.best_accuracy:.1f}%")
            
            if self.patience_counter >= self.config['patience']:
                self.logger.info(f"\n⏹️ Early stopping (no improvement for {self.config['patience']} epochs)")
                break
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"✅ TRAINING COMPLETE!")
        self.logger.info(f"   Best Recall: {self.best_recall:.2f}%")
        self.logger.info(f"   Best Accuracy: {self.best_accuracy:.2f}%")
        self.logger.info(f"   Model saved to: {self.config['checkpoint_dir']}/best_model.pth")
        self.logger.info("=" * 60)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bulletproof ConvNeXt Training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/stage1/train")
    parser.add_argument("--val_dir", type=str, default="data/stage1/val")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    # Model
    parser.add_argument("--variant", type=str, default="convnext_large",
                       choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"], help="Mixed precision mode")
    parser.add_argument("--weighted_sampling", action="store_true", help="Use weighted sampling for class imbalance")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    
    args = parser.parse_args()
    
    config = vars(args)
    config['compile'] = not args.no_compile
    
    # Setup logging
    logger = setup_logging(Path("logs"))
    
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    
    # Train
    trainer = Trainer(config, logger)
    trainer.train()


if __name__ == "__main__":
    main()
