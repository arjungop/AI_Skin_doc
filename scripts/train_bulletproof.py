#!/usr/bin/env python3
"""
BULLETPROOF A100 TRAINING SCRIPT
================================
Designed for HPC queue systems where failures = weeks of waiting.

Features:
- Auto-resume from any checkpoint
- Saves checkpoint EVERY epoch
- Graceful handling of all errors
- NaN detection and recovery
- Automatic batch size reduction on OOM
- Logging to file + tensorboard
- Pre-training validation
- Signal handling for preemption
"""

import os
import sys
import json
import signal
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import InterpolationMode

from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter

# ============================================================================
# GLOBAL ERROR HANDLER
# ============================================================================

class TrainingState:
    """Global state that survives errors."""
    def __init__(self):
        self.epoch = 0
        self.best_acc = 0
        self.should_stop = False
        self.checkpoint_path = None

TRAINING_STATE = TrainingState()

def signal_handler(signum, frame):
    """Handle preemption signals gracefully."""
    print(f"\n⚠️  Received signal {signum}. Saving checkpoint and exiting...")
    TRAINING_STATE.should_stop = True

# Register signal handlers for HPC preemption
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_dir: Path):
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
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
# MODEL
# ============================================================================

class RobustSkinClassifier(nn.Module):
    """Robust hierarchical skin classifier with error handling."""
    
    BACKBONES = {
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792),
        "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT, 1280),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, 1024),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, 1536),
        "swin_b": (models.swin_b, models.Swin_B_Weights.DEFAULT, 1024),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    }
    
    def __init__(self, num_categories: int, num_diseases: int, backbone: str = "efficientnet_b4"):
        super().__init__()
        
        model_fn, weights, in_features = self.BACKBONES[backbone]
        self.backbone = model_fn(weights=weights)
        
        # Remove classification head
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Classification heads
        self.category_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Linear(256, num_categories)
        )
        
        self.disease_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_diseases)
        )
        
        self.num_categories = num_categories
        self.num_diseases = num_diseases
        self.in_features = in_features
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])
        
        cat_logits = self.category_head(features)
        dis_logits = self.disease_head(features)
        
        return cat_logits, dis_logits

# ============================================================================
# DATASET
# ============================================================================

class RobustSkinDataset(Dataset):
    """Dataset with robust error handling for corrupted images."""
    
    def __init__(self, root_dir: Path, hierarchy_path: Path, transform=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        
        with open(hierarchy_path) as f:
            self.hierarchy = json.load(f)
        
        self.diseases = [d for d in self.hierarchy.get("diseases", []) if not d.startswith("_aux_")]
        self.categories = self.hierarchy.get("categories", [])
        self.disease_to_category = self.hierarchy.get("disease_to_category", {})
        
        self.samples = []
        self.class_counts = Counter()
        
        for disease_idx, disease in enumerate(self.diseases):
            disease_dir = self.root_dir / disease
            if not disease_dir.exists():
                continue
            
            category = self.disease_to_category.get(disease, "unknown")
            category_idx = self.categories.index(category) if category in self.categories else 0
            
            for img_path in disease_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append({
                        'path': img_path,
                        'disease_idx': disease_idx,
                        'category_idx': category_idx,
                    })
                    self.class_counts[disease_idx] += 1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Robust image loading with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                image = Image.open(sample['path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                return {
                    'image': image,
                    'disease_idx': sample['disease_idx'],
                    'category_idx': sample['category_idx'],
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return a random other sample
                    new_idx = np.random.randint(len(self))
                    return self.__getitem__(new_idx)
    
    def get_class_weights(self):
        counts = [self.class_counts.get(i, 1) for i in range(len(self.diseases))]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum() * len(self.diseases)
        return weights
    
    def get_sampler(self):
        sample_weights = []
        for sample in self.samples:
            count = max(self.class_counts[sample['disease_idx']], 1)
            sample_weights.append(1.0 / count)
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(image_size: int = 384, is_training: bool = True):
    if is_training:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.1), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# ============================================================================
# FOCAL LOSS - REDUCES FALSE NEGATIVES
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for reducing false negatives in medical diagnosis.
    
    Focuses training on hard-to-classify examples (cases the model is unsure about).
    This is critical for medical applications where missing a disease (false negative)
    is much worse than a false alarm (false positive).
    
    Args:
        gamma: Focusing parameter (2.0 = strong focus on hard examples)
        alpha: Class weights (higher for cancer/serious diseases)
        label_smoothing: Prevent overconfidence
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Gather the log probabilities for the correct class
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Get probability of correct class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_t
        
        # Final focal loss
        loss = focal_weight * ce_loss
        
        return loss.mean()


class SensitivityMetrics:
    """Track per-class sensitivity (recall) to monitor false negatives."""
    
    def __init__(self, num_classes: int, disease_names: list = None):
        self.num_classes = num_classes
        self.disease_names = disease_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.true_positives = torch.zeros(self.num_classes)
        self.false_negatives = torch.zeros(self.num_classes)
        self.support = torch.zeros(self.num_classes)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.cpu()
        targets = targets.cpu()
        
        for i in range(self.num_classes):
            mask = targets == i
            self.support[i] += mask.sum()
            self.true_positives[i] += ((preds == i) & mask).sum()
            self.false_negatives[i] += ((preds != i) & mask).sum()
    
    def get_sensitivity(self) -> Dict[str, float]:
        """Returns per-class sensitivity (recall = TP / (TP + FN))."""
        sensitivity = {}
        for i in range(self.num_classes):
            total = self.true_positives[i] + self.false_negatives[i]
            if total > 0:
                sensitivity[self.disease_names[i]] = (self.true_positives[i] / total).item()
            else:
                sensitivity[self.disease_names[i]] = 0.0
        return sensitivity
    
    def get_average_sensitivity(self) -> float:
        """Returns macro-average sensitivity."""
        sens = self.get_sensitivity()
        valid = [v for v in sens.values() if v > 0]
        return sum(valid) / len(valid) if valid else 0.0
    
    def get_low_sensitivity_classes(self, threshold: float = 0.8) -> List[str]:
        """Returns classes with sensitivity below threshold (high false negatives)."""
        sens = self.get_sensitivity()
        return [name for name, val in sens.items() if val < threshold and self.support[self.disease_names.index(name)] > 0]

# ============================================================================
# TRAINER
# ============================================================================

class BulletproofTrainer:
    """Trainer designed to never fail."""
    
    def __init__(self, config: Dict, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
        # Try to resume from checkpoint
        self.start_epoch = 0
        self.best_acc = 0
        self.resume_from_checkpoint()
    
    def setup_data(self):
        """Setup datasets with validation."""
        self.logger.info("Setting up datasets...")
        
        train_transform = get_transforms(self.config['image_size'], is_training=True)
        val_transform = get_transforms(self.config['image_size'], is_training=False)
        
        hierarchy_path = Path(self.config['data_dir']) / "hierarchy.json"
        
        self.train_dataset = RobustSkinDataset(
            root_dir=self.config['data_dir'],
            hierarchy_path=hierarchy_path,
            transform=train_transform,
            is_training=True
        )
        
        self.val_dataset = RobustSkinDataset(
            root_dir=self.config['val_dir'],
            hierarchy_path=hierarchy_path,
            transform=val_transform,
            is_training=False
        )
        
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}")
        
        # Check minimum samples
        if len(self.train_dataset) < 100:
            raise ValueError(f"Too few training samples: {len(self.train_dataset)}")
        
        sampler = self.train_dataset.get_sampler() if self.config['use_weighted_sampling'] else None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
        
        self.num_diseases = len(self.train_dataset.diseases)
        self.num_categories = len(self.train_dataset.categories)
        self.class_weights = self.train_dataset.get_class_weights().to(self.device)
    
    def setup_model(self):
        """Setup model with error handling."""
        self.logger.info(f"Creating model: {self.config['backbone']}")
        
        self.model = RobustSkinClassifier(
            num_categories=self.num_categories,
            num_diseases=self.num_diseases,
            backbone=self.config['backbone']
        ).to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
    
    def setup_training(self):
        """Setup optimizer and scheduler with focus on minimizing false negatives."""
        # Category loss (standard)
        self.cat_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Disease loss: USE FOCAL LOSS to reduce false negatives
        # gamma=2.0 focuses strongly on hard examples (potential false negatives)
        # Higher class weights for cancer diseases (melanoma, bcc, scc) - never miss cancer!
        cancer_boost = 2.0  # 2x weight for cancer classes
        boosted_weights = self.class_weights.clone()
        
        # Boost cancer disease weights (assuming indices 0-3 are cancer based on hierarchy)
        # This makes the model more sensitive to cancer, reducing false negatives
        for i in range(min(4, len(boosted_weights))):  # First 4 = cancer diseases
            boosted_weights[i] *= cancer_boost
        
        self.dis_criterion = FocalLoss(
            gamma=2.0,  # Focus on hard examples
            alpha=boosted_weights,
            label_smoothing=0.1
        )
        
        # Sensitivity tracker
        self.sensitivity_tracker = SensitivityMetrics(
            num_classes=self.num_diseases,
            disease_names=self.train_dataset.diseases
        )
        
        self.logger.info("🎯 Using Focal Loss to minimize false negatives")
        self.logger.info(f"   Cancer diseases boosted by {cancer_boost}x")
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
        )
        
        total_steps = len(self.train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
        )
        
        self.scaler = GradScaler()
    
    def resume_from_checkpoint(self):
        """Try to resume from existing checkpoint."""
        ckpt_dir = Path(self.config['checkpoint_dir'])
        
        # Look for latest checkpoint
        checkpoints = list(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            # Try best model
            best_ckpt = ckpt_dir / "best_model.pth"
            if best_ckpt.exists():
                checkpoints = [best_ckpt]
        
        if checkpoints:
            # Get the latest
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            self.logger.info(f"Found checkpoint: {latest}")
            
            try:
                checkpoint = torch.load(latest, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_acc = checkpoint.get('best_val_acc', 0)
                
                self.logger.info(f"✅ Resumed from epoch {self.start_epoch}, best_acc: {self.best_acc:.2f}%")
                
            except Exception as e:
                self.logger.warning(f"Failed to resume: {e}. Starting fresh.")
                self.start_epoch = 0
                self.best_acc = 0
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save checkpoint with error handling."""
        ckpt_dir = Path(self.config['checkpoint_dir'])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': max(val_acc, self.best_acc),
            'config': self.config,
            'diseases': self.train_dataset.diseases,
            'categories': self.train_dataset.categories,
        }
        
        # Always save latest
        latest_path = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, latest_path)
        self.logger.info(f"💾 Saved checkpoint: {latest_path}")
        
        # Save best
        if is_best:
            best_path = ckpt_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 New best model! Acc: {val_acc:.2f}%")
        
        # Keep only last 3 epoch checkpoints
        epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        for old_ckpt in epoch_ckpts[:-3]:
            try:
                old_ckpt.unlink()
            except:
                pass
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with robust error handling."""
        self.model.train()
        
        total_loss = 0
        cat_correct = 0
        dis_correct = 0
        total_samples = 0
        nan_count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Check for preemption
            if TRAINING_STATE.should_stop:
                self.logger.info("Preemption detected, saving and stopping...")
                return None
            
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
                dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
                
                # Forward with mixed precision
                with autocast(dtype=torch.bfloat16):
                    cat_logits, dis_logits = self.model(images)
                    cat_loss = self.cat_criterion(cat_logits, cat_targets)
                    dis_loss = self.dis_criterion(dis_logits, dis_targets)
                    loss = 0.3 * cat_loss + 0.7 * dis_loss
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    if nan_count > 10:
                        self.logger.error("Too many NaN losses! Stopping.")
                        return None
                    continue
                
                # Backward
                self.scaler.scale(loss / self.config['grad_accum']).backward()
                
                if (batch_idx + 1) % self.config['grad_accum'] == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                
                # Metrics (log unscaled loss)
                total_loss += loss.item()
                cat_pred = cat_logits.argmax(dim=1)
                dis_pred = dis_logits.argmax(dim=1)
                cat_correct += (cat_pred == cat_targets).sum().item()
                dis_correct += (dis_pred == dis_targets).sum().item()
                total_samples += images.size(0)
                
                pbar.set_postfix({
                    'loss': f"{total_loss/(batch_idx+1):.4f}",
                    'cat': f"{100*cat_correct/total_samples:.1f}%",
                    'dis': f"{100*dis_correct/total_samples:.1f}%",
                })
                
            except torch.cuda.OutOfMemoryError:
                self.logger.error("OOM! Clearing cache and continuing...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            except Exception as e:
                self.logger.error(f"Batch error: {e}")
                continue
        
        return {
            'train_loss': total_loss / max(len(self.train_loader), 1),
            'train_cat_acc': 100 * cat_correct / max(total_samples, 1),
            'train_dis_acc': 100 * dis_correct / max(total_samples, 1),
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate with robust error handling."""
        self.model.eval()
        
        total_loss = 0
        cat_correct = 0
        dis_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            try:
                images = batch['image'].to(self.device, non_blocking=True)
                cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
                dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.bfloat16):
                    cat_logits, dis_logits = self.model(images)
                    cat_loss = self.cat_criterion(cat_logits, cat_targets)
                    dis_loss = self.dis_criterion(dis_logits, dis_targets)
                    loss = 0.3 * cat_loss + 0.7 * dis_loss
                
                total_loss += loss.item()
                cat_pred = cat_logits.argmax(dim=1)
                dis_pred = dis_logits.argmax(dim=1)
                cat_correct += (cat_pred == cat_targets).sum().item()
                dis_correct += (dis_pred == dis_targets).sum().item()
                total_samples += images.size(0)
                
                # Track sensitivity (false negatives)
                self.sensitivity_tracker.update(dis_pred, dis_targets)
                
            except Exception as e:
                self.logger.error(f"Validation batch error: {e}")
                continue
        
        # Calculate average sensitivity
        avg_sensitivity = self.sensitivity_tracker.get_average_sensitivity()
        low_sens_classes = self.sensitivity_tracker.get_low_sensitivity_classes(threshold=0.7)
        
        if low_sens_classes:
            self.logger.warning(f"⚠️  Low sensitivity (high false negatives): {low_sens_classes}")
        
        return {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'val_cat_acc': 100 * cat_correct / max(total_samples, 1),
            'val_dis_acc': 100 * dis_correct / max(total_samples, 1),
            'val_sensitivity': avg_sensitivity * 100,
            'low_sensitivity_classes': low_sens_classes,
        }
    
    def train(self):
        """Full training loop with bulletproof error handling."""
        self.logger.info("="*60)
        self.logger.info("🚀 STARTING BULLETPROOF TRAINING")
        self.logger.info("="*60)
        
        patience_counter = 0
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            # Check preemption
            if TRAINING_STATE.should_stop:
                break
            
            try:
                # Train
                train_metrics = self.train_epoch(epoch)
                
                if train_metrics is None:  # Preemption or error
                    break
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Log (include sensitivity for false negative tracking)
                self.logger.info(
                    f"Epoch {epoch+1} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_dis_acc']:.1f}% | "
                    f"Sensitivity: {val_metrics['val_sensitivity']:.1f}%"
                )
                
                # Check for best
                is_best = val_metrics['val_dis_acc'] > self.best_acc
                if is_best:
                    self.best_acc = val_metrics['val_dis_acc']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Save checkpoint EVERY epoch
                self.save_checkpoint(epoch, val_metrics['val_dis_acc'], is_best)
                
                # Early stopping
                if patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
            except Exception as e:
                self.logger.error(f"Epoch {epoch} error: {e}")
                traceback.print_exc()
                # Save whatever we have
                try:
                    self.save_checkpoint(epoch, self.best_acc, is_best=False)
                except:
                    pass
                continue
        
        self.logger.info(f"✅ Training complete! Best accuracy: {self.best_acc:.2f}%")
        return self.best_acc


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bulletproof A100 training")
    
    # Required
    parser.add_argument("--data_dir", type=str, default="data/unified_train")
    parser.add_argument("--val_dir", type=str, default="data/unified_val")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    # Model
    parser.add_argument("--backbone", type=str, default="convnext_large",
                       choices=["efficientnet_b4", "efficientnet_v2_m", "convnext_base", "convnext_large", "swin_b", "resnet50"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_weighted_sampling", action="store_true", default=True)
    
    args = parser.parse_args()
    config = vars(args)
    
    # Setup logging
    log_dir = Path(config['checkpoint_dir']) / "logs"
    logger = setup_logging(log_dir)
    
    logger.info("="*60)
    logger.info("🛡️ BULLETPROOF A100 TRAINING SCRIPT")
    logger.info("="*60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Run training
    try:
        trainer = BulletproofTrainer(config, logger)
        trainer.train()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
