#!/usr/bin/env python3
"""
ConvNeXt Training v2 - Optimized for HIGH ACCURACY + LOW FALSE NEGATIVES
Target: 93%+ accuracy, minimum false negatives for cancer classes

Key Features:
- Focal Loss (focuses on hard samples)
- Class-weighted loss (penalizes cancer misses heavily)
- Recall-optimized training
- Per-class sensitivity tracking
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

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
import random

# ============================================================================
# 20 TARGET CLASSES (ordered by clinical priority)
# ============================================================================
TARGET_CLASSES = [
    # Cancer (CRITICAL - false negatives are dangerous)
    "melanoma", "bcc", "scc", "ak",
    # Benign (must differentiate from cancer)
    "nevus", "seborrheic_keratosis", "angioma", "dermatofibroma",
    # Inflammatory (common conditions)
    "eczema", "psoriasis", "acne", "dermatitis", "urticaria",
    # Infectious (treatable)
    "candida", "herpes", "scabies", "impetigo",
    # Other
    "alopecia", "hyperpigmentation", "healthy"
]

# Cancer classes get HIGHER weight for false negative penalty
CANCER_CLASSES = {"melanoma", "bcc", "scc", "ak"}
CANCER_WEIGHT_MULTIPLIER = 3.0  # 3x penalty for missing cancer


# ============================================================================
# FOCAL LOSS - Better for imbalanced data, focuses on hard samples
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss - reduces loss for well-classified samples, focuses on hard ones."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (higher = more focus on hard samples)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ============================================================================
# MODEL
# ============================================================================
class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-Large optimized for 20-class skin disease classification."""
    
    def __init__(self, num_classes=20, dropout=0.5, variant="convnext_large"):
        super().__init__()
        
        variants = {
            "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, 1024),
            "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, 1536),
        }
        
        model_fn, weights, in_features = variants[variant]
        self.backbone = model_fn(weights=weights)
        self.backbone.classifier = nn.Identity()
        
        # Strong classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])
        return self.classifier(features)


# ============================================================================
# DATASET
# ============================================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_classes=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.classes = target_classes or TARGET_CLASSES
        
        self.samples = []
        self.class_counts = Counter()
        
        for idx, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                print(f"  ⚠️ Class not found: {cls}")
                continue
            
            for img in cls_dir.iterdir():
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img, idx))
                    self.class_counts[idx] += 1
        
        print(f"[{root_dir}] {len(self.samples)} samples, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            return self.__getitem__(random.randint(0, len(self)-1))
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def get_class_weights(self):
        """Get weights with extra penalty for cancer classes."""
        counts = [self.class_counts.get(i, 1) for i in range(len(self.classes))]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        
        # Apply extra weight for cancer classes
        for i, cls in enumerate(self.classes):
            if cls in CANCER_CLASSES:
                weights[i] *= CANCER_WEIGHT_MULTIPLIER
        
        weights = weights / weights.sum() * len(self.classes)
        return weights
    
    def get_sampler(self):
        weights = [1.0 / self.class_counts[lbl] for _, lbl in self.samples]
        # Boost cancer class sampling
        for i, (_, lbl) in enumerate(self.samples):
            if self.classes[lbl] in CANCER_CLASSES:
                weights[i] *= CANCER_WEIGHT_MULTIPLIER
        return WeightedRandomSampler(weights, len(weights), replacement=True)


# ============================================================================
# TRANSFORMS
# ============================================================================
def get_transforms(size=384, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25),
        ])
    return T.Compose([
        T.Resize(int(size * 1.1), interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ============================================================================
# METRICS - Focus on RECALL (minimize false negatives)
# ============================================================================
def compute_metrics(preds, targets, num_classes, class_names):
    """Compute per-class recall, accuracy, and identify worst classes."""
    
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    
    metrics = {"accuracy": (preds == targets).float().mean().item() * 100}
    
    # Per-class recall (sensitivity)
    recalls = []
    for i in range(num_classes):
        mask = targets == i
        if mask.sum() > 0:
            recall = (preds[mask] == i).float().mean().item()
            recalls.append(recall)
            metrics[f"recall_{class_names[i]}"] = recall * 100
        else:
            recalls.append(0)
    
    metrics["mean_recall"] = np.mean(recalls) * 100
    
    # Cancer recall (CRITICAL)
    cancer_mask = torch.tensor([class_names[t] in CANCER_CLASSES for t in targets])
    if cancer_mask.sum() > 0:
        cancer_correct = (preds[cancer_mask] == targets[cancer_mask]).float().mean().item()
        metrics["cancer_recall"] = cancer_correct * 100
    
    # Worst classes
    worst_idx = np.argsort(recalls)[:3]
    metrics["worst_classes"] = [(class_names[i], recalls[i]*100) for i in worst_idx]
    
    return metrics


# ============================================================================
# TRAINER
# ============================================================================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print(f"🖥️ GPU: {torch.cuda.get_device_name()}")
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _setup_data(self):
        train_tf = get_transforms(self.config['image_size'], train=True)
        val_tf = get_transforms(self.config['image_size'], train=False)
        
        self.train_ds = SkinDataset(self.config['data_dir'], train_tf, TARGET_CLASSES)
        self.val_ds = SkinDataset(self.config['val_dir'], val_tf, TARGET_CLASSES)
        
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.config['batch_size'],
            sampler=self.train_ds.get_sampler(),
            num_workers=8, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.config['batch_size'] * 2,
            shuffle=False, num_workers=8, pin_memory=True
        )
        
        self.class_weights = self.train_ds.get_class_weights().to(self.device)
        self.num_classes = len(TARGET_CLASSES)
    
    def _setup_model(self):
        self.model = ConvNeXtClassifier(
            num_classes=self.num_classes,
            dropout=self.config['dropout'],
            variant=self.config['variant']
        ).to(self.device)
        
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model: {self.config['variant']}, Params: {params:,}")
    
    def _setup_training(self):
        # Focal Loss with class weights
        self.criterion = FocalLoss(alpha=self.class_weights, gamma=2.0)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'],
            total_steps=len(self.train_loader) * self.config['epochs'],
            pct_start=0.1
        )
        
        self.scaler = GradScaler()
        self.best_recall = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss, all_preds, all_targets = 0, [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            with autocast(dtype=torch.bfloat16):
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        metrics = compute_metrics(all_preds, all_targets, self.num_classes, TARGET_CLASSES)
        return total_loss / len(self.train_loader), metrics
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        all_preds, all_targets = [], []
        
        for imgs, labels in tqdm(self.val_loader, desc="Validating"):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with autocast(dtype=torch.bfloat16):
                logits = self.model(imgs)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
        
        metrics = compute_metrics(all_preds, all_targets, self.num_classes, TARGET_CLASSES)
        
        # Save best based on MEAN RECALL (not accuracy)
        if metrics['mean_recall'] > self.best_recall:
            self.best_recall = metrics['mean_recall']
            self.patience_counter = 0
            self._save(epoch)
        else:
            self.patience_counter += 1
        
        return metrics
    
    def _save(self, epoch):
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'best_recall': self.best_recall,
            'classes': TARGET_CLASSES,
            'config': self.config
        }, f"{self.config['checkpoint_dir']}/best_model.pth")
        print(f"  💾 Saved best model (recall: {self.best_recall:.2f}%)")
    
    def train(self):
        print("\n" + "="*60)
        print("🚀 TRAINING - TARGET: 93%+ Accuracy, Low False Negatives")
        print("="*60)
        
        for epoch in range(self.config['epochs']):
            train_loss, train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)
            
            print(f"\n📈 Epoch {epoch+1}:")
            print(f"   Train: Acc={train_m['accuracy']:.1f}%, Recall={train_m['mean_recall']:.1f}%")
            print(f"   Val:   Acc={val_m['accuracy']:.1f}%, Recall={val_m['mean_recall']:.1f}%")
            print(f"   Cancer Recall: {val_m.get('cancer_recall', 0):.1f}%")
            print(f"   Worst: {val_m['worst_classes']}")
            print(f"   Best Recall: {self.best_recall:.1f}%")
            
            if self.patience_counter >= self.config['patience']:
                print(f"\n⏹️ Early stopping")
                break
        
        print(f"\n✅ Done! Best recall: {self.best_recall:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/stage1/train")
    parser.add_argument("--val_dir", default="data/stage1/val")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--variant", default="convnext_large", choices=["convnext_base", "convnext_large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    
    Trainer(vars(args)).train()


if __name__ == "__main__":
    main()
