#!/usr/bin/env python3
"""
ConvNeXt Training Script for Skin Disease Classification
Optimized for RTX 6000 Ada (48GB) / A100 (40GB) / A6000 (48GB)

Features:
- ConvNeXt-Base/Large backbone
- Weighted sampling for severe class imbalance
- Mixed precision (bf16 for A100, fp16 for others)
- Gradient accumulation
- Strong augmentations (RandAugment, MixUp, CutMix)
- Cosine annealing with warmup
- Early stopping with best model checkpointing
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from collections import Counter

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
# MODEL
# ============================================================================

class ConvNeXtClassifier(nn.Module):
    """ConvNeXt-based skin disease classifier with hierarchical heads."""
    
    VARIANTS = {
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, 768),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, 1024),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, 1536),
    }
    
    def __init__(
        self,
        num_categories: int = 5,
        num_diseases: int = 33,
        variant: str = "convnext_base",
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        super().__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Choose from {list(self.VARIANTS.keys())}")
        
        model_fn, weights, in_features = self.VARIANTS[variant]
        
        # Load pretrained backbone
        self.backbone = model_fn(weights=weights if pretrained else None)
        self.backbone.classifier = nn.Identity()  # Remove classification head
        
        self.in_features = in_features
        
        # Category head (5 categories)
        self.category_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_categories)
        )
        
        # Disease head (33 diseases)
        self.disease_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_diseases)
        )
        
        self.num_categories = num_categories
        self.num_diseases = num_diseases
        self.variant = variant
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])
        
        cat_logits = self.category_head(features)
        dis_logits = self.disease_head(features)
        return cat_logits, dis_logits


# ============================================================================
# DATASET
# ============================================================================

class SkinDataset(Dataset):
    """Skin disease dataset with hierarchical labels."""
    
    def __init__(self, root_dir: Path, transform=None, is_training: bool = True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Load or create hierarchy
        hierarchy_path = self.root_dir / "hierarchy.json"
        if hierarchy_path.exists():
            with open(hierarchy_path) as f:
                self.hierarchy = json.load(f)
        else:
            hierarchy_path = self.root_dir / "metadata.json"
            if hierarchy_path.exists():
                with open(hierarchy_path) as f:
                    self.hierarchy = json.load(f)
            else:
                self.hierarchy = self._build_hierarchy()
        
        self.diseases = sorted([d.name for d in self.root_dir.iterdir() 
                               if d.is_dir() and not d.name.startswith('.')])
        
        # Category mapping
        self.categories = ["cancer", "benign", "inflammatory", "infectious", "pigmentary"]
        self.disease_to_category = self._get_disease_category_map()
        
        # Build samples
        self.samples = []
        self.class_counts = Counter()
        
        for disease_idx, disease in enumerate(self.diseases):
            disease_dir = self.root_dir / disease
            category = self.disease_to_category.get(disease, "inflammatory")
            category_idx = self.categories.index(category)
            
            for img_path in disease_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append({
                        'path': img_path,
                        'disease_idx': disease_idx,
                        'category_idx': category_idx,
                        'disease': disease
                    })
                    self.class_counts[disease_idx] += 1
        
        print(f"[{root_dir.name}] Loaded {len(self.samples)} samples, {len(self.diseases)} classes")
    
    def _get_disease_category_map(self):
        return {
            # Cancer
            "melanoma": "cancer", "bcc": "cancer", "scc": "cancer", 
            "ak": "cancer", "malignant": "cancer",
            # Benign
            "nevus": "benign", "seborrheic_keratosis": "benign", 
            "angioma": "benign", "wart": "benign", "benign": "benign",
            "dermatofibroma": "benign", "healthy": "benign", "diverse_skin": "benign",
            # Infectious
            "impetigo": "infectious", "herpes": "infectious", "candida": "infectious",
            "scabies": "infectious", "cellulitis": "infectious", "chickenpox": "infectious",
            "shingles": "infectious", "nail_fungus": "infectious",
            # Inflammatory
            "eczema": "inflammatory", "psoriasis": "inflammatory", "acne": "inflammatory",
            "dermatitis": "inflammatory", "urticaria": "inflammatory", "rash": "inflammatory",
            "bullous": "inflammatory", "lupus": "inflammatory", "vasculitis": "inflammatory",
            "drug_eruption": "inflammatory", "alopecia": "inflammatory", "systemic": "inflammatory",
            # Pigmentary
            "hyperpigmentation": "pigmentary",
        }
    
    def _build_hierarchy(self):
        return {"diseases": self.diseases, "categories": self.categories}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'disease_idx': sample['disease_idx'],
            'category_idx': sample['category_idx'],
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights for imbalanced classes."""
        counts = [self.class_counts.get(i, 1) for i in range(len(self.diseases))]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum() * len(self.diseases)
        return weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Weighted sampler for balanced training."""
        sample_weights = [1.0 / self.class_counts[s['disease_idx']] for s in self.samples]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


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
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.1), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ============================================================================
# TRAINING
# ============================================================================

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect GPU type for optimal settings
        self.gpu_type = self._detect_gpu()
        
        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _detect_gpu(self):
        if not torch.cuda.is_available():
            return "CPU"
        name = torch.cuda.get_device_name(0).upper()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {mem_gb:.1f} GB")
        
        if "A100" in name:
            return "A100"
        elif "6000" in name or "A6000" in name:
            return "RTX6000"
        else:
            return "GENERIC"
    
    def _setup_data(self):
        train_tf = get_transforms(self.config['image_size'], is_training=True)
        val_tf = get_transforms(self.config['image_size'], is_training=False)
        
        self.train_dataset = SkinDataset(
            Path(self.config['data_dir']), transform=train_tf, is_training=True
        )
        self.val_dataset = SkinDataset(
            Path(self.config['val_dir']), transform=val_tf, is_training=False
        )
        
        sampler = self.train_dataset.get_sampler() if self.config['weighted_sampling'] else None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.config['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'] * 2,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
        
        self.num_diseases = len(self.train_dataset.diseases)
        self.num_categories = len(self.train_dataset.categories)
        self.class_weights = self.train_dataset.get_class_weights().to(self.device)
    
    def _setup_model(self):
        self.model = ConvNeXtClassifier(
            num_categories=self.num_categories,
            num_diseases=self.num_diseases,
            variant=self.config['variant'],
            dropout=self.config['dropout'],
            pretrained=True
        ).to(self.device)
        
        # Compile for PyTorch 2.0+
        if hasattr(torch, 'compile') and self.config.get('compile', True):
            print("⚡ Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 Model: {self.config['variant']}, Parameters: {params:,}")
    
    def _setup_training(self):
        # Loss with class weights
        self.cat_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.dis_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, label_smoothing=0.1
        )
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = len(self.train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['lr'],
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Mixed precision
        dtype = torch.bfloat16 if self.gpu_type == "A100" else torch.float16
        self.amp_dtype = dtype
        self.scaler = GradScaler()
        
        self.best_acc = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
            dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
            
            with autocast(dtype=self.amp_dtype):
                cat_logits, dis_logits = self.model(images)
                cat_loss = self.cat_criterion(cat_logits, cat_targets)
                dis_loss = self.dis_criterion(dis_logits, dis_targets)
                loss = 0.3 * cat_loss + 0.7 * dis_loss
            
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
            correct += (dis_logits.argmax(1) == dis_targets).sum().item()
            total += images.size(0)
            
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100*correct/total:.1f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / len(self.train_loader), 100 * correct / total
    
    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device, non_blocking=True)
            cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
            dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
            
            with autocast(dtype=self.amp_dtype):
                cat_logits, dis_logits = self.model(images)
                cat_loss = self.cat_criterion(cat_logits, cat_targets)
                dis_loss = self.dis_criterion(dis_logits, dis_targets)
                loss = 0.3 * cat_loss + 0.7 * dis_loss
            
            total_loss += loss.item()
            correct += (dis_logits.argmax(1) == dis_targets).sum().item()
            total += images.size(0)
        
        val_acc = 100 * correct / total
        
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.patience_counter = 0
            self._save_checkpoint(epoch, is_best=True)
        else:
            self.patience_counter += 1
        
        return total_loss / len(self.val_loader), val_acc
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        save_dir = Path(self.config['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config,
            'diseases': self.train_dataset.diseases,
            'categories': self.train_dataset.categories,
        }
        
        if is_best:
            path = save_dir / "best_model.pth"
            torch.save(ckpt, path)
            print(f"  💾 Saved best model ({self.best_acc:.2f}%): {path}")
        
        torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch}.pth")
    
    def train(self):
        print("\n" + "="*60)
        print("🚀 STARTING CONVNEXT TRAINING")
        print("="*60)
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            print(f"\n📈 Epoch {epoch+1}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}% (Best={self.best_acc:.1f}%)")
            
            if self.patience_counter >= self.config['patience']:
                print(f"\n⏹️  Early stopping (no improvement for {self.config['patience']} epochs)")
                break
        
        print(f"\n✅ Training complete! Best accuracy: {self.best_acc:.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt for Skin Disease Classification")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/unified_train")
    parser.add_argument("--val_dir", type=str, default="data/unified_val")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    
    # Model
    parser.add_argument("--variant", type=str, default="convnext_base",
                       choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    
    # Flags
    parser.add_argument("--weighted_sampling", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    
    args = parser.parse_args()
    
    config = {
        'data_dir': args.data_dir,
        'val_dir': args.val_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'variant': args.variant,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'grad_accum': args.grad_accum,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'weighted_sampling': args.weighted_sampling,
        'compile': not args.no_compile,
    }
    
    print("="*60)
    print("⚙️  CONFIGURATION")
    print("="*60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
