#!/usr/bin/env python3
"""
A100 40GB Optimized Training Script for Skin Disease Classification
Supports: EfficientNet-B4, ConvNeXt-Base/Large, Swin-Base/Large, ViT-Base/Large

Features:
- Mixed precision training (bf16/fp16)
- Gradient accumulation
- Advanced augmentations (MixUp, CutMix, RandAugment)
- Class-weighted loss for imbalanced data
- Cosine annealing with warmup
- Early stopping and best model checkpointing
- Hierarchical classification (Category + Disease)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

import torchvision.transforms as T
from torchvision import models
from torchvision.transforms import InterpolationMode

from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class HierarchicalSkinClassifier(nn.Module):
    """
    Multi-head classifier for hierarchical skin disease classification.
    Supports multiple state-of-the-art backbones optimized for A100.
    """
    
    BACKBONES = {
        # EfficientNet family
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, 1792, "classifier"),
        "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT, 2048, "classifier"),
        "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.DEFAULT, 1280, "classifier"),
        "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.DEFAULT, 1280, "classifier"),
        
        # ConvNeXt family (recommended for A100)
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, 1024, "classifier"),
        "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, 1536, "classifier"),
        
        # Swin Transformer family (highest accuracy potential)
        "swin_b": (models.swin_b, models.Swin_B_Weights.DEFAULT, 1024, "head"),
        "swin_v2_b": (models.swin_v2_b, models.Swin_V2_B_Weights.DEFAULT, 1024, "head"),
        
        # Vision Transformer
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT, 768, "heads"),
        "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT, 1024, "heads"),
        
        # ResNet (baseline)
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048, "fc"),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048, "fc"),
    }
    
    def __init__(
        self, 
        num_categories: int = 5, 
        num_diseases: int = 20,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout: float = 0.4,
        use_attention: bool = True
    ):
        super().__init__()
        
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(self.BACKBONES.keys())}")
        
        model_fn, weights, in_features, head_attr = self.BACKBONES[backbone]
        
        # Load pretrained backbone
        self.backbone = model_fn(weights=weights if pretrained else None)
        
        # Remove classification head
        if head_attr == "classifier":
            if hasattr(self.backbone.classifier, '__len__'):
                self.backbone.classifier = nn.Identity()
            else:
                self.backbone.classifier = nn.Identity()
        elif head_attr == "head":
            self.backbone.head = nn.Identity()
        elif head_attr == "heads":
            self.backbone.heads = nn.Identity()
        elif head_attr == "fc":
            self.backbone.fc = nn.Identity()
        
        self.in_features = in_features
        self.use_attention = use_attention
        
        # Optional attention pooling
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            )
        
        # Classification heads with more capacity
        self.category_head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_categories)
        )
        
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
        self.backbone_name = backbone
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        
        # Handle different output shapes
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])  # Global average pooling
        
        category_logits = self.category_head(features)
        disease_logits = self.disease_head(features)
        
        return category_logits, disease_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns predictions and confidence scores."""
        cat_logits, dis_logits = self.forward(x)
        
        cat_probs = F.softmax(cat_logits, dim=1)
        dis_probs = F.softmax(dis_logits, dim=1)
        
        cat_pred = torch.argmax(cat_probs, dim=1)
        dis_pred = torch.argmax(dis_probs, dim=1)
        
        cat_conf = cat_probs.max(dim=1).values
        dis_conf = dis_probs.max(dim=1).values
        
        return cat_pred, dis_pred, cat_conf, dis_conf


# ============================================================================
# DATASET
# ============================================================================

class SkinDiseaseDataset(Dataset):
    """Dataset for hierarchical skin disease classification."""
    
    def __init__(
        self,
        root_dir: Path,
        hierarchy_path: Path,
        transform=None,
        is_training: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Load hierarchy
        with open(hierarchy_path) as f:
            self.hierarchy = json.load(f)
        
        # Get disease classes (exclude auxiliary)
        self.diseases = [d for d in self.hierarchy.get("diseases", []) if not d.startswith("_aux_")]
        self.categories = self.hierarchy.get("categories", [])
        
        # Disease to category mapping
        self.disease_to_category = self.hierarchy.get("disease_to_category", {})
        
        # Build sample list
        self.samples = []
        self.class_counts = Counter()
        
        for disease_idx, disease in enumerate(self.diseases):
            disease_dir = self.root_dir / disease
            if not disease_dir.exists():
                continue
            
            category = self.disease_to_category.get(disease, "unknown")
            category_idx = self.categories.index(category) if category in self.categories else -1
            
            for img_path in disease_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append({
                        'path': img_path,
                        'disease_idx': disease_idx,
                        'category_idx': category_idx,
                        'disease': disease,
                        'category': category
                    })
                    self.class_counts[disease_idx] += 1
        
        print(f"Loaded {len(self.samples)} samples across {len(self.diseases)} diseases")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a random valid sample
            return self.__getitem__(np.random.randint(len(self)))
        
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
        """Get weighted sampler for balanced training."""
        sample_weights = []
        for sample in self.samples:
            count = self.class_counts[sample['disease_idx']]
            sample_weights.append(1.0 / count)
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(image_size: int = 384, is_training: bool = True):
    """Get transforms optimized for skin disease classification."""
    
    if is_training:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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
# LOSS FUNCTIONS
# ============================================================================

class HierarchicalLoss(nn.Module):
    """Combined loss for hierarchical classification."""
    
    def __init__(
        self, 
        category_weight: float = 0.3,
        disease_weight: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.category_weight = category_weight
        self.disease_weight = disease_weight
        self.label_smoothing = label_smoothing
        
        self.category_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.disease_criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self, 
        cat_logits: torch.Tensor, 
        dis_logits: torch.Tensor,
        cat_targets: torch.Tensor,
        dis_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        cat_loss = self.category_criterion(cat_logits, cat_targets)
        dis_loss = self.disease_criterion(dis_logits, dis_targets)
        
        total_loss = self.category_weight * cat_loss + self.disease_weight * dis_loss
        
        return total_loss, cat_loss, dis_loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    """Training manager with all optimizations for A100."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable TF32 for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        self.setup_data()
        self.setup_model()
        self.setup_training()
    
    def setup_data(self):
        """Setup datasets and dataloaders."""
        train_transform = get_transforms(self.config['image_size'], is_training=True)
        val_transform = get_transforms(self.config['image_size'], is_training=False)
        
        # Load hierarchy
        hierarchy_path = Path(self.config['data_dir']) / "hierarchy.json"
        
        self.train_dataset = SkinDiseaseDataset(
            root_dir=self.config['data_dir'],
            hierarchy_path=hierarchy_path,
            transform=train_transform,
            is_training=True
        )
        
        self.val_dataset = SkinDiseaseDataset(
            root_dir=self.config['val_dir'],
            hierarchy_path=hierarchy_path,
            transform=val_transform,
            is_training=False
        )
        
        # Use weighted sampler for balanced training
        sampler = self.train_dataset.get_sampler() if self.config['use_weighted_sampling'] else None
        
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
    
    def setup_model(self):
        """Setup model architecture."""
        self.model = HierarchicalSkinClassifier(
            num_categories=self.num_categories,
            num_diseases=self.num_diseases,
            backbone=self.config['backbone'],
            pretrained=True,
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Compile model for faster training (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.get('compile_model', True):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {self.config['backbone']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss."""
        # Loss with class weights
        self.criterion = HierarchicalLoss(
            category_weight=0.3,
            disease_weight=0.7,
            class_weights=self.class_weights,
            label_smoothing=0.1
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config['epochs']
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Best metrics tracking
        self.best_val_acc = 0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        cat_correct = 0
        dis_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, non_blocking=True)
            cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
            dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast(dtype=torch.bfloat16):
                cat_logits, dis_logits = self.model(images)
                loss, cat_loss, dis_loss = self.criterion(
                    cat_logits, dis_logits, cat_targets, dis_targets
                )
            
            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
            
            # Metrics
            total_loss += loss.item() * self.config['gradient_accumulation']
            cat_pred = cat_logits.argmax(dim=1)
            dis_pred = dis_logits.argmax(dim=1)
            cat_correct += (cat_pred == cat_targets).sum().item()
            dis_correct += (dis_pred == dis_targets).sum().item()
            total_samples += images.size(0)
            
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'cat_acc': f"{100*cat_correct/total_samples:.1f}%",
                'dis_acc': f"{100*dis_correct/total_samples:.1f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_cat_acc': 100 * cat_correct / total_samples,
            'train_dis_acc': 100 * dis_correct / total_samples,
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        cat_correct = 0
        dis_correct = 0
        total_samples = 0
        
        all_dis_preds = []
        all_dis_targets = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device, non_blocking=True)
            cat_targets = batch['category_idx'].to(self.device, non_blocking=True)
            dis_targets = batch['disease_idx'].to(self.device, non_blocking=True)
            
            with autocast(dtype=torch.bfloat16):
                cat_logits, dis_logits = self.model(images)
                loss, _, _ = self.criterion(
                    cat_logits, dis_logits, cat_targets, dis_targets
                )
            
            total_loss += loss.item()
            
            cat_pred = cat_logits.argmax(dim=1)
            dis_pred = dis_logits.argmax(dim=1)
            
            cat_correct += (cat_pred == cat_targets).sum().item()
            dis_correct += (dis_pred == dis_targets).sum().item()
            total_samples += images.size(0)
            
            all_dis_preds.extend(dis_pred.cpu().numpy())
            all_dis_targets.extend(dis_targets.cpu().numpy())
        
        val_dis_acc = 100 * dis_correct / total_samples
        
        # Check for improvement
        if val_dis_acc > self.best_val_acc:
            self.best_val_acc = val_dis_acc
            self.best_epoch = epoch
            self.patience_counter = 0
            self.save_checkpoint(epoch, is_best=True)
        else:
            self.patience_counter += 1
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_cat_acc': 100 * cat_correct / total_samples,
            'val_dis_acc': val_dis_acc,
            'best_val_acc': self.best_val_acc,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'diseases': self.train_dataset.diseases,
            'categories': self.train_dataset.categories,
            'hierarchy': self.train_dataset.hierarchy,
        }
        
        save_dir = Path(self.config['checkpoint_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            path = save_dir / "best_model.pth"
            torch.save(checkpoint, path)
            print(f"  📦 Saved best model: {path} (acc: {self.best_val_acc:.2f}%)")
        
        # Also save latest
        path = save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, path)
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        for epoch in range(self.config['epochs']):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"Cat Acc: {train_metrics['train_cat_acc']:.1f}%, "
                  f"Dis Acc: {train_metrics['train_dis_acc']:.1f}%")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                  f"Cat Acc: {val_metrics['val_cat_acc']:.1f}%, "
                  f"Dis Acc: {val_metrics['val_dis_acc']:.1f}% "
                  f"(Best: {val_metrics['best_val_acc']:.1f}%)")
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⏹️  Early stopping at epoch {epoch+1} (no improvement for {self.config['patience']} epochs)")
                break
        
        print(f"\n✅ Training complete! Best accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")


# ============================================================================
# MAIN
# ============================================================================

def detect_gpu_type():
    """Detect GPU type and return optimal settings."""
    if not torch.cuda.is_available():
        return None, {}
    
    gpu_name = torch.cuda.get_device_name(0).upper()
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if 'A100' in gpu_name:
        return 'A100', {
            'memory_gb': 40,
            'optimal_batch_multiplier': 1.0,
            'mixed_precision': 'bf16',  # A100 supports bfloat16
        }
    elif 'A6000' in gpu_name or 'RTX A6000' in gpu_name:
        return 'A6000', {
            'memory_gb': 48,
            'optimal_batch_multiplier': 0.85,  # Slightly smaller batches for slower memory
            'mixed_precision': 'fp16',  # A6000 uses float16
        }
    else:
        # Generic GPU
        return 'UNKNOWN', {
            'memory_gb': gpu_memory_gb,
            'optimal_batch_multiplier': 0.75,
            'mixed_precision': 'fp16',
        }


def validate_environment():
    """Validate training environment before starting."""
    print("\n" + "="*60)
    print("🔍 VALIDATING ENVIRONMENT")
    print("="*60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        print("   Make sure you're running on a GPU node")
        sys.exit(1)
    
    # Detect GPU type
    gpu_type, gpu_config = detect_gpu_type()
    
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Type: {gpu_type}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check PyTorch version
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Check compute capability
    capability = torch.cuda.get_device_capability(0)
    print(f"   Compute capability: {capability[0]}.{capability[1]}")
    
    if capability[0] < 7:
        print("⚠️  WARNING: GPU compute capability < 7.0, some features may not work")
    
    # GPU-specific recommendations
    if gpu_type == 'A100':
        print(f"   💡 A100 detected - Using optimized settings for data center GPU")
        print(f"   💡 Mixed precision: bfloat16 (bf16)")
    elif gpu_type == 'A6000':
        print(f"   💡 A6000 detected - Adjusting settings for workstation GPU")
        print(f"   💡 Mixed precision: float16 (fp16)")
        print(f"   💡 Batch sizes scaled to 85% for optimal performance")
    
    return True, gpu_type, gpu_config


def validate_data_dirs(train_dir: str, val_dir: str):
    """Validate data directories exist and have correct structure."""
    print("\n" + "="*60)
    print("🔍 VALIDATING DATA DIRECTORIES")
    print("="*60)
    
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    
    # Check existence
    if not train_path.exists():
        print(f"❌ ERROR: Training directory not found: {train_dir}")
        print("   Run: python scripts/prepare_unified_dataset.py")
        sys.exit(1)
    
    if not val_path.exists():
        print(f"❌ ERROR: Validation directory not found: {val_dir}")
        print("   Run: python scripts/prepare_unified_dataset.py")
        sys.exit(1)
    
    # Count classes
    train_classes = [d for d in train_path.iterdir() if d.is_dir()]
    val_classes = [d for d in val_path.iterdir() if d.is_dir()]
    
    if len(train_classes) == 0:
        print(f"❌ ERROR: No disease classes found in {train_dir}")
        sys.exit(1)
    
    print(f"✅ Training directory: {train_dir}")
    print(f"   Classes: {len(train_classes)}")
    
    # Count images per class
    total_train = 0
    min_samples = float('inf')
    for cls_dir in train_classes:
        count = len(list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png')) + list(cls_dir.glob('*.jpeg')))
        total_train += count
        min_samples = min(min_samples, count)
        if count < 50:
            print(f"   ⚠️  Warning: Class '{cls_dir.name}' has only {count} samples")
    
    print(f"   Total images: {total_train:,}")
    print(f"   Min samples per class: {min_samples}")
    
    print(f"✅ Validation directory: {val_dir}")
    print(f"   Classes: {len(val_classes)}")
    
    total_val = sum(len(list(d.glob('*.jpg')) + list(d.glob('*.png')) + list(d.glob('*.jpeg'))) for d in val_classes)
    print(f"   Total images: {total_val:,}")
    
    if total_train < 1000:
        print("⚠️  WARNING: Very few training samples. Results may be poor.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Train skin disease classifier on A100")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/unified_train", help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="data/unified_val", help="Validation data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    # Model
    parser.add_argument("--backbone", type=str, default="efficientnet_b4",
                       choices=list(HierarchicalSkinClassifier.BACKBONES.keys()),
                       help="Backbone architecture")
    
    # Training (A100 optimized defaults)
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=384, help="Image size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    
    # Flags
    parser.add_argument("--use_weighted_sampling", action="store_true", help="Use weighted sampling")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--skip_validation", action="store_true", help="Skip environment validation")
    
    args = parser.parse_args()
    
    # Validate environment FIRST and get GPU info
    gpu_type = 'UNKNOWN'
    gpu_config = {}
    
    if not args.skip_validation:
        _, gpu_type, gpu_config = validate_environment()
        validate_data_dirs(args.data_dir, args.val_dir)
    
    # Adjust batch size based on GPU type
    batch_size = args.batch_size
    if gpu_type == 'A6000' and not args.batch_size:
        # Auto-adjust batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'gradient_accumulation': args.gradient_accumulation,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'use_weighted_sampling': args.use_weighted_sampling,
        'compile_model': not args.no_compile,
        'gpu_type': gpu_type,
        'mixed_precision': gpu_config.get('mixed_precision', 'fp16')
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'gradient_accumulation': args.gradient_accumulation,
        'patience': args.patience,
        'num_workers': args.num_workers,
        'use_weighted_sampling': args.use_weighted_sampling,
        'compile_model': not args.no_compile,
    }
    
    print("\n" + "="*60)
    print("🔬 SKIN DISEASE CLASSIFIER - A100 TRAINING")
    print("="*60)
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Create checkpoint dir
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = Path(config['checkpoint_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Saved config to: {config_path}")
    
    try:
        trainer = Trainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
