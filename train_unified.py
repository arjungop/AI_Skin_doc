#!/usr/bin/env python3
"""
UNIFIED SKIN DISEASE TRAINING SCRIPT
This script is designed to be called by `train_unified.sh`.
It handles argument parsing, data loading, model creation, and the training loop.
"""

import os
import sys
import argparse
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torchvision import models, transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================
TARGET_CLASSES = [
    "melanoma", "bcc", "scc", "ak", "nevus", "seborrheic_keratosis", 
    "angioma", "eczema", "psoriasis", "acne", "dermatitis", "urticaria", 
    "bullous", "candida", "herpes", "scabies", "impetigo", "nail_fungus", 
    "chickenpox", "shingles", "wart", "alopecia", "hyperpigmentation", 
    "lupus", "healthy"
]

CANCER_CLASSES = {"melanoma", "bcc", "scc", "ak"}
CANCER_WEIGHT = 3.0

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# DATASET
# ============================================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.classes = TARGET_CLASSES
        self.samples = []
        self.class_counts = Counter()
        
        print(f"Scanning {root_dir}...")
        for idx, cls in enumerate(self.classes):
            cls_dir = self.root / cls
            if not cls_dir.exists():
                print(f"Warning: Class {cls} not found in {root_dir}")
                continue
            
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), idx))
                    self.class_counts[idx] += 1
                    
        print(f"Found {len(self.samples)} images across {len(self.class_counts)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a random sample instead of crashing
            return self.__getitem__(np.random.randint(0, len(self)))

    def get_weighted_sampler(self):
        weights = []
        for _, label in self.samples:
            count = max(self.class_counts[label], 1)
            weight = 1.0 / count
            if self.classes[label] in CANCER_CLASSES:
                weight *= CANCER_WEIGHT
            weights.append(weight)
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def get_class_weights(self):
        counts = [max(self.class_counts[i], 1) for i in range(len(self.classes))]
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        # Apply cancer weight
        for i, cls in enumerate(self.classes):
            if cls in CANCER_CLASSES:
                weights[i] *= CANCER_WEIGHT
        # Normalize
        weights = weights / weights.sum() * len(self.classes)
        return weights

# ============================================================================
# MODEL
# ============================================================================
class SkinClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, dropout=0.5):
        super().__init__()
        
        if "convnext_large" in backbone_name:
            self.backbone = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
            in_features = 1536
        elif "convnext_base" in backbone_name:
            self.backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            in_features = 1024
        elif "convnext_small" in backbone_name:
            self.backbone = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
            in_features = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# MAIN TRAINING FUNC
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Skin Disease Training")
    
    # Arguments MUST match train_unified.sh
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--backbone", type=str, default="convnext_large")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4) # Matches --lr in shell script
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weighted_sampling", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"])
    
    args = parser.parse_args()
    logger = setup_logging("logs")
    
    logger.info("="*50)
    logger.info("Starting Unified Training")
    logger.info(f"Arguments: {vars(args)}")
    logger.info("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Transforms
    train_tf = T.Compose([
        T.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tf = T.Compose([
        T.Resize(int(args.image_size * 1.14)),
        T.CenterCrop(args.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    logger.info("Loading datasets...")
    train_ds = SkinDataset(args.data_dir, transform=train_tf)
    val_ds = SkinDataset(args.val_dir, transform=val_tf)
    
    # DataLoaders
    sampler = train_ds.get_weighted_sampler() if args.weighted_sampling else None
    shuffle = sampler is None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model Setup
    model = SkinClassifier(args.backbone, num_classes=len(TARGET_CLASSES))
    model = model.to(device)
    
    # Optimizer & Loss
    class_weights = train_ds.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Mixed Precision
    scaler = GradScaler(enabled=(args.mixed_precision != "no"))
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    
    # Training Loop
    best_acc = 0.0
    patience_counter = 0
    
    logger.info("Staring training loop...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # Training Phase
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for i, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                
                with autocast(dtype=amp_dtype, enabled=(args.mixed_precision != "no")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / args.grad_accum
                
                scaler.scale(loss).backward()
                
                if (i + 1) % args.grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * args.grad_accum
                pbar.set_postfix({'loss': loss.item() * args.grad_accum})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                with autocast(dtype=amp_dtype, enabled=(args.mixed_precision != "no")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        scheduler.step(val_acc)
        
        # Save Checkpoint
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_acc': val_acc
        }
        
        # Save latest
        torch.save(state, os.path.join(args.checkpoint_dir, "latest.pth"))
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(state, os.path.join(args.checkpoint_dir, "best_model.pth"))
            logger.info("✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
