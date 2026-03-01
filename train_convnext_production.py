#!/usr/bin/env python3
"""
Production ConvNeXt Training Script - Skin Disease Classification
Optimized for RTX 6000 Ada (48GB VRAM)
29-class DermNet-based classification

Features:
- Mixed precision training (AMP)
- Gradient checkpointing for memory efficiency
- Advanced augmentation
- Automatic checkpoint management
- Early stopping with patience
- Comprehensive logging
- TensorBoard integration
- Resume from checkpoint support
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class Config:
    """Training configuration"""
    # Model - ConvNeXt Large for maximum accuracy
    model_name = 'convnext_large'
    img_size = 384
    drop_rate = 0.4
    drop_path_rate = 0.3
    
    # Training - Optimized for top 20 classes
    batch_size = 48  # Reduced for Large model
    epochs = 50
    lr = 3e-3  # Slightly lower LR for Large model
    weight_decay = 0.05
    warmup_epochs = 5
    label_smoothing = 0.1
    
    # Data - Top 20 classes from unified dataset
    data_dir = '/dist_home/suryansh/arjungop/AI_Skin_doc/data'
    train_dir = 'top20_train'
    val_dir = 'top20_val'
    test_dir = 'top20_test'
    num_workers = 16
    
    # Checkpointing
    save_dir = 'checkpoints/convnext_large_top20'
    save_every = 5  # Save checkpoint every N epochs
    keep_last_n = 3  # Keep only last N checkpoints
    
    # Early stopping
    patience = 15
    min_delta = 0.001
    
    # Logging
    log_dir = 'logs'
    tensorboard_dir = 'runs/convnext_large_top20'
    
    # System
    seed = 42
    mixed_precision = True
    gradient_checkpointing = True  # Enabled for Large model memory efficiency
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ==================== Utility Functions ====================
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU available, using CPU")
    return device


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== Dataset ====================
class SkinDiseaseDataset(Dataset):
    """Skin disease classification dataset"""
    
    def __init__(self, root_dir, transform=None, verbose=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory not found: {self.root_dir}")
        
        if verbose:
            print(f"📂 Loading dataset from: {self.root_dir}")
        
        # Scan directory structure
        for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            self.classes.append(class_name)
            self.class_to_idx[class_name] = class_idx
            
            count = 0
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in class_dir.glob(ext):
                    self.samples.append((str(img_path), class_idx))
                    count += 1
            
            if verbose and count > 0:
                print(f"  ✓ {class_name}: {count:,} images")
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        if verbose:
            print(f"✅ Total: {len(self.samples):,} images, {len(self.classes)} classes\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            # Return black image as fallback
            image = Image.new('RGB', (384, 384), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== Data Augmentation ====================
def get_transforms(img_size=384, split='train'):
    """Get data augmentation transforms"""
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ==================== Model ====================
def create_model(num_classes, config):
    """Create ConvNeXt model"""
    print(f"\n🔨 Creating {config.model_name}...")
    
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
    )
    
    if config.gradient_checkpointing:
        model.set_grad_checkpointing(enable=True)
        print("✓ Gradient checkpointing enabled")
    
    params = count_parameters(model)
    print(f"✅ Model created! Trainable parameters: {params:,}\n")
    
    return model


# ==================== Training Functions ====================
def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, config, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{config.epochs} [TRAIN]', ncols=100)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if config.mixed_precision:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}', 'acc': f'{acc:.2f}%'})
        
        # TensorBoard logging
        if writer and batch_idx % 50 == 0:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, config, split='VAL'):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{config.epochs} [{split}]', ncols=100)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{running_loss/(batch_idx+1):.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


# ==================== Checkpoint Management ====================
def save_checkpoint(state, config, is_best=False, filename='checkpoint.pth'):
    """Save training checkpoint"""
    os.makedirs(config.save_dir, exist_ok=True)
    filepath = os.path.join(config.save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(config.save_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f"💾 Best model saved to {best_path}")
    
    # Keep only last N checkpoints
    checkpoints = sorted(Path(config.save_dir).glob('checkpoint_epoch_*.pth'))
    if len(checkpoints) > config.keep_last_n:
        for old_ckpt in checkpoints[:-config.keep_last_n]:
            old_ckpt.unlink()


def load_checkpoint(model, optimizer, scheduler, config, checkpoint_path):
    """Load checkpoint for resuming training"""
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return 0, {}
    
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint.get('history', {})
    
    print(f"✅ Resumed from epoch {checkpoint['epoch']}")
    return start_epoch, history


# ==================== Learning Rate Scheduling ====================
def get_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler with warmup"""
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.epochs * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== Main Training Loop ====================
def main(args):
    # Configuration
    config = Config(**vars(args))
    set_seed(config.seed)
    device = get_device()
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(config.tensorboard_dir)
    
    # Print configuration
    print("\n" + "="*70)
    print("🔥 CONVNEXT PRODUCTION TRAINING 🔥")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.train_dir} ({config.data_dir})")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"Mixed Precision: {config.mixed_precision}")
    print(f"Save Directory: {config.save_dir}")
    print("="*70 + "\n")
    
    # Load datasets
    print("📊 Loading datasets...")
    train_dataset = SkinDiseaseDataset(
        os.path.join(config.data_dir, config.train_dir),
        get_transforms(config.img_size, 'train'),
        verbose=True
    )
    
    val_dataset = SkinDiseaseDataset(
        os.path.join(config.data_dir, config.val_dir),
        get_transforms(config.img_size, 'val'),
        verbose=True
    )
    
    num_classes = len(train_dataset.classes)
    
    # Save class mapping
    class_mapping = {
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'num_classes': num_classes,
        'idx_to_class': {v: k for k, v in train_dataset.class_to_idx.items()}
    }
    with open(os.path.join(config.save_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Save config
    with open(os.path.join(config.save_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Model
    model = create_model(num_classes, config)
    model = model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.mixed_precision)
    
    # Resume from checkpoint if exists
    start_epoch = 1
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    resume_path = os.path.join(config.save_dir, 'checkpoint_latest.pth')
    if os.path.exists(resume_path) and not args.fresh_start:
        start_epoch, history = load_checkpoint(model, optimizer, scheduler, config, resume_path)
    
    # Training state
    best_val_acc = max(history.get('val_acc', [0]))
    patience_counter = 0
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, config, writer
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, config
        )
        
        # Update scheduler
        for _ in range(len(train_loader)):
            scheduler.step()
        
        # Save history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        eta = elapsed / (epoch - start_epoch + 1) * (config.epochs - epoch)
        
        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{config.epochs} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*70}\n")
        
        # Check for improvement
        is_best = val_acc > best_val_acc + config.min_delta
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"🎉 New best validation accuracy: {best_val_acc:.2f}%\n")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{config.patience}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'history': history,
            'config': config.to_dict(),
        }
        
        # Save latest checkpoint
        save_checkpoint(checkpoint, config, is_best=is_best, filename='checkpoint_latest.pth')
        
        # Save periodic checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(checkpoint, config, filename=f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {config.save_dir}")
    print(f"TensorBoard logs: {config.tensorboard_dir}")
    print("="*70 + "\n")
    
    writer.close()
    
    # Save final history
    with open(os.path.join(config.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    return best_val_acc


# ==================== Entry Point ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ConvNeXt for Skin Disease Classification')
    
    # Model args
    parser.add_argument('--model_name', type=str, default='convnext_base', help='ConvNeXt model variant')
    parser.add_argument('--img_size', type=int, default=384, help='Input image size')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=4e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    
    # Data args
    parser.add_argument('--data_dir', type=str, 
                       default='/dist_home/suryansh/arjungop/AI_Skin_doc/data',
                       help='Data directory')
    parser.add_argument('--train_dir', type=str, default='main_train', help='Training data subdirectory')
    parser.add_argument('--val_dir', type=str, default='main_val', help='Validation data subdirectory')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader workers')
    
    # Checkpoint args
    parser.add_argument('--save_dir', type=str, default='checkpoints/convnext_base_production',
                       help='Checkpoint save directory')
    parser.add_argument('--fresh_start', action='store_true', help='Start fresh (ignore existing checkpoints)')
    
    # System args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='Use mixed precision training')
    
    args = parser.parse_args()
    
    try:
        best_acc = main(args)
        print(f"\n✅ Training finished successfully! Best accuracy: {best_acc:.2f}%")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error:")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
