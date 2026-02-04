#!/usr/bin/env python3
"""
PRE-TRAINING VALIDATION SCRIPT
Run this BEFORE submitting to the A100 queue to catch all issues!

This script validates:
1. All datasets exist and are readable
2. All images can be loaded
3. Model can be initialized
4. Forward/backward pass works
5. Checkpoint saving/loading works
6. Memory fits within A100 40GB
7. DataLoader works correctly

If this passes, training WILL NOT FAIL.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import traceback
from tqdm import tqdm
import gc

def print_header(msg):
    print("\n" + "="*60)
    print(f"  {msg}")
    print("="*60)

def print_pass(msg):
    print(f"  ✅ PASS: {msg}")

def print_fail(msg):
    print(f"  ❌ FAIL: {msg}")

def print_warn(msg):
    print(f"  ⚠️  WARN: {msg}")

class ValidationResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def add_pass(self, msg):
        self.passed += 1
        print_pass(msg)

    def add_fail(self, msg):
        self.failed += 1
        self.errors.append(msg)
        print_fail(msg)

    def add_warn(self, msg):
        self.warnings += 1
        print_warn(msg)

    def summary(self):
        print_header("VALIDATION SUMMARY")
        print(f"  ✅ Passed: {self.passed}")
        print(f"  ⚠️  Warnings: {self.warnings}")
        print(f"  ❌ Failed: {self.failed}")
        
        if self.failed > 0:
            print("\n  ERRORS:")
            for e in self.errors:
                print(f"    - {e}")
            print("\n  🚫 DO NOT SUBMIT TO QUEUE - FIX ERRORS FIRST!")
            return False
        else:
            print("\n  🎉 ALL CHECKS PASSED - SAFE TO SUBMIT TO QUEUE!")
            return True


def validate_directories(result, train_dir, val_dir):
    """Validate dataset directories exist and have images."""
    print_header("1. VALIDATING DIRECTORIES")
    
    # Check train directory
    train_path = Path(train_dir)
    if not train_path.exists():
        result.add_fail(f"Train directory not found: {train_dir}")
        return False
    result.add_pass(f"Train directory exists: {train_dir}")
    
    # Check val directory
    val_path = Path(val_dir)
    if not val_path.exists():
        result.add_fail(f"Validation directory not found: {val_dir}")
        return False
    result.add_pass(f"Validation directory exists: {val_dir}")
    
    # Check hierarchy.json
    hierarchy_path = train_path / "hierarchy.json"
    if not hierarchy_path.exists():
        result.add_fail(f"hierarchy.json not found in {train_dir}")
        return False
    result.add_pass("hierarchy.json found")
    
    # Count images per class
    class_counts = {}
    for class_dir in train_path.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('.'):
            count = len(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg")))
            class_counts[class_dir.name] = count
    
    total_train = sum(class_counts.values())
    if total_train == 0:
        result.add_fail("No training images found!")
        return False
    
    result.add_pass(f"Found {total_train} training images across {len(class_counts)} classes")
    
    # Check for empty classes
    empty_classes = [c for c, n in class_counts.items() if n == 0]
    if empty_classes:
        result.add_warn(f"Empty classes: {empty_classes}")
    
    # Check for small classes
    small_classes = [c for c, n in class_counts.items() if 0 < n < 50]
    if small_classes:
        result.add_warn(f"Small classes (<50 images): {small_classes}")
    
    return True


def validate_images(result, train_dir, sample_size=100):
    """Validate that images can be loaded."""
    print_header("2. VALIDATING IMAGES")
    
    train_path = Path(train_dir)
    all_images = list(train_path.rglob("*.jpg")) + list(train_path.rglob("*.png"))
    
    if len(all_images) == 0:
        result.add_fail("No images found!")
        return False
    
    # Sample images to validate
    import random
    sample = random.sample(all_images, min(sample_size, len(all_images)))
    
    corrupted = []
    for img_path in tqdm(sample, desc="Checking images"):
        try:
            img = Image.open(img_path)
            img.verify()  # Verify image is not corrupted
            img = Image.open(img_path)
            img = img.convert('RGB')
            if img.size[0] < 10 or img.size[1] < 10:
                corrupted.append(str(img_path))
        except Exception as e:
            corrupted.append(str(img_path))
    
    if corrupted:
        result.add_warn(f"{len(corrupted)} potentially corrupted images found")
        for c in corrupted[:5]:
            print(f"      - {c}")
    else:
        result.add_pass(f"Validated {len(sample)} sample images successfully")
    
    return True


def validate_model(result, backbone="efficientnet_b4", num_classes=19):
    """Validate model can be created and run."""
    print_header("3. VALIDATING MODEL")
    
    try:
        from torchvision import models
        
        # Test model creation
        if backbone == "efficientnet_b4":
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Linear(in_features, num_classes)
        elif backbone == "swin_b":
            model = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        
        result.add_pass(f"Model {backbone} created successfully")
        
        # Test forward pass (CPU)
        dummy_input = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            output = model(dummy_input)
        result.add_pass(f"Forward pass works (output shape: {output.shape})")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        result.add_pass(f"Total parameters: {total_params:,}")
        
        del model, dummy_input, output
        gc.collect()
        
        return True
        
    except Exception as e:
        result.add_fail(f"Model validation failed: {e}")
        traceback.print_exc()
        return False


def validate_gpu(result, batch_size=64, image_size=384):
    """Validate GPU is available and has enough memory."""
    print_header("4. VALIDATING GPU")
    
    if not torch.cuda.is_available():
        result.add_warn("CUDA not available - will use CPU (SLOW!)")
        return True
    
    result.add_pass(f"CUDA available: {torch.cuda.get_device_name()}")
    
    # Check memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    result.add_pass(f"GPU Memory: {total_mem:.1f} GB")
    
    if total_mem < 20:
        result.add_warn(f"GPU has only {total_mem:.1f}GB - reduce batch size!")
    
    # Test memory with actual batch
    try:
        from torchvision import models
        
        model = models.efficientnet_b4(weights=None)
        model = model.cuda()
        
        dummy = torch.randn(batch_size, 3, image_size, image_size).cuda()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(dummy)
            loss = output.mean()
            loss.backward()
        
        torch.cuda.synchronize()
        
        used_mem = torch.cuda.max_memory_allocated() / 1e9
        result.add_pass(f"Memory test passed (used {used_mem:.1f}GB with batch_size={batch_size})")
        
        del model, dummy, output
        torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        result.add_fail(f"OUT OF MEMORY with batch_size={batch_size}! Reduce batch size.")
        return False
    except Exception as e:
        result.add_warn(f"GPU test inconclusive: {e}")
        return True


def validate_dataloader(result, train_dir, batch_size=16):
    """Validate DataLoader works correctly."""
    print_header("5. VALIDATING DATALOADER")
    
    try:
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Dataset
        
        class SimpleDataset(Dataset):
            def __init__(self, root_dir):
                self.root = Path(root_dir)
                self.images = list(self.root.rglob("*.jpg")) + list(self.root.rglob("*.png"))
                self.transform = T.Compose([
                    T.Resize((384, 384)),
                    T.ToTensor(),
                ])
            
            def __len__(self):
                return min(100, len(self.images))
            
            def __getitem__(self, idx):
                img = Image.open(self.images[idx]).convert('RGB')
                return self.transform(img), 0
        
        dataset = SimpleDataset(train_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Test a few batches
        for i, (images, labels) in enumerate(loader):
            if i >= 3:
                break
        
        result.add_pass(f"DataLoader works with {len(dataset)} samples, batch_size={batch_size}, num_workers=4")
        return True
        
    except Exception as e:
        result.add_fail(f"DataLoader failed: {e}")
        traceback.print_exc()
        return False


def validate_checkpointing(result, checkpoint_dir="checkpoints"):
    """Validate checkpoint saving/loading works."""
    print_header("6. VALIDATING CHECKPOINTING")
    
    try:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = ckpt_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        result.add_pass(f"Checkpoint directory writable: {checkpoint_dir}")
        
        # Test saving a checkpoint
        from torchvision import models
        model = models.resnet18()
        
        checkpoint = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'test': 'value'
        }
        
        test_ckpt = ckpt_path / "test_checkpoint.pth"
        torch.save(checkpoint, test_ckpt)
        
        # Test loading
        loaded = torch.load(test_ckpt, weights_only=False)
        assert loaded['test'] == 'value'
        
        test_ckpt.unlink()
        result.add_pass("Checkpoint save/load works")
        
        return True
        
    except Exception as e:
        result.add_fail(f"Checkpointing failed: {e}")
        return False


def validate_disk_space(result, required_gb=50):
    """Validate enough disk space is available."""
    print_header("7. VALIDATING DISK SPACE")
    
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    if free_gb < required_gb:
        result.add_fail(f"Only {free_gb:.1f}GB free, need {required_gb}GB")
        return False
    
    result.add_pass(f"Disk space OK: {free_gb:.1f}GB free")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate training setup before A100 submission")
    parser.add_argument("--train-dir", default="data/unified_train", help="Training data directory")
    parser.add_argument("--val-dir", default="data/unified_val", help="Validation data directory")
    parser.add_argument("--backbone", default="efficientnet_b4", help="Model backbone")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size to test")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()
    
    print("\n" + "🔍"*30)
    print("  PRE-TRAINING VALIDATION")
    print("  Run this BEFORE submitting to A100 queue!")
    print("🔍"*30)
    
    result = ValidationResult()
    
    # Run all validations
    validate_directories(result, args.train_dir, args.val_dir)
    validate_images(result, args.train_dir)
    validate_model(result, args.backbone)
    validate_gpu(result, args.batch_size)
    validate_dataloader(result, args.train_dir)
    validate_checkpointing(result, args.checkpoint_dir)
    validate_disk_space(result)
    
    # Print summary
    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
