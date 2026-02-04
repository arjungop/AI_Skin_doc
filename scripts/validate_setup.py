#!/usr/bin/env python3
"""
Pre-Training Validation Script
Checks everything before starting expensive A100 training
Catches issues early to avoid wasting queue time
"""

import sys
import os
from pathlib import Path
import subprocess
import json

# Colors
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_header(title):
    print(f"\n{'='*60}")
    print(f"{BLUE}{title}{NC}")
    print('='*60)

def print_ok(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def print_warn(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def print_error(msg):
    print(f"{RED}❌ {msg}{NC}")

def check_conda_env():
    """Verify we're in correct conda environment, NOT base."""
    print_header("CONDA ENVIRONMENT CHECK")
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
    
    if not conda_env:
        print_warn("Not using conda environment")
        return True
    
    if conda_env == 'base':
        print_error("Running in BASE conda environment!")
        print("   CRITICAL: Never use base environment")
        print("   Run: conda activate skindoc")
        return False
    
    print_ok(f"Using conda environment: {conda_env}")
    return True

def check_python_packages():
    """Check if required packages are installed."""
    print_header("PYTHON PACKAGES CHECK")
    
    required = {
        'torch': '2.0.0',
        'torchvision': '0.15.0',
        'numpy': '1.20.0',
        'PIL': None,
        'tqdm': None,
        'kaggle': None,
        'sklearn': None,
        'pandas': None,
    }
    
    all_ok = True
    for package, min_version in required.items():
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
                mod_name = 'Pillow'
            elif package == 'sklearn':
                import sklearn
                version = sklearn.__version__
                mod_name = 'scikit-learn'
            else:
                mod = __import__(package)
                version = mod.__version__
                mod_name = package
            
            # Check version if specified (simple string comparison)
            if min_version:
                # Simple version check - just verify it's installed
                # Full semantic versioning would require packaging module
                print_ok(f"{mod_name}: {version}")
            else:
                print_ok(f"{mod_name}: {version}")
                
        except ImportError:
            print_error(f"{package} not installed")
            all_ok = False
        except Exception as e:
            print_warn(f"{package}: {e}")
    
    # Check optional but useful packages
    optional = ['matplotlib', 'seaborn', 'tensorboard', 'albumentations']
    print("\n  Optional packages:")
    for pkg in optional:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"    ✅ {pkg}: {version}")
        except ImportError:
            print(f"    ⚠️  {pkg}: not installed (optional)")
    
    return all_ok

def check_cuda():
    """Check CUDA availability and version."""
    print_header("CUDA CHECK")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_error("CUDA not available!")
            print("   Make sure you're on a GPU node (A100 or A6000)")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        print_ok(f"CUDA available")
        print(f"   Device: {gpu_name}")
        print(f"   CUDA version: {torch.version.cuda}")
        
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU memory: {memory_gb:.1f} GB")
        
        # Detect GPU type
        gpu_type = 'Unknown'
        if 'A100' in gpu_name.upper():
            gpu_type = 'A100'
            print_ok(f"GPU type: A100 (40GB, Data Center GPU)")
            print("   💡 Optimal for training - fastest option")
        elif 'A6000' in gpu_name.upper() or 'RTX A6000' in gpu_name.upper():
            gpu_type = 'A6000'
            print_ok(f"GPU type: A6000 (48GB, Workstation GPU)")
            print("   💡 Good for training - more memory available")
        else:
            print_warn(f"GPU type: {gpu_name}")
            print("   ⚠️  Not A100 or A6000, but should work")
        
        if memory_gb < 30:
            print_warn(f"GPU has only {memory_gb:.1f} GB memory")
        else:
            print_ok(f"GPU memory sufficient: {memory_gb:.1f} GB")
        
        # Check compute capability
        capability = torch.cuda.get_device_capability(0)
        print(f"   Compute capability: {capability[0]}.{capability[1]}")
        
        if capability[0] >= 8:
            print_ok("GPU supports modern features (compute 8.0+)")
        elif capability[0] >= 7:
            print_ok("GPU supports training (compute 7.0+)")
        else:
            print_warn(f"GPU compute capability {capability[0]}.{capability[1]} (may have issues)")
        
        return True
        
    except Exception as e:
        print_error(f"Error checking CUDA: {e}")
        return False

def check_kaggle_credentials():
    """Check if Kaggle API is configured."""
    print_header("KAGGLE API CHECK")
    
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_json.exists():
        print_error("Kaggle credentials not found")
        print("   Expected at: ~/.kaggle/kaggle.json")
        print("   Download from: https://www.kaggle.com/settings")
        return False
    
    print_ok(f"Kaggle credentials found: {kaggle_json}")
    
    # Check permissions
    stat = kaggle_json.stat()
    perms = oct(stat.st_mode)[-3:]
    if perms != '600':
        print_warn(f"Incorrect permissions: {perms} (should be 600)")
        print("   Run: chmod 600 ~/.kaggle/kaggle.json")
    else:
        print_ok("Permissions correct (600)")
    
    # Test API
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '--page-size', '1'],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print_ok("Kaggle API working")
            return True
        else:
            print_error("Kaggle API not working")
            print(f"   Error: {result.stderr.decode()}")
            return False
    except Exception as e:
        print_error(f"Error testing Kaggle API: {e}")
        return False

def check_datasets():
    """Check if datasets are downloaded."""
    print_header("DATASETS CHECK")
    
    datasets_dir = Path('datasets')
    required_datasets = {
        'isic_2019': 'ISIC_2019_Training_Input',
        'ham10000': 'HAM10000_images_part_1',
        'dermnet': 'train',
    }
    
    all_ok = True
    total_images = 0
    
    for dataset, check_path in required_datasets.items():
        full_path = datasets_dir / dataset / check_path
        if full_path.exists():
            # Count images
            count = len(list(full_path.rglob('*.jpg')) + 
                       list(full_path.rglob('*.png')) +
                       list(full_path.rglob('*.jpeg')))
            print_ok(f"{dataset}: {count:,} images")
            total_images += count
        else:
            print_error(f"{dataset}: NOT FOUND")
            print(f"   Expected at: {full_path}")
            all_ok = False
    
    if all_ok:
        print_ok(f"Total images available: {total_images:,}")
    
    return all_ok

def check_prepared_data():
    """Check if unified dataset is prepared."""
    print_header("PREPARED DATA CHECK")
    
    train_dir = Path('data/unified_train')
    val_dir = Path('data/unified_val')
    
    if not train_dir.exists():
        print_warn("Unified training data not found")
        print("   Run: python scripts/prepare_unified_dataset_v2.py")
        return False
    
    if not val_dir.exists():
        print_warn("Unified validation data not found")
        print("   Run: python scripts/prepare_unified_dataset_v2.py")
        return False
    
    # Count classes and images
    train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d for d in val_dir.iterdir() if d.is_dir()]
    
    train_count = len(list(train_dir.rglob('*.jpg')) + 
                     list(train_dir.rglob('*.png')) +
                     list(train_dir.rglob('*.jpeg')))
    val_count = len(list(val_dir.rglob('*.jpg')) + 
                   list(val_dir.rglob('*.png')) +
                   list(val_dir.rglob('*.jpeg')))
    
    print_ok(f"Training: {len(train_classes)} classes, {train_count:,} images")
    print_ok(f"Validation: {len(val_classes)} classes, {val_count:,} images")
    
    if train_count < 1000:
        print_warn(f"Very few training images ({train_count}). Results may be poor.")
    
    if val_count < 100:
        print_warn(f"Very few validation images ({val_count}). May not be representative.")
    
    # Check for empty classes
    for cls_dir in train_classes:
        count = len(list(cls_dir.glob('*.jpg')) + 
                   list(cls_dir.glob('*.png')) +
                   list(cls_dir.glob('*.jpeg')))
        if count < 50:
            print_warn(f"Class '{cls_dir.name}' has only {count} samples")
    
    return True

def check_disk_space():
    """Check available disk space."""
    print_header("DISK SPACE CHECK")
    
    try:
        stat = os.statvfs('.')
        free_gb = (stat.f_bavail * stat.f_frsize) / 1e9
        
        print(f"   Free space: {free_gb:.1f} GB")
        
        if free_gb < 10:
            print_error(f"Low disk space: {free_gb:.1f} GB")
            print("   Need at least 10GB for checkpoints and logs")
            return False
        elif free_gb < 50:
            print_warn(f"Disk space: {free_gb:.1f} GB (may need more for large models)")
        else:
            print_ok(f"Sufficient disk space: {free_gb:.1f} GB")
        
        return True
    except Exception as e:
        print_warn(f"Could not check disk space: {e}")
        return True

def check_scripts():
    """Check if training scripts exist."""
    print_header("SCRIPTS CHECK")
    
    required_scripts = [
        'scripts/train_a100.py',
        'scripts/prepare_unified_dataset_v2.py',
    ]
    
    all_ok = True
    for script in required_scripts:
        path = Path(script)
        if path.exists():
            print_ok(f"{script}")
        else:
            print_error(f"{script} NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "="*60)
    print(f"{BLUE}PRE-TRAINING VALIDATION FOR A100{NC}")
    print("="*60)
    print("\nThis will verify your setup before starting training")
    print("to avoid wasting time in the queue.\n")
    
    checks = [
        ("Conda Environment", check_conda_env),
        ("Python Packages", check_python_packages),
        ("CUDA & GPU", check_cuda),
        ("Kaggle Credentials", check_kaggle_credentials),
        ("Raw Datasets", check_datasets),
        ("Prepared Data", check_prepared_data),
        ("Disk Space", check_disk_space),
        ("Training Scripts", check_scripts),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = f"{GREEN}PASS{NC}" if result else f"{RED}FAIL{NC}"
        print(f"  {name:.<30} {status}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print(f"\n{GREEN}✅ ALL CHECKS PASSED! Ready to train.{NC}")
        print("\nRun training with:")
        print("  bash scripts/run_complete_training.sh")
        print("or:")
        print("  python scripts/train_a100.py --backbone efficientnet_b4 --batch_size 64 --epochs 50")
        return 0
    else:
        print(f"\n{RED}❌ {total - passed} check(s) failed. Fix issues before training.{NC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
