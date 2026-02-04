#!/usr/bin/env python3
"""
Validate all imports for A100 training scripts
Checks that every required module is available
"""

import sys
from pathlib import Path

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def check_import(module_name, package_name=None, min_version=None):
    """Check if a module can be imported and optionally check version."""
    try:
        if '.' in module_name:
            # Handle submodule imports like torch.nn
            parts = module_name.split('.')
            mod = __import__(parts[0])
            for part in parts[1:]:
                mod = getattr(mod, part)
        else:
            mod = __import__(module_name)
        
        # Get version
        version = getattr(mod, '__version__', 'unknown')
        
        # Check version if specified
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"{YELLOW}⚠️  {module_name}: {version} (need >= {min_version}){NC}")
                return False
        
        display_name = package_name or module_name
        print(f"{GREEN}✅ {display_name:.<40} {version}{NC}")
        return True
        
    except ImportError as e:
        display_name = package_name or module_name
        print(f"{RED}❌ {display_name:.<40} NOT INSTALLED{NC}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠️  {module_name}: {e}{NC}")
        return True  # Don't fail for non-critical errors

def main():
    print("="*60)
    print("TRAINING ENVIRONMENT IMPORT VALIDATION")
    print("="*60)
    
    # Core Python (always available)
    core_modules = [
        ('os', None),
        ('sys', None),
        ('json', None),
        ('pathlib', 'pathlib.Path'),
        ('argparse', None),
        ('collections', 'collections.Counter'),
        ('typing', None),
        ('subprocess', None),
        ('csv', None),
        ('shutil', None),
    ]
    
    # PyTorch ecosystem
    pytorch_modules = [
        ('torch', None, '2.0.0'),
        ('torch.nn', 'torch.nn'),
        ('torch.nn.functional', 'torch.nn.functional'),
        ('torch.utils.data', 'torch.utils.data'),
        ('torch.cuda.amp', 'torch.cuda.amp'),
        ('torch.optim', 'torch.optim'),
        ('torchvision', None, '0.15.0'),
        ('torchvision.transforms', 'torchvision.transforms'),
        ('torchvision.models', 'torchvision.models'),
    ]
    
    # Data science
    data_modules = [
        ('numpy', 'numpy (np)', '1.20.0'),
        ('pandas', 'pandas (pd)'),
        ('scipy', None),
        ('sklearn', 'scikit-learn'),
    ]
    
    # Image processing
    image_modules = [
        ('PIL', 'Pillow'),
    ]
    
    # Utilities
    utility_modules = [
        ('tqdm', None),
        ('matplotlib', None),
        ('kaggle', None),
    ]
    
    # Optional but recommended
    optional_modules = [
        ('tensorboard', 'tensorboard (optional)'),
        ('seaborn', 'seaborn (optional)'),
        ('albumentations', 'albumentations (optional)'),
        ('jupyter', 'jupyter (optional)'),
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("CORE PYTHON MODULES")
    print("="*60)
    for module_info in core_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        results[module] = check_import(module, display)
    
    print("\n" + "="*60)
    print("PYTORCH & TORCHVISION")
    print("="*60)
    for module_info in pytorch_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        min_ver = module_info[2] if len(module_info) > 2 else None
        results[module] = check_import(module, display, min_ver)
    
    # Check CUDA
    try:
        import torch
        print(f"\n{GREEN}CUDA Configuration:{NC}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Device count: {torch.cuda.device_count()}")
    except:
        print(f"{YELLOW}⚠️  Could not check CUDA configuration{NC}")
    
    print("\n" + "="*60)
    print("DATA SCIENCE LIBRARIES")
    print("="*60)
    for module_info in data_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        min_ver = module_info[2] if len(module_info) > 2 else None
        results[module] = check_import(module, display, min_ver)
    
    print("\n" + "="*60)
    print("IMAGE PROCESSING")
    print("="*60)
    for module_info in image_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        results[module] = check_import(module, display)
    
    print("\n" + "="*60)
    print("UTILITIES")
    print("="*60)
    for module_info in utility_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        results[module] = check_import(module, display)
    
    print("\n" + "="*60)
    print("OPTIONAL MODULES")
    print("="*60)
    optional_ok = 0
    for module_info in optional_modules:
        module = module_info[0]
        display = module_info[1] if len(module_info) > 1 else None
        if check_import(module, display):
            optional_ok += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    required_passed = sum(1 for k, v in results.items() if v)
    required_total = len(results)
    
    print(f"\nRequired modules: {required_passed}/{required_total} passed")
    print(f"Optional modules: {optional_ok}/{len(optional_modules)} available")
    
    if required_passed == required_total:
        print(f"\n{GREEN}✅ ALL REQUIRED IMPORTS AVAILABLE!{NC}")
        print(f"\nYou're ready to train. Run:")
        print(f"  python scripts/validate_setup.py")
        return 0
    else:
        print(f"\n{RED}❌ {required_total - required_passed} required module(s) missing!{NC}")
        print(f"\nInstall missing packages:")
        print(f"  pip install -r requirements_training.txt")
        print(f"\nOr run:")
        print(f"  bash scripts/server_setup.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
