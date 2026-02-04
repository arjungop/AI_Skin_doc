#!/usr/bin/env python3
"""
Verify all Kaggle dataset links are valid
Checks that datasets exist before downloading
"""

import subprocess
import sys

# Dataset slugs used in server_setup.sh
DATASETS = {
    'ISIC 2019': 'andrewmvd/isic-2019',
    'HAM10000': 'kmader/skin-cancer-mnist-ham10000',
    'DermNet': 'shubhamgoel27/dermnet',
    'Fitzpatrick17k': 'mmaximillian/fitzpatrick17k-images',
    'PAD-UFES-20': 'mahdavi1202/pad-ufes-20',
    '20 Skin Diseases': 'haroonalam16/20-skin-diseases-dataset',
    'Massive Skin Disease': 'kylegraupe/skin-disease-balanced-dataset',
}

GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def check_kaggle_api():
    """Check if kaggle is installed and configured."""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def verify_dataset(name, slug):
    """Verify a dataset exists on Kaggle."""
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'list', '-s', slug],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and slug in result.stdout:
            return True, "Available"
        else:
            return False, "Not found or inaccessible"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    print("="*60)
    print("KAGGLE DATASET LINK VERIFICATION")
    print("="*60)
    
    if not check_kaggle_api():
        print(f"\n{RED}ERROR: Kaggle CLI not available or not configured{NC}")
        print("\nSetup Kaggle API:")
        print("  1. Download kaggle.json from https://www.kaggle.com/settings")
        print("  2. mv kaggle.json ~/.kaggle/")
        print("  3. chmod 600 ~/.kaggle/kaggle.json")
        print("  4. pip install kaggle")
        sys.exit(1)
    
    print(f"\n{GREEN}✅ Kaggle API configured{NC}\n")
    print("Verifying dataset links...\n")
    
    results = {}
    for name, slug in DATASETS.items():
        print(f"Checking {name:.<40} ", end='', flush=True)
        valid, msg = verify_dataset(name, slug)
        results[name] = valid
        
        if valid:
            print(f"{GREEN}✅ {msg}{NC}")
        else:
            print(f"{RED}❌ {msg}{NC}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    valid_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nDatasets verified: {valid_count}/{total_count}")
    
    if valid_count == total_count:
        print(f"{GREEN}✅ All dataset links are valid!{NC}")
        return 0
    else:
        print(f"{YELLOW}⚠️  Some datasets couldn't be verified{NC}")
        print("\nNote: Datasets may still work if:")
        print("  - Your Kaggle account needs to accept competition rules")
        print("  - Dataset is temporarily unavailable")
        print("  - API rate limit reached")
        return 1

if __name__ == "__main__":
    sys.exit(main())
