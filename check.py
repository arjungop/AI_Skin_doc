#!/usr/bin/env python3
"""
Check what datasets are actually usable
"""

import os
from pathlib import Path

def count_images(directory):
    """Count images recursively"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    count = 0
    
    if not os.path.exists(directory):
        return 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                count += 1
    
    return count

print("="*70)
print("  SafeDerm Dataset Status Check")
print("="*70)

datasets = {
    'ISIC 2019': './isic_data/isic_2019/ISIC_2019_Training_Input',
    'ISIC 2018': './datasets/ISIC2018_Task3_Training_Input',
    'DermNet': './datasets/dermnet_main',
    'Diverse Derm': './datasets/diverse_derm',
    'Skin Disease': './datasets/skin_disease'
}

total = 0

for name, path in datasets.items():
    print(f"\n📁 {name}")
    print(f"   Path: {path}")
    
    count = count_images(path)
    
    if count > 0:
        print(f"   Status: ✅ {count:,} images")
        total += count
        
        # Show sample folders
        if os.path.exists(path):
            subdirs = [d.name for d in Path(path).iterdir() if d.is_dir()][:5]
            if subdirs:
                print(f"   Folders: {', '.join(subdirs)}...")
    else:
        print(f"   Status: ⚠️  No images found")
        
        # Check for zips
        if os.path.exists(path):
            zips = [f for f in os.listdir(path) if f.endswith('.zip')]
            if zips:
                print(f"   Found: {zips[0]} (NEEDS EXTRACTION)")

print("\n" + "="*70)
print(f"📊 Total Images: {total:,}")
print("="*70)

if total >= 30000:
    print("\n🎉 Perfect! Ready to organize.")
    print("\nNext: python organize_safederm_final.py")
elif total >= 10000:
    print("\n✅ Good enough to start.")
    print("\nNext: python organize_safederm_final.py")
else:
    print("\n⚠️  Need to extract datasets first")
