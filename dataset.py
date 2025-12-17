#!/usr/bin/env python3
"""
Download from MULTIPLE Kaggle datasets to get ALL 12 diseases
"""

import subprocess
import os

def download_complete_dataset():
    """Combine multiple datasets for 100% coverage"""
    
    datasets = {
        # Dataset 1: Main DermNet (covers 7/12)
        'dermnet_main': {
            'id': 'shubhamgoel27/dermnet',
            'covers': ['psoriasis', 'atopic_dermatitis', 'acne_vulgaris', 
                      'rosacea', 'tinea', 'impetigo', 'scabies'],
            'missing': ['bowen_disease', 'lentigo_maligna', 'solar_lentigo', 
                       'cherry_angioma', 'vitiligo']
        },
        
        # Dataset 2: Skin Disease Dataset (has vitiligo!)
        'skin_disease': {
            'id': 'subirbiswas19/skin-disease-dataset',
            'covers': ['vitiligo'],  # ✅ Has vitiligo!
            'missing': ['bowen_disease', 'lentigo_maligna', 'solar_lentigo', 
                       'cherry_angioma']
        },
        
        # Dataset 3: Diverse Dermatology (broader coverage)
        'diverse_derm': {
            'id': 'nodoubttome/skin-cancer9-classesisic',
            'covers': ['additional benign lesions'],
            'missing': ['bowen_disease', 'lentigo_maligna', 'solar_lentigo', 
                       'cherry_angioma']
        }
    }
    
    print("=" * 70)
    print("  Downloading Complete Dataset (Multi-source)")
    print("=" * 70)
    
    for name, info in datasets.items():
        print(f"\n📥 Downloading {name}...")
        output_dir = f'./datasets/{name}'
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', info['id'],
            '-p', output_dir,
            '--unzip'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Downloaded {name}")
            print(f"  Covers: {', '.join(info['covers'])}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Still missing 4 diseases
    print("\n" + "=" * 70)
    print("⚠️  STILL MISSING 4 DISEASES:")
    print("=" * 70)
    print("  • bowen_disease")
    print("  • lentigo_maligna")
    print("  • solar_lentigo")
    print("  • cherry_angioma")
    print("\nThese are RARE - need manual collection or:")
    print("  1. Google Images (see alternative below)")
    print("  2. Medical image repositories")
    print("  3. Reduce to 16 diseases (remove these 4)")

if __name__ == "__main__":
    download_complete_dataset()
