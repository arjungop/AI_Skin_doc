#!/usr/bin/env python3
"""
Extract DermNet and other datasets (excluding ISIC)
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm

class DatasetExtractor:
    def __init__(self):
        self.extracted_count = 0
        
    def extract_zip(self, zip_path, extract_to, description):
        """Extract zip with progress bar"""
        
        print(f"\n{'='*70}")
        print(f"📦 Extracting: {description}")
        print(f"{'='*70}")
        print(f"From: {zip_path}")
        print(f"To: {extract_to}")
        
        if not os.path.exists(zip_path):
            print(f"❌ File not found: {zip_path}")
            return False
        
        os.makedirs(extract_to, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                total_files = len(members)
                
                print(f"\nExtracting {total_files:,} files...")
                
                for member in tqdm(members, desc="Extracting", unit="file"):
                    try:
                        zip_ref.extract(member, extract_to)
                    except Exception:
                        continue
            
            print(f"✓ Extraction complete!")
            
            # Count images
            image_count = self.count_images(extract_to)
            print(f"📊 Found {image_count:,} images")
            
            self.extracted_count += 1
            return True
            
        except zipfile.BadZipFile:
            print(f"❌ Corrupted zip file")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def count_images(self, directory):
        """Count image files"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        count = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in extensions:
                    count += 1
        
        return count
    
    def extract_all(self):
        """Extract all non-ISIC datasets"""
        
        print("=" * 70)
        print("  Extracting DermNet & Other Datasets")
        print("  (ISIC excluded - you already have it)")
        print("=" * 70)
        
        # Define datasets to extract (NO ISIC)
        datasets = [
            {
                'zip': './datasets/dermnet_main/dermnet.zip',
                'extract_to': './data/DermNet/',
                'name': 'DermNet (19,500 images)'
            },
            {
                'zip': './datasets/skin_disease/skin-disease-dataset.zip',
                'extract_to': './data/SkinDisease/',
                'name': 'Skin Disease Dataset (6,700 images)'
            },
            {
                'zip': './datasets/diverse_derm/skin-cancer9-classesisic.zip',
                'extract_to': './data/DiverseDerm/',
                'name': 'Diverse Dermatology (15,000 images)'
            }
        ]
        
        # Extract each
        for dataset in datasets:
            if os.path.exists(dataset['zip']):
                self.extract_zip(
                    dataset['zip'],
                    dataset['extract_to'],
                    dataset['name']
                )
            else:
                print(f"\n⚠️  Not found: {dataset['name']}")
                print(f"   Expected at: {dataset['zip']}")
        
        # Summary
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)
        print(f"\n✅ Successfully extracted: {self.extracted_count} dataset(s)")
        
        # Count images in each
        print("\n📊 Dataset Contents:")
        
        total_new = 0
        for dataset in datasets:
            if os.path.exists(dataset['extract_to']):
                count = self.count_images(dataset['extract_to'])
                if count > 0:
                    print(f"   {dataset['name']:35s}: {count:,} images")
                    total_new += count
        
        print(f"\n{'='*70}")
        print(f"Total NEW Images: {total_new:,}")
        print(f"{'='*70}")
        
        if total_new > 20000:
            print("\n🎉 Excellent! Combined with ISIC, you have 45,000+ images!")
        elif total_new > 10000:
            print("\n✅ Good! Combined with ISIC, you have 35,000+ images!")
        else:
            print("\n⚠️  Some datasets may not have extracted properly")
        
        print("\nNext steps:")
        print("  1. Verify extraction: ls -la ./data/DermNet/")
        print("  2. Organize data: python organize_safederm.py")
        print("  3. Start training!")

if __name__ == "__main__":
    try:
        extractor = DatasetExtractor()
        extractor.extract_all()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

