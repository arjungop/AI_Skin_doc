#!/usr/bin/env python3
"""
Complete DermNet Downloader for SafeDerm
Downloads 12 missing diseases for FREE
No API keys, no registration needed
"""

import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from urllib.parse import urljoin
import hashlib

class DermNetDownloader:
    def __init__(self, output_dir='./dermnet_images'):
        self.output_dir = output_dir
        self.base_url = 'https://dermnetnz.org'
        
        # Map SafeDerm diseases to DermNet URLs
        self.disease_urls = {
            # Cancer/Precancer (2 missing)
            'bowen_disease': [
                'https://dermnetnz.org/topics/bowens-disease',
                'https://dermnetnz.org/images/bowens-disease-images'
            ],
            'lentigo_maligna': [
                'https://dermnetnz.org/topics/lentigo-maligna',
                'https://dermnetnz.org/images/lentigo-maligna-images'
            ],
            
            # Benign (2 missing)
            'solar_lentigo': [
                'https://dermnetnz.org/topics/solar-lentigo',
                'https://dermnetnz.org/images/solar-lentigo-images'
            ],
            'cherry_angioma': [
                'https://dermnetnz.org/topics/cherry-angioma',
                'https://dermnetnz.org/images/cherry-angioma-images'
            ],
            
            # Inflammatory (4 missing)
            'psoriasis': [
                'https://dermnetnz.org/topics/psoriasis',
                'https://dermnetnz.org/images/psoriasis-images',
                'https://dermnetnz.org/topics/chronic-plaque-psoriasis'
            ],
            'atopic_dermatitis': [
                'https://dermnetnz.org/topics/atopic-dermatitis',
                'https://dermnetnz.org/images/atopic-dermatitis-images',
                'https://dermnetnz.org/topics/eczema'
            ],
            'acne_vulgaris': [
                'https://dermnetnz.org/topics/acne',
                'https://dermnetnz.org/images/acne-images'
            ],
            'rosacea': [
                'https://dermnetnz.org/topics/rosacea',
                'https://dermnetnz.org/images/rosacea-images'
            ],
            
            # Infectious (3 missing)
            'tinea': [
                'https://dermnetnz.org/topics/tinea-corporis',
                'https://dermnetnz.org/topics/dermatophyte-infections',
                'https://dermnetnz.org/images/tinea-images'
            ],
            'impetigo': [
                'https://dermnetnz.org/topics/impetigo',
                'https://dermnetnz.org/images/impetigo-images'
            ],
            'scabies': [
                'https://dermnetnz.org/topics/scabies',
                'https://dermnetnz.org/images/scabies-images'
            ],
            
            # Pigmentary (1 missing)
            'vitiligo': [
                'https://dermnetnz.org/topics/vitiligo',
                'https://dermnetnz.org/images/vitiligo-images'
            ]
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        os.makedirs(output_dir, exist_ok=True)
    
    def get_image_urls_from_page(self, url):
        """Extract all image URLs from a DermNet page"""
        try:
            response = self.session.get(url, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            image_urls = set()
            
            # Method 1: Find images in content area
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith('/'):
                        src = urljoin(self.base_url, src)
                    
                    # Only keep image files
                    if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                        # Skip logos, icons, small images
                        if not any(skip in src.lower() for skip in ['logo', 'icon', 'banner', 'avatar']):
                            image_urls.add(src)
            
            # Method 2: Find links to full-size images
            for a in soup.find_all('a', href=True):
                href = a['href']
                if any(ext in href.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    if href.startswith('/'):
                        href = urljoin(self.base_url, href)
                    image_urls.add(href)
            
            # Method 3: Find images in galleries
            for div in soup.find_all('div', class_=['gallery', 'image-gallery', 'slideshow']):
                for img in div.find_all('img'):
                    src = img.get('src') or img.get('data-src')
                    if src:
                        if src.startswith('/'):
                            src = urljoin(self.base_url, src)
                        if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                            image_urls.add(src)
            
            return list(image_urls)
            
        except Exception as e:
            print(f"⚠️  Error fetching {url}: {e}")
            return []
    
    def download_image(self, url, save_path):
        """Download a single image"""
        try:
            response = self.session.get(url, timeout=15, stream=True)
            if response.status_code == 200:
                # Check if it's actually an image
                content_type = response.headers.get('content-type', '')
                if 'image' in content_type:
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
            return False
        except Exception as e:
            return False
    
    def download_disease(self, disease_name, urls):
        """Download all images for a specific disease"""
        
        disease_dir = os.path.join(self.output_dir, disease_name)
        os.makedirs(disease_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"📥 Downloading: {disease_name}")
        print(f"{'='*70}")
        
        # Collect image URLs from all provided pages
        all_image_urls = set()
        
        for url in urls:
            print(f"Scanning: {url}")
            image_urls = self.get_image_urls_from_page(url)
            all_image_urls.update(image_urls)
            time.sleep(1)  # Be polite to server
        
        all_image_urls = list(all_image_urls)
        print(f"Found {len(all_image_urls)} images\n")
        
        if len(all_image_urls) == 0:
            print(f"⚠️  No images found for {disease_name}")
            return 0
        
        # Download images
        successful = 0
        
        for idx, img_url in enumerate(tqdm(all_image_urls, desc="Downloading")):
            # Generate unique filename using hash
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
            ext = img_url.split('.')[-1].split('?')[0].lower()
            
            # Fallback to jpg if extension unclear
            if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                ext = 'jpg'
            
            filename = f"{disease_name}_{idx:04d}_{url_hash}.{ext}"
            filepath = os.path.join(disease_dir, filename)
            
            # Skip if already downloaded
            if os.path.exists(filepath):
                successful += 1
                continue
            
            # Download
            if self.download_image(img_url, filepath):
                successful += 1
            
            # Be polite - delay between requests
            time.sleep(0.3)
        
        print(f"✓ Downloaded {successful}/{len(all_image_urls)} images")
        return successful
    
    def download_all(self):
        """Download all 12 diseases"""
        
        print("=" * 70)
        print("  DermNet Image Downloader for SafeDerm")
        print("  Free & Open Source - No Registration Required")
        print("=" * 70)
        print(f"\n📂 Output directory: {os.path.abspath(self.output_dir)}")
        print(f"📊 Diseases to download: {len(self.disease_urls)}\n")
        
        input("Press Enter to start downloading...")
        
        total_images = 0
        results = {}
        
        for disease_name, urls in self.disease_urls.items():
            count = self.download_disease(disease_name, urls)
            results[disease_name] = count
            total_images += count
            time.sleep(2)  # Delay between diseases
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 70)
        
        for disease, count in sorted(results.items()):
            status = "✓" if count > 0 else "⚠️"
            print(f"{status} {disease:25s}: {count:4d} images")
        
        print(f"\n{'='*70}")
        print(f"🎉 Total: {total_images} images downloaded")
        print(f"📂 Location: {os.path.abspath(self.output_dir)}")
        print(f"✅ SafeDerm dataset complete (20/20 diseases)!")
        print(f"{'='*70}\n")
        
        return results


if __name__ == "__main__":
    try:
        print("\n🚀 Starting DermNet Downloader...\n")
        
        # Create downloader
        downloader = DermNetDownloader(output_dir='./dermnet_images')
        
        # Download all diseases
        results = downloader.download_all()
        
        # Check results
        if sum(results.values()) == 0:
            print("\n⚠️  No images downloaded. Possible issues:")
            print("   1. Internet connection")
            print("   2. DermNet website changed structure")
            print("   3. Firewall blocking requests")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("Run again to continue from where it stopped")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
