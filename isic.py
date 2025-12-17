import os
import requests
import zipfile
from tqdm import tqdm
import time

class ISICDownloader:
    def __init__(self, output_dir='./isic_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_file(self, url, destination, max_retries=3):
        """Download file with retry logic and validation"""
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}...")
                
                # Check if file exists and get its size
                if os.path.exists(destination):
                    existing_size = os.path.getsize(destination)
                    print(f"Found existing file ({existing_size:,} bytes)")
                else:
                    existing_size = 0
                
                # Get expected file size from server
                response = requests.head(url, timeout=10)
                expected_size = int(response.headers.get('content-length', 0))
                print(f"Expected size: {expected_size:,} bytes")
                
                # If file is complete, skip download
                if existing_size == expected_size and existing_size > 0:
                    print(f"✓ File already complete!")
                    return True
                
                # Download with resume support
                headers = {}
                mode = 'wb'
                initial = 0
                
                if existing_size > 0 and existing_size < expected_size:
                    print(f"Resuming from {existing_size:,} bytes...")
                    headers['Range'] = f'bytes={existing_size}-'
                    mode = 'ab'
                    initial = existing_size
                
                # Start download
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                
                if response.status_code not in [200, 206]:
                    print(f"❌ Server returned status {response.status_code}")
                    if attempt < max_retries - 1:
                        # Delete corrupted file and retry
                        if os.path.exists(destination):
                            os.remove(destination)
                        time.sleep(5)
                        continue
                    return False
                
                # Download with progress bar
                with open(destination, mode) as f:
                    with tqdm(
                        total=expected_size,
                        initial=initial,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=os.path.basename(destination)
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                # Verify download
                final_size = os.path.getsize(destination)
                if final_size == expected_size:
                    print(f"✓ Download complete and verified!")
                    return True
                else:
                    print(f"⚠️  Size mismatch: got {final_size:,}, expected {expected_size:,}")
                    if attempt < max_retries - 1:
                        os.remove(destination)
                        time.sleep(5)
                        continue
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                return False
            
            except KeyboardInterrupt:
                print("\n⚠️  Download interrupted by user")
                raise
            
            except Exception as e:
                print(f"❌ Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return False
        
        return False
    
    def verify_zip(self, zip_path):
        """Verify zip file integrity"""
        try:
            print(f"\n🔍 Verifying {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Test zip integrity
                corrupt = zip_ref.testzip()
                if corrupt is not None:
                    print(f"❌ Corrupted file found: {corrupt}")
                    return False
                print(f"✓ Zip file is valid")
                return True
        except zipfile.BadZipFile:
            print(f"❌ Not a valid zip file")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def download_isic_2019(self):
        """Download ISIC 2019 Challenge Dataset"""
        print("=" * 70)
        print("  ISIC 2019 Dataset Downloader")
        print("=" * 70)
        
        year_dir = os.path.join(self.output_dir, 'isic_2019')
        os.makedirs(year_dir, exist_ok=True)
        
        # Files to download
        files = {
            'images': {
                'filename': 'ISIC_2019_Training_Input.zip',
                'url': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip',
                'size_gb': 9.1
            },
            'labels': {
                'filename': 'ISIC_2019_Training_GroundTruth.csv',
                'url': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv',
                'size_gb': 0.001
            },
            'metadata': {
                'filename': 'ISIC_2019_Training_Metadata.csv',
                'url': 'https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv',
                'size_gb': 0.002
            }
        }
        
        # Download each file
        for file_type, file_info in files.items():
            filename = file_info['filename']
            url = file_info['url']
            destination = os.path.join(year_dir, filename)
            
            print(f"\n{'='*70}")
            print(f"📥 Downloading: {filename} (~{file_info['size_gb']} GB)")
            print(f"{'='*70}")
            
            success = self.download_file(url, destination)
            
            if not success:
                print(f"\n❌ Failed to download {filename}")
                print("Try running the script again - it will resume from where it stopped")
                return False
            
            # Verify zip files
            if filename.endswith('.zip'):
                if not self.verify_zip(destination):
                    print(f"\n❌ {filename} is corrupted!")
                    print("Deleting and will retry on next run...")
                    os.remove(destination)
                    return False
        
        # Extract images
        zip_path = os.path.join(year_dir, files['images']['filename'])
        extract_dir = os.path.join(year_dir, 'ISIC_2019_Training_Input')
        
        if os.path.exists(extract_dir):
            print(f"\n✓ Images already extracted to {extract_dir}")
        else:
            print(f"\n{'='*70}")
            print("📦 Extracting images (this will take 5-10 minutes)...")
            print(f"{'='*70}")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    members = zip_ref.namelist()
                    for member in tqdm(members, desc="Extracting", unit="file"):
                        zip_ref.extract(member, year_dir)
                
                print("\n✓ Extraction complete!")
                
            except Exception as e:
                print(f"\n❌ Extraction failed: {e}")
                return False
        
        # Summary
        print(f"\n{'='*70}")
        print("🎉 ISIC 2019 Dataset Ready!")
        print(f"{'='*70}")
        print(f"\n📂 Location: {os.path.abspath(year_dir)}")
        print(f"\n📊 Dataset contains:")
        print(f"   • 25,331 dermoscopic images")
        print(f"   • 8 disease classes:")
        print(f"     - Melanoma (MEL)")
        print(f"     - Melanocytic nevus (NV)")
        print(f"     - Basal cell carcinoma (BCC)")
        print(f"     - Actinic keratosis (AK)")
        print(f"     - Benign keratosis (BKL)")
        print(f"     - Dermatofibroma (DF)")
        print(f"     - Vascular lesion (VASC)")
        print(f"     - Squamous cell carcinoma (SCC)")
        print(f"\n✅ Ready for SafeDerm training!\n")
        
        return True


if __name__ == "__main__":
    try:
        downloader = ISICDownloader(output_dir='./isic_data')
        success = downloader.download_isic_2019()
        
        if not success:
            print("\n⚠️  Download incomplete. Run the script again to resume.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Run again to resume download.")
        exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        exit(1)
