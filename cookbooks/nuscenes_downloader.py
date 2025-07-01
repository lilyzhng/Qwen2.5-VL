#!/usr/bin/env python3
"""
nuScenes Mini Dataset Downloader

This script downloads the nuScenes mini dataset, which includes:
- Mini dataset (metadata and annotations)
- Camera images
- Radar data
- Lidar data
- Maps

Total size: ~4GB
"""

import os
import sys
import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import hashlib
import argparse

# nuScenes mini dataset URLs
NUSCENES_MINI_URLS = {
    "mini_meta": {
        "url": "https://www.nuscenes.org/data/v1.0-mini.tgz",
        "filename": "v1.0-mini.tgz",
        "size_mb": 32,
        "description": "Mini dataset metadata and annotations"
    },
    "mini_keyframes": {
        "url": "https://www.nuscenes.org/data/samples.tar",
        "filename": "samples.tar", 
        "size_mb": 1800,
        "description": "Camera images, radar and lidar keyframes"
    },
    "mini_sweeps": {
        "url": "https://www.nuscenes.org/data/sweeps.tar",
        "filename": "sweeps.tar",
        "size_mb": 2300,
        "description": "Camera, radar and lidar sweeps"
    },
    "maps": {
        "url": "https://www.nuscenes.org/data/nuScenes-map-expansion-v1.3.zip",
        "filename": "nuScenes-map-expansion-v1.3.zip",
        "size_mb": 60,
        "description": "Map expansion pack"
    }
}

class NuScenesDownloader:
    def __init__(self, download_dir="./nuscenes_mini", create_structure=True):
        self.download_dir = Path(download_dir)
        self.create_structure = create_structure
        
        # Create directory structure
        if self.create_structure:
            self.setup_directory_structure()
    
    def setup_directory_structure(self):
        """Create the expected nuScenes directory structure"""
        dirs_to_create = [
            self.download_dir,
            self.download_dir / "v1.0-mini",
            self.download_dir / "samples",
            self.download_dir / "sweeps", 
            self.download_dir / "maps"
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def download_file(self, url, filepath, description=""):
        """Download a file with progress bar"""
        print(f"\nDownloading: {description}")
        print(f"URL: {url}")
        print(f"Destination: {filepath}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Successfully downloaded: {filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Failed to download {filepath}: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error downloading {filepath}: {e}")
            return False
    
    def extract_archive(self, filepath, extract_to=None):
        """Extract tar or zip archive"""
        if extract_to is None:
            extract_to = self.download_dir
        
        print(f"\nExtracting: {filepath}")
        
        try:
            if filepath.suffix.lower() in ['.tgz', '.tar']:
                with tarfile.open(filepath, 'r:*') as tar:
                    tar.extractall(path=extract_to)
            elif filepath.suffix.lower() == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {filepath}")
                return False
            
            print(f"✓ Successfully extracted: {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to extract {filepath}: {e}")
            return False
    
    def verify_file_size(self, filepath, expected_size_mb):
        """Verify downloaded file size"""
        if not filepath.exists():
            return False
        
        actual_size_mb = filepath.stat().st_size / (1024 * 1024)
        expected_range = (expected_size_mb * 0.95, expected_size_mb * 1.05)  # 5% tolerance
        
        if expected_range[0] <= actual_size_mb <= expected_range[1]:
            print(f"✓ File size verification passed: {actual_size_mb:.1f}MB")
            return True
        else:
            print(f"✗ File size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
            return False
    
    def download_component(self, component_key, extract=True, keep_archive=False):
        """Download and optionally extract a dataset component"""
        if component_key not in NUSCENES_MINI_URLS:
            print(f"Unknown component: {component_key}")
            return False
        
        component = NUSCENES_MINI_URLS[component_key]
        filepath = self.download_dir / component["filename"]
        
        # Skip if file already exists and has correct size
        if filepath.exists() and self.verify_file_size(filepath, component["size_mb"]):
            print(f"File already exists and verified: {filepath}")
        else:
            # Download the file
            success = self.download_file(
                component["url"], 
                filepath, 
                component["description"]
            )
            
            if not success:
                return False
            
            # Verify file size
            if not self.verify_file_size(filepath, component["size_mb"]):
                print(f"Warning: File size verification failed for {filepath}")
        
        # Extract if requested
        if extract:
            success = self.extract_archive(filepath)
            if not success:
                return False
            
            # Remove archive if not keeping it
            if not keep_archive:
                try:
                    filepath.unlink()
                    print(f"Removed archive: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not remove archive {filepath}: {e}")
        
        return True
    
    def download_all(self, extract=True, keep_archives=False):
        """Download all components of the nuScenes mini dataset"""
        print("=" * 60)
        print("nuScenes Mini Dataset Downloader")
        print("=" * 60)
        print(f"Download directory: {self.download_dir.absolute()}")
        print(f"Expected total size: ~4GB")
        print("=" * 60)
        
        total_components = len(NUSCENES_MINI_URLS)
        successful_downloads = 0
        
        for i, (component_key, component) in enumerate(NUSCENES_MINI_URLS.items(), 1):
            print(f"\n[{i}/{total_components}] Processing: {component['description']}")
            print(f"Expected size: {component['size_mb']}MB")
            
            success = self.download_component(
                component_key, 
                extract=extract, 
                keep_archive=keep_archives
            )
            
            if success:
                successful_downloads += 1
            else:
                print(f"✗ Failed to process component: {component_key}")
        
        # Summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Successful downloads: {successful_downloads}/{total_components}")
        
        if successful_downloads == total_components:
            print("✓ All components downloaded successfully!")
            print(f"\nDataset ready at: {self.download_dir.absolute()}")
            self.print_usage_info()
        else:
            print(f"✗ {total_components - successful_downloads} components failed")
            return False
        
        return True
    
    def print_usage_info(self):
        """Print information about using the downloaded dataset"""
        print("\n" + "-" * 40)
        print("USAGE INFORMATION")
        print("-" * 40)
        print("To use the nuScenes mini dataset in Python:")
        print()
        print("1. Install nuscenes-devkit:")
        print("   pip install nuscenes-devkit")
        print()
        print("2. Load the dataset:")
        print("   from nuscenes.nuscenes import NuScenes")
        print(f"   nusc = NuScenes(version='v1.0-mini', dataroot='{self.download_dir.absolute()}', verbose=True)")
        print()
        print("3. Explore the data:")
        print("   print(f'Number of scenes: {len(nusc.scene)}')")
        print("   nusc.list_scenes()")

def main():
    parser = argparse.ArgumentParser(description='Download nuScenes mini dataset')
    parser.add_argument('--output-dir', '-o', default='./nuscenes_mini',
                       help='Output directory for dataset (default: ./nuscenes_mini)')
    parser.add_argument('--keep-archives', action='store_true',
                       help='Keep downloaded archive files after extraction')
    parser.add_argument('--no-extract', action='store_true',
                       help='Download only, do not extract archives')
    parser.add_argument('--component', choices=list(NUSCENES_MINI_URLS.keys()),
                       help='Download only specific component')
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = NuScenesDownloader(download_dir=args.output_dir)
    
    try:
        if args.component:
            # Download specific component
            success = downloader.download_component(
                args.component, 
                extract=not args.no_extract,
                keep_archive=args.keep_archives
            )
        else:
            # Download all components
            success = downloader.download_all(
                extract=not args.no_extract,
                keep_archives=args.keep_archives
            )
        
        if success:
            print("\n🎉 Download completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Download failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()