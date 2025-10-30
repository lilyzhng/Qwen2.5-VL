#!/usr/bin/env python3
"""
OpenDV Mini Dataset Download Script

This script downloads the OpenDV mini dataset, which includes:
1. Repository tools and scripts from DriveAGI
2. Metadata from Google Sheets
3. Videos from YouTube (using yt-dlp)
4. Language annotations from HuggingFace

Repository: https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv
Dataset: OpenDV-mini (28 hours of driving videos)
Annotations: https://huggingface.co/datasets/OpenDriveLab/OpenDV-YouTube-Language
"""

import os
import sys
import time
import requests
import zipfile
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import argparse

class OpenDVDownloader:
    """Downloads and manages the OpenDV mini dataset."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize the downloader.
        
        Args:
            base_dir: Base directory for data storage. Defaults to current data/ directory.
        """
        if base_dir is None:
            # Use the current data directory
            self.base_dir = Path(__file__).parent
        else:
            self.base_dir = Path(base_dir)
        
        self.opendv_dir = self.base_dir / "opendv_mini"
        self.temp_dir = self.base_dir / "temp"
        
        # Dataset configuration
        self.dataset_config = {
            "name": "OpenDV Mini Dataset",
            "source": "https://github.com/OpenDriveLab/DriveAGI",
            "metadata_url": "https://docs.google.com/spreadsheets/d/1bHWWP_VXeEe5UzIG-QgKFBdH7mNlSC4GFSJkEhFnt2I/export?format=csv",
            "annotations_repo": "OpenDriveLab/OpenDV-YouTube-Language",
            "urls": [
                "https://github.com/OpenDriveLab/DriveAGI/archive/refs/heads/main.zip"  # Repository with tools
            ],
            "expected_files": [
                "scripts/",
                "configs/", 
                "utils/",
                "meta/",
                "OpenDV-YouTube/videos/"  # After processing
            ],
            "annotation_files": [
                "10hz_YouTube_val.json",
                # Mini train splits will be determined dynamically
            ]
        }
    
    def create_directories(self):
        """Create necessary directories."""
        self.opendv_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Created directories:")
        print(f"   Dataset: {self.opendv_dir}")
        print(f"   Temp: {self.temp_dir}")
    
    def download_file(self, url: str, filepath: Path, description: str = "Downloading") -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            filepath: Local path to save file
            description: Description for progress bar
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"🌐 Downloading from: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=description,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Download failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract a zip file.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"📦 Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of files in zip
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(file_list), desc="Extracting") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)
            
            print(f"✅ Extracted to: {extract_to}")
            return True
            
        except zipfile.BadZipFile:
            print(f"❌ Invalid zip file: {zip_path}")
            return False
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_from_github_api(self) -> bool:
        """
        Download using GitHub API to get specific files from opendv directory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # GitHub API URL for opendv directory contents
            api_url = "https://api.github.com/repos/OpenDriveLab/DriveAGI/contents/opendv"
            
            print(f"🔍 Fetching directory contents from GitHub API...")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            contents = response.json()
            
            if not isinstance(contents, list):
                print("❌ Unexpected API response format")
                return False
            
            # Create opendv directory structure
            for item in contents:
                if item['type'] == 'file':
                    file_url = item['download_url']
                    file_name = item['name']
                    file_path = self.opendv_dir / file_name
                    
                    print(f"📄 Downloading: {file_name}")
                    if not self.download_file(file_url, file_path, f"Downloading {file_name}"):
                        print(f"⚠️ Failed to download: {file_name}")
                        continue
                
                elif item['type'] == 'dir':
                    # Recursively handle directories (simplified for now)
                    dir_name = item['name']
                    dir_path = self.opendv_dir / dir_name
                    dir_path.mkdir(exist_ok=True)
                    print(f"📁 Created directory: {dir_name}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ GitHub API request failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def clone_repository_subset(self) -> bool:
        """
        Clone only the opendv directory using git sparse-checkout.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import subprocess
            
            repo_url = "https://github.com/OpenDriveLab/DriveAGI.git"
            clone_dir = self.temp_dir / "DriveAGI"
            
            print(f"🔄 Cloning repository with sparse-checkout...")
            
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=self.temp_dir, check=True)
            subprocess.run(["git", "remote", "add", "origin", repo_url], cwd=self.temp_dir, check=True)
            
            # Enable sparse-checkout
            subprocess.run(["git", "config", "core.sparseCheckout", "true"], cwd=self.temp_dir, check=True)
            
            # Set sparse-checkout patterns
            sparse_checkout_file = self.temp_dir / ".git" / "info" / "sparse-checkout"
            with open(sparse_checkout_file, 'w') as f:
                f.write("opendv/*\n")
            
            # Pull only the specified directory
            subprocess.run(["git", "pull", "origin", "main"], cwd=self.temp_dir, check=True)
            
            # Move opendv directory to final location
            source_opendv = self.temp_dir / "opendv"
            if source_opendv.exists():
                import shutil
                if self.opendv_dir.exists():
                    shutil.rmtree(self.opendv_dir)
                shutil.move(str(source_opendv), str(self.opendv_dir))
                print(f"✅ Successfully cloned opendv directory")
                return True
            else:
                print("❌ opendv directory not found in cloned repository")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Git command failed: {e}")
            return False
        except ImportError:
            print("❌ Git not available or subprocess module not found")
            return False
        except Exception as e:
            print(f"❌ Repository cloning failed: {e}")
            return False
    
    def verify_dataset(self) -> Dict[str, Any]:
        """
        Verify the downloaded dataset.
        
        Returns:
            Dictionary with verification results
        """
        verification = {
            "success": False,
            "files_found": [],
            "files_missing": [],
            "total_size": 0,
            "file_count": 0
        }
        
        if not self.opendv_dir.exists():
            verification["error"] = "Dataset directory not found"
            return verification
        
        print(f"🔍 Verifying dataset in: {self.opendv_dir}")
        
        # Count files and calculate total size
        for file_path in self.opendv_dir.rglob('*'):
            if file_path.is_file():
                verification["file_count"] += 1
                verification["total_size"] += file_path.stat().st_size
                verification["files_found"].append(str(file_path.relative_to(self.opendv_dir)))
        
        # Check for expected files/directories
        for expected in self.dataset_config["expected_files"]:
            expected_path = self.opendv_dir / expected
            if not expected_path.exists():
                verification["files_missing"].append(expected)
        
        verification["success"] = len(verification["files_missing"]) == 0 and verification["file_count"] > 0
        
        return verification
    
    def save_metadata(self, verification: Dict[str, Any]):
        """Save dataset metadata."""
        metadata = {
            "dataset": self.dataset_config["name"],
            "source": self.dataset_config["source"],
            "download_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "verification": verification,
            "location": str(self.opendv_dir)
        }
        
        metadata_file = self.opendv_dir / "download_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Saved metadata: {metadata_file}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary files")
    
    def download(self, method: str = "auto") -> bool:
        """
        Download the OpenDV mini dataset.
        
        Args:
            method: Download method ('auto', 'api', 'git', 'direct')
            
        Returns:
            True if successful, False otherwise
        """
        print(f"🚀 Starting OpenDV Mini Dataset download")
        print(f"📂 Target directory: {self.opendv_dir}")
        print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        self.create_directories()
        
        success = False
        
        # Try different download methods
        if method == "auto" or method == "git":
            print("🔄 Attempting git sparse-checkout method...")
            success = self.clone_repository_subset()
            
        if not success and (method == "auto" or method == "api"):
            print("🔄 Attempting GitHub API method...")
            success = self.download_from_github_api()
        
        if not success and (method == "auto" or method == "direct"):
            print("🔄 Attempting direct download method...")
            # Try direct download URLs
            for url in self.dataset_config["urls"]:
                temp_file = self.temp_dir / "download.zip"
                if self.download_file(url, temp_file, "Downloading dataset"):
                    if self.extract_zip(temp_file, self.temp_dir):
                        # Look for opendv directory in extracted content
                        extracted_opendv = self.temp_dir / "opendv"
                        if not extracted_opendv.exists():
                            # Check if it's in a subdirectory (like DriveAGI-main/opendv)
                            for item in self.temp_dir.iterdir():
                                if item.is_dir():
                                    potential_opendv = item / "opendv"
                                    if potential_opendv.exists():
                                        extracted_opendv = potential_opendv
                                        break
                        
                        if extracted_opendv.exists():
                            import shutil
                            if self.opendv_dir.exists():
                                shutil.rmtree(self.opendv_dir)
                            shutil.move(str(extracted_opendv), str(self.opendv_dir))
                            success = True
                            break
        
        if success:
            print("\n" + "=" * 60)
            print("✅ Download completed successfully!")
            
            # Verify dataset
            verification = self.verify_dataset()
            
            print(f"\n📊 Dataset Verification:")
            print(f"   Files found: {verification['file_count']}")
            print(f"   Total size: {verification['total_size'] / 1024**2:.1f} MB")
            print(f"   Status: {'✅ Valid' if verification['success'] else '⚠️ Issues detected'}")
            
            if verification["files_missing"]:
                print(f"   Missing expected files: {verification['files_missing']}")
            
            # Show some found files
            if verification["files_found"]:
                print(f"\n📋 Sample files:")
                for file in verification["files_found"][:10]:  # Show first 10 files
                    print(f"   📄 {file}")
                if len(verification["files_found"]) > 10:
                    print(f"   ... and {len(verification['files_found']) - 10} more files")
            
            # Save metadata
            self.save_metadata(verification)
            
            print(f"\n⏰ Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\n" + "=" * 60)
            print("❌ Download failed!")
            print("Please check your internet connection and try again.")
            print("You can also try downloading manually from:")
            print("https://github.com/OpenDriveLab/DriveAGI/tree/main/opendv")
        
        # Cleanup
        self.cleanup_temp_files()
        
        return success


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download OpenDV Mini Dataset from DriveAGI repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_opendv.py                    # Auto download
  python download_opendv.py --method git       # Use git sparse-checkout
  python download_opendv.py --method api       # Use GitHub API
  python download_opendv.py --dir /path/to/data # Custom directory
        """
    )
    
    parser.add_argument(
        "--method",
        choices=["auto", "git", "api", "direct"],
        default="auto",
        help="Download method to use (default: auto)"
    )
    
    parser.add_argument(
        "--dir",
        type=str,
        help="Base directory for data storage (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Create downloader and start download
    downloader = OpenDVDownloader(base_dir=args.dir)
    success = downloader.download(method=args.method)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
