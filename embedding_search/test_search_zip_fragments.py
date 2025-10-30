#!/usr/bin/env python3
"""
Test script to verify search engine handles zip fragments correctly.
"""

import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

# Add the core module to path
sys.path.append(str(Path(__file__).parent))

from core.search import _path_exists, _is_zip_path, _parse_zip_path

def create_test_data():
    """Create test data with zip fragments."""
    print("🔧 Creating test data...")
    
    # Create test directory and images
    test_dir = Path("test_search_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create subfolders
    (test_dir / "camera_front").mkdir(exist_ok=True)
    (test_dir / "camera_rear").mkdir(exist_ok=True)
    
    # Create test images
    def create_test_image(path, color, text):
        img = np.full((224, 224, 3), color, dtype=np.uint8)
        cv2.putText(img, text, (50, 112), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(path), img)
    
    # Create images in subfolders
    for i in range(3):
        create_test_image(test_dir / "camera_front" / f"frame_{i:03d}.jpg", (0, 100, 200), f"Front {i}")
        create_test_image(test_dir / "camera_rear" / f"frame_{i:03d}.jpg", (200, 100, 0), f"Rear {i}")
    
    # Create zip file
    import subprocess
    subprocess.run(["zip", "-r", str(test_dir / "sensor_data.zip"), "camera_front/", "camera_rear/"], 
                   cwd=test_dir, capture_output=True)
    
    # Create test parquet file with zip fragments
    test_data = {
        'slice_id': ['video_001', 'video_002', 'video_003'],
        'sensor_frame_zip': [
            str(test_dir / "sensor_data.zip"),
            str(test_dir / "sensor_data.zip#camera_front"),
            str(test_dir / "sensor_data.zip#camera_rear")
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_parquet(test_dir / "test_input.parquet", index=False)
    
    print(f"✅ Created test data in {test_dir}")
    return test_dir

def test_path_exists():
    """Test the _path_exists function."""
    print("\n🧪 Testing _path_exists function:")
    
    test_dir = Path("test_search_data")
    
    test_cases = [
        (str(test_dir / "sensor_data.zip"), True, "Regular zip file"),
        (str(test_dir / "sensor_data.zip#camera_front"), True, "Zip with fragment (camera_front)"),
        (str(test_dir / "sensor_data.zip#camera_rear"), True, "Zip with fragment (camera_rear)"),
        (str(test_dir / "sensor_data.zip#nonexistent"), True, "Zip with nonexistent fragment (still valid zip)"),
        (str(test_dir / "nonexistent.zip"), False, "Nonexistent zip file"),
        (str(test_dir / "nonexistent.zip#camera_front"), False, "Nonexistent zip with fragment"),
    ]
    
    for path, expected, description in test_cases:
        result = _path_exists(path)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {description}: {result} (expected: {expected})")
        if result != expected:
            print(f"      Path: {path}")

def test_parquet_loading():
    """Test loading video files from parquet with zip fragments."""
    print("\n🧪 Testing parquet loading with zip fragments:")
    
    try:
        from core.search import VideoSearchEngine
        from core.config import VideoRetrievalConfig
        
        # Create minimal config
        config = VideoRetrievalConfig()
        config.input_path = "test_search_data/test_input.parquet"
        
        # Create search engine (without full initialization)
        search_engine = VideoSearchEngine(config)
        
        # Test loading video files
        video_files = search_engine._get_video_files_from_parquet()
        
        print(f"✅ Successfully loaded {len(video_files)} video files")
        for i, file_path in enumerate(video_files):
            print(f"   {i+1}. {file_path}")
            
        # Verify all 3 files were loaded (including zip fragments)
        if len(video_files) == 3:
            print("✅ All zip fragment paths were correctly processed")
        else:
            print(f"❌ Expected 3 files, got {len(video_files)}")
            
    except Exception as e:
        print(f"❌ Error testing parquet loading: {e}")

def test_zip_fragment_parsing():
    """Test zip fragment parsing."""
    print("\n🧪 Testing zip fragment parsing:")
    
    test_cases = [
        ("test.zip", True, "test.zip", None),
        ("test.zip#camera_front", True, "test.zip", "camera_front"),
        ("path/to/data.zip#sensor/camera", True, "path/to/data.zip", "sensor/camera"),
        ("video.mp4", False, "video.mp4", None),
    ]
    
    for path, expected_is_zip, expected_zip, expected_subfolder in test_cases:
        is_zip = _is_zip_path(path)
        zip_path, subfolder = _parse_zip_path(path)
        
        zip_match = str(zip_path) == expected_zip
        subfolder_match = subfolder == expected_subfolder
        is_zip_match = is_zip == expected_is_zip
        
        all_match = zip_match and subfolder_match and is_zip_match
        status = "✅" if all_match else "❌"
        
        print(f"   {status} {path}")
        if not all_match:
            print(f"      Expected: is_zip={expected_is_zip}, zip={expected_zip}, subfolder={expected_subfolder}")
            print(f"      Got:      is_zip={is_zip}, zip={zip_path}, subfolder={subfolder}")

def cleanup():
    """Clean up test data."""
    print("\n🧹 Cleaning up test data...")
    import shutil
    test_dir = Path("test_search_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("✅ Cleanup complete")

def main():
    """Run all tests."""
    print("🧪 Testing Search Engine Zip Fragment Support")
    print("=" * 60)
    
    try:
        # Create test data
        test_dir = create_test_data()
        
        # Run tests
        test_zip_fragment_parsing()
        test_path_exists()
        test_parquet_loading()
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    main()
