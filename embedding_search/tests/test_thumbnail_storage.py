#!/usr/bin/env python3
"""
Test suite for thumbnail storage in parquet files and Streamlit integration.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import base64
import io
from unittest.mock import Mock, patch
import cv2
from PIL import Image

from core.database import ParquetVectorDatabase
from core.config import VideoRetrievalConfig
from core.visualizer import VideoResultsVisualizer


class TestThumbnailStorage(unittest.TestCase):
    """Test thumbnail storage and retrieval in parquet files."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.test_dir / "test_db.parquet"
        self.config = VideoRetrievalConfig()
        self.config.thumbnail_size = (480, 270)  # 16:9 aspect ratio
        
        # Create a test video file (dummy MP4)
        self.test_video_path = self.test_dir / "test_video.mp4"
        self._create_test_video()
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_video(self):
        """Create a simple test video file."""
        # Create a simple test video with OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, 20.0, (640, 480))
        
        # Create 30 frames with different colors
        for i in range(30):
            # Create a frame with gradient colors
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 8) % 256  # Red channel
            frame[:, :, 1] = (i * 16) % 256  # Green channel  
            frame[:, :, 2] = (i * 32) % 256  # Blue channel
            
            # Add some text to make frames distinguishable
            cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        print(f"✅ Created test video: {self.test_video_path}")
    
    def test_database_schema_includes_thumbnail_columns(self):
        """Test that the database schema includes thumbnail columns."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Check that empty database has thumbnail columns
        expected_columns = ['thumbnail', 'thumbnail_size']
        for col in expected_columns:
            self.assertIn(col, db.df.columns, f"Column '{col}' missing from database schema")
        
        print("✅ Database schema includes thumbnail columns")
    
    def test_thumbnail_extraction_and_storage(self):
        """Test that thumbnails are extracted and stored properly."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Create test embedding
        test_embedding = np.random.rand(768).astype('float32')
        
        # Add embedding with video that should generate a thumbnail
        success = db.add_embedding(
            "test_video.mp4",
            self.test_video_path,
            test_embedding,
            {"category": "test", "num_frames": 8}
        )
        
        self.assertTrue(success, "Failed to add embedding")
        
        # Check that thumbnail data was stored
        row = db.df[db.df['slice_id'] == 'test_video.mp4'].iloc[0]
        
        # Verify thumbnail column is not empty
        thumbnail_b64 = row['thumbnail']
        self.assertIsInstance(thumbnail_b64, str, "Thumbnail should be stored as string")
        self.assertGreater(len(thumbnail_b64), 0, "Thumbnail data should not be empty")
        
        # Verify thumbnail size is stored
        thumbnail_size = row['thumbnail_size']
        self.assertIsInstance(thumbnail_size, (tuple, list), "Thumbnail size should be tuple/list")
        self.assertEqual(len(thumbnail_size), 2, "Thumbnail size should have width and height")
        self.assertEqual(tuple(thumbnail_size), self.config.thumbnail_size, 
                        f"Thumbnail size should match config: {self.config.thumbnail_size}")
        
        print(f"✅ Thumbnail stored: {len(thumbnail_b64)} chars, size: {thumbnail_size}")
    
    def test_thumbnail_base64_validity(self):
        """Test that stored thumbnail is valid base64 and can be decoded."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add test embedding
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        
        # Get thumbnail data
        thumbnail_b64 = db.get_thumbnail_base64("test_video.mp4")
        self.assertIsNotNone(thumbnail_b64, "Should retrieve thumbnail base64")
        self.assertGreater(len(thumbnail_b64), 0, "Thumbnail base64 should not be empty")
        
        # Test that it's valid base64
        try:
            thumbnail_bytes = base64.b64decode(thumbnail_b64)
            self.assertGreater(len(thumbnail_bytes), 0, "Decoded thumbnail should have data")
        except Exception as e:
            self.fail(f"Thumbnail base64 is invalid: {e}")
        
        # Test that it's a valid image
        try:
            thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
            width, height = thumbnail_pil.size
            self.assertEqual((width, height), self.config.thumbnail_size, 
                           "Decoded image size should match config")
            print(f"✅ Valid thumbnail image: {width}x{height}")
        except Exception as e:
            self.fail(f"Thumbnail is not a valid image: {e}")
    
    def test_thumbnail_numpy_array_retrieval(self):
        """Test retrieving thumbnail as numpy array."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add test embedding
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        
        # Get thumbnail as numpy array
        thumbnail_array = db.get_thumbnail("test_video.mp4")
        self.assertIsNotNone(thumbnail_array, "Should retrieve thumbnail array")
        self.assertIsInstance(thumbnail_array, np.ndarray, "Should return numpy array")
        
        # Check array properties
        expected_shape = (*self.config.thumbnail_size[::-1], 3)  # (height, width, channels)
        self.assertEqual(thumbnail_array.shape, expected_shape, 
                        f"Array shape should be {expected_shape}")
        self.assertEqual(thumbnail_array.dtype, np.uint8, "Array should be uint8")
        
        print(f"✅ Thumbnail array: {thumbnail_array.shape}, dtype: {thumbnail_array.dtype}")
    
    def test_parquet_file_persistence(self):
        """Test that thumbnails persist when saved to and loaded from parquet file."""
        # Create and populate database
        db1 = ParquetVectorDatabase(self.test_db_path, self.config)
        test_embedding = np.random.rand(768).astype('float32')
        
        db1.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        original_thumbnail = db1.get_thumbnail_base64("test_video.mp4")
        
        # Save to parquet
        db1.save()
        self.assertTrue(self.test_db_path.exists(), "Parquet file should be created")
        
        # Load in new instance
        db2 = ParquetVectorDatabase(self.test_db_path, self.config)
        loaded_thumbnail = db2.get_thumbnail_base64("test_video.mp4")
        
        # Compare thumbnails
        self.assertEqual(original_thumbnail, loaded_thumbnail, 
                        "Loaded thumbnail should match original")
        
        print("✅ Thumbnail data persists across save/load cycles")
    
    def test_streamlit_integration_format(self):
        """Test that thumbnail format is compatible with Streamlit display."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add test embedding
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        
        # Simulate getting search results (as would be returned by search engine)
        search_result = {
            'slice_id': 'test_video.mp4',
            'video_path': str(self.test_video_path),
            'similarity_score': 0.95,
            'thumbnail': db.get_thumbnail_base64("test_video.mp4"),
            'thumbnail_size': self.config.thumbnail_size
        }
        
        # Test the format that Streamlit expects
        thumbnail_b64 = search_result.get('thumbnail', '')
        self.assertGreater(len(thumbnail_b64), 0, "Search result should contain thumbnail")
        
        # Test HTML format for Streamlit
        html_img_tag = f'<img src="data:image/jpeg;base64,{thumbnail_b64}" style="width:100%">'
        self.assertIn('data:image/jpeg;base64,', html_img_tag, 
                     "Should create valid HTML img tag with base64 data")
        
        print("✅ Thumbnail format is compatible with Streamlit HTML display")
    
    def test_multiple_videos_thumbnails(self):
        """Test handling thumbnails for multiple videos."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Create multiple test videos and add them
        slice_ids = []
        for i in range(3):
            slice_id = f"test_video_{i}.mp4"
            video_path = self.test_dir / slice_id
            
            # Create a simple video file (copy the original for simplicity)
            shutil.copy2(self.test_video_path, video_path)
            
            # Add embedding
            test_embedding = np.random.rand(768).astype('float32')
            db.add_embedding(slice_id, video_path, test_embedding)
            slice_ids.append(slice_id)
        
        # Verify all thumbnails are stored
        for slice_id in slice_ids:
            thumbnail = db.get_thumbnail_base64(slice_id)
            self.assertIsNotNone(thumbnail, f"Should have thumbnail for {slice_id}")
            self.assertGreater(len(thumbnail), 0, f"Thumbnail should not be empty for {slice_id}")
        
        print(f"✅ Successfully stored thumbnails for {len(slice_ids)} videos")
    
    def test_parquet_file_structure(self):
        """Test the actual parquet file structure contains thumbnail data."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add test data
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        db.save()
        
        # Read parquet file directly with pandas
        df = pd.read_parquet(self.test_db_path)
        
        # Verify structure
        self.assertIn('thumbnail', df.columns, "Parquet should contain thumbnail column")
        self.assertIn('thumbnail_size', df.columns, "Parquet should contain thumbnail_size column")
        
        # Verify data
        row = df.iloc[0]
        self.assertIsInstance(row['thumbnail'], str, "Thumbnail should be string in parquet")
        self.assertGreater(len(row['thumbnail']), 0, "Thumbnail data should not be empty")
        
        print("✅ Parquet file structure contains thumbnail columns with data")


def run_comprehensive_test():
    """Run all thumbnail tests and provide summary."""
    print("🎬 COMPREHENSIVE THUMBNAIL STORAGE TEST")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestThumbnailStorage)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL THUMBNAIL TESTS PASSED!")
        print("✅ Thumbnails are properly stored in parquet files")
        print("✅ Thumbnails can be retrieved and displayed in Streamlit")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
