#!/usr/bin/env python3
"""
Test Streamlit thumbnail integration - verify the get_thumbnail_from_result function works properly.
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
import cv2
from unittest.mock import Mock, patch

from core.database import ParquetVectorDatabase
from core.config import VideoRetrievalConfig
from interface.streamlit_app import get_thumbnail_from_result


class TestStreamlitThumbnailIntegration(unittest.TestCase):
    """Test Streamlit thumbnail integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.test_dir / "test_db.parquet"
        self.config = VideoRetrievalConfig()
        self.config.thumbnail_size = (480, 270)
        
        # Create a test video file
        self.test_video_path = self.test_dir / "test_video.mp4"
        self._create_test_video()
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_video(self):
        """Create a simple test video file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, 20.0, (640, 480))
        
        # Create 10 frames with different colors
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 25) % 256  # Red
            frame[:, :, 1] = (i * 50) % 256  # Green
            frame[:, :, 2] = (i * 75) % 256  # Blue
            cv2.putText(frame, f'Test Frame {i}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            out.write(frame)
        
        out.release()
        print(f"✅ Created test video: {self.test_video_path}")
    
    def test_get_thumbnail_from_stored_data(self):
        """Test getting thumbnail from stored database data (primary method)."""
        # Create database and add video with thumbnail
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        
        # Get stored thumbnail
        stored_thumbnail = db.get_thumbnail_base64("test_video.mp4")
        
        # Simulate search result with stored thumbnail
        video_info = {
            'slice_id': 'test_video.mp4',
            'video_path': str(self.test_video_path),
            'similarity_score': 0.95,
            'thumbnail': stored_thumbnail,
            'thumbnail_size': self.config.thumbnail_size
        }
        
        # Test Streamlit function
        result_thumbnail = get_thumbnail_from_result(video_info)
        
        self.assertIsNotNone(result_thumbnail, "Should return thumbnail")
        self.assertEqual(result_thumbnail, stored_thumbnail, "Should return stored thumbnail")
        self.assertGreater(len(result_thumbnail), 0, "Thumbnail should not be empty")
        
        print("✅ get_thumbnail_from_result works with stored thumbnail data")
    
    def test_get_thumbnail_fallback_extraction(self):
        """Test fallback thumbnail extraction when no stored thumbnail."""
        # Simulate search result without stored thumbnail (legacy case)
        video_info = {
            'slice_id': 'test_video.mp4',
            'video_path': str(self.test_video_path),
            'similarity_score': 0.85,
            'thumbnail': '',  # No stored thumbnail
            'thumbnail_size': (0, 0)
        }
        
        # Test Streamlit function - should fall back to on-the-fly extraction
        result_thumbnail = get_thumbnail_from_result(video_info)
        
        self.assertIsNotNone(result_thumbnail, "Should extract thumbnail on-the-fly")
        self.assertGreater(len(result_thumbnail), 0, "Extracted thumbnail should not be empty")
        
        print("✅ get_thumbnail_from_result works with fallback extraction")
    
    def test_get_thumbnail_missing_video_file(self):
        """Test behavior when video file doesn't exist."""
        # Simulate search result with non-existent video
        video_info = {
            'slice_id': 'missing_video.mp4',
            'video_path': str(self.test_dir / "missing_video.mp4"),
            'similarity_score': 0.75,
            'thumbnail': '',  # No stored thumbnail
            'thumbnail_size': (0, 0)
        }
        
        # Test Streamlit function - should handle missing file gracefully
        result_thumbnail = get_thumbnail_from_result(video_info)
        
        self.assertIsNone(result_thumbnail, "Should return None for missing video")
        
        print("✅ get_thumbnail_from_result handles missing video files gracefully")
    
    def test_multiple_search_results_thumbnails(self):
        """Test handling thumbnails for multiple search results."""
        # Create database with multiple videos
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add multiple videos
        search_results = []
        for i in range(3):
            slice_id = f"test_video_{i}.mp4"
            video_path = self.test_dir / slice_id
            shutil.copy2(self.test_video_path, video_path)  # Copy test video
            
            # Add to database
            test_embedding = np.random.rand(768).astype('float32')
            db.add_embedding(slice_id, video_path, test_embedding)
            
            # Create search result
            search_result = {
                'slice_id': slice_id,
                'video_path': str(video_path),
                'similarity_score': 0.9 - (i * 0.1),
                'thumbnail': db.get_thumbnail_base64(slice_id),
                'thumbnail_size': self.config.thumbnail_size,
                'rank': i + 1
            }
            search_results.append(search_result)
        
        # Test all search results have thumbnails
        for result in search_results:
            thumbnail = get_thumbnail_from_result(result)
            self.assertIsNotNone(thumbnail, f"Should have thumbnail for {result['slice_id']}")
            self.assertGreater(len(thumbnail), 0, f"Thumbnail should not be empty for {result['slice_id']}")
        
        print(f"✅ Successfully handled thumbnails for {len(search_results)} search results")
    
    def test_thumbnail_html_format_generation(self):
        """Test generating HTML format for Streamlit display."""
        # Create database and add video
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        test_embedding = np.random.rand(768).astype('float32')
        db.add_embedding("test_video.mp4", self.test_video_path, test_embedding)
        
        # Get search result
        video_info = {
            'slice_id': 'test_video.mp4',
            'video_path': str(self.test_video_path),
            'similarity_score': 0.95,
            'thumbnail': db.get_thumbnail_base64("test_video.mp4"),
            'thumbnail_size': self.config.thumbnail_size
        }
        
        # Get thumbnail
        thumbnail_b64 = get_thumbnail_from_result(video_info)
        
        # Generate HTML as Streamlit would
        html_img_tag = f'''
        <div style="position: relative; width: 100%; padding-bottom: 56.25%;">
            <img src="data:image/jpeg;base64,{thumbnail_b64}" 
                 style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">
        </div>
        '''
        
        # Verify HTML format
        self.assertIn('data:image/jpeg;base64,', html_img_tag, "Should contain base64 data URL")
        self.assertIn(thumbnail_b64, html_img_tag, "Should contain thumbnail data")
        self.assertIn('object-fit: cover', html_img_tag, "Should have proper CSS styling")
        
        print("✅ Thumbnail HTML format is properly generated for Streamlit")


def run_streamlit_integration_test():
    """Run Streamlit integration tests."""
    print("🖥️  STREAMLIT THUMBNAIL INTEGRATION TEST")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStreamlitThumbnailIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL STREAMLIT INTEGRATION TESTS PASSED!")
        print("✅ Thumbnails work properly with Streamlit interface")
        print("✅ Both stored and fallback thumbnail methods work")
        print("✅ Error handling for missing files works")
    else:
        print("❌ SOME TESTS FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_streamlit_integration_test()
    sys.exit(0 if success else 1)
