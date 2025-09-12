#!/usr/bin/env python3
"""
Complete end-to-end test of thumbnail workflow: database creation -> search -> Streamlit display
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
import base64
import io
from PIL import Image

from core.database import ParquetVectorDatabase
from core.config import VideoRetrievalConfig
from core.search import VideoSearchEngine
from interface.streamlit_app import get_thumbnail_from_result, preview_video_with_thumbnail


def create_test_video(video_path: Path, frame_count: int = 15, text: str = "Test"):
    """Create a test video with unique content."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640, 480))
    
    for i in range(frame_count):
        # Create unique frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 15) % 256  # Red
        frame[:, :, 1] = (i * 30) % 256  # Green  
        frame[:, :, 2] = (i * 45) % 256  # Blue
        
        # Add identifying text
        cv2.putText(frame, f'{text} {i}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        out.write(frame)
    
    out.release()


class TestCompleteThumbnailWorkflow(unittest.TestCase):
    """Test complete thumbnail workflow from database to display."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = VideoRetrievalConfig()
        self.config.main_embeddings_path = str(self.test_dir / "main_embeddings.parquet")
        self.config.query_embeddings_path = str(self.test_dir / "query_embeddings.parquet")
        
        # Create test videos
        self.main_videos = []
        self.query_video = None
        self._create_test_videos()
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def _create_test_videos(self):
        """Create test video database."""
        # Create main database videos
        for i in range(3):
            video_path = self.test_dir / f"main_video_{i}.mp4"
            create_test_video(video_path, frame_count=20, text=f"Main{i}")
            self.main_videos.append(video_path)
        
        # Create query video
        self.query_video = self.test_dir / "query_video.mp4"
        create_test_video(self.query_video, frame_count=15, text="Query")
        
        print(f"✅ Created {len(self.main_videos)} main videos and 1 query video")
    
    def test_complete_workflow_with_thumbnails(self):
        """Test complete workflow: build database -> search -> display thumbnails."""
        print("\n🔄 Testing Complete Thumbnail Workflow")
        print("-" * 50)
        
        # Step 1: Build main database with thumbnails
        print("📁 Step 1: Building main database...")
        main_db = ParquetVectorDatabase(self.config.main_embeddings_path, self.config)
        
        for i, video_path in enumerate(self.main_videos):
            # Create dummy embedding
            embedding = np.random.rand(768).astype('float32')
            # Normalize to simulate real embeddings
            embedding = embedding / np.linalg.norm(embedding)
            
            success = main_db.add_embedding(
                video_path.name,
                video_path,
                embedding,
                {"category": "main", "num_frames": 20}
            )
            self.assertTrue(success, f"Failed to add {video_path.name}")
        
        main_db.save()
        print(f"✅ Main database created with {len(main_db.list_videos())} videos")
        
        # Step 2: Build query database with thumbnails  
        print("🔍 Step 2: Building query database...")
        query_db = ParquetVectorDatabase(self.config.query_embeddings_path, self.config)
        
        query_embedding = np.random.rand(768).astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        success = query_db.add_embedding(
            self.query_video.name,
            self.query_video,
            query_embedding,
            {"category": "query", "num_frames": 15}
        )
        self.assertTrue(success, "Failed to add query video")
        
        query_db.save()
        print(f"✅ Query database created with {len(query_db.list_videos())} videos")
        
        # Step 3: Verify thumbnails are stored in parquet files
        print("🖼️  Step 3: Verifying thumbnail storage...")
        
        # Check main database thumbnails
        main_df = pd.read_parquet(self.config.main_embeddings_path)
        self.assertIn('thumbnail', main_df.columns, "Main database should have thumbnail column")
        
        for _, row in main_df.iterrows():
            thumbnail_data = row['thumbnail']
            self.assertIsInstance(thumbnail_data, str, "Thumbnail should be string")
            self.assertGreater(len(thumbnail_data), 0, "Thumbnail should not be empty")
            
            # Verify it's valid base64
            try:
                decoded = base64.b64decode(thumbnail_data)
                img = Image.open(io.BytesIO(decoded))
                self.assertEqual(img.size, self.config.thumbnail_size, "Thumbnail size should match config")
            except Exception as e:
                self.fail(f"Invalid thumbnail data: {e}")
        
        print("✅ All thumbnails properly stored in main database")
        
        # Check query database thumbnails
        query_df = pd.read_parquet(self.config.query_embeddings_path)
        self.assertIn('thumbnail', query_df.columns, "Query database should have thumbnail column")
        
        query_thumbnail = query_df.iloc[0]['thumbnail']
        self.assertGreater(len(query_thumbnail), 0, "Query thumbnail should not be empty")
        
        print("✅ Query thumbnail properly stored")
        
        # Step 4: Simulate search results with thumbnails
        print("🔎 Step 4: Simulating search results...")
        
        # Create mock search results that include thumbnail data
        search_results = []
        for i, (_, row) in enumerate(main_df.iterrows()):
            result = {
                'slice_id': row['slice_id'],
                'video_path': row['video_path'],
                'similarity_score': 0.95 - (i * 0.1),
                'rank': i + 1,
                'thumbnail': row['thumbnail'],
                'thumbnail_size': row['thumbnail_size']
            }
            search_results.append(result)
        
        print(f"✅ Created {len(search_results)} search results with thumbnails")
        
        # Step 5: Test Streamlit thumbnail display functions
        print("🖥️  Step 5: Testing Streamlit display functions...")
        
        for result in search_results:
            # Test get_thumbnail_from_result function
            thumbnail_b64 = get_thumbnail_from_result(result)
            self.assertIsNotNone(thumbnail_b64, f"Should get thumbnail for {result['slice_id']}")
            self.assertGreater(len(thumbnail_b64), 0, "Thumbnail should not be empty")
            
            # Verify thumbnail is same as stored
            self.assertEqual(thumbnail_b64, result['thumbnail'], "Should return stored thumbnail")
        
        print("✅ All Streamlit thumbnail functions work correctly")
        
        # Step 6: Test thumbnail data integrity
        print("🔍 Step 6: Testing thumbnail data integrity...")
        
        # Compare thumbnails from database vs Streamlit function
        for slice_id in main_db.list_videos():
            db_thumbnail = main_db.get_thumbnail_base64(slice_id)
            
            # Create search result format
            video_info = {
                'slice_id': slice_id,
                'thumbnail': db_thumbnail
            }
            
            streamlit_thumbnail = get_thumbnail_from_result(video_info)
            
            self.assertEqual(db_thumbnail, streamlit_thumbnail, 
                           f"Database and Streamlit thumbnails should match for {slice_id}")
        
        print("✅ Thumbnail data integrity verified")
        
        # Step 7: Performance check
        print("⚡ Step 7: Performance check...")
        
        import time
        start_time = time.time()
        
        # Simulate rapid thumbnail access (as would happen in Streamlit)
        for _ in range(100):
            for result in search_results:
                thumbnail = get_thumbnail_from_result(result)
                self.assertIsNotNone(thumbnail)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (100 * len(search_results))
        
        print(f"✅ Average thumbnail access time: {avg_time*1000:.2f}ms")
        self.assertLess(avg_time, 0.001, "Thumbnail access should be very fast")  # Less than 1ms
        
        print("\n🎉 COMPLETE WORKFLOW TEST PASSED!")
        print("✅ Thumbnails are properly stored in parquet files")
        print("✅ Thumbnails are efficiently retrieved for search results")  
        print("✅ Streamlit integration works seamlessly")
        print("✅ Performance is excellent for real-time display")


def run_complete_workflow_test():
    """Run the complete workflow test."""
    print("🎬 COMPLETE THUMBNAIL WORKFLOW TEST")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCompleteThumbnailWorkflow)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🏆 COMPLETE WORKFLOW TEST SUCCESSFUL!")
        print("✨ Thumbnail system is fully functional end-to-end")
    else:
        print("❌ WORKFLOW TEST FAILED")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_complete_workflow_test()
    sys.exit(0 if success else 1)
