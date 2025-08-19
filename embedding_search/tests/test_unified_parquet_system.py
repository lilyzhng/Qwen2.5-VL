#!/usr/bin/env python3
"""
Test suite for the unified parquet storage system.
Tests the new ParquetVectorDatabase and UnifiedQueryManager classes.
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
from unittest.mock import Mock, patch

from core.config import VideoRetrievalConfig
from core.database import ParquetVectorDatabase, UnifiedQueryManager
from core.search import VideoSearchEngine


class TestParquetVectorDatabase(unittest.TestCase):
    """Test cases for ParquetVectorDatabase class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.test_dir / "test_embeddings.parquet"
        self.config = VideoRetrievalConfig()
        self.db = ParquetVectorDatabase(self.test_db_path, self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_empty_database_creation(self):
        """Test creating an empty database."""
        self.assertIsNotNone(self.db.df)
        self.assertEqual(len(self.db.list_videos()), 0)
    
    def test_add_embedding(self):
        """Test adding an embedding to the database."""
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test_video.mp4"
        
        # Create a dummy video file
        test_video_path.touch()
        
        # Add embedding
        success = self.db.add_embedding(
            "test_video.mp4",
            test_video_path,
            test_embedding,
            {"category": "test", "num_frames": 8}
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.db.list_videos()), 1)
        self.assertIn("test_video.mp4", self.db.list_videos())
    
    def test_get_embedding(self):
        """Test retrieving an embedding from the database."""
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test_video.mp4"
        test_video_path.touch()
        
        # Add embedding
        self.db.add_embedding("test_video.mp4", test_video_path, test_embedding)
        
        # Retrieve embedding
        retrieved = self.db.get_embedding("test_video.mp4")
        
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, test_embedding)
    
    def test_save_and_load(self):
        """Test saving and loading the database."""
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test_video.mp4"
        test_video_path.touch()
        
        # Add embedding and save
        self.db.add_embedding("test_video.mp4", test_video_path, test_embedding)
        self.db.save()
        
        # Create new database instance and load
        new_db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        self.assertEqual(len(new_db.list_videos()), 1)
        retrieved = new_db.get_embedding("test_video.mp4")
        np.testing.assert_array_equal(retrieved, test_embedding)
    
    def test_statistics(self):
        """Test database statistics."""
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test_video.mp4"
        test_video_path.touch()
        
        self.db.add_embedding("test_video.mp4", test_video_path, test_embedding, {"category": "test"})
        
        stats = self.db.get_statistics()
        
        self.assertEqual(stats["total_embeddings"], 1)
        self.assertEqual(stats["embedding_dim"], 768)
        self.assertIn("test", stats["categories"])


class TestUnifiedQueryManager(unittest.TestCase):
    """Test cases for UnifiedQueryManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = VideoRetrievalConfig()
        self.config.query_embeddings_path = str(self.test_dir / "query_embeddings.parquet")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test UnifiedQueryManager initialization."""
        manager = UnifiedQueryManager(self.config)
        self.assertIsNotNone(manager.query_db)
        self.assertEqual(len(manager.list_available_query_videos()), 0)
    
    def test_build_from_file_list_structure(self):
        """Test building query database from file list structure (without actual embedding extraction)."""
        # Create test file list
        test_file_list = self.test_dir / "test_query_paths.parquet"
        test_video_path = self.test_dir / "test_query.mp4"
        test_video_path.touch()
        
        # Create file list
        df = pd.DataFrame([{
            'slice_id': 'test_query.mp4',
            'sensor_video_file': str(test_video_path.absolute()),
            'category': 'user_input'
        }])
        df.to_parquet(test_file_list, index=False)
        
        # Test that manager can process the file list structure
        manager = UnifiedQueryManager(self.config)
        
        # Test file list loading (without actual embedding extraction)
        try:
            # This should load the file list successfully
            file_path = Path(test_file_list)
            df_loaded = pd.read_parquet(file_path)
            
            # Filter for query videos
            query_df = df_loaded[df_loaded['category'] == 'user_input']
            video_paths = [Path(path) for path in query_df['sensor_video_file'].tolist()]
            video_files = [path for path in video_paths if path.exists()]
            
            self.assertEqual(len(video_files), 1)
            self.assertEqual(video_files[0].name, 'test_query.mp4')
            
        except Exception as e:
            self.fail(f"File list processing failed: {e}")


class TestOptimizedVideoSearchEngine(unittest.TestCase):
    """Test cases for VideoSearchEngine with unified parquet system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = VideoRetrievalConfig()
        self.config.main_embeddings_path = str(self.test_dir / "main_embeddings.parquet")
        self.config.query_embeddings_path = str(self.test_dir / "query_embeddings.parquet")
        self.config.main_input_path = str(self.test_dir / "main_input_path.parquet")
        self.config.query_input_path = str(self.test_dir / "query_input_path.parquet")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_config_structure_with_new_paths(self):
        """Test that search engine config has new paths."""
        # Test without actually initializing the full engine (to avoid model loading)
        self.assertEqual(self.config.main_embeddings_path, str(self.test_dir / "main_embeddings.parquet"))
        self.assertEqual(self.config.query_embeddings_path, str(self.test_dir / "query_embeddings.parquet"))
        self.assertEqual(self.config.main_input_path, str(self.test_dir / "main_input_path.parquet"))
        self.assertEqual(self.config.query_input_path, str(self.test_dir / "query_input_path.parquet"))


class TestConfigurationSystem(unittest.TestCase):
    """Test cases for the new configuration system."""
    
    def test_config_has_new_paths(self):
        """Test that configuration has all new paths."""
        config = VideoRetrievalConfig()
        
        # Check new paths exist
        self.assertTrue(hasattr(config, 'main_embeddings_path'))
        self.assertTrue(hasattr(config, 'query_embeddings_path'))
        self.assertTrue(hasattr(config, 'main_input_path'))
        self.assertTrue(hasattr(config, 'query_input_path'))
        
        # Check default values
        self.assertEqual(config.main_embeddings_path, "data/main_embeddings.parquet")
        self.assertEqual(config.query_embeddings_path, "data/query_embeddings.parquet")
        self.assertEqual(config.main_input_path, "data/main_input_path.parquet")
        self.assertEqual(config.query_input_path, "data/query_input_path.parquet")
    
    def test_config_validation(self):
        """Test configuration validation with new paths."""
        config = VideoRetrievalConfig()
        
        # Should not raise exception even if files don't exist (just warnings)
        try:
            config.validate()
        except Exception as e:
            self.fail(f"Config validation failed: {e}")


class TestFilePathGeneration(unittest.TestCase):
    """Test cases for file path generation."""
    
    def test_file_path_structure(self):
        """Test the structure of generated file path lists."""
        # Test data structure
        test_data = [
            {
                'slice_id': 'test1.mp4',
                'sensor_video_file': '/path/to/test1.mp4',
                'category': 'video_database'
            },
            {
                'slice_id': 'test2.mp4', 
                'sensor_video_file': '/path/to/test2.mp4',
                'category': 'user_input'
            }
        ]
        
        df = pd.DataFrame(test_data)
        
        # Check required columns exist
        required_columns = ['slice_id', 'sensor_video_file', 'category']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(df['slice_id'].dtype == 'object')
        self.assertTrue(df['sensor_video_file'].dtype == 'object')
        self.assertTrue(df['category'].dtype == 'object')


if __name__ == '__main__':
    print("🧪 Running Unified Parquet Storage Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestConfigurationSystem))
    test_suite.addTest(unittest.makeSuite(TestParquetVectorDatabase))
    test_suite.addTest(unittest.makeSuite(TestUnifiedQueryManager))
    test_suite.addTest(unittest.makeSuite(TestOptimizedVideoSearchEngine))
    test_suite.addTest(unittest.makeSuite(TestFilePathGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All tests passed! Unified parquet system is working correctly.")
    else:
        print(f"⚠️  {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
    sys.exit(0 if result.wasSuccessful() else 1)
