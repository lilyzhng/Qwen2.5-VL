#!/usr/bin/env python3
"""
Test suite for core/database.py - ParquetVectorDatabase and UnifiedQueryManager.
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

from core.database import ParquetVectorDatabase, UnifiedQueryManager
from core.config import VideoRetrievalConfig


class TestParquetVectorDatabase(unittest.TestCase):
    """Test ParquetVectorDatabase class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.test_dir / "test_db.parquet"
        self.config = VideoRetrievalConfig()
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_empty_database_initialization(self):
        """Test creating a new empty database."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        self.assertIsNotNone(db.df)
        self.assertEqual(len(db.list_videos()), 0)
        self.assertFalse(self.test_db_path.exists())  # Not saved yet
    
    def test_add_and_retrieve_embedding(self):
        """Test adding and retrieving embeddings."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Create test data
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test.mp4"
        test_video_path.touch()  # Create dummy file
        
        # Add embedding
        success = db.add_embedding(
            "test.mp4",
            test_video_path,
            test_embedding,
            {"category": "test", "num_frames": 8}
        )
        
        self.assertTrue(success)
        
        # Retrieve embedding
        retrieved = db.get_embedding("test.mp4")
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, test_embedding)
    
    def test_save_and_load_persistence(self):
        """Test database persistence across sessions."""
        # Create and populate database
        db1 = ParquetVectorDatabase(self.test_db_path, self.config)
        test_embedding = np.random.rand(768).astype('float32')
        test_video_path = self.test_dir / "test.mp4"
        test_video_path.touch()
        
        db1.add_embedding("test.mp4", test_video_path, test_embedding)
        db1.save()
        
        # Load in new instance
        db2 = ParquetVectorDatabase(self.test_db_path, self.config)
        
        self.assertEqual(len(db2.list_videos()), 1)
        retrieved = db2.get_embedding("test.mp4")
        np.testing.assert_array_equal(retrieved, test_embedding)
    
    def test_multiple_embeddings(self):
        """Test handling multiple embeddings."""
        db = ParquetVectorDatabase(self.test_db_path, self.config)
        
        # Add multiple embeddings
        for i in range(3):
            embedding = np.random.rand(768).astype('float32')
            video_path = self.test_dir / f"test{i}.mp4"
            video_path.touch()
            
            db.add_embedding(f"test{i}.mp4", video_path, embedding)
        
        # Check all are stored
        videos = db.list_videos()
        self.assertEqual(len(videos), 3)
        
        # Check all can be retrieved
        for i in range(3):
            retrieved = db.get_embedding(f"test{i}.mp4")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.shape, (768,))


class TestUnifiedQueryManager(unittest.TestCase):
    """Test UnifiedQueryManager class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = VideoRetrievalConfig()
        self.config.query_embeddings_path = str(self.test_dir / "query_embeddings.parquet")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_manager_initialization(self):
        """Test UnifiedQueryManager initialization."""
        manager = UnifiedQueryManager(self.config)
        
        self.assertIsNotNone(manager.query_db)
        self.assertEqual(manager.config.query_embeddings_path, self.config.query_embeddings_path)
    
    def test_empty_query_list(self):
        """Test listing videos from empty database."""
        manager = UnifiedQueryManager(self.config)
        videos = manager.list_available_query_videos()
        
        self.assertEqual(len(videos), 0)
        self.assertIsInstance(videos, list)
    
    def test_statistics_empty_database(self):
        """Test getting statistics from empty database."""
        manager = UnifiedQueryManager(self.config)
        stats = manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_embeddings"], 0)


if __name__ == '__main__':
    print("🧪 Testing core/database.py")
    print("=" * 60)
    
    unittest.main(verbosity=2)
