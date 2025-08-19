#!/usr/bin/env python3
"""
Test suite for core/search.py - VideoSearchEngine with unified parquet storage.
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

from core.search import VideoSearchEngine
from core.config import VideoRetrievalConfig


class TestVideoSearchEngineUnified(unittest.TestCase):
    """Test VideoSearchEngine with unified parquet storage."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = VideoRetrievalConfig()
        
        # Set test paths
        self.config.main_embeddings_path = str(self.test_dir / "main_embeddings.parquet")
        self.config.query_embeddings_path = str(self.test_dir / "query_embeddings.parquet")
        self.config.main_file_path = str(self.test_dir / "main_file_path.parquet")
        self.config.query_file_path = str(self.test_dir / "query_file_path.parquet")
        
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('core.search.CosmosVideoEmbedder')
    def test_initialization_with_unified_config(self, mock_embedder):
        """Test search engine initialization with unified parquet configuration."""
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        
        try:
            engine = VideoSearchEngine(config=self.config)
            
            # Check that engine uses new paths
            self.assertEqual(engine.config.main_embeddings_path, self.config.main_embeddings_path)
            self.assertEqual(engine.config.query_embeddings_path, self.config.query_embeddings_path)
            self.assertEqual(engine.config.main_file_path, self.config.main_file_path)
            self.assertEqual(engine.config.query_file_path, self.config.query_file_path)
            
        except Exception as e:
            self.fail(f"Search engine initialization failed: {e}")
    
    def test_path_resolution(self):
        """Test path resolution with new configuration."""
        engine = VideoSearchEngine.__new__(VideoSearchEngine)
        engine.config = self.config
        engine.project_root = self.test_dir.parent
        
        # Test relative path resolution
        resolved = engine._resolve_path("data/test.parquet")
        expected = engine.project_root / "data/test.parquet"
        self.assertEqual(resolved, expected)
        
        # Test absolute path
        abs_path = self.test_dir / "absolute.parquet"
        resolved_abs = engine._resolve_path(abs_path)
        self.assertEqual(resolved_abs, abs_path)
    
    def test_get_video_files_from_file_list(self):
        """Test loading video files from main_file_path.parquet."""
        # Create test file list
        test_videos = [
            {
                'slice_id': 'test1.mp4',
                'sensor_video_file': str(self.test_dir / 'test1.mp4'),
                'category': 'video_database'
            },
            {
                'slice_id': 'test2.mp4',
                'sensor_video_file': str(self.test_dir / 'test2.mp4'),
                'category': 'video_database'
            }
        ]
        
        # Create actual video files
        for video in test_videos:
            Path(video['sensor_video_file']).touch()
        
        # Save file list
        df = pd.DataFrame(test_videos)
        df.to_parquet(self.config.main_file_path, index=False)
        
        # Test loading
        engine = VideoSearchEngine.__new__(VideoSearchEngine)
        engine.config = self.config
        engine.project_root = self.test_dir.parent
        
        video_files = engine._get_video_files("dummy_directory")
        
        self.assertEqual(len(video_files), 2)
        self.assertTrue(all(isinstance(vf, Path) for vf in video_files))
        self.assertTrue(all(vf.exists() for vf in video_files))


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation with new unified paths."""
    
    def test_new_config_structure(self):
        """Test that new configuration structure is correct."""
        config = VideoRetrievalConfig()
        
        # Check all new attributes exist
        required_attrs = [
            'main_embeddings_path',
            'query_embeddings_path', 
            'main_file_path',
            'query_file_path'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")
            self.assertIsNotNone(getattr(config, attr), f"Attribute {attr} is None")
    
    def test_config_validation_without_legacy_paths(self):
        """Test config validation works without legacy directory paths."""
        config = VideoRetrievalConfig()
        
        # Should not raise exception
        try:
            config.validate()
        except Exception as e:
            self.fail(f"Config validation failed: {e}")


if __name__ == '__main__':
    print("🧪 Testing core/search.py - VideoSearchEngine")
    print("=" * 60)
    
    unittest.main(verbosity=2)
