#!/usr/bin/env python3
"""
Test script to verify query cache functionality without running full search.
"""

import os
# Fix OpenMP library conflict issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from core.query_cache import QueryDatabaseManager
    from core.config import VideoRetrievalConfig
    
    def test_query_cache():
        """Test query cache functionality."""
        
        # Initialize manager
        config = VideoRetrievalConfig()
        manager = QueryDatabaseManager(config)
        
        # Test filename
        test_filename = "car2cyclist_2.mp4"
        
        print("🔍 Testing Query Cache Functionality")
        print("=" * 50)
        
        # Check if video is cached
        is_cached = manager.query_cache.is_cached(test_filename)
        print(f"📁 Is '{test_filename}' cached? {is_cached}")
        
        if is_cached:
            # Get embedding
            embedding = manager.get_query_embedding(test_filename)
            if embedding is not None:
                print(f"✅ Successfully retrieved embedding: shape {embedding.shape}")
                print(f"📊 Embedding stats: min={embedding.min():.3f}, max={embedding.max():.3f}, norm={embedding.dot(embedding):.3f}")
            else:
                print("❌ Failed to retrieve embedding")
        else:
            print(f"⚠️  Video not cached. Run: python main.py build-query")
        
        # Get available videos
        available = manager.list_available_query_videos()
        print(f"\n📋 Available query videos ({len(available)}):")
        for i, filename in enumerate(available, 1):
            print(f"  {i}. {filename}")
        
        # Get cache statistics
        stats = manager.get_statistics()
        print(f"\n📊 Cache Statistics:")
        if 'cache' in stats:
            cache_stats = stats['cache']
            print(f"  Cached videos: {cache_stats.get('total_videos', 0)}")
            print(f"  Cache size: {cache_stats.get('cache_size_mb', 0):.2f} MB")
            print(f"  Total accesses: {cache_stats.get('total_accesses', 0)}")
        
        return is_cached and embedding is not None
    
    if __name__ == "__main__":
        success = test_query_cache()
        if success:
            print("\n✅ Query cache test PASSED")
            sys.exit(0)
        else:
            print("\n❌ Query cache test FAILED")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and have installed dependencies")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
