#!/usr/bin/env python3
"""
Example usage of the improved NVIDIA Cosmos Video Retrieval System.
"""

from video_search_v2 import VideoSearchEngine
from config import VideoRetrievalConfig
from exceptions import VideoNotFoundError, NoResultsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Basic usage example."""
    print("\n=== Basic Usage Example ===")
    
    # Create search engine with default config
    search_engine = VideoSearchEngine()
    
    # Build database
    video_dir = "/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database"
    search_engine.build_database(video_dir)
    
    # Search by text
    results = search_engine.search_by_text("car approaching cyclist", top_k=3)
    
    print("\nSearch results:")
    for result in results:
        print(f"- {result['video_name']} (score: {result['similarity_score']:.3f})")


def example_with_config():
    """Example using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom config
    config = VideoRetrievalConfig(
        batch_size=8,  # Process more videos in parallel
        similarity_threshold=0.5,  # Only show results with >0.5 similarity
        log_level="DEBUG"  # More verbose logging
    )
    
    # Create search engine with custom config
    search_engine = VideoSearchEngine(config=config)
    
    # Search with filters
    query_video = "/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/user_input/car2cyclist_2.mp4"
    
    try:
        results = search_engine.search_by_video(query_video, top_k=5)
        print(f"\nFound {len(results)} similar videos")
    except NoResultsError:
        print("No videos found above similarity threshold")


def example_error_handling():
    """Example with proper error handling."""
    print("\n=== Error Handling Example ===")
    
    search_engine = VideoSearchEngine()
    
    try:
        # Try to search with non-existent video
        results = search_engine.search_by_video("non_existent_video.mp4")
    except VideoNotFoundError as e:
        print(f"Handled error: {e}")
    
    try:
        # Try empty text search
        results = search_engine.search_by_text("")
    except Exception as e:
        print(f"Handled error: {e}")


def example_batch_processing():
    """Example showing batch processing benefits."""
    print("\n=== Batch Processing Example ===")
    
    from pathlib import Path
    
    # Get all videos
    video_dir = Path("/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database")
    video_files = list(video_dir.glob("*.mp4"))
    
    # Create embedder with different batch sizes
    from video_embedder_v2 import CosmosVideoEmbedder
    
    # Small batch size
    config1 = VideoRetrievalConfig(batch_size=1)
    embedder1 = CosmosVideoEmbedder(config1)
    
    # Large batch size
    config2 = VideoRetrievalConfig(batch_size=4)
    embedder2 = CosmosVideoEmbedder(config2)
    
    print(f"Processing {len(video_files)} videos...")
    print("Batch size 1: Sequential processing")
    print("Batch size 4: Parallel processing (faster!)")


def example_export_import():
    """Example of database export and import."""
    print("\n=== Export/Import Example ===")
    
    search_engine = VideoSearchEngine()
    
    # Get database info
    info = search_engine.get_database_info()
    print(f"Database contains {info['num_videos']} videos")
    
    # Export search results
    results = search_engine.search_by_text("car", top_k=3)
    export_path = search_engine.export_results(
        results, 
        "search_results_export",
        format='csv'  # Export as CSV
    )
    print(f"Results exported to: {export_path}")


def example_extensibility():
    """Example showing how to extend with custom components."""
    print("\n=== Extensibility Example ===")
    
    from base import SearchStrategy
    import numpy as np
    
    # Custom search strategy with threshold
    class ThresholdedSearch(SearchStrategy):
        def __init__(self, min_similarity: float = 0.7):
            self.min_similarity = min_similarity
        
        def search(self, query_embedding, database, top_k=5, filters=None):
            # Get all results
            all_results = database.compute_similarity(query_embedding, top_k=len(database.embeddings))
            
            # Filter by threshold
            filtered = [(idx, sim, meta) for idx, sim, meta in all_results if sim >= self.min_similarity]
            
            return filtered[:top_k]
    
    # Use custom search strategy
    custom_strategy = ThresholdedSearch(min_similarity=0.8)
    search_engine = VideoSearchEngine(search_strategy=custom_strategy)
    
    print("Using custom search strategy with high similarity threshold (0.8)")


if __name__ == "__main__":
    print("NVIDIA Cosmos Video Retrieval System - Improved Version Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_with_config()
    example_error_handling()
    example_batch_processing()
    example_export_import()
    example_extensibility()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey improvements demonstrated:")
    print("✓ Batch processing for faster embedding extraction")
    print("✓ Custom exceptions with proper error handling")
    print("✓ Safe JSON + numpy serialization")
    print("✓ Configuration management")
    print("✓ Extensible architecture with abstract base classes")
    print("\nSee IMPROVEMENTS.md for more details on all improvements.")
