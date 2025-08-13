#!/usr/bin/env python3
"""
Quick demo script for NVIDIA Cosmos Video Retrieval System.
This script demonstrates the basic functionality without command-line arguments.
"""

from video_search import VideoSearchEngine
from video_visualizer import VideoResultsVisualizer
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 80)
    print("NVIDIA Cosmos Video Retrieval System - Quick Demo")
    print("=" * 80)
    
    # Initialize the search engine
    print("\n1. Initializing search engine...")
    search_engine = VideoSearchEngine("video_embeddings.pkl")
    
    # Define paths
    video_db_dir = "/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database"
    query_video = "/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/user_input/car2cyclist_2.mp4"
    
    # Build or load database
    print("\n2. Building video embeddings database...")
    search_engine.build_database(video_db_dir)
    
    # Get database info
    info = search_engine.get_database_info()
    print(f"\nDatabase contains {info['num_videos']} videos:")
    for i, name in enumerate(info['video_names'], 1):
        print(f"  {i}. {name}")
    
    # Video-to-video search
    print("\n3. Performing video-to-video search...")
    print(f"Query video: {Path(query_video).name}")
    
    video_results = search_engine.search_by_video(query_video, top_k=5)
    
    print("\nVideo Search Results:")
    for result in video_results:
        print(f"  Rank {result['rank']}: {result['video_name']} (Score: {result['similarity_score']:.4f})")
    
    # Text-to-video search
    print("\n4. Performing text-to-video search...")
    text_query = "car approaching cyclist"
    print(f"Query text: '{text_query}'")
    
    text_results = search_engine.search_by_text(text_query, top_k=3)
    
    print("\nText Search Results:")
    for result in text_results:
        print(f"  Rank {result['rank']}: {result['video_name']} (Score: {result['similarity_score']:.4f})")
    
    # Visualize results
    print("\n5. Creating visualizations...")
    visualizer = VideoResultsVisualizer()
    
    # Video search visualization
    video_vis_path = visualizer.visualize_video_search_results(query_video, video_results[:3])
    print(f"Video search visualization saved to: {video_vis_path}")
    
    # Text search visualization
    text_vis_path = visualizer.visualize_text_search_results(text_query, text_results)
    print(f"Text search visualization saved to: {text_vis_path}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
