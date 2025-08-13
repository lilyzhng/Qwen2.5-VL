#!/usr/bin/env python3
"""
Main script for NVIDIA Cosmos Video Retrieval System.
"""

import argparse
import logging
from pathlib import Path
import sys
from video_search import VideoSearchEngine
from video_visualizer import VideoResultsVisualizer
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Cosmos Video Retrieval System - Search for similar videos using embeddings"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build database command
    build_parser = subparsers.add_parser('build', help='Build video embeddings database')
    build_parser.add_argument(
        '--video-dir', '-d',
        type=str,
        default='/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database',
        help='Directory containing video files'
    )
    build_parser.add_argument(
        '--force-rebuild', '-f',
        action='store_true',
        help='Force rebuild database from scratch'
    )
    build_parser.add_argument(
        '--db-path',
        type=str,
        default='video_embeddings.pkl',
        help='Path to save the database'
    )
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for similar videos')
    search_parser.add_argument(
        '--query-video', '-q',
        type=str,
        help='Path to query video file'
    )
    search_parser.add_argument(
        '--query-text', '-t',
        type=str,
        help='Text query for video search'
    )
    search_parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of top results to return'
    )
    search_parser.add_argument(
        '--db-path',
        type=str,
        default='video_embeddings.pkl',
        help='Path to the database'
    )
    search_parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization of results'
    )
    search_parser.add_argument(
        '--export-dir', '-e',
        type=str,
        help='Directory to export results'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show database information')
    info_parser.add_argument(
        '--db-path',
        type=str,
        default='video_embeddings.pkl',
        help='Path to the database'
    )
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with example query')
    demo_parser.add_argument(
        '--db-path',
        type=str,
        default='video_embeddings.pkl',
        help='Path to the database'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize search engine
    search_engine = VideoSearchEngine(args.db_path if hasattr(args, 'db_path') else 'video_embeddings.pkl')
    
    if args.command == 'build':
        logger.info("Building video embeddings database...")
        search_engine.build_database(args.video_dir, args.force_rebuild)
        
        # Show database info
        info = search_engine.get_database_info()
        logger.info(f"Database built successfully!")
        logger.info(f"Total videos: {info['num_videos']}")
        logger.info(f"Embedding dimension: {info['embedding_dim']}")
        
    elif args.command == 'search':
        if not args.query_video and not args.query_text:
            logger.error("Please provide either --query-video or --query-text")
            sys.exit(1)
        
        # Perform search
        if args.query_video:
            logger.info(f"Searching by video: {args.query_video}")
            results = search_engine.search_by_video(args.query_video, args.top_k)
        else:
            logger.info(f"Searching by text: '{args.query_text}'")
            results = search_engine.search_by_text(args.query_text, args.top_k)
        
        # Display results
        print("\n" + "="*80)
        print("SEARCH RESULTS")
        print("="*80)
        
        for result in results:
            print(f"\nRank {result['rank']}:")
            print(f"  Video: {result['video_name']}")
            print(f"  Path: {result['video_path']}")
            print(f"  Similarity Score: {result['similarity_score']:.4f}")
        
        # Visualize if requested
        if args.visualize:
            visualizer = VideoResultsVisualizer()
            
            if args.query_video:
                vis_path = visualizer.visualize_video_search_results(
                    args.query_video, results
                )
            else:
                vis_path = visualizer.visualize_text_search_results(
                    args.query_text, results
                )
            
            logger.info(f"Visualization saved to: {vis_path}")
        
        # Export if requested
        if args.export_dir:
            export_path = search_engine.export_results(
                results, args.export_dir, copy_videos=True
            )
            logger.info(f"Results exported to: {export_path}")
    
    elif args.command == 'info':
        info = search_engine.get_database_info()
        
        print("\n" + "="*80)
        print("DATABASE INFORMATION")
        print("="*80)
        print(f"Total videos: {info['num_videos']}")
        print(f"Embedding dimension: {info['embedding_dim']}")
        print("\nVideos in database:")
        for i, video_name in enumerate(info['video_names'], 1):
            print(f"  {i}. {video_name}")
    
    elif args.command == 'demo':
        # Run demo with sample query
        logger.info("Running demo...")
        
        # First build database if needed
        db_path = Path(args.db_path)
        if not db_path.exists():
            logger.info("Building database first...")
            video_db_dir = '/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database'
            search_engine.build_database(video_db_dir)
        
        # Use the sample input video
        query_video = '/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/user_input/car2cyclist_2.mp4'
        
        if Path(query_video).exists():
            logger.info(f"Searching for videos similar to: {Path(query_video).name}")
            results = search_engine.search_by_video(query_video, top_k=5)
            
            print("\n" + "="*80)
            print("DEMO RESULTS - Video similar to 'car2cyclist_2.mp4'")
            print("="*80)
            
            for result in results:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['video_name']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            # Also demo text search
            print("\n" + "="*80)
            print("DEMO RESULTS - Text search: 'car approaching cyclist'")
            print("="*80)
            
            text_results = search_engine.search_by_text("car approaching cyclist", top_k=3)
            for result in text_results:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['video_name']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
        else:
            logger.error(f"Demo query video not found: {query_video}")

if __name__ == "__main__":
    main()
