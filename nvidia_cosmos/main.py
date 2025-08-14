#!/usr/bin/env python3
"""
Improved main script for NVIDIA Cosmos Video Retrieval System.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import yaml

from video_search import VideoSearchEngine
from video_visualizer import VideoResultsVisualizer
from config import VideoRetrievalConfig
from exceptions import (
    VideoRetrievalError, VideoNotFoundError, NoResultsError
)

# Configure logging based on config
def setup_logging(config: VideoRetrievalConfig):
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=config.log_format
    )

logger = logging.getLogger(__name__)


def handle_error(e: Exception):
    """Handle errors gracefully with user-friendly messages."""
    if isinstance(e, VideoNotFoundError):
        logger.error(f"Video not found: {e}")
        return 1
    elif isinstance(e, NoResultsError):
        logger.warning(f"No results found: {e}")
        return 0
    elif isinstance(e, VideoRetrievalError):
        logger.error(f"Error: {e}")
        return 1
    else:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 2


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA Cosmos Video Retrieval System - Search for similar videos using embeddings"
    )
    
    # Global arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build database command
    build_parser = subparsers.add_parser('build', help='Build video embeddings database')
    build_parser.add_argument(
        '--video-dir', '-d',
        type=str,
        help='Directory containing video files'
    )
    build_parser.add_argument(
        '--force-rebuild', '-f',
        action='store_true',
        help='Force rebuild database from scratch'
    )
    build_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for processing videos'
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
        help='Number of top results to return'
    )
    search_parser.add_argument(
        '--threshold', '-th',
        type=float,
        help='Similarity threshold (0-1)'
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
    search_parser.add_argument(
        '--export-format',
        choices=['json', 'csv'],
        default='json',
        help='Export format'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show database information')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import legacy database')
    import_parser.add_argument(
        '--legacy-path', '-l',
        type=str,
        required=True,
        help='Path to legacy pickle database'
    )
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with example query')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate example configuration file')
    config_parser.add_argument(
        '--output', '-o',
        type=str,
        default='config.yaml',
        help='Output path for configuration file'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Load configuration
    config = VideoRetrievalConfig()
    
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config = VideoRetrievalConfig.from_yaml(args.config)
        elif config_path.suffix == '.json':
            config = VideoRetrievalConfig.from_json(args.config)
        else:
            logger.error("Configuration file must be YAML or JSON")
            return 1
    
    # Override config with command line arguments
    if hasattr(args, 'video_dir') and args.video_dir:
        config.video_database_dir = args.video_dir
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'top_k') and args.top_k:
        config.default_top_k = args.top_k
    if hasattr(args, 'threshold') and args.threshold:
        config.similarity_threshold = args.threshold
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    
    # Setup logging
    setup_logging(config)
    
    # Handle config generation
    if args.command == 'config':
        config.to_yaml(args.output)
        logger.info(f"Example configuration saved to {args.output}")
        return 0
    
    try:
        # Initialize search engine
        search_engine = VideoSearchEngine(config=config)
        
        if args.command == 'build':
            logger.info("Building video embeddings database...")
            search_engine.build_database(config.video_database_dir, args.force_rebuild)
            
            # Show database info
            info = search_engine.get_database_info()
            logger.info(f"Database built successfully!")
            logger.info(f"Total videos: {info['num_videos']}")
            logger.info(f"Embedding dimension: {info['embedding_dim']}")
            logger.info(f"Database size: {info.get('database_size_mb', 0):.2f} MB")
            
        elif args.command == 'search':
            if not args.query_video and not args.query_text:
                logger.error("Please provide either --query-video or --query-text")
                return 1
            
            # Perform search
            if args.query_video:
                logger.info(f"Searching by video: {args.query_video}")
                results = search_engine.search_by_video(args.query_video)
            else:
                logger.info(f"Searching by text: '{args.query_text}'")
                results = search_engine.search_by_text(args.query_text)
            
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
                        args.query_video, results[:5]  # Top 5 for visualization
                    )
                else:
                    vis_path = visualizer.visualize_text_search_results(
                        args.query_text, results[:5]
                    )
                
                logger.info(f"Visualization saved to: {vis_path}")
            
            # Export if requested
            if args.export_dir:
                export_path = search_engine.export_results(
                    results, args.export_dir, 
                    copy_videos=True,
                    format=args.export_format
                )
                logger.info(f"Results exported to: {export_path}")
        
        elif args.command == 'info':
            info = search_engine.get_database_info()
            
            print("\n" + "="*80)
            print("DATABASE INFORMATION")
            print("="*80)
            print(f"Version: {info.get('version', 'unknown')}")
            print(f"Total videos: {info['num_videos']}")
            print(f"Embedding dimension: {info['embedding_dim']}")
            print(f"Database size: {info.get('database_size_mb', 0):.2f} MB")
            
            if info['num_videos'] > 0:
                print("\nVideos in database:")
                for i, video_name in enumerate(info['video_names'][:10], 1):
                    print(f"  {i}. {video_name}")
                if len(info['video_names']) > 10:
                    print(f"  ... and {len(info['video_names']) - 10} more")
        
        elif args.command == 'import':
            logger.info(f"Importing legacy database from {args.legacy_path}")
            search_engine.import_legacy_database(args.legacy_path)
            logger.info("Import completed successfully")
        
        elif args.command == 'demo':
            # Run demo with sample query
            logger.info("Running demo...")
            
            # First build database if needed
            db_info = search_engine.get_database_info()
            if db_info['num_videos'] == 0:
                logger.info("Building database first...")
                search_engine.build_database(config.video_database_dir)
            
            # Use the sample input video
            query_video = Path(config.user_input_dir) / 'car2cyclist_2.mp4'
            
            if query_video.exists():
                logger.info(f"Searching for videos similar to: {query_video.name}")
                results = search_engine.search_by_video(query_video)
                
                print("\n" + "="*80)
                print(f"DEMO RESULTS - Videos similar to '{query_video.name}'")
                print("="*80)
                
                for result in results[:3]:
                    print(f"\nRank {result['rank']}:")
                    print(f"  Video: {result['video_name']}")
                    print(f"  Similarity Score: {result['similarity_score']:.4f}")
                
                # Also demo text search
                print("\n" + "="*80)
                print("DEMO RESULTS - Text search: 'car approaching cyclist'")
                print("="*80)
                
                text_results = search_engine.search_by_text("car approaching cyclist")
                for result in text_results[:3]:
                    print(f"\nRank {result['rank']}:")
                    print(f"  Video: {result['video_name']}")
                    print(f"  Similarity Score: {result['similarity_score']:.4f}")
            else:
                logger.error(f"Demo query video not found: {query_video}")
                return 1
    
    except Exception as e:
        return handle_error(e)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
