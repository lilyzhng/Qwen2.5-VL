#!/usr/bin/env python3
"""
ALFA 0.1 - Similarity Search Interface.
"""

import os
import sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import argparse
import logging
import json
import yaml
import pandas as pd
from core.search import VideoSearchEngine
from core.visualizer import VideoResultsVisualizer
from core.config import VideoRetrievalConfig
from core.exceptions import (
    VideoRetrievalError, VideoNotFoundError, NoResultsError
)

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
        description="ALFA 0.1 - Search for similar videos using embeddings"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    build_parser = subparsers.add_parser('build-main', help='Build video embeddings database')
    build_parser.add_argument(
        '--video-dir', '-d',
        type=str,
        help='Directory containing video files'
    )
    build_parser.add_argument(
        '--data-path-file',
        type=str,
        help='Path to video index file containing video file paths (column: sensor_video_file)'
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
    build_query_parser = subparsers.add_parser('build-query', help='Build query video embeddings database')
    build_query_parser.add_argument(
        '--query-dir', '-qd',
        type=str,
        help='Directory containing query video files'
    )
    build_query_parser.add_argument(
        '--data-path-file',
        type=str,
        help='Path to query video index file containing video file paths (column: sensor_video_file)'
    )
    build_query_parser.add_argument(
        '--force-rebuild', '-f',
        action='store_true',
        help='Force rebuild query database from scratch'
    )
    
    search_parser = subparsers.add_parser('search', help='Search for similar videos')
    search_parser.add_argument(
        '--query-video', '-q',
        type=str,
        help='Path to query video file'
    )
    search_parser.add_argument(
        '--query-filename', '-qf',
        type=str,
        help='Query video filename (uses pre-computed embeddings)'
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

    
    info_parser = subparsers.add_parser('info', help='Show database information')
    
    query_info_parser = subparsers.add_parser('query-info', help='Show query database information')
    
    list_query_parser = subparsers.add_parser('list-queries', help='List available query videos')
    
    demo_parser = subparsers.add_parser('demo', help='Run demo with example query')
    
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
    if hasattr(args, 'data_path_file') and args.data_path_file:
        config.main_file_path = args.data_path_file
    if hasattr(args, 'batch_size') and args.batch_size:
        config.batch_size = args.batch_size
    if hasattr(args, 'top_k') and args.top_k:
        config.default_top_k = args.top_k
    if hasattr(args, 'threshold') and args.threshold:
        config.similarity_threshold = args.threshold
    
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    
    setup_logging(config)
    
    if args.command == 'config':
        config.to_yaml(args.output)
        logger.info(f"Example configuration saved to {args.output}")
        return 0
    
    try:
        search_engine = VideoSearchEngine(config=config)
        
        if args.command == 'build-main':
            logger.info("Building video embeddings database...")
            video_source = args.video_dir if hasattr(args, 'video_dir') and args.video_dir else "from main_file_path"
            search_engine.build_database(video_source, args.force_rebuild)
            
            info = search_engine.get_database_info()
            logger.info(f"Database built successfully!")
            logger.info(f"Total videos: {info['num_videos']}")
            logger.info(f"Embedding dimension: {info['embedding_dim']}")
            logger.info(f"Database size: {info.get('database_size_mb', 0):.2f} MB")
            
        elif args.command == 'build-query':
            logger.info("Building query video embeddings database...")
            # Use query_file_path from config, query_dir argument is optional override
            query_dir = args.query_dir if hasattr(args, 'query_dir') and args.query_dir else None
            stats = search_engine.build_query_database(query_dir, args.force_rebuild)
            
            logger.info(f"Query database built successfully!")
            logger.info(f"Processed: {stats['processed']} videos")
            logger.info(f"Already cached: {stats['cached']} videos")
            logger.info(f"Errors: {stats['errors']} videos")
            if stats.get('orphaned_cleaned', 0) > 0:
                logger.info(f"Cleaned up: {stats['orphaned_cleaned']} orphaned files")
            
        elif args.command == 'search':
            if not args.query_video and not args.query_text and not args.query_filename:
                logger.error("Please provide --query-video, --query-text, or --query-filename")
                return 1
            
            if args.query_filename:
                logger.info(f"Searching by filename (pre-computed): {args.query_filename}")
                results = search_engine.search_by_filename(args.query_filename)
            elif args.query_video:
                logger.info(f"Searching by video: {args.query_video}")
                results = search_engine.search_by_video(args.query_video)
            else:
                logger.info(f"Searching by text: '{args.query_text}'")
                results = search_engine.search_by_text(args.query_text)
            
            print("\n" + "="*80)
            print("SEARCH RESULTS")
            print("="*80)
            
            for result in results:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['video_name']}")
                print(f"  Path: {result['video_path']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            if args.visualize:
                visualizer = VideoResultsVisualizer()
                
                if args.query_video:
                    vis_path = visualizer.visualize_video_search_results(
                        args.query_video, results[:5],  # Top 5 for visualization
                        show_interactive=True, keep_open=True
                    )
                else:
                    vis_path = visualizer.visualize_text_search_results(
                        args.query_text, results[:5], 
                        show_interactive=True, keep_open=True
                    )
                
                logger.info(f"Visualization saved to: {vis_path}")
            

        
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
        
        elif args.command == 'query-info':
            stats = search_engine.get_statistics()
            query_stats = stats.get('query_database', {})
            
            print("\n" + "="*80)
            print("QUERY DATABASE INFORMATION")
            print("="*80)
            
            if 'cache' in query_stats:
                cache_info = query_stats['cache']
                print(f"Cached videos: {cache_info.get('total_videos', 0)}")
                print(f"Cache size: {cache_info.get('cache_size_mb', 0):.2f} MB")
                print(f"Total accesses: {cache_info.get('total_accesses', 0)}")
                print(f"Average accesses per video: {cache_info.get('avg_accesses', 0):.1f}")
            
            if 'database' in query_stats:
                db_info = query_stats['database']
                print(f"Database videos: {db_info.get('num_videos', 0)}")
                print(f"Database size: {db_info.get('database_size_mb', 0):.2f} MB")
            
            print(f"Total available: {query_stats.get('total_available', 0)}")
        
        elif args.command == 'list-queries':
            available_videos = search_engine.get_query_videos_list()
            
            print("\n" + "="*80)
            print("AVAILABLE QUERY VIDEOS")
            print("="*80)
            
            if available_videos:
                for i, filename in enumerate(available_videos, 1):
                    print(f"  {i}. {filename}")
                print(f"\nTotal: {len(available_videos)} query videos available")
            else:
                print("No query videos found. Use 'build-query' command to process query videos.")
        
        elif args.command == 'demo':
            logger.info("Running demo...")
            
            db_info = search_engine.get_database_info()
            if db_info['num_videos'] == 0:
                logger.info("Building database first...")
                search_engine.build_database("from main_file_path")
            
            # Demo text search
            print("\n" + "="*80)
            print("DEMO RESULTS - Text search: 'car approaching cyclist'")
            print("="*80)
            
            text_results = search_engine.search_by_text("car approaching cyclist")
            for result in text_results[:3]:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['video_name']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            logger.info("Demo completed successfully!")
    
    except Exception as e:
        return handle_error(e)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
