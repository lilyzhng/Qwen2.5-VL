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
import numpy as np
import pandas as pd
from core.search import VideoSearchEngine
from core.visualizer import VideoResultsVisualizer
from core.config import VideoRetrievalConfig
from core.cluster import EmbeddingClusterer
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
    
    build_parser = subparsers.add_parser('build', help='Build unified video embeddings database')
    build_parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to unified input parquet file containing video file paths (column: sensor_video_file)'
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
    
    search_parser = subparsers.add_parser('search', help='Search for similar videos')
    search_parser.add_argument(
        '--query-video', '-q',
        type=str,
        help='Path to query video file'
    )
    search_parser.add_argument(
        '--query-slice-id', '-qid',
        type=str,
        help='Query video slice ID (8 digits, uses pre-computed embeddings)'
    )
    search_parser.add_argument(
        '--query-text', '-t',
        type=str,
        help='Text query for search'
    )
    search_parser.add_argument(
        '--search-mode', '-m',
        type=str,
        choices=['video', 'text', 'joint'],
        default='auto',
        help='Search mode: video, text, joint, or auto (default: auto - infers from inputs)'
    )
    search_parser.add_argument(
        '--text-weight-alpha', '-a',
        type=float,
        default=0.5,
        help='Text weight for joint search (0.0=video only, 1.0=text only, default=0.5)'
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
    
    demo_parser = subparsers.add_parser('demo', help='Run demo with example query')
    
    cluster_parser = subparsers.add_parser('cluster', help='Generate clustering and 2D coordinates for embeddings')
    cluster_parser.add_argument(
        '--database-type', '-dt',
        type=str,
        choices=['main', 'query', 'both'],
        default='main',
        help='Which database to cluster (default: main)'
    )
    cluster_parser.add_argument(
        '--pca-components', '-pc',
        type=int,
        default=50,
        help='Number of PCA components for dimensionality reduction (default: 50)'
    )
    cluster_parser.add_argument(
        '--umap-neighbors', '-un',
        type=int,
        default=15,
        help='Number of neighbors for UMAP (default: 15)'
    )
    cluster_parser.add_argument(
        '--umap-min-dist', '-umd',
        type=float,
        default=0.1,
        help='Minimum distance for UMAP (default: 0.1)'
    )
    cluster_parser.add_argument(
        '--dbscan-eps', '-de',
        type=float,
        default=0.5,
        help='DBSCAN epsilon parameter (default: 0.5)'
    )
    cluster_parser.add_argument(
        '--dbscan-min-samples', '-dms',
        type=int,
        default=5,
        help='DBSCAN minimum samples parameter (default: 5)'
    )
    cluster_parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization of clusters'
    )
    
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
    if hasattr(args, 'input_path') and args.input_path:
        config.input_path = args.input_path
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
        
        if args.command == 'build':
            logger.info("Building unified video embeddings database from parquet input...")
            # Override config input_path with command line argument
            if args.input_path:
                config.input_path = args.input_path
            
            search_engine.build_database_from_parquet(args.force_rebuild)
            
            info = search_engine.get_database_info()
            logger.info(f"Database built successfully!")
            logger.info(f"Total input files: {info['num_inputs']}")
            logger.info(f"Embedding dimension: {info['embedding_dim']}")
            logger.info(f"Database size: {info.get('database_size_mb', 0):.2f} MB")
            
        elif args.command == 'search':
            # Determine search mode
            search_mode = args.search_mode
            
            # Auto-detect search mode if not specified
            if search_mode == 'auto':
                if args.query_slice_id:
                    search_mode = 'video'
                elif args.query_video:
                    search_mode = 'video'
                elif args.query_text and (args.query_video or args.query_slice_id):
                    search_mode = 'joint'
                elif args.query_text:
                    search_mode = 'text'
                else:
                    logger.error("Please provide --query-video, --query-text, --query-slice-id, or both text and video for joint search")
                    return 1
            
            # Execute search based on mode
            if search_mode == 'joint':
                if not args.query_text:
                    logger.error("Joint search requires --query-text")
                    return 1
                if not (args.query_video or args.query_slice_id):
                    logger.error("Joint search requires --query-video or --query-slice-id")
                    return 1
                
                # Use slice_id or extract from video path for joint search
                video_identifier = args.query_slice_id if args.query_slice_id else Path(args.query_video).name
                
                logger.info(f"Joint search: text='{args.query_text}', video={video_identifier}, alpha={args.text_weight_alpha}")
                results = search_engine.search_by_joint(
                    args.query_text, 
                    video_identifier,
                    alpha=args.text_weight_alpha,
                    top_k=args.top_k
                )
                
            elif search_mode == 'video':
                if args.query_slice_id:
                    logger.info(f"Searching by slice_id (pre-computed): {args.query_slice_id}")
                    results = search_engine.search_by_filename(args.query_slice_id)
                elif args.query_video:
                    logger.info(f"Searching by video: {args.query_video}")
                    results = search_engine.search_by_video(args.query_video)
                else:
                    logger.error("Video search requires --query-video or --query-slice-id")
                    return 1
                    
            elif search_mode == 'text':
                if not args.query_text:
                    logger.error("Text search requires --query-text")
                    return 1
                logger.info(f"Searching by text: '{args.query_text}'")
                results = search_engine.search_by_text(args.query_text)
                
            else:
                logger.error(f"Invalid search mode: {search_mode}")
                return 1
            
            print("\n" + "="*80)
            print("SEARCH RESULTS")
            print("="*80)
            
            for result in results:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['slice_id']}")
                print(f"  Path: {result['video_path']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            if args.visualize:
                visualizer = VideoResultsVisualizer()
                
                if search_mode == 'joint':
                    # Joint search visualization - use text visualization with joint info
                    video_identifier = args.query_slice_id if args.query_slice_id else Path(args.query_video).name
                    joint_query_text = f"Joint: '{args.query_text}' + {video_identifier} (α={args.text_weight_alpha})"
                    vis_path = visualizer.visualize_text_search_results(
                        joint_query_text, results[:5], 
                        show_interactive=True, keep_open=True
                    )
                elif search_mode == 'video' and args.query_video:
                    vis_path = visualizer.visualize_video_search_results(
                        args.query_video, results[:5],  # Top 5 for visualization
                        show_interactive=True, keep_open=True
                    )
                else:
                    # Text search or video search by slice_id
                    query_display = args.query_text if args.query_text else args.query_slice_id
                    vis_path = visualizer.visualize_text_search_results(
                        query_display, results[:5], 
                        show_interactive=True, keep_open=True
                    )
                
                logger.info(f"Visualization saved to: {vis_path}")
            

        
        elif args.command == 'info':
            info = search_engine.get_database_info()
            
            print("\n" + "="*80)
            print("DATABASE INFORMATION")
            print("="*80)
            print(f"Version: {info.get('version', 'unknown')}")
            print(f"Total input files: {info['num_inputs']}")
            print(f"Embedding dimension: {info['embedding_dim']}")
            print(f"Database size: {info.get('database_size_mb', 0):.2f} MB")
            
            if info['num_inputs'] > 0:
                print("\nVideos in database:")
                for i, slice_id in enumerate(info['slice_ids'][:10], 1):
                    print(f"  {i}. {slice_id}")
                if len(info['slice_ids']) > 10:
                    print(f"  ... and {len(info['slice_ids']) - 10} more")
        

        
        elif args.command == 'demo':
            logger.info("Running demo...")
            
            db_info = search_engine.get_database_info()
            if db_info['num_inputs'] == 0:
                logger.info("Building database first...")
                search_engine.build_unified_database_from_file_list(config.input_path)
            
            # Demo text search
            print("\n" + "="*80)
            print("DEMO RESULTS - Text search: 'car approaching cyclist'")
            print("="*80)
            
            text_results = search_engine.search_by_text("car approaching cyclist")
            for result in text_results[:3]:
                print(f"\nRank {result['rank']}:")
                print(f"  Video: {result['slice_id']}")
                print(f"  Similarity Score: {result['similarity_score']:.4f}")
            
            logger.info("Demo completed successfully!")
        
        elif args.command == 'cluster':
            logger.info("Generating clusters for unified database...")
            
            # Use unified embeddings path
            db_path = search_engine._resolve_path(config.embeddings_path)
            
            if not db_path.exists():
                logger.error(f"Unified database not found at {db_path}. Please run 'build' command first.")
                return 1
            
            databases_to_process = [('unified', db_path)]
            
            for db_type, db_path in databases_to_process:
                db_path = Path(db_path)
                if not db_path.exists():
                    logger.warning(f"{db_type.capitalize()} database not found at {db_path}. Skipping...")
                    continue
                
                logger.info(f"Processing {db_type} database: {db_path}")
                
                # Load parquet file
                df = pd.read_parquet(db_path)
                logger.info(f"Loaded {len(df)} embeddings from {db_type} database")
                
                # Extract embeddings
                embeddings = np.vstack(df['embedding'].values)
                logger.info(f"Embeddings shape: {embeddings.shape}")
                
                # Initialize clusterer with command line arguments
                clusterer = EmbeddingClusterer(
                    pca_components=args.pca_components,
                    umap_n_neighbors=args.umap_neighbors,
                    umap_min_dist=args.umap_min_dist,
                    dbscan_eps=args.dbscan_eps,
                    dbscan_min_samples=args.dbscan_min_samples
                )
                
                # Perform clustering
                coords_2d, cluster_labels = clusterer.fit_transform(embeddings)
                
                # Add cluster information to dataframe
                df['cluster_id'] = cluster_labels
                df['x'] = coords_2d[:, 0]
                df['y'] = coords_2d[:, 1]
                
                # Save updated parquet file
                df.to_parquet(db_path, index=False)
                logger.info(f"Updated {db_path} with cluster information")
                
                # Get and display statistics
                stats = clusterer.get_cluster_statistics()
                print(f"\n{db_type.upper()} DATABASE CLUSTERING RESULTS")
                print("="*50)
                print(f"Total points: {len(df)}")
                print(f"Number of clusters: {stats['n_clusters']}")
                print(f"Noise points: {stats['n_noise_points']}")
                print(f"PCA explained variance: {stats['pca_explained_variance']:.2%}")
                print("\nCluster sizes:")
                for cluster, size in stats['cluster_sizes'].items():
                    print(f"  {cluster}: {size} points")
                
                # Save detailed results as JSON
                json_path = db_path.parent / f"{db_path.stem}_cluster_results.json"
                metadata = [{'slice_id': row['slice_id'], 'video_path': row['video_path']} 
                           for _, row in df.iterrows()]
                clusterer.save_results(json_path, metadata=metadata)
                logger.info(f"Saved detailed clustering results to {json_path}")
                
                # Generate visualization if requested
                if args.visualize:
                    try:
                        from core.cluster import visualize_clusters
                        viz_path = db_path.parent / f"{db_path.stem}_cluster_visualization.png"
                        visualize_clusters(
                            coords_2d, 
                            cluster_labels,
                            save_path=viz_path,
                            point_names=df['slice_id'].tolist()
                        )
                        logger.info(f"Saved visualization to {viz_path}")
                    except ImportError:
                        logger.warning("Visualization requires matplotlib. Install it to generate plots.")
            
            logger.info("Clustering completed successfully!")
    
    except Exception as e:
        return handle_error(e)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
