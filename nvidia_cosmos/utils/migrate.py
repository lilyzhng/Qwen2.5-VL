#!/usr/bin/env python3
"""
Migration script to update existing code to use the improved components.
"""

import sys
import logging
from pathlib import Path
import argparse

from search import OptimizedVideoSearchEngine as VideoSearchEngine
from config import VideoRetrievalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_database(old_db_path: str, new_db_path: str):
    """Migrate from pickle database to safe JSON + numpy format."""
    logger.info(f"Migrating database from {old_db_path} to {new_db_path}")
    
    # Create search engine with new components
    config = VideoRetrievalConfig()
    config.database_path = new_db_path
    
    search_engine = VideoSearchEngine(config=config)
    
    # Import legacy database
    search_engine.import_legacy_database(old_db_path)
    
    logger.info("Migration completed successfully!")
    
    # Show database info
    info = search_engine.get_database_info()
    logger.info(f"Migrated {info['num_videos']} videos")
    logger.info(f"New database size: {info.get('database_size_mb', 0):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate to improved NVIDIA Cosmos Video Retrieval System"
    )
    
    parser.add_argument(
        '--old-db',
        type=str,
        default='video_embeddings.pkl',
        help='Path to old pickle database'
    )
    
    parser.add_argument(
        '--new-db',
        type=str,
        default='video_embeddings',
        help='Base path for new database (without extension)'
    )
    
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate a configuration file'
    )
    
    args = parser.parse_args()
    
    if args.generate_config:
        config = VideoRetrievalConfig()
        config.to_yaml('config.yaml')
        logger.info("Configuration file saved to config.yaml")
        logger.info("Edit this file to customize your settings")
        return 0
    
    # Check if old database exists
    old_db_path = Path(args.old_db)
    if not old_db_path.exists():
        logger.error(f"Old database not found: {old_db_path}")
        logger.info("If you haven't built a database yet, use the new main.py directly")
        return 1
    
    try:
        migrate_database(args.old_db, args.new_db)
        
        print("\n" + "="*80)
        print("MIGRATION SUCCESSFUL!")
        print("="*80)
        print(f"Your database has been migrated to the new safe format.")
        print(f"The new database files are:")
        print(f"  - {args.new_db}.json (metadata)")
        print(f"  - {args.new_db}.npy (embeddings)")
        print()
        print("To use the improved system:")
        print("  1. Review the new features in IMPROVEMENTS.md")
        print("  2. Use the updated main.py")
        print("  3. Example: python main.py search --query-text 'car approaching cyclist'")
        print()
        print("The old database file is preserved and can be deleted if migration was successful.")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
