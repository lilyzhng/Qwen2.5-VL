"""
Query Video Cache Manager for pre-computed embeddings.
Provides persistent storage and fast lookup for query video embeddings.
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import logging
from datetime import datetime
import hashlib

from .database import SafeVideoDatabase
from .config import VideoRetrievalConfig
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)


class QueryVideoCache:
    """
    Persistent cache for query video embeddings using SQLite + file storage.
    
    This class manages pre-computed embeddings for query videos, enabling
    instant lookup without real-time computation during user interactions.
    """
    
    def __init__(self, cache_path: Union[str, Path] = "data/query_cache", 
                 config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize query video cache.
        
        Args:
            cache_path: Base path for cache files
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True)
        
        # SQLite database for metadata and lookup
        self.db_path = self.cache_path / "query_cache.db"
        
        # Directory for embedding files
        self.embeddings_dir = self.cache_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"QueryVideoCache initialized at {self.cache_path}")
    
    def _init_database(self):
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    embedding_file TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    modification_time REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Create index for fast filename lookup
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_filename 
                ON query_embeddings(filename)
            """)
            
            conn.commit()
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Compute file hash for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_embedding_filename(self, video_filename: str) -> str:
        """Generate embedding file name from video filename."""
        return f"{Path(video_filename).stem}_embedding.npy"
    
    def is_cached(self, filename: str) -> bool:
        """
        Check if embedding is cached and up-to-date.
        
        Args:
            filename: Video filename
            
        Returns:
            True if cached and current, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_hash, modification_time, file_path FROM query_embeddings WHERE filename = ?",
                (filename,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
            
            cached_hash, cached_mtime, file_path = result
            
            # Check if file still exists
            if not Path(file_path).exists():
                logger.warning(f"Cached file no longer exists: {file_path}")
                self.remove_from_cache(filename)
                return False
            
            # Check if file has been modified
            current_mtime = Path(file_path).stat().st_mtime
            if current_mtime != cached_mtime:
                logger.info(f"File modified, invalidating cache: {filename}")
                self.remove_from_cache(filename)
                return False
            
            # Verify embedding file exists
            embedding_file = self.embeddings_dir / self._get_embedding_filename(filename)
            if not embedding_file.exists():
                logger.warning(f"Embedding file missing: {embedding_file}")
                self.remove_from_cache(filename)
                return False
            
            return True
    
    def get_embedding(self, filename: str) -> Optional[np.ndarray]:
        """
        Get cached embedding by filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Embedding array if cached, None otherwise
        """
        if not self.is_cached(filename):
            return None
        
        try:
            embedding_file = self.embeddings_dir / self._get_embedding_filename(filename)
            embedding = np.load(embedding_file)
            
            # Update access statistics
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE query_embeddings 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE filename = ?
                """, (datetime.now().isoformat(), filename))
                conn.commit()
            
            logger.debug(f"Retrieved cached embedding for: {filename}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading cached embedding for {filename}: {e}")
            self.remove_from_cache(filename)
            return None
    
    def store_embedding(self, filename: str, file_path: Union[str, Path], 
                       embedding: np.ndarray) -> bool:
        """
        Store embedding in cache.
        
        Args:
            filename: Video filename
            file_path: Full path to video file
            embedding: Embedding array
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Validate inputs
            if not file_path.exists():
                raise ValueError(f"Video file not found: {file_path}")
            
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Embedding must be numpy array")
            
            # Get file information
            file_stat = file_path.stat()
            file_hash = self._get_file_hash(file_path)
            
            # Save embedding to file
            embedding_file = self.embeddings_dir / self._get_embedding_filename(filename)
            np.save(embedding_file, embedding)
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO query_embeddings 
                    (filename, file_path, file_hash, embedding_file, embedding_dim,
                     file_size, modification_time, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    filename,
                    str(file_path),
                    file_hash,
                    str(embedding_file),
                    embedding.shape[0],
                    file_stat.st_size,
                    file_stat.st_mtime,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            logger.info(f"Cached embedding for: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for {filename}: {e}")
            return False
    
    def remove_from_cache(self, filename: str) -> bool:
        """
        Remove embedding from cache.
        
        Args:
            filename: Video filename
            
        Returns:
            True if removed, False if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get embedding file path
                cursor = conn.execute(
                    "SELECT embedding_file FROM query_embeddings WHERE filename = ?",
                    (filename,)
                )
                result = cursor.fetchone()
                
                if result:
                    embedding_file = Path(result[0])
                    
                    # Remove embedding file
                    if embedding_file.exists():
                        embedding_file.unlink()
                    
                    # Remove from database
                    conn.execute(
                        "DELETE FROM query_embeddings WHERE filename = ?",
                        (filename,)
                    )
                    conn.commit()
                    
                    logger.info(f"Removed from cache: {filename}")
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Error removing from cache {filename}: {e}")
            return False
    
    def list_cached_videos(self) -> List[Dict[str, Any]]:
        """
        Get list of all cached videos.
        
        Returns:
            List of video information dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT filename, file_path, embedding_dim, file_size,
                           created_at, last_accessed, access_count
                    FROM query_embeddings
                    ORDER BY last_accessed DESC
                """)
                
                videos = []
                for row in cursor.fetchall():
                    videos.append({
                        'filename': row[0],
                        'file_path': row[1],
                        'embedding_dim': row[2],
                        'file_size': row[3],
                        'created_at': row[4],
                        'last_accessed': row[5],
                        'access_count': row[6]
                    })
                
                return videos
                
        except Exception as e:
            logger.error(f"Error listing cached videos: {e}")
            return []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_videos,
                        SUM(file_size) as total_file_size,
                        SUM(access_count) as total_accesses,
                        AVG(access_count) as avg_accesses,
                        MAX(last_accessed) as last_access
                    FROM query_embeddings
                """)
                
                result = cursor.fetchone()
                
                # Calculate cache directory size
                cache_size = 0
                if self.embeddings_dir.exists():
                    for file in self.embeddings_dir.glob("*.npy"):
                        cache_size += file.stat().st_size
                
                return {
                    'total_videos': result[0] or 0,
                    'total_file_size_mb': (result[1] or 0) / (1024 * 1024),
                    'cache_size_mb': cache_size / (1024 * 1024),
                    'total_accesses': result[2] or 0,
                    'avg_accesses': result[3] or 0,
                    'last_access': result[4],
                    'cache_path': str(self.cache_path)
                }
                
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {
                'total_videos': 0,
                'total_file_size_mb': 0,
                'cache_size_mb': 0,
                'total_accesses': 0,
                'avg_accesses': 0,
                'last_access': None,
                'cache_path': str(self.cache_path),
                'error': str(e)
            }
    
    def clear_cache(self) -> bool:
        """
        Clear all cached embeddings.
        
        Returns:
            True if cleared successfully
        """
        try:
            # Remove all embedding files
            for file in self.embeddings_dir.glob("*.npy"):
                file.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM query_embeddings")
                conn.commit()
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def cleanup_orphaned_files(self) -> int:
        """
        Remove orphaned embedding files that don't have database entries.
        
        Returns:
            Number of files cleaned up
        """
        try:
            # Get all embedding files from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT embedding_file FROM query_embeddings")
                db_files = {Path(row[0]).name for row in cursor.fetchall()}
            
            # Find orphaned files
            orphaned_count = 0
            for file in self.embeddings_dir.glob("*.npy"):
                if file.name not in db_files:
                    file.unlink()
                    orphaned_count += 1
                    logger.debug(f"Removed orphaned file: {file.name}")
            
            if orphaned_count > 0:
                logger.info(f"Cleaned up {orphaned_count} orphaned files")
            
            return orphaned_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


class QueryDatabaseManager:
    """
    Manager for building and maintaining query video databases.
    Combines SafeVideoDatabase with QueryVideoCache for comprehensive query management.
    """
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """Initialize query database manager."""
        self.config = config or VideoRetrievalConfig()
        
        # Query video database (similar to main database but for query videos)
        self.query_database = SafeVideoDatabase(
            database_path="query_embeddings",
            config=self.config
        )
        
        # Fast cache for frequent queries
        self.query_cache = QueryVideoCache(
            cache_path="data/query_cache",
            config=self.config
        )
        
        logger.info("QueryDatabaseManager initialized")
    
    def build_query_database(self, query_video_dir: Union[str, Path], 
                           force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build database for query videos.
        
        Args:
            query_video_dir: Directory containing query videos
            force_rebuild: Whether to rebuild from scratch
            
        Returns:
            Build statistics
        """
        from .embedder import CosmosVideoEmbedder
        from .optimizations import batch_normalize_embeddings
        
        query_dir = Path(query_video_dir)
        if not query_dir.exists():
            raise ValueError(f"Query video directory not found: {query_dir}")
        
        # Get video files
        video_files = []
        for ext in self.config.supported_formats:
            video_files.extend(query_dir.glob(f"*{ext}"))
            video_files.extend(query_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"No video files found in {query_dir}")
            return {"processed": 0, "cached": 0, "errors": 0}
        
        logger.info(f"Found {len(video_files)} query video files")
        
        # Initialize embedder
        embedder = CosmosVideoEmbedder(self.config)
        
        # Process videos
        processed = 0
        cached = 0
        errors = 0
        embeddings_data = []
        
        for video_path in video_files:
            filename = video_path.name
            
            try:
                # Check if already processed and current
                if not force_rebuild:
                    if self.query_cache.is_cached(filename):
                        cached += 1
                        logger.debug(f"Already cached: {filename}")
                        continue
                
                # Extract embedding
                logger.info(f"Processing: {filename}")
                embedding = embedder.extract_video_embedding(video_path)
                
                # Normalize
                embedding = embedding.astype('float32').reshape(1, -1)
                batch_normalize_embeddings(embedding)
                embedding = embedding[0]
                
                # Store in cache
                if self.query_cache.store_embedding(filename, video_path, embedding):
                    # Also add to database for completeness
                    embeddings_data.append({
                        "video_path": str(video_path),
                        "embedding": embedding,
                        "embedding_dim": embedding.shape[0],
                        "num_frames": self.config.num_frames
                    })
                    processed += 1
                else:
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                errors += 1
        
        # Update database if we have new embeddings
        if embeddings_data:
            self.query_database.add_embeddings(embeddings_data)
            self.query_database.save()
        
        # Cleanup orphaned files
        orphaned = self.query_cache.cleanup_orphaned_files()
        
        stats = {
            "processed": processed,
            "cached": cached,
            "errors": errors,
            "orphaned_cleaned": orphaned,
            "total_files": len(video_files)
        }
        
        logger.info(f"Query database build complete: {stats}")
        return stats
    
    def get_query_embedding(self, filename: str) -> Optional[np.ndarray]:
        """
        Get embedding for query video by filename.
        
        Args:
            filename: Video filename
            
        Returns:
            Embedding if found, None otherwise
        """
        # Try cache first (fastest)
        embedding = self.query_cache.get_embedding(filename)
        if embedding is not None:
            return embedding
        
        # Fall back to database
        result = self.query_database.get_embedding_by_filename(filename)
        if result is not None:
            embedding, metadata = result
            # Store in cache for future use
            video_path = Path(metadata['video_path'])
            if video_path.exists():
                self.query_cache.store_embedding(filename, video_path, embedding)
            return embedding
        
        return None
    
    def list_available_query_videos(self) -> List[str]:
        """Get list of available query video filenames."""
        cached_videos = self.query_cache.list_cached_videos()
        db_videos = self.query_database.list_available_videos()
        
        # Combine and deduplicate
        all_filenames = set()
        
        for video in cached_videos:
            all_filenames.add(video['filename'])
        
        for video in db_videos:
            all_filenames.add(video['filename'])
        
        return sorted(list(all_filenames))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics for query system."""
        cache_stats = self.query_cache.get_cache_statistics()
        db_stats = self.query_database.get_statistics()
        
        return {
            "cache": cache_stats,
            "database": db_stats,
            "total_available": len(self.list_available_query_videos())
        }
