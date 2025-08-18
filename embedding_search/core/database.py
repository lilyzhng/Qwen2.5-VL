"""
Unified Parquet Database System for Video Embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import logging
from datetime import datetime
import hashlib

from .config import VideoRetrievalConfig
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)


class ParquetVectorDatabase:
    """
    Unified Parquet-based vector database for both main and query embeddings.
    """
    
    def __init__(self, database_path: Union[str, Path], config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize Parquet vector database.
        
        Args:
            database_path: Path to the parquet database file
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.database_path = Path(database_path)
        self.df = None
        
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()
        
    def load(self):
        """Load database from parquet file."""
        if self.database_path.exists():
            try:
                self.df = pd.read_parquet(self.database_path)
                # Set video_name as index for fast lookups
                if 'video_name' in self.df.columns and self.df.index.name != 'video_name':
                    self.df = self.df.set_index('video_name', drop=False)
                logger.info(f"Loaded {len(self.df)} embeddings from {self.database_path}")
            except Exception as e:
                logger.warning(f"Could not load database {self.database_path}: {e}")
                self._create_empty_database()
        else:
            self._create_empty_database()
    
    def _create_empty_database(self):
        """Create empty database with proper schema."""
        self.df = pd.DataFrame({
            'video_name': [],
            'video_path': [],
            'embedding': [],
            'embedding_dim': [],
            'num_frames': [],
            'file_hash': [],
            'file_size': [],
            'created_at': [],
            'last_accessed': [],
            'access_count': [],
            'category': []
        })
        
        # Set index if not empty
        if len(self.df) > 0:
            self.df = self.df.set_index('video_name', drop=False)
        
    def add_embedding(self, video_name: str, video_path: Union[str, Path], 
                     embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add or update an embedding in the database.
        
        Args:
            video_name: Name of the video file
            video_path: Full path to video file
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            video_path = Path(video_path)
            
            # Get file information
            file_stat = video_path.stat() if video_path.exists() else None
            file_hash = self._get_file_hash(video_path) if video_path.exists() else None
            
            # Prepare row data
            row_data = {
                'video_name': video_name,
                'video_path': str(video_path.absolute()),
                'embedding': embedding.tolist(),
                'embedding_dim': embedding.shape[0],
                'num_frames': metadata.get('num_frames', self.config.num_frames) if metadata else self.config.num_frames,
                'file_hash': file_hash,
                'file_size': file_stat.st_size if file_stat else 0,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 1,
                'category': metadata.get('category', 'query') if metadata else 'query'
            }
            
            # Add additional metadata
            if metadata:
                for key, value in metadata.items():
                    if key not in row_data:
                        row_data[key] = value
            
            # Initialize DataFrame if needed
            if self.df is None or len(self.df) == 0:
                self._create_empty_database()
                
            # Convert to DataFrame for single row
            new_row = pd.DataFrame([row_data])
            
            # Add to DataFrame
            if len(self.df) == 0:
                self.df = new_row.set_index('video_name', drop=False)
            else:
                # Update existing or add new
                if video_name in self.df.index:
                    # Update existing
                    for key, value in row_data.items():
                        self.df.loc[video_name, key] = value
                else:
                    # Add new
                    new_row = new_row.set_index('video_name', drop=False)
                    self.df = pd.concat([self.df, new_row])
            
            logger.info(f"Added embedding for: {video_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding for {video_name}: {e}")
            return False
    
    def get_embedding(self, video_name: str) -> Optional[np.ndarray]:
        """
        Get embedding by video name.
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Embedding array or None if not found
        """
        if self.df is None or len(self.df) == 0 or video_name not in self.df.index:
            return None
            
        try:
            row = self.df.loc[video_name]
            embedding = np.array(row['embedding'], dtype='float32')
            
            # Update access information
            self.df.loc[video_name, 'last_accessed'] = datetime.now().isoformat()
            self.df.loc[video_name, 'access_count'] = int(self.df.loc[video_name, 'access_count']) + 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error retrieving embedding for {video_name}: {e}")
            return None
    
    def list_videos(self) -> List[str]:
        """Get list of all video names in the database."""
        if self.df is None or len(self.df) == 0:
            return []
        return self.df['video_name'].tolist()
    
    def save(self):
        """Save database to parquet file."""
        if self.df is not None and len(self.df) > 0:
            try:
                # Reset index before saving to avoid index column issues
                df_to_save = self.df.reset_index(drop=True)
                df_to_save.to_parquet(self.database_path, index=False)
                logger.info(f"Saved {len(df_to_save)} embeddings to {self.database_path}")
            except Exception as e:
                logger.error(f"Error saving database: {e}")
                raise DatabaseError(f"Failed to save database: {e}")
        else:
            logger.info("No data to save")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Compute file hash for integrity checking."""
        if not file_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.df is None or len(self.df) == 0:
            return {"total_embeddings": 0, "categories": {}}
            
        stats = {
            "total_embeddings": len(self.df),
            "categories": self.df['category'].value_counts().to_dict() if 'category' in self.df.columns else {},
            "embedding_dim": int(self.df['embedding_dim'].iloc[0]) if len(self.df) > 0 else 0,
            "database_size_mb": round(self.database_path.stat().st_size / (1024 * 1024), 2) if self.database_path.exists() else 0
        }
        
        return stats


class UnifiedQueryManager:
    """
    Unified manager for query embeddings using pure Parquet storage.
    Replaces the complex QueryDatabaseManager with a simpler, more efficient system.
    """
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """Initialize unified query manager."""
        self.config = config or VideoRetrievalConfig()
        
        # Use new parquet database
        self.query_db = ParquetVectorDatabase(
            self.config.query_embeddings_path,
            self.config
        )
        
        logger.info("UnifiedQueryManager initialized with Parquet storage")
    
    def build_query_database_from_file_list(self, query_file_path: Union[str, Path],
                                          force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build query database from file path list (Parquet or CSV).
        
        Args:
            query_file_path: Path to file containing query video paths
            force_rebuild: Whether to rebuild from scratch
            
        Returns:
            Build statistics
        """
        from .embedder import CosmosVideoEmbedder
        from .faiss_backend import batch_normalize_embeddings
        
        file_path = Path(query_file_path)
        if not file_path.exists():
            raise ValueError(f"Query file path list not found: {file_path}")
        
        # Load video paths from file
        if file_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if 'sensor_video_file' not in df.columns:
            raise ValueError(f"Column 'sensor_video_file' not found in {file_path}")
        
        # Filter for query videos only
        if 'category' in df.columns:
            query_df = df[df['category'] == 'user_input']
        else:
            query_df = df
        
        video_paths = [Path(path) for path in query_df['sensor_video_file'].tolist()]
        video_files = [path for path in video_paths if path.exists()]
        
        if not video_files:
            logger.warning(f"No valid query video files found in {file_path}")
            return {"processed": 0, "cached": 0, "errors": 0}
        
        logger.info(f"Found {len(video_files)} query video files from file list")
        
        # Initialize embedder
        embedder = CosmosVideoEmbedder(self.config)
        
        stats = {"processed": 0, "cached": 0, "errors": 0}
        
        for video_path in video_files:
            filename = video_path.name
            
            try:
                # Check if already processed
                if not force_rebuild and self.query_db.get_embedding(filename) is not None:
                    stats["cached"] += 1
                    logger.debug(f"Already cached: {filename}")
                    continue
                
                # Extract embedding
                logger.info(f"Processing: {filename}")
                embedding = embedder.extract_video_embedding(video_path)
                
                # Normalize
                embedding = embedding.astype('float32').reshape(1, -1)
                batch_normalize_embeddings(embedding)
                embedding = embedding[0]
                
                # Store in database
                metadata = {
                    "num_frames": self.config.num_frames,
                    "category": "query"
                }
                
                if self.query_db.add_embedding(filename, video_path, embedding, metadata):
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                stats["errors"] += 1
        
        # Save database
        self.query_db.save()
        
        logger.info(f"Query database build complete: {stats}")
        return stats

    def build_query_database(self, query_video_dir: Union[str, Path], 
                           force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build query database from video directory.
        
        Args:
            query_video_dir: Directory containing query videos
            force_rebuild: Whether to rebuild from scratch
            
        Returns:
            Build statistics
        """
        from .embedder import CosmosVideoEmbedder
        from .faiss_backend import batch_normalize_embeddings
        
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
        
        stats = {"processed": 0, "cached": 0, "errors": 0}
        
        for video_path in video_files:
            filename = video_path.name
            
            try:
                # Check if already processed
                if not force_rebuild and self.query_db.get_embedding(filename) is not None:
                    stats["cached"] += 1
                    logger.debug(f"Already cached: {filename}")
                    continue
                
                # Extract embedding
                logger.info(f"Processing: {filename}")
                embedding = embedder.extract_video_embedding(video_path)
                
                # Normalize
                embedding = embedding.astype('float32').reshape(1, -1)
                batch_normalize_embeddings(embedding)
                embedding = embedding[0]
                
                # Store in database
                metadata = {
                    "num_frames": self.config.num_frames,
                    "category": "query"
                }
                
                if self.query_db.add_embedding(filename, video_path, embedding, metadata):
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                stats["errors"] += 1
        
        # Save database
        self.query_db.save()
        
        logger.info(f"Query database build complete: {stats}")
        return stats
    
    def get_query_embedding(self, filename: str) -> Optional[np.ndarray]:
        """Get query embedding by filename."""
        return self.query_db.get_embedding(filename)
    
    def list_available_query_videos(self) -> List[str]:
        """Get list of available query video filenames."""
        return self.query_db.list_videos()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query database statistics."""
        return self.query_db.get_statistics()
