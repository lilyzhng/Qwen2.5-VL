"""
Video Search Engine.
"""

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
import time
from functools import wraps

from .base import EmbeddingModel
from .embedder import CosmosVideoEmbedder
from .faiss_backend import (
    VideoDatabase, FaissSearchStrategy, EmbeddingCache,
    batch_normalize_embeddings
)
from .config import VideoRetrievalConfig
from .exceptions import VideoNotFoundError, SearchError, NoResultsError
from .database import UnifiedQueryManager

logger = logging.getLogger(__name__)


def time_it(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


class VideoSearchEngine:
    """
    a. FAISS for efficient similarity search
    b. Embedding normalization using faiss.normalize_L2
    c. Caching for repeated queries
    """
    
    def __init__(self,
                 config: Optional[VideoRetrievalConfig] = None,
                 embedder: Optional[EmbeddingModel] = None,
                 use_gpu_faiss: bool = False):
        """
        Initialize optimized search engine.
        
        Args:
            config: Configuration object
            embedder: Embedding model (uses CosmosVideoEmbedder if None)
            use_gpu_faiss: Whether to use GPU acceleration for FAISS
        """
        self.config = config or VideoRetrievalConfig()
        self.embedder = embedder or CosmosVideoEmbedder(self.config)
        
        self.project_root = Path(__file__).parent.parent
        
        db_path = self._resolve_path(self.config.main_embeddings_path)
        
        self.database = VideoDatabase(
            db_path,
            self.config,
            use_faiss=True
        )
        
        # FAISS provides faster similarity search with optional GPU acceleration
        self.search_strategy = FaissSearchStrategy(use_gpu=use_gpu_faiss)
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(
            cache_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000
        )
        
        self.query_manager = UnifiedQueryManager(self.config)
        
        self._model_cache = {}
        
        logger.info(f"VideoSearchEngine initialized (GPU FAISS: {use_gpu_faiss})")

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to project root if not absolute."""
        path = Path(path)
        if not path.is_absolute():
            return self.project_root / path
        return path

    def _get_video_files(self, video_directory: Union[str, Path]) -> List[Path]:
        """
        Get video files either from CSV file or directory scanning.
        
        Args:
            video_directory: Directory containing video files (used as fallback)
            
        Returns:
            List of video file paths
        """
        video_files = []

        if hasattr(self.config, 'main_file_path') and self.config.main_file_path:
            index_path = self._resolve_path(self.config.main_file_path)
        else:
            index_path = None
            
        if index_path and index_path.exists():
            try:
                logger.info(f"Loading video files from index: {index_path}")
                if index_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(index_path)
                else:
                    df = pd.read_csv(index_path)
                
                if 'sensor_video_file' not in df.columns:
                    logger.warning(f"Column 'sensor_video_file' not found in index file. Available columns: {list(df.columns)}")
                    logger.info("Falling back to directory scanning")
                else:
                    video_paths = df['sensor_video_file'].drop_duplicates().tolist()
                    video_files = [Path(path) for path in video_paths if Path(path).exists()]
                    
                    if video_files:
                        logger.info(f"Loaded {len(video_files)} video files from index")
                        return video_files
                    else:
                        logger.warning("No valid video files found in index, falling back to directory scanning")
            except Exception as e:
                logger.warning(f"Error reading index file: {e}")
                logger.info("Falling back to directory scanning")
        
        video_dir = Path(video_directory)
        if not video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")
        
        logger.info(f"Scanning directory for video files: {video_dir}")
        for ext in self.config.supported_formats:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
        
        return video_files

    @time_it
    def build_database(self, video_directory: Union[str, Path], 
                      force_rebuild: bool = False,
                      save_format: str = "parquet"):
        """
        Build video embeddings database.
        
        Args:
            video_directory: Directory containing video files
            force_rebuild: If True, rebuild from scratch
            save_format: Format to save database ("parquet")
        """
        if not force_rebuild:
            try:
                if save_format == "parquet":
                    db_path = self._resolve_path(self.config.main_embeddings_path)
                    self.database.load_from_parquet(db_path)
                else:
                    self.database.load()
                logger.info("Loaded existing database")
                return
            except Exception as e:
                logger.info(f"Could not load existing database: {e}")
        
        video_files = self._get_video_files(video_directory)
        
        if not video_files:
            logger.warning(f"No video files found")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        if hasattr(self.database, 'metadata'):
            existing_paths = {m.get("video_path") for m in self.database.metadata}
            new_videos = [v for v in video_files if str(v) not in existing_paths]
        else:
            new_videos = video_files
        
        if new_videos:
            logger.info(f"Processing {len(new_videos)} new videos")
            
            embeddings_data = self._extract_embeddings_optimized(new_videos)
            
            if embeddings_data:
                self.database.add_embeddings(embeddings_data)
                
                if save_format == "parquet":
                    self.database.save_as_parquet()
                else:
                    self.database.save()
                    
                logger.info(f"Database updated with {len(embeddings_data)} videos")
        else:
            logger.info("All videos already in database")

    @time_it
    def build_query_database(self, query_video_directory: Union[str, Path] = None,
                           force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Build database for query videos with pre-computed embeddings.
        
        Args:
            query_video_directory: Directory containing query videos (uses config default if None)
            force_rebuild: If True, rebuild from scratch
            
        Returns:
            Build statistics
        """
        if hasattr(self.config, 'query_file_path') and self.config.query_file_path:
            query_file_path = self._resolve_path(self.config.query_file_path)
            if query_file_path.exists():
                logger.info(f"Loading query videos from file path list: {query_file_path}")
                return self.query_manager.build_query_database_from_file_list(query_file_path, force_rebuild)
        
        if query_video_directory:
            query_dir = self._resolve_path(query_video_directory)
        else:
            raise ValueError("No query video directory provided and no query_file_path configured")
        
        if not query_dir.exists():
            logger.warning(f"Query video directory not found: {query_dir}")
            return {"processed": 0, "cached": 0, "errors": 1}
        
        logger.info(f"Building query database from: {query_dir}")
        return self.query_manager.build_query_database(query_dir, force_rebuild)

    def search_by_filename(self, filename: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Search using pre-computed query video embedding by filename.
        
        Args:
            filename: Query video filename (e.g., "car2cyclist_2.mp4")
            top_k: Number of results
            
        Returns:
            Search results
        """
        top_k = top_k or self.config.default_top_k
        
        try:
            query_embedding = self.query_manager.get_query_embedding(filename)
            
            if query_embedding is not None:
                logger.info(f"Using pre-computed embedding for: {filename}")
                try:
                    return self._search_by_embedding(query_embedding, top_k)
                except Exception as e:
                    logger.error(f"Error during pre-computed search: {e}")
                    logger.info("Falling back to real-time computation...")
            else:
                logger.warning(f"No pre-computed embedding found for {filename}")
        except Exception as e:
            logger.error(f"Error accessing query cache: {e}")
        
        logger.info(f"Falling back to real-time processing for: {filename}")
        
        if hasattr(self.config, 'query_file_path') and self.config.query_file_path:
            query_file_path = self._resolve_path(self.config.query_file_path)
            if query_file_path.exists():
                if query_file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(query_file_path)
                else:
                    df = pd.read_csv(query_file_path)
                
                matching_rows = df[df['video_name'] == filename]
                if len(matching_rows) > 0:
                    query_path = Path(matching_rows.iloc[0]['sensor_video_file'])
                    if query_path.exists():
                        return self.search_by_video(query_path, top_k)
        
        raise VideoNotFoundError(f"Query video not found: {filename}. Please ensure it's listed in query_file_path.")

    def _extract_embeddings_optimized(self, video_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract embeddings with optimizations:
        - Batch processing
        - Caching
        - Normalization using FAISS
        """
        embeddings_data = []
        
        uncached_paths = []
        for path in video_paths:
            cache_key = str(path)
            cached_embedding = self.embedding_cache.get(cache_key)
            
            if cached_embedding is not None:
                embeddings_data.append({
                    "video_path": str(path),
                    "embedding": cached_embedding,
                    "embedding_dim": cached_embedding.shape[0],
                    "from_cache": True
                })
            else:
                uncached_paths.append(path)
        
        logger.info(f"Found {len(embeddings_data)} cached embeddings")
        
        if uncached_paths:
            new_embeddings = self.embedder.extract_video_embeddings_batch(
                uncached_paths,
                batch_size=self.config.batch_size
            )
            
            for emb_data in new_embeddings:
                embedding = emb_data["embedding"].astype('float32')
                embedding = embedding.reshape(1, -1)
                batch_normalize_embeddings(embedding)
                emb_data["embedding"] = embedding[0]
                
                self.embedding_cache.put(emb_data["video_path"], emb_data["embedding"])
            
            embeddings_data.extend(new_embeddings)
        
        return embeddings_data

    @time_it
    def search_by_video(self, query_video_path: Union[str, Path],
                       top_k: Optional[int] = None,
                       use_cache: bool = True) -> List[Dict]:
        """
        Search for similar videos with caching support.
        
        Args:
            query_video_path: Path to query video
            top_k: Number of results
            use_cache: Whether to use cached embeddings
            
        Returns:
            Search results
        """
        query_path = Path(query_video_path)
        if not query_path.exists():
            raise VideoNotFoundError(f"Query video not found: {query_path}")
        
        top_k = top_k or self.config.default_top_k
        
        try:
            cache_key = str(query_path)
            query_embedding = None
            
            if use_cache:
                query_embedding = self.embedding_cache.get(cache_key)
                
            if query_embedding is None:
                logger.info(f"Extracting embedding for: {query_path.name}")
                query_embedding = self.embedder.extract_video_embedding(query_path)
                
                query_embedding = query_embedding.astype('float32').reshape(1, -1)
                batch_normalize_embeddings(query_embedding)
                query_embedding = query_embedding[0]
                
                if use_cache:
                    self.embedding_cache.put(cache_key, query_embedding)
            else:
                logger.info(f"Using cached embedding for: {query_path.name}")
            
            return self._search_by_embedding(query_embedding, top_k)
            
        except Exception as e:
            raise SearchError(f"Search failed: {str(e)}")
    
    @time_it
    def search_by_text(self, query_text: str,
                      top_k: Optional[int] = None) -> List[Dict]:
        """
        Search videos using text query (following official implementation pattern).
        
        Args:
            query_text: Text query
            top_k: Number of results
            
        Returns:
            Search results
        """
        if not query_text or not query_text.strip():
            raise ValueError("Text query cannot be empty")
        
        top_k = top_k or self.config.default_top_k
        
        try:
            with torch.no_grad():
                text_embedding = self.embedder.extract_text_embedding(query_text)
                
            text_embedding = text_embedding.astype('float32').reshape(1, -1)
            batch_normalize_embeddings(text_embedding)
            text_embedding = text_embedding[0]
            
            return self._search_by_embedding(text_embedding, top_k)
            
        except Exception as e:
            raise SearchError(f"Text search failed: {str(e)}")
    
    def _search_by_embedding(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Internal search method using FAISS.
        
        Args:
            query_embedding: Normalized query embedding
            top_k: Number of results
            
        Returns:
            Formatted search results
        """
        if not hasattr(self.database, 'embedding_matrix') or self.database.embedding_matrix is None:
            db_path = self._resolve_path(self.config.main_embeddings_path)
            self.database.load_from_parquet(db_path)
        
        results = self.search_strategy.search(
            query_embedding,
            self.database,
            top_k
        )
        
        if not results:
            raise NoResultsError("No results found")
        
        formatted_results = []
        for idx, similarity, metadata in results:
            if similarity < self.config.similarity_threshold:
                continue
                
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata.get("video_path", ""),
                "video_name": metadata.get("video_name", ""),
                "similarity_score": similarity,
                "metadata": metadata,
                "thumbnail": metadata.get("thumbnail", ""),
                "thumbnail_size": metadata.get("thumbnail_size", (0, 0))
            }
            formatted_results.append(result)
        
        if not formatted_results:
            raise NoResultsError(f"No results above threshold {self.config.similarity_threshold}")
        
        return formatted_results
    
    def get_neighbors(self, video_idx: int, k: int = 5, ignore_self: bool = True) -> List[int]:
        """
        Get k nearest neighbors for a video by index (as in official implementation).
        
        Args:
            video_idx: Index of video in database
            k: Number of neighbors
            ignore_self: Whether to exclude the query video itself
            
        Returns:
            List of neighbor indices
        """
        if not hasattr(self.database, 'embedding_matrix') or video_idx >= len(self.database.embedding_matrix):
            return []
        
        query_embedding = self.database.embedding_matrix[video_idx]
        
        search_k = k + 1 if ignore_self else k
        
        results = self.search_strategy.search(
            query_embedding,
            self.database,
            search_k
        )
        
        indices = [r[0] for r in results]
        
        if ignore_self and indices and indices[0] == video_idx:
            indices = indices[1:]
        
        return indices[:k]
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        if hasattr(self.database, '_clear_cache'):
            self.database._clear_cache()
        logger.info("Caches cleared")
    
    def get_query_videos_list(self) -> List[str]:
        """Get list of available query video filenames."""
        return self.query_manager.list_available_query_videos()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.database.get_statistics()
        
        stats.update({
            "cache_size": len(self.embedding_cache._cache),
            "cache_capacity": self.embedding_cache.cache_size,
            "using_gpu": torch.cuda.is_available() and self.config.device == "cuda",
            "search_backend": "FAISS" if self.database.use_faiss else "NumPy"
        })
        
        query_stats = self.query_manager.get_statistics()
        stats["query_database"] = query_stats
        
        return stats

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information in the format expected by main.py."""
        try:
            stats = self.get_statistics()
            
            video_names = []
            if hasattr(self.database, 'metadata') and self.database.metadata:
                video_names = [meta.get('video_name', '') for meta in self.database.metadata]
            
            database_size_mb = 0
            if hasattr(self.database, 'embedding_matrix') and self.database.embedding_matrix is not None:
                matrix_size = self.database.embedding_matrix.nbytes / (1024 * 1024)
                metadata_size = len(str(self.database.metadata)) / (1024 * 1024)
                database_size_mb = matrix_size + metadata_size
            
            info = {
                'num_videos': stats.get('num_videos', 0),
                'embedding_dim': stats.get('embedding_dim', 0),
                'database_size_mb': database_size_mb,
                'video_names': video_names,
                'version': '2.0',
                'search_backend': stats.get('search_backend', 'Unknown'),
                'cache_size': stats.get('cache_size', 0),
                'using_gpu': stats.get('using_gpu', False)
            }
            
            return info
            
        except Exception as e:
            return {
                'num_videos': 0,
                'embedding_dim': 0,
                'database_size_mb': 0,
                'video_names': [],
                'version': '2.0',
                'search_backend': 'Unknown',
                'cache_size': 0,
                'using_gpu': False,
                'error': str(e)
            }
