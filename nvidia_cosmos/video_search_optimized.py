"""
Optimized Video Search Engine incorporating techniques from the official NVIDIA implementation.
Reference: https://huggingface.co/spaces/nvidia/Cosmos-Embed1/blob/main/src/streamlit_app.py
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
import time
from functools import wraps

from base import EmbeddingModel
from video_embedder import CosmosVideoEmbedder
from optimizations import (
    OptimizedVideoDatabase, FaissSearchStrategy, OptimizedEmbeddingCache,
    batch_normalize_embeddings
)
from config import VideoRetrievalConfig
from exceptions import VideoNotFoundError, SearchError, NoResultsError

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


class OptimizedVideoSearchEngine:
    """
    Optimized search engine using techniques from the official NVIDIA implementation:
    - FAISS for efficient similarity search
    - Embedding normalization using faiss.normalize_L2
    - Caching for repeated queries
    - Parquet format for efficient storage
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
        
        # Use optimized database with FAISS
        self.database = OptimizedVideoDatabase(
            self.config.database_path,
            self.config,
            use_faiss=True
        )
        
        # Use FAISS search strategy
        self.search_strategy = FaissSearchStrategy(use_gpu=use_gpu_faiss)
        
        # Initialize embedding cache
        self.embedding_cache = OptimizedEmbeddingCache(
            cache_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000
        )
        
        # Model caching (singleton pattern from official implementation)
        self._model_cache = {}
        
        logger.info(f"OptimizedVideoSearchEngine initialized (GPU FAISS: {use_gpu_faiss})")
    
    @time_it
    def build_database(self, video_directory: Union[str, Path], 
                      force_rebuild: bool = False,
                      save_format: str = "parquet"):
        """
        Build video embeddings database with optimizations.
        
        Args:
            video_directory: Directory containing video files
            force_rebuild: If True, rebuild from scratch
            save_format: Format to save database ("parquet" recommended)
        """
        video_dir = Path(video_directory)
        if not video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")
        
        # Try to load existing database
        if not force_rebuild:
            try:
                if save_format == "parquet":
                    self.database.load_from_parquet(
                        self.config.database_path + ".parquet"
                    )
                else:
                    self.database.load()
                logger.info("Loaded existing database")
                return
            except Exception as e:
                logger.info(f"Could not load existing database: {e}")
        
        # Get video files
        video_files = []
        for ext in self.config.supported_formats:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Check for new videos only
        if hasattr(self.database, 'metadata'):
            existing_paths = {m.get("video_path") for m in self.database.metadata}
            new_videos = [v for v in video_files if str(v) not in existing_paths]
        else:
            new_videos = video_files
        
        if new_videos:
            logger.info(f"Processing {len(new_videos)} new videos")
            
            # Extract embeddings with batch processing
            embeddings_data = self._extract_embeddings_optimized(new_videos)
            
            if embeddings_data:
                # Add to database
                self.database.add_embeddings(embeddings_data)
                
                # Save in specified format
                if save_format == "parquet":
                    self.database.save_as_parquet()
                else:
                    self.database.save()
                    
                logger.info(f"Database updated with {len(embeddings_data)} videos")
        else:
            logger.info("All videos already in database")
    
    def _extract_embeddings_optimized(self, video_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Extract embeddings with optimizations:
        - Batch processing
        - Caching
        - Normalization using FAISS
        """
        embeddings_data = []
        
        # Check cache first
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
        
        # Extract uncached embeddings
        if uncached_paths:
            new_embeddings = self.embedder.extract_video_embeddings_batch(
                uncached_paths,
                batch_size=self.config.batch_size
            )
            
            # Normalize embeddings using FAISS (as in official implementation)
            for emb_data in new_embeddings:
                embedding = emb_data["embedding"].astype('float32')
                embedding = embedding.reshape(1, -1)
                batch_normalize_embeddings(embedding)
                emb_data["embedding"] = embedding[0]
                
                # Cache the embedding
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
            # Check cache if enabled
            cache_key = str(query_path)
            query_embedding = None
            
            if use_cache:
                query_embedding = self.embedding_cache.get(cache_key)
                
            if query_embedding is None:
                # Extract embedding
                logger.info(f"Extracting embedding for: {query_path.name}")
                query_embedding = self.embedder.extract_video_embedding(query_path)
                
                # Normalize using FAISS
                query_embedding = query_embedding.astype('float32').reshape(1, -1)
                batch_normalize_embeddings(query_embedding)
                query_embedding = query_embedding[0]
                
                # Cache it
                if use_cache:
                    self.embedding_cache.put(cache_key, query_embedding)
            else:
                logger.info(f"Using cached embedding for: {query_path.name}")
            
            # Perform search
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
            # Extract text embedding (with no_grad as in official implementation)
            with torch.no_grad():
                text_embedding = self.embedder.extract_text_embedding(query_text)
                
            # Normalize
            text_embedding = text_embedding.astype('float32').reshape(1, -1)
            batch_normalize_embeddings(text_embedding)
            text_embedding = text_embedding[0]
            
            # Search
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
        # Ensure database is loaded
        if not hasattr(self.database, 'embedding_matrix') or self.database.embedding_matrix is None:
            # Try loading from parquet first
            try:
                self.database.load_from_parquet(self.config.database_path + ".parquet")
            except:
                self.database.load()
        
        # Use FAISS search
        results = self.search_strategy.search(
            query_embedding,
            self.database,
            top_k
        )
        
        if not results:
            raise NoResultsError("No results found")
        
        # Format results
        formatted_results = []
        for idx, similarity, metadata in results:
            # Apply threshold
            if similarity < self.config.similarity_threshold:
                continue
                
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata.get("video_path", ""),
                "video_name": metadata.get("video_name", ""),
                "similarity_score": similarity,
                "metadata": metadata
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
        
        # Get embedding
        query_embedding = self.database.embedding_matrix[video_idx]
        
        # Adjust k if ignoring self
        search_k = k + 1 if ignore_self else k
        
        # Search
        results = self.search_strategy.search(
            query_embedding,
            self.database,
            search_k
        )
        
        # Extract indices
        indices = [r[0] for r in results]
        
        # Remove self if needed
        if ignore_self and indices and indices[0] == video_idx:
            indices = indices[1:]
        
        return indices[:k]
    
    def clear_cache(self):
        """Clear all caches."""
        self.embedding_cache.clear()
        if hasattr(self.database, '_clear_cache'):
            self.database._clear_cache()
        logger.info("Caches cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self.database.get_statistics()
        
        # Add cache statistics
        stats.update({
            "cache_size": len(self.embedding_cache._cache),
            "cache_capacity": self.embedding_cache.cache_size,
            "using_gpu": torch.cuda.is_available() and self.config.device == "cuda",
            "search_backend": "FAISS" if self.database.use_faiss else "NumPy"
        })
        
        return stats
