"""
FAISS-based search backend provides similarity search, caching, and database optimizations using FAISS.
"""

import numpy as np
import faiss
import torch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
from functools import lru_cache
import pandas as pd

from .base import SearchStrategy
from .config import VideoRetrievalConfig
from typing import Union, Any

logger = logging.getLogger(__name__)


class FaissSearchStrategy(SearchStrategy):
    """
    Optimized search using FAISS library as shown in the official implementation.
    FAISS provides highly optimized similarity search.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize FAISS search strategy.
        
        Args:
            use_gpu: Whether to use GPU acceleration for FAISS
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index = None
        self.dimension = None
        
    def build_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings (n_samples, n_features)
        """
        embeddings = embeddings.astype('float32')
        
        # Normalize embeddings as done in the official implementation
        faiss.normalize_L2(embeddings)
        
        self.dimension = embeddings.shape[1]
        
        # Create FAISS index - using IndexFlatIP for inner product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if self.use_gpu:
            # Move index to GPU for faster search
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.index.add(embeddings)
        logger.info(f"Built FAISS index with {len(embeddings)} embeddings")
        
    def search(self, 
               query_embedding: np.ndarray, 
               database: Any, 
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, Dict]]:
        """
        Perform optimized search using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            database: Database backend (used for metadata)
            top_k: Number of results to return
            filters: Optional filters (not implemented in basic version)
            
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        if self.index is None:
            # Build index if not already built
            if hasattr(database, 'embedding_matrix') and database.embedding_matrix is not None:
                self.build_index(database.embedding_matrix)
            else:
                logger.warning("No embeddings available for FAISS index")
                return []
        
        # Prepare query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, top_k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            similarity = float(distances[0][i])
            
            if hasattr(database, 'metadata') and idx < len(database.metadata):
                metadata = database.metadata[idx]
            else:
                metadata = {}
                
            results.append((idx, similarity, metadata))
            
        return results


# VideoDatabase class removed - using ParquetVectorDatabase instead


class EmbeddingCache:
    """
    Caching mechanism for embeddings inspired by Streamlit's caching in the official implementation.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.cache_size = cache_size
        self._cache = {}
        self._access_count = {}
        
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None
        
    def put(self, key: str, embedding: np.ndarray):
        """Put embedding in cache with LRU eviction."""
        if len(self._cache) >= self.cache_size:
            # Evict least recently used
            lru_key = min(self._access_count.keys(), key=lambda k: self._access_count[k])
            del self._cache[lru_key]
            del self._access_count[lru_key]
            
        self._cache[key] = embedding
        self._access_count[key] = 1
        
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()


# Additional optimization utilities
def batch_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Batch normalize embeddings using FAISS as in the official implementation.
    This is more efficient than normalizing one by one.
    """
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    return embeddings


def create_index(embeddings: np.ndarray, index_type: str = "Flat") -> faiss.Index:
    """
    Create an optimized FAISS index based on the number of embeddings.
    
    Args:
        embeddings: Embeddings to index
        index_type: Type of index ("Flat", "IVF", "HNSW")
        
    Returns:
        FAISS index
    """
    n_samples, dimension = embeddings.shape
    embeddings = embeddings.astype('float32')
    
    if index_type == "Flat" or n_samples < 10000:
        # Use flat index for small datasets (exact search)
        index = faiss.IndexFlatIP(dimension)
    elif index_type == "IVF" or n_samples < 100000:
        # Use IVF for medium datasets (approximate search)
        nlist = int(np.sqrt(n_samples))
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
    else:
        # Use HNSW for large datasets (approximate search with good recall)
        index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        
    index.add(embeddings)
    return index
