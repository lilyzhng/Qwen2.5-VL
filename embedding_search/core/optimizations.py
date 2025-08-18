"""
Optimizations inspired by the official NVIDIA Cosmos-Embed1 implementation.
Reference: https://huggingface.co/spaces/nvidia/Cosmos-Embed1/blob/main/src/streamlit_app.py
"""

import numpy as np
import faiss
import torch
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
from functools import lru_cache
import pandas as pd

from .base import DatabaseBackend, SearchStrategy
from .config import VideoRetrievalConfig
from typing import Union

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
               database: DatabaseBackend, 
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


class OptimizedVideoDatabase(DatabaseBackend):
    """
    Optimized database with caching and efficient storage inspired by the official implementation.
    """
    
    def __init__(self, database_path: Union[str, Path] = "data/video_embeddings",
                 config: Optional[VideoRetrievalConfig] = None,
                 use_faiss: bool = True):
        """
        Initialize optimized database.
        
        Args:
            database_path: Base path for database files
            config: Configuration object
            use_faiss: Whether to use FAISS for search
        """
        self.database_path = Path(database_path)
        self.config = config or VideoRetrievalConfig()
        self.use_faiss = use_faiss
        
        # Storage
        self.embeddings = []
        self.metadata = []
        self.embedding_matrix = None
        
        # FAISS search strategy
        self.faiss_search = FaissSearchStrategy() if use_faiss else None
        
        # Caching for frequently accessed data
        self._cache = {}
        
    def add_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """Add embeddings with automatic index updates."""
        if not embeddings_data:
            return
            
        for data in embeddings_data:
            # Ensure normalization as in official implementation
            embedding = data["embedding"].astype('float32')
            faiss.normalize_L2(embedding.reshape(1, -1))
            
            self.embeddings.append(embedding)
            self.metadata.append({
                "video_path": str(data["video_path"]),
                "video_name": Path(data["video_path"]).name,
                **{k: v for k, v in data.items() if k not in ["embedding", "video_path"]}
            })
        
        self._update_embedding_matrix()
        self._clear_cache()
        
    def _update_embedding_matrix(self):
        """Update embedding matrix and rebuild FAISS index if needed."""
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings).astype('float32')
            
            # Rebuild FAISS index
            if self.faiss_search:
                self.faiss_search.build_index(self.embedding_matrix)
        else:
            self.embedding_matrix = None
            
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Compute similarity using FAISS if available, otherwise fallback to numpy."""
        if self.use_faiss and self.faiss_search:
            return self.faiss_search.search(query_embedding, self, top_k)
        else:
            # Fallback to numpy implementation
            return self._numpy_similarity(query_embedding, top_k)
            
    def _numpy_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Numpy-based similarity computation as fallback."""
        if self.embedding_matrix is None:
            return []
            
        # Normalize query
        query_norm = query_embedding.astype('float32')
        faiss.normalize_L2(query_norm.reshape(1, -1))
        
        # Compute similarities
        similarities = np.dot(self.embedding_matrix, query_norm)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.metadata[idx].copy()
            ))
            
        return results
    
    def save_as_parquet(self, path: Optional[Union[str, Path]] = None):
        """
        Save database in Parquet format as used in the official implementation.
        This format is more efficient than JSON for large datasets.
        """
        save_path = Path(path) if path else self.database_path.with_suffix('.parquet')
        
        if not self.embeddings:
            logger.warning("No embeddings to save")
            return
            
        # Create DataFrame
        data = []
        for emb, meta in zip(self.embeddings, self.metadata):
            row = meta.copy()
            row['embedding'] = emb.tolist()
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_parquet(save_path, index=False)
        
        logger.info(f"Database saved to {save_path} in Parquet format")
        
    def load_from_parquet(self, path: Union[str, Path]):
        """
        Load database from Parquet format as used in the official implementation.
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {load_path}")
            
        df = pd.read_parquet(load_path)
        
        # Extract embeddings
        self.embeddings = []
        self.metadata = []
        
        for _, row in df.iterrows():
            embedding = np.array(row['embedding'], dtype='float32')
            self.embeddings.append(embedding)
            
            # Extract metadata (all columns except embedding)
            meta = row.to_dict()
            del meta['embedding']
            self.metadata.append(meta)
            
        self._update_embedding_matrix()
        logger.info(f"Loaded {len(self.embeddings)} embeddings from {load_path}")
    
    @lru_cache(maxsize=128)
    def get_cached_similarity(self, query_hash: str, top_k: int) -> List[Tuple[int, float, Dict]]:
        """Cache similarity results for repeated queries."""
        # This would be called with a hash of the query embedding
        # Implementation would depend on how you want to handle caching
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics with additional FAISS information."""
        stats = {
            "num_videos": len(self.embeddings),
            "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else 0,
            "using_faiss": self.use_faiss,
            "index_type": type(self.faiss_search.index).__name__ if self.faiss_search and self.faiss_search.index else "None"
        }
        return stats
    
    def clear(self):
        """Clear database and caches."""
        self.embeddings = []
        self.metadata = []
        self.embedding_matrix = None
        self._cache.clear()
        if self.faiss_search:
            self.faiss_search.index = None
            
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save using Parquet format for better performance."""
        self.save_as_parquet(path)
        
    def load(self, path: Optional[Union[str, Path]] = None):
        """Load from Parquet if available, otherwise try other formats."""
        load_path = Path(path) if path else self.database_path
        
        # Try Parquet first
        parquet_path = load_path.with_suffix('.parquet')
        if parquet_path.exists():
            self.load_from_parquet(parquet_path)
            return
            
        # Fallback to other formats
        raise NotImplementedError("Only Parquet format is supported in optimized database")
        
    def remove_video(self, video_path: str) -> bool:
        """Remove video with index rebuild."""
        for i, meta in enumerate(self.metadata):
            if meta.get("video_path") == video_path:
                del self.embeddings[i]
                del self.metadata[i]
                self._update_embedding_matrix()
                self._clear_cache()
                return True
        return False
        
    def _clear_cache(self):
        """Clear internal caches."""
        self._cache.clear()
        if hasattr(self.get_cached_similarity, 'cache_clear'):
            self.get_cached_similarity.cache_clear()


class OptimizedEmbeddingCache:
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


def create_optimized_index(embeddings: np.ndarray, index_type: str = "Flat") -> faiss.Index:
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
