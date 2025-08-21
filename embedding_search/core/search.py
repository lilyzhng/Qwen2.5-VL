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
    FaissSearchStrategy, EmbeddingCache,
    batch_normalize_embeddings
)
from .config import VideoRetrievalConfig
from .exceptions import VideoNotFoundError, SearchError, NoResultsError
from .database import UnifiedQueryManager, ParquetVectorDatabase

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
        Initialize search engine.
        
        Args:
            config: Configuration object
            embedder: Embedding model (uses CosmosVideoEmbedder if None)
            use_gpu_faiss: Whether to use GPU acceleration for FAISS
        """
        self.config = config or VideoRetrievalConfig()
        self.embedder = embedder or CosmosVideoEmbedder(self.config)
        
        self.project_root = Path(__file__).parent.parent
        
        db_path = self._resolve_path(self.config.main_embeddings_path)
        
        # Use ParquetVectorDatabase for main database to support thumbnails
        self.database = ParquetVectorDatabase(db_path, self.config)
        
        # FAISS provides faster similarity search with optional GPU acceleration
        self.search_strategy = FaissSearchStrategy(use_gpu=use_gpu_faiss)
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(
            cache_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000
        )
        
        self.query_manager = UnifiedQueryManager(self.config)
        
        self._model_cache = {}
        
        # Initialize feature visualizer for advanced analysis
        try:
            from .feature_visualizer import FeatureVisualizer
            self.feature_visualizer = FeatureVisualizer(self.embedder, self.config)
            logger.info("Feature visualizer initialized")
        except ImportError as e:
            logger.warning(f"Feature visualizer not available: {e}")
            self.feature_visualizer = None
        
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

        if hasattr(self.config, 'main_input_path') and self.config.main_input_path:
            index_path = self._resolve_path(self.config.main_input_path)
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
                # ParquetVectorDatabase loads automatically in __init__
                if hasattr(self.database, 'df') and self.database.df is not None and len(self.database.df) > 0:
                    logger.info("Loaded existing database")
                    return
            except Exception as e:
                logger.info(f"Could not load existing database: {e}")
        
        video_files = self._get_video_files(video_directory)
        
        if not video_files:
            logger.warning(f"No video files found")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Load slice_id mapping if available
        path_to_slice_id = {}
        if hasattr(self.config, 'main_input_path') and self.config.main_input_path:
            index_path = self._resolve_path(self.config.main_input_path)
            if index_path and index_path.exists():
                try:
                    if index_path.suffix.lower() == '.parquet':
                        df = pd.read_parquet(index_path)
                    else:
                        df = pd.read_csv(index_path)
                    
                    if 'slice_id' in df.columns and 'sensor_video_file' in df.columns:
                        for _, row in df.iterrows():
                            path_to_slice_id[Path(row['sensor_video_file'])] = row['slice_id']
                        logger.info(f"Loaded slice_id mapping for {len(path_to_slice_id)} videos")
                except Exception as e:
                    logger.warning(f"Error loading slice_id mapping: {e}")
        
        # Check existing videos in ParquetVectorDatabase using only slice_id as key
        existing_slice_ids = set()
        if hasattr(self.database, 'df') and self.database.df is not None and len(self.database.df) > 0:
            existing_slice_ids = set(self.database.df['slice_id'].tolist())
        
        # Filter videos that are not already in the database by slice_id
        new_videos = []
        for v in video_files:
            slice_id = path_to_slice_id.get(v)
            if slice_id is None:
                logger.warning(f"No slice_id found for video path: {v}")
                continue
                
            # Skip if the slice_id is already in the database
            if slice_id in existing_slice_ids:
                continue
                
            new_videos.append(v)
        
        if new_videos:
            logger.info(f"Processing {len(new_videos)} new videos")
            
            embeddings_data = self._extract_embeddings(new_videos)
            
            if embeddings_data:
                # Add embeddings one by one to ParquetVectorDatabase
                for emb_data in embeddings_data:
                    video_path = Path(emb_data["video_path"])
                    # Use slice_id from mapping - it must exist in the parquet file
                    slice_id = path_to_slice_id.get(video_path)
                    if slice_id is None:
                        logger.error(f"No slice_id found for video path: {video_path}")
                        continue
                    embedding = emb_data["embedding"]
                    metadata = {
                        "num_frames": self.config.num_frames,
                        "category": "main",
                        **{k: v for k, v in emb_data.items() if k not in ["video_path", "embedding"]}
                    }
                    self.database.add_embedding(slice_id, video_path, embedding, metadata)
                
                # Save the database
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
        if hasattr(self.config, 'query_input_path') and self.config.query_input_path:
            query_input_path = self._resolve_path(self.config.query_input_path)
            if query_input_path.exists():
                logger.info(f"Loading query videos from file path list: {query_input_path}")
                return self.query_manager.build_query_database_from_file_list(query_input_path, force_rebuild)
        
        if query_video_directory:
            query_dir = self._resolve_path(query_video_directory)
        else:
            raise ValueError("No query video directory provided and no query_input_path configured")
        
        if not query_dir.exists():
            logger.warning(f"Query video directory not found: {query_dir}")
            return {"processed": 0, "cached": 0, "errors": 1}
        
        logger.info(f"Building query database from: {query_dir}")
        return self.query_manager.build_query_database(query_dir, force_rebuild)

    def search_by_filename(self, slice_id: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Search using pre-computed query video embedding by slice_id.
        
        Args:
            slice_id: Unique identifier for the slice (e.g., "0859975219")
            top_k: Number of results
            
        Returns:
            Search results
        """
        top_k = top_k or self.config.default_top_k
        
        try:
            query_embedding = self.query_manager.get_query_embedding(slice_id)
            
            if query_embedding is not None:
                logger.info(f"Using pre-computed embedding for slice_id: {slice_id}")
                try:
                    return self._search_by_embedding(query_embedding, top_k)
                except Exception as e:
                    logger.error(f"Error during pre-computed search: {e}")
                    logger.info("Falling back to real-time computation...")
            else:
                logger.warning(f"No pre-computed embedding found for slice_id: {slice_id}")
        except Exception as e:
            logger.error(f"Error accessing query cache: {e}")
        
        logger.info(f"Falling back to real-time processing for slice_id: {slice_id}")
        
        if hasattr(self.config, 'query_input_path') and self.config.query_input_path:
            query_input_path = self._resolve_path(self.config.query_input_path)
            if query_input_path.exists():
                if query_input_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(query_input_path)
                else:
                    df = pd.read_csv(query_input_path)
                
                matching_rows = df[df['slice_id'] == slice_id]
                if len(matching_rows) > 0:
                    query_path = Path(matching_rows.iloc[0]['sensor_video_file'])
                    if query_path.exists():
                        return self.search_by_video(query_path, top_k)
        
        raise VideoNotFoundError(f"Query video not found for slice_id: {slice_id}. Please ensure it's listed in query_input_path.")

    def _extract_embeddings(self, video_paths: List[Path]) -> List[Dict[str, Any]]:
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
        Search videos using text query.
        
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
    
    @time_it
    def search_by_joint(self, query_text: str, query_video_slice_id: str,
                       alpha: float = 0.5, top_k: Optional[int] = None) -> List[Dict]:
        """
        Joint search combining text and video embeddings with alpha weighting.
        Uses precomputed video embeddings by default.
        
        Formula: q = normalize(alpha * E_text + (1 - alpha) * E_video)
        
        Args:
            query_text: Text query
            query_video_slice_id: Slice ID of query video (uses precomputed embedding)
            alpha: Weight for text embedding (0.0 = video only, 1.0 = text only)
            top_k: Number of results
            
        Returns:
            Search results
        """
        if not query_text or not query_text.strip():
            raise ValueError("Text query cannot be empty")
        
        if not query_video_slice_id or not query_video_slice_id.strip():
            raise ValueError("Video slice_id cannot be empty")
        
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0.0 and 1.0")
        
        top_k = top_k or self.config.default_top_k
        
        try:
            # Extract text embedding
            with torch.no_grad():
                text_embedding = self.embedder.extract_text_embedding(query_text)
            
            text_embedding = text_embedding.astype('float32').reshape(1, -1)
            batch_normalize_embeddings(text_embedding)
            text_embedding = text_embedding[0]
            
            # Get precomputed video embedding
            video_embedding = self.query_manager.get_query_embedding(query_video_slice_id)
            if video_embedding is None:
                raise VideoNotFoundError(f"No precomputed embedding found for slice_id: {query_video_slice_id}. Please build query database first.")
            
            logger.info(f"Using precomputed embedding for joint search: {query_video_slice_id}")
            
            # Combine embeddings with alpha weighting
            joint_embedding = alpha * text_embedding + (1 - alpha) * video_embedding
            
            # Normalize the combined embedding to unit vector
            joint_embedding = joint_embedding / np.linalg.norm(joint_embedding)
            
            logger.info(f"Joint search: alpha={alpha:.2f}, text_weight={alpha:.2f}, video_weight={1-alpha:.2f}")
            
            return self._search_by_embedding(joint_embedding, top_k)
            
        except Exception as e:
            raise SearchError(f"Joint search failed: {str(e)}")
    
    def _calculate_similarity_with_logit_scale(self, query_emb: np.ndarray, db_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarity using model's logit scale for more accurate scoring.
        
        Args:
            query_emb: Query embedding (1D)
            db_embeddings: Database embeddings (2D: num_videos x embedding_dim)
            
        Returns:
            Similarity scores
        """
        try:
            if hasattr(self.embedder.model, 'logit_scale'):
                # Convert to torch tensors
                query_tensor = torch.from_numpy(query_emb).to(self.embedder.device, dtype=self.embedder.dtype)
                db_tensor = torch.from_numpy(db_embeddings).to(self.embedder.device, dtype=self.embedder.dtype)
                
                with torch.no_grad():
                    # Use logit scale for better similarity calculation (as in official demo)
                    logit_scale = self.embedder.model.logit_scale.exp()
                    similarities = torch.softmax(logit_scale * query_tensor @ db_tensor.T, dim=-1)
                    return similarities.cpu().numpy()
            else:
                logger.debug("Model has no logit_scale, using standard cosine similarity")
                # Fallback to standard cosine similarity
                return query_emb @ db_embeddings.T
        except Exception as e:
            logger.warning(f"Failed to use logit_scale similarity, falling back to cosine: {e}")
            return query_emb @ db_embeddings.T

    def _search_by_embedding(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Internal search method using FAISS with enhanced similarity calculation.
        
        Args:
            query_embedding: Normalized query embedding
            top_k: Number of results
            
        Returns:
            Formatted search results
        """
        # Ensure database is loaded
        if self.database.embedding_matrix is None:
            logger.warning("No embeddings in database for search")
        
        # Try using enhanced similarity calculation first
        try:
            if hasattr(self.embedder.model, 'logit_scale') and self.database.embedding_matrix is not None:
                # Use logit scale similarity for better results
                similarities = self._calculate_similarity_with_logit_scale(
                    query_embedding, 
                    self.database.embedding_matrix
                )
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for i, idx in enumerate(top_indices):
                    similarity = float(similarities[idx])
                    metadata = self.database.metadata[idx] if hasattr(self.database, 'metadata') else {}
                    results.append((idx, similarity, metadata))
            else:
                # Fallback to FAISS search
                results = self.search_strategy.search(
                    query_embedding,
                    self.database,
                    top_k
                )
        except Exception as e:
            logger.warning(f"Enhanced similarity failed, using FAISS: {e}")
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
                "slice_id": metadata.get("slice_id", ""),
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
        Get k nearest neighbors for a video by index.
        
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
        # Get stats from ParquetVectorDatabase
        stats = self.database.get_statistics()
        
        # Map ParquetVectorDatabase stats to expected format
        if 'total_embeddings' in stats:
            stats['num_videos'] = stats.pop('total_embeddings', 0)
        
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
            
            slice_ids = []
            database_size_mb = 0
            
            # Get data from ParquetVectorDatabase
            if self.database.df is not None:
                slice_ids = self.database.df['slice_id'].tolist() if 'slice_id' in self.database.df.columns else []
                # Calculate database size from parquet file
                if self.database.database_path.exists():
                    database_size_mb = self.database.database_path.stat().st_size / (1024 * 1024)
            
            info = {
                'num_videos': stats.get('num_videos', 0),
                'embedding_dim': stats.get('embedding_dim', 0),
                'database_size_mb': database_size_mb,
                'slice_ids': slice_ids,
                'version': '2.0',
                'search_backend': stats.get('search_backend', 'Unknown'),
                'cache_size': stats.get('cache_size', 0),
                'using_gpu': stats.get('using_gpu', False),
                'has_feature_visualizer': self.feature_visualizer is not None,
                'supports_logit_scale': hasattr(self.embedder.model, 'logit_scale'),
                'precision': str(self.embedder.dtype)
            }
            
            return info
            
        except Exception as e:
            return {
                'num_videos': 0,
                'embedding_dim': 0,
                'database_size_mb': 0,
                'slice_ids': [],
                'version': '2.0',
                'search_backend': 'Unknown',
                'cache_size': 0,
                'using_gpu': False,
                'has_feature_visualizer': False,
                'supports_logit_scale': False,
                'precision': 'unknown',
                'error': str(e)
            }
    
    def create_pca_visualization(self, video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Create PCA feature visualization for a video (from official demo).
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save visualization
            
        Returns:
            Tuple of (original_frames, pca_frames) or None if feature visualizer not available
        """
        if self.feature_visualizer is None:
            logger.warning("Feature visualizer not available")
            return None
        
        try:
            return self.feature_visualizer.create_pca_visualization(video_path, output_path)
        except Exception as e:
            logger.error(f"Failed to create PCA visualization: {e}")
            return None
    
    def analyze_temporal_stability(self, video_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal stability of video features.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Temporal stability metrics or None if not available
        """
        if self.feature_visualizer is None:
            logger.warning("Feature visualizer not available")
            return None
        
        try:
            return self.feature_visualizer.analyze_temporal_stability(video_path)
        except Exception as e:
            logger.error(f"Failed to analyze temporal stability: {e}")
            return None
