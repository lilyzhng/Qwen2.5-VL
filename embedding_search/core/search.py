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
from .database import ParquetVectorDatabase

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
        
        # Use unified embeddings path
        db_path = self._resolve_path(self.config.embeddings_path)
        
        # Use ParquetVectorDatabase for unified database to support thumbnails
        self.database = ParquetVectorDatabase(db_path, self.config)
        
        # FAISS provides faster similarity search with optional GPU acceleration
        self.search_strategy = FaissSearchStrategy(use_gpu=use_gpu_faiss)
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache(
            cache_size=self.config.cache_size if hasattr(self.config, 'cache_size') else 1000
        )
        
        # No need for separate query manager with unified system
        
        self._model_cache = {}
        
        logger.info(f"VideoSearchEngine initialized (GPU FAISS: {use_gpu_faiss})")

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path relative to project root if not absolute."""
        path = Path(path)
        if not path.is_absolute():
            return self.project_root / path
        return path

    def _get_video_files_from_parquet(self) -> List[Path]:
        """
        Get video files from parquet input file, supporting both sensor_video_file and sensor_frame_zip columns.
        
        Returns:
            List of video/frame file paths
        """
        if not hasattr(self.config, 'input_path') or not self.config.input_path:
            raise ValueError("input_path must be specified in config")
            
        index_path = self._resolve_path(self.config.input_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Input parquet file not found: {index_path}")
            
        if index_path.suffix.lower() != '.parquet':
            raise ValueError(f"Only parquet input files are supported, got: {index_path.suffix}")
        
        try:
            logger.info(f"Loading video files from parquet: {index_path}")
            df = pd.read_parquet(index_path)
            
            # Check for both sensor_video_file and sensor_frame_zip columns
            video_column = None
            frame_column = None
            
            if 'sensor_video_file' in df.columns:
                video_column = 'sensor_video_file'
            if 'sensor_frame_zip' in df.columns:
                frame_column = 'sensor_frame_zip'
                
            if not video_column and not frame_column:
                raise ValueError(f"Neither 'sensor_video_file' nor 'sensor_frame_zip' columns found in parquet file. Available columns: {list(df.columns)}")
            
            # Collect all input paths
            all_paths = []
            
            # Add video files if column exists
            if video_column:
                video_paths = df[video_column].dropna().drop_duplicates().tolist()
                all_paths.extend(video_paths)
                logger.info(f"Found {len(video_paths)} video file entries")
            
            # Add frame files if column exists  
            if frame_column:
                frame_paths = df[frame_column].dropna().drop_duplicates().tolist()
                all_paths.extend(frame_paths)
                logger.info(f"Found {len(frame_paths)} frame file entries")
            
            # Filter for existing files
            valid_files = [Path(path) for path in all_paths if Path(path).exists()]
            
            missing_files = [path for path in all_paths if not Path(path).exists()]
            if missing_files:
                logger.warning(f"Found {len(missing_files)} missing input files")
                for missing in missing_files[:5]:  # Show first 5 missing files
                    logger.warning(f"  Missing: {missing}")
                if len(missing_files) > 5:
                    logger.warning(f"  ... and {len(missing_files) - 5} more")
            
            logger.info(f"Loaded {len(valid_files)} valid input files from parquet")
            return valid_files
            
        except Exception as e:
            raise RuntimeError(f"Error reading parquet file {index_path}: {e}")

    def _get_video_files(self, video_directory: Union[str, Path]) -> List[Path]:
        """
        Legacy method - use _get_video_files_from_parquet instead.
        """
        logger.warning("_get_video_files is deprecated. Use _get_video_files_from_parquet instead.")
        return self._get_video_files_from_parquet()

    @time_it
    def build_database_from_parquet(self, force_rebuild: bool = False):
        """
        Build video embeddings database from parquet input file.
        
        Args:
            force_rebuild: If True, rebuild from scratch
        """
        if not force_rebuild:
            try:
                # ParquetVectorDatabase loads automatically in __init__
                if hasattr(self.database, 'df') and self.database.df is not None and len(self.database.df) > 0:
                    logger.info("Loaded existing database")
                    return
            except Exception as e:
                logger.info(f"Could not load existing database: {e}")
        
        video_files = self._get_video_files_from_parquet()
        
        if not video_files:
            logger.warning(f"No video files found")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Load slice_id mapping from parquet
        path_to_slice_id = {}
        index_path = self._resolve_path(self.config.input_path)
        try:
            df = pd.read_parquet(index_path)
            if 'slice_id' in df.columns and 'sensor_video_file' in df.columns:
                for _, row in df.iterrows():
                    path_to_slice_id[Path(row['sensor_video_file'])] = row['slice_id']
                logger.info(f"Loaded slice_id mapping for {len(path_to_slice_id)} videos")
            else:
                raise ValueError("Required columns 'slice_id' and 'sensor_video_file' not found in parquet")
        except Exception as e:
            raise RuntimeError(f"Error loading slice_id mapping from parquet: {e}")
        
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
        if hasattr(self.config, 'input_path') and self.config.input_path:
            input_path = self._resolve_path(self.config.input_path)
            if input_path.exists():
                logger.info(f"Loading query videos from unified input path: {input_path}")
                return self.build_unified_database_from_file_list(input_path, force_rebuild)
        
        if query_video_directory:
            query_dir = self._resolve_path(query_video_directory)
        else:
            raise ValueError("No query video directory provided and no input_path configured")
        
        if not query_dir.exists():
            logger.warning(f"Query video directory not found: {query_dir}")
            return {"processed": 0, "cached": 0, "errors": 1}
        
        logger.info(f"Building unified database from: {query_dir}")
        return self.build_unified_database_from_file_list(self.config.input_path, force_rebuild)

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
            query_embedding = self.database.get_embedding(slice_id)
            
            if query_embedding is not None:
                logger.info(f"Using pre-computed embedding for slice_id: {slice_id}")
                try:
                    return self._search_by_embedding(query_embedding, top_k, exclude_slice_id=slice_id)
                except Exception as e:
                    logger.error(f"Error during pre-computed search: {e}")
                    logger.info("Falling back to real-time computation...")
            else:
                logger.warning(f"No pre-computed embedding found for slice_id: {slice_id}")
        except Exception as e:
            logger.error(f"Error accessing query cache: {e}")
        
        logger.info(f"Falling back to real-time processing for slice_id: {slice_id}")
        
        if hasattr(self.config, 'input_path') and self.config.input_path:
            input_path = self._resolve_path(self.config.input_path)
            if input_path.exists():
                if input_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(input_path)
                else:
                    df = pd.read_csv(input_path)
                
                matching_rows = df[df['slice_id'] == slice_id]
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    
                    # Check for sensor_video_file first, then sensor_frame_zip
                    query_path = None
                    if 'sensor_video_file' in row and pd.notna(row['sensor_video_file']):
                        query_path = Path(row['sensor_video_file'])
                    elif 'sensor_frame_zip' in row and pd.notna(row['sensor_frame_zip']):
                        query_path = Path(row['sensor_frame_zip'])
                    
                    if query_path and query_path.exists():
                        return self.search_by_video(query_path, top_k)
                    else:
                        logger.warning(f"Query file not found for slice_id: {slice_id}, path: {query_path}")
        
        raise VideoNotFoundError(f"Query video/frame file not found for slice_id: {slice_id}. Please ensure it's listed in input_path.")

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
        Search for similar videos with caching support and self-matching avoidance.
        
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
        
        # Extract slice_id from video path for self-matching avoidance
        query_slice_id = query_path.name
        
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
            
            # Pass slice_id to avoid self-matching
            return self._search_by_embedding(query_embedding, top_k, exclude_slice_id=query_slice_id)
            
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
            
            # Get precomputed video embedding from unified database
            video_embedding = self.database.get_embedding(query_video_slice_id)
            if video_embedding is None:
                raise VideoNotFoundError(f"No precomputed embedding found for slice_id: {query_video_slice_id}. Please build unified database first.")
            
            logger.info(f"Using precomputed embedding for joint search: {query_video_slice_id}")
            
            # Combine embeddings with alpha weighting
            joint_embedding = alpha * text_embedding + (1 - alpha) * video_embedding
            
            # Normalize the combined embedding to unit vector
            joint_embedding = joint_embedding / np.linalg.norm(joint_embedding)
            
            logger.info(f"Joint search: alpha={alpha:.2f}, text_weight={alpha:.2f}, video_weight={1-alpha:.2f}")
            
            return self._search_by_embedding(joint_embedding, top_k)
            
        except Exception as e:
            raise SearchError(f"Joint search failed: {str(e)}")
    
    def _search_by_embedding(self, query_embedding: np.ndarray, top_k: int, 
                           exclude_slice_id: Optional[str] = None) -> List[Dict]:
        """
        Internal search method using FAISS with self-matching avoidance.
        
        Args:
            query_embedding: Normalized query embedding
            top_k: Number of results
            exclude_slice_id: Slice ID to exclude from results (for self-matching avoidance)
            
        Returns:
            Formatted search results
        """
        # Ensure database is loaded
        if self.database.embedding_matrix is None:
            logger.warning("No embeddings in database for search")
        
        # Request more results to account for potential self-matches
        search_k = top_k + 1 if exclude_slice_id else top_k
        
        results = self.search_strategy.search(
            query_embedding,
            self.database,
            search_k
        )
        
        if not results:
            raise NoResultsError("No results found")
        
        formatted_results = []
        for idx, similarity, metadata in results:
            if similarity < self.config.similarity_threshold:
                continue
            
            # Skip self-matching if exclude_slice_id is provided
            current_slice_id = metadata.get("slice_id", "")
            if exclude_slice_id and current_slice_id == exclude_slice_id:
                logger.debug(f"Skipping self-match for slice_id: {current_slice_id}")
                continue
                
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata.get("video_path", ""),
                "slice_id": current_slice_id,
                "similarity_score": similarity,
                "metadata": metadata,
                "thumbnail": metadata.get("thumbnail", ""),
                "thumbnail_size": metadata.get("thumbnail_size", (0, 0))
            }
            formatted_results.append(result)
            
            # Stop when we have enough results
            if len(formatted_results) >= top_k:
                break
        
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
        """Get list of available video filenames from unified database."""
        return self.database.list_videos()
    
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
        
        # No separate query database in unified system
        stats["unified_database"] = True
        
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
                'using_gpu': stats.get('using_gpu', False)
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
                'error': str(e)
            }
