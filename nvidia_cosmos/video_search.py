"""
Improved Video Search Engine with extensibility and better error handling.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
import shutil
import json

from base import EmbeddingModel, DatabaseBackend, SearchStrategy
from video_embedder import CosmosVideoEmbedder
from video_database import SafeVideoDatabase
from config import VideoRetrievalConfig
from exceptions import (
    VideoNotFoundError, SearchError, InvalidQueryError, NoResultsError
)

logger = logging.getLogger(__name__)


class CosineSimlaritySearch(SearchStrategy):
    """Standard cosine similarity search strategy."""
    
    def search(self, 
               query_embedding: np.ndarray, 
               database: DatabaseBackend, 
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, Dict]]:
        """
        Perform cosine similarity search.
        
        Args:
            query_embedding: Query embedding vector
            database: Database backend
            top_k: Number of results to return
            filters: Optional filters (not implemented in basic version)
            
        Returns:
            Search results
        """
        if filters:
            logger.warning("Filters not implemented in basic cosine similarity search")
        
        return database.compute_similarity(query_embedding, top_k)


class VideoSearchEngine:
    """Improved search engine with dependency injection and configuration."""
    
    def __init__(self, 
                 config: Optional[VideoRetrievalConfig] = None,
                 embedder: Optional[EmbeddingModel] = None,
                 database: Optional[DatabaseBackend] = None,
                 search_strategy: Optional[SearchStrategy] = None):
        """
        Initialize the video search engine.
        
        Args:
            config: Configuration object
            embedder: Embedding model (uses CosmosVideoEmbedder if None)
            database: Database backend (uses SafeVideoDatabase if None)
            search_strategy: Search strategy (uses CosineSimlaritySearch if None)
        """
        self.config = config or VideoRetrievalConfig()
        
        # Initialize components
        self.embedder = embedder or CosmosVideoEmbedder(self.config)
        self.database = database or SafeVideoDatabase(self.config.database_path, self.config)
        self.search_strategy = search_strategy or CosineSimlaritySearch()
        
        logger.info("VideoSearchEngine initialized")
    
    def build_database(self, video_directory: Union[str, Path], force_rebuild: bool = False):
        """
        Build or update the video embeddings database from a directory.
        
        Args:
            video_directory: Directory containing video files
            force_rebuild: If True, rebuild database from scratch
        """
        video_dir = Path(video_directory)
        if not video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")
        
        # Clear database if force rebuild
        if force_rebuild:
            logger.info("Force rebuild: clearing existing database")
            self.database.clear()
        else:
            # Try to load existing database
            try:
                self.database.load()
                logger.info("Loaded existing database")
            except Exception as e:
                logger.info(f"Could not load existing database: {e}")
        
        # Get all video files
        video_files = []
        for ext in self.config.supported_formats:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))  # Handle uppercase extensions
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Check which videos are already in database
        existing_stats = self.database.get_statistics()
        existing_paths = {meta["video_path"] for meta in self.database.metadata} if hasattr(self.database, 'metadata') else set()
        new_videos = [v for v in video_files if str(v) not in existing_paths]
        
        if new_videos:
            logger.info(f"Processing {len(new_videos)} new videos")
            
            # Extract embeddings for new videos using batch processing
            embeddings_data = self.embedder.extract_video_embeddings_batch(new_videos)
            
            if embeddings_data:
                # Add to database
                self.database.add_embeddings(embeddings_data)
                
                # Save database
                self.database.save()
                logger.info(f"Database updated with {len(embeddings_data)} new videos")
            else:
                logger.warning("No embeddings were successfully extracted")
        else:
            logger.info("All videos already in database")
    
    def search_by_video(self, query_video_path: Union[str, Path], 
                       top_k: Optional[int] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for similar videos using a query video.
        
        Args:
            query_video_path: Path to the query video
            top_k: Number of top similar videos to return
            filters: Optional filters for search
            
        Returns:
            List of search results with similarity scores and metadata
        """
        query_path = Path(query_video_path)
        if not query_path.exists():
            raise VideoNotFoundError(f"Query video not found: {query_path}")
        
        top_k = top_k or self.config.default_top_k
        
        try:
            # Extract embedding for query video
            logger.info(f"Extracting embedding for query video: {query_path.name}")
            query_embedding = self.embedder.extract_video_embedding(query_path)
            
            # Perform search
            return self._search_by_embedding(query_embedding, top_k, filters)
            
        except VideoNotFoundError:
            raise
        except Exception as e:
            raise SearchError(f"Search failed for video {query_path}: {str(e)}")
    
    def search_by_text(self, query_text: str, 
                      top_k: Optional[int] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for videos using a text query.
        
        Args:
            query_text: Text description of the video content
            top_k: Number of top similar videos to return
            filters: Optional filters for search
            
        Returns:
            List of search results with similarity scores and metadata
        """
        if not query_text or not query_text.strip():
            raise InvalidQueryError("Text query cannot be empty")
        
        top_k = top_k or self.config.default_top_k
        
        try:
            # Extract text embedding
            logger.info(f"Extracting embedding for text query: '{query_text}'")
            text_embedding = self.embedder.extract_text_embedding(query_text)
            
            # Perform search
            return self._search_by_embedding(text_embedding, top_k, filters)
            
        except Exception as e:
            raise SearchError(f"Search failed for text query '{query_text}': {str(e)}")
    
    def _search_by_embedding(self, query_embedding: np.ndarray, 
                           top_k: int,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Internal method to search by embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            filters: Optional filters
            
        Returns:
            Formatted search results
        """
        # Load database if needed
        if not hasattr(self.database, 'embedding_matrix') or self.database.embedding_matrix is None:
            try:
                self.database.load()
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                raise SearchError("Database not available")
        
        # Perform search using strategy
        results = self.search_strategy.search(query_embedding, self.database, top_k, filters)
        
        if not results:
            raise NoResultsError("No results found")
        
        # Format results
        formatted_results = []
        for idx, similarity, metadata in results:
            # Filter by similarity threshold
            if similarity < self.config.similarity_threshold:
                continue
            
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata["video_path"],
                "video_name": metadata["video_name"],
                "similarity_score": similarity,
                "metadata": metadata
            }
            formatted_results.append(result)
        
        if not formatted_results:
            raise NoResultsError(f"No results above similarity threshold {self.config.similarity_threshold}")
        
        return formatted_results
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database."""
        try:
            # Try to load if not loaded
            if not hasattr(self.database, 'embeddings') or not self.database.embeddings:
                self.database.load()
        except Exception as e:
            logger.warning(f"Could not load database: {e}")
        
        return self.database.get_statistics()
    
    def export_results(self, results: List[Dict], output_dir: Union[str, Path], 
                      copy_videos: bool = True,
                      format: str = 'json') -> Path:
        """
        Export search results to a directory.
        
        Args:
            results: Search results to export
            output_dir: Directory to export results to
            copy_videos: If True, copy video files to output directory
            format: Export format ('json', 'csv', 'html')
            
        Returns:
            Path to the export directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format == 'json':
            results_file = output_path / "search_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        elif format == 'csv':
            import csv
            results_file = output_path / "search_results.csv"
            with open(results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['rank', 'video_name', 'similarity_score', 'video_path'])
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        'rank': result['rank'],
                        'video_name': result['video_name'],
                        'similarity_score': result['similarity_score'],
                        'video_path': result['video_path']
                    })
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Copy videos if requested
        if copy_videos:
            videos_dir = output_path / "videos"
            videos_dir.mkdir(exist_ok=True)
            
            for result in results:
                src_path = Path(result["video_path"])
                if src_path.exists():
                    dst_path = videos_dir / f"{result['rank']:02d}_{result['video_name']}"
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logger.error(f"Failed to copy {src_path}: {e}")
        
        logger.info(f"Results exported to {output_path}")
        return output_path
    
    def import_legacy_database(self, legacy_path: Union[str, Path]):
        """Import a database from the legacy pickle format."""
        if isinstance(self.database, SafeVideoDatabase):
            self.database.import_from_legacy_format(legacy_path)
            self.database.save()
            logger.info("Legacy database imported and converted to safe format")
        else:
            logger.warning("Database backend does not support legacy import")
