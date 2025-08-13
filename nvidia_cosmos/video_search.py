"""
Video Similarity Search module for finding similar videos using embeddings.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import logging
from video_embedder import VideoEmbedder
from video_database import VideoDatabase
import shutil
import os

logger = logging.getLogger(__name__)

class VideoSearchEngine:
    """Search engine for finding similar videos using embeddings."""
    
    def __init__(self, database_path: str = "video_embeddings.pkl"):
        """
        Initialize the video search engine.
        
        Args:
            database_path: Path to the video embeddings database
        """
        self.embedder = VideoEmbedder()
        self.database = VideoDatabase(database_path)
        
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
        
        # Load existing database if not forcing rebuild
        if not force_rebuild and self.database.database_path.exists():
            self.database.load()
            logger.info("Loaded existing database")
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return
        
        logger.info(f"Found {len(video_files)} video files")
        
        # Check which videos are already in database
        existing_paths = {meta["video_path"] for meta in self.database.metadata}
        new_videos = [v for v in video_files if str(v) not in existing_paths]
        
        if new_videos:
            logger.info(f"Processing {len(new_videos)} new videos")
            
            # Extract embeddings for new videos
            embeddings_data = self.embedder.extract_embeddings_batch(new_videos)
            
            # Add to database
            self.database.add_embeddings(embeddings_data)
            
            # Save database
            self.database.save()
        else:
            logger.info("All videos already in database")
    
    def search_by_video(self, query_video_path: Union[str, Path], top_k: int = 5) -> List[Dict]:
        """
        Search for similar videos using a query video.
        
        Args:
            query_video_path: Path to the query video
            top_k: Number of top similar videos to return
            
        Returns:
            List of search results with similarity scores and metadata
        """
        query_path = Path(query_video_path)
        if not query_path.exists():
            raise FileNotFoundError(f"Query video not found: {query_path}")
        
        # Extract embedding for query video
        logger.info(f"Extracting embedding for query video: {query_path.name}")
        query_data = self.embedder.extract_embedding(query_path)
        query_embedding = query_data["embedding"]
        
        # Load database if not already loaded
        if self.database.embedding_matrix is None:
            self.database.load()
        
        # Search for similar videos
        results = self.database.compute_similarity(query_embedding, top_k)
        
        # Format results
        formatted_results = []
        for idx, similarity, metadata in results:
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata["video_path"],
                "video_name": metadata["video_name"],
                "similarity_score": similarity,
                "metadata": metadata
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Search for videos using a text query.
        
        Args:
            query_text: Text description of the video content
            top_k: Number of top similar videos to return
            
        Returns:
            List of search results with similarity scores and metadata
        """
        # Extract text embedding
        logger.info(f"Extracting embedding for text query: '{query_text}'")
        text_embedding = self.embedder.get_text_embedding(query_text)
        
        # Load database if not already loaded
        if self.database.embedding_matrix is None:
            self.database.load()
        
        # Search for similar videos
        results = self.database.compute_similarity(text_embedding, top_k)
        
        # Format results
        formatted_results = []
        for idx, similarity, metadata in results:
            result = {
                "rank": len(formatted_results) + 1,
                "video_path": metadata["video_path"],
                "video_name": metadata["video_name"],
                "similarity_score": similarity,
                "metadata": metadata
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_database_info(self) -> Dict:
        """Get information about the current database."""
        if self.database.embedding_matrix is None:
            self.database.load()
        
        return self.database.get_statistics()
    
    def export_results(self, results: List[Dict], output_dir: Union[str, Path], 
                      copy_videos: bool = True) -> Path:
        """
        Export search results to a directory.
        
        Args:
            results: Search results to export
            output_dir: Directory to export results to
            copy_videos: If True, copy video files to output directory
            
        Returns:
            Path to the export directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results metadata
        import json
        results_file = output_path / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Copy videos if requested
        if copy_videos:
            videos_dir = output_path / "videos"
            videos_dir.mkdir(exist_ok=True)
            
            for result in results:
                src_path = Path(result["video_path"])
                if src_path.exists():
                    dst_path = videos_dir / f"{result['rank']:02d}_{result['video_name']}"
                    shutil.copy2(src_path, dst_path)
        
        logger.info(f"Results exported to {output_path}")
        return output_path
