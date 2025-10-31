"""
Abstract base classes for the alpha 0.1 retrieval.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def extract_video_embedding(self, video_path: Path) -> np.ndarray:
        """
        Extract embedding from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding from text.
        
        Args:
            text: Text query
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def extract_video_embeddings_batch(self, video_paths: List[Path], batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Extract embeddings from multiple videos in batches.
        
        Args:
            video_paths: List of video file paths
            batch_size: Number of videos to process in each batch
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    def add_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """Add embeddings to the database."""
        pass
    
    @abstractmethod
    def save(self, path: Optional[Path] = None):
        """Save the database to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Optional[Path] = None):
        """Load the database from disk."""
        pass
    
    @abstractmethod
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Compute similarity between query and database embeddings.
        
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all data from the database."""
        pass
    
    @abstractmethod
    def remove_video(self, video_path: str) -> bool:
        """Remove a video from the database by path."""
        pass


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    def search(self, 
               query_embedding: np.ndarray, 
               database: DatabaseBackend, 
               top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[int, float, Dict]]:
        """
        Perform search using the given strategy.
        
        Args:
            query_embedding: Query embedding vector
            database: Database backend instance
            top_k: Number of top results to return
            filters: Optional filters to apply
            
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        pass


class VideoProcessor(ABC):
    """Abstract base class for video processing."""
    
    @abstractmethod
    def load_frames(self, video_path: Path, num_frames: int = 8) -> np.ndarray:
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract
            
        Returns:
            Array of frames
        """
        pass
    
    @abstractmethod
    def validate_video(self, video_path: Path) -> bool:
        """
        Validate if a video file is valid and can be processed.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if valid, False otherwise
        """
        pass
