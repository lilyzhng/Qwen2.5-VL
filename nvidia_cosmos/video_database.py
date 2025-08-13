"""
Video Database Manager for storing and loading video embeddings.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoDatabase:
    """Manages video embeddings database for efficient similarity search."""
    
    def __init__(self, database_path: Union[str, Path] = "video_embeddings.pkl"):
        """
        Initialize the video database.
        
        Args:
            database_path: Path to save/load the database
        """
        self.database_path = Path(database_path)
        self.embeddings = []
        self.metadata = []
        self.embedding_matrix = None
        
    def add_embeddings(self, embeddings_data: List[Dict[str, np.ndarray]]):
        """
        Add video embeddings to the database.
        
        Args:
            embeddings_data: List of dictionaries containing embeddings and metadata
        """
        for data in embeddings_data:
            self.embeddings.append(data["embedding"])
            
            # Store metadata
            meta = {
                "video_path": data["video_path"],
                "video_name": Path(data["video_path"]).name,
                "embedding_dim": data["embedding_dim"],
                "num_frames": data["num_frames"],
                "added_at": datetime.now().isoformat()
            }
            self.metadata.append(meta)
        
        # Update embedding matrix
        self._update_embedding_matrix()
        logger.info(f"Added {len(embeddings_data)} embeddings to database")
    
    def _update_embedding_matrix(self):
        """Update the embedding matrix for efficient similarity computation."""
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings)
        else:
            self.embedding_matrix = None
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save the database to disk.
        
        Args:
            path: Optional path to save the database
        """
        save_path = Path(path) if path else self.database_path
        
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "embedding_matrix": self.embedding_matrix
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Database saved to {save_path}")
    
    def load(self, path: Optional[Union[str, Path]] = None):
        """
        Load the database from disk.
        
        Args:
            path: Optional path to load the database from
        """
        load_path = Path(path) if path else self.database_path
        
        if not load_path.exists():
            logger.warning(f"No database found at {load_path}")
            return
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.embedding_matrix = data["embedding_matrix"]
        
        logger.info(f"Database loaded from {load_path} with {len(self.embeddings)} videos")
    
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Compute cosine similarity between query and all database embeddings.
        
        Args:
            query_embedding: Query video embedding
            top_k: Number of top similar videos to return
            
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        if self.embedding_matrix is None:
            logger.warning("No embeddings in database")
            return []
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(self.embedding_matrix, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.metadata[idx]
            ))
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about the database."""
        if not self.embeddings:
            return {"num_videos": 0}
        
        return {
            "num_videos": len(self.embeddings),
            "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else 0,
            "video_names": [m["video_name"] for m in self.metadata]
        }
    
    def clear(self):
        """Clear all data from the database."""
        self.embeddings = []
        self.metadata = []
        self.embedding_matrix = None
        logger.info("Database cleared")
    
    def remove_video(self, video_path: str) -> bool:
        """
        Remove a video from the database by path.
        
        Args:
            video_path: Path of the video to remove
            
        Returns:
            True if video was found and removed, False otherwise
        """
        for i, meta in enumerate(self.metadata):
            if meta["video_path"] == video_path:
                del self.embeddings[i]
                del self.metadata[i]
                self._update_embedding_matrix()
                logger.info(f"Removed video: {video_path}")
                return True
        
        logger.warning(f"Video not found in database: {video_path}")
        return False
