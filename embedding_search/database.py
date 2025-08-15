"""
Improved Video Database Manager with safe serialization and better error handling.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any
import logging
from datetime import datetime
import hashlib

from base import DatabaseBackend
from config import VideoRetrievalConfig
from exceptions import (
    DatabaseError, DatabaseNotFoundError, DatabaseCorruptedError
)

logger = logging.getLogger(__name__)


class SafeVideoDatabase(DatabaseBackend):
    """Video database with JSON + numpy serialization for safety."""
    
    def __init__(self, database_path: Union[str, Path] = "video_embeddings", 
                 config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize the video database.
        
        Args:
            database_path: Base path for database files (without extension)
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.database_path = Path(database_path)
        
        # Separate files for metadata and embeddings
        self.metadata_path = self.database_path.with_suffix('.json')
        self.embeddings_path = self.database_path.with_suffix('.npy')
        
        self.embeddings = []
        self.metadata = []
        self.embedding_matrix = None
        
        # Version for compatibility checking
        self.version = "2.0"
    
    def _compute_checksum(self, data: np.ndarray) -> str:
        """Compute checksum for data integrity."""
        return hashlib.sha256(data.tobytes()).hexdigest()
    
    def add_embeddings(self, embeddings_data: List[Dict[str, Any]]):
        """
        Add video embeddings to the database with validation.
        
        Args:
            embeddings_data: List of dictionaries containing embeddings and metadata
        """
        if not embeddings_data:
            logger.warning("No embeddings to add")
            return
        
        # Validate embeddings
        embedding_dim = None
        for data in embeddings_data:
            if 'embedding' not in data or 'video_path' not in data:
                raise ValueError("Missing required fields in embedding data")
            
            # Check embedding dimension consistency
            if embedding_dim is None:
                embedding_dim = data['embedding'].shape[0]
            elif data['embedding'].shape[0] != embedding_dim:
                raise ValueError(f"Inconsistent embedding dimensions: expected {embedding_dim}, got {data['embedding'].shape[0]}")
        
        # Add to database
        for data in embeddings_data:
            # Ensure embedding is normalized
            embedding = data["embedding"]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            self.embeddings.append(embedding)
            
            # Store metadata
            meta = {
                "video_path": str(data["video_path"]),
                "video_name": Path(data["video_path"]).name,
                "embedding_dim": data.get("embedding_dim", embedding_dim),
                "num_frames": data.get("num_frames", 0),
                "added_at": datetime.now().isoformat(),
                "checksum": self._compute_checksum(embedding)
            }
            self.metadata.append(meta)
        
        # Update embedding matrix
        self._update_embedding_matrix()
        logger.info(f"Added {len(embeddings_data)} embeddings to database")
    
    def _update_embedding_matrix(self):
        """Update the embedding matrix for efficient similarity computation."""
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings)
            # Ensure all embeddings are normalized
            norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
            self.embedding_matrix = self.embedding_matrix / np.maximum(norms, 1e-8)
        else:
            self.embedding_matrix = None
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """
        Save the database using JSON for metadata and numpy for embeddings.
        
        Args:
            path: Optional base path for saving
        """
        save_path = Path(path) if path else self.database_path
        metadata_path = save_path.with_suffix('.json')
        embeddings_path = save_path.with_suffix('.npy')
        
        try:
            # Save embeddings as numpy array
            if self.embeddings:
                embeddings_array = np.vstack(self.embeddings)
                np.save(embeddings_path, embeddings_array)
                logger.info(f"Saved {len(self.embeddings)} embeddings to {embeddings_path}")
            
            # Prepare metadata
            db_metadata = {
                "version": self.version,
                "created_at": datetime.now().isoformat(),
                "num_videos": len(self.metadata),
                "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else 0,
                "videos": self.metadata
            }
            
            # Save metadata as JSON
            with open(metadata_path, 'w') as f:
                json.dump(db_metadata, f, indent=2)
            
            logger.info(f"Database metadata saved to {metadata_path}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to save database: {str(e)}")
    
    def load(self, path: Optional[Union[str, Path]] = None):
        """
        Load the database from JSON metadata and numpy embeddings.
        
        Args:
            path: Optional base path for loading
        """
        load_path = Path(path) if path else self.database_path
        metadata_path = load_path.with_suffix('.json')
        embeddings_path = load_path.with_suffix('.npy')
        
        # Check if files exist
        if not metadata_path.exists():
            raise DatabaseNotFoundError(f"Database metadata not found at {metadata_path}")
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                db_metadata = json.load(f)
            
            # Check version compatibility
            if db_metadata.get('version', '1.0') != self.version:
                logger.warning(f"Database version mismatch: expected {self.version}, got {db_metadata.get('version')}")
            
            self.metadata = db_metadata.get('videos', [])
            
            # Load embeddings if file exists
            if embeddings_path.exists():
                embeddings_array = np.load(embeddings_path)
                
                # Verify dimensions
                expected_count = db_metadata.get('num_videos', 0)
                if len(embeddings_array) != expected_count:
                    raise DatabaseCorruptedError(
                        f"Embedding count mismatch: expected {expected_count}, got {len(embeddings_array)}"
                    )
                
                # Verify checksums for data integrity
                self.embeddings = []
                for i, (embedding, meta) in enumerate(zip(embeddings_array, self.metadata)):
                    if 'checksum' in meta:
                        computed_checksum = self._compute_checksum(embedding)
                        if computed_checksum != meta['checksum']:
                            logger.warning(f"Checksum mismatch for video {meta['video_name']}")
                    
                    self.embeddings.append(embedding)
                
                self._update_embedding_matrix()
            else:
                logger.warning(f"Embeddings file not found at {embeddings_path}")
                self.embeddings = []
                self.embedding_matrix = None
            
            logger.info(f"Database loaded with {len(self.embeddings)} videos")
            
        except json.JSONDecodeError as e:
            raise DatabaseCorruptedError(f"Failed to parse database metadata: {str(e)}")
        except Exception as e:
            if isinstance(e, (DatabaseNotFoundError, DatabaseCorruptedError)):
                raise
            raise DatabaseError(f"Failed to load database: {str(e)}")
    
    def compute_similarity(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Compute cosine similarity between query and all database embeddings.
        
        Args:
            query_embedding: Query video embedding
            top_k: Number of top similar videos to return
            
        Returns:
            List of tuples (index, similarity_score, metadata)
        """
        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            logger.warning("No embeddings in database")
            return []
        
        # Ensure query is normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities (embeddings are already normalized)
        similarities = np.dot(self.embedding_matrix, query_norm)
        
        # Handle edge cases
        top_k = min(top_k, len(similarities))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.metadata[idx].copy()  # Return a copy to prevent modification
            ))
        
        return results
    
    def get_embedding_by_filename(self, filename: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get pre-computed embedding by video filename.
        
        Args:
            filename: Video filename (e.g., "car2cyclist_2.mp4")
            
        Returns:
            Tuple of (embedding, metadata) if found, None otherwise
        """
        for i, meta in enumerate(self.metadata):
            if Path(meta['video_path']).name == filename:
                if i < len(self.embeddings):
                    return self.embeddings[i], meta
        return None
    
    def get_embedding_by_path(self, video_path: Union[str, Path]) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Get pre-computed embedding by full video path.
        
        Args:
            video_path: Full path to video file
            
        Returns:
            Tuple of (embedding, metadata) if found, None otherwise
        """
        video_path_str = str(video_path)
        for i, meta in enumerate(self.metadata):
            if meta['video_path'] == video_path_str:
                if i < len(self.embeddings):
                    return self.embeddings[i], meta
        return None
    
    def list_available_videos(self) -> List[Dict[str, str]]:
        """
        Get list of all available videos with their filenames and paths.
        
        Returns:
            List of dictionaries with 'filename', 'path', and 'added_at' keys
        """
        videos = []
        for meta in self.metadata:
            videos.append({
                'filename': Path(meta['video_path']).name,
                'path': meta['video_path'],
                'added_at': meta.get('added_at', 'unknown')
            })
        return videos

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the database."""
        if not self.embeddings:
            return {
                "num_videos": 0,
                "embedding_dim": 0,
                "video_names": [],
                "database_size_mb": 0,
                "version": self.version
            }
        
        # Calculate database size
        db_size = 0
        if self.metadata_path.exists():
            db_size += self.metadata_path.stat().st_size
        if self.embeddings_path.exists():
            db_size += self.embeddings_path.stat().st_size
        
        return {
            "num_videos": len(self.embeddings),
            "embedding_dim": self.embeddings[0].shape[0],
            "video_names": [m["video_name"] for m in self.metadata],
            "database_size_mb": db_size / (1024 * 1024),
            "version": self.version,
            "created_dates": [m.get("added_at", "unknown") for m in self.metadata]
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
        video_path = str(video_path)  # Ensure string comparison
        
        for i, meta in enumerate(self.metadata):
            if meta["video_path"] == video_path:
                del self.embeddings[i]
                del self.metadata[i]
                self._update_embedding_matrix()
                logger.info(f"Removed video: {video_path}")
                return True
        
        logger.warning(f"Video not found in database: {video_path}")
        return False
    
    def export_to_legacy_format(self, path: Union[str, Path]):
        """Export database to legacy pickle format for compatibility."""
        import pickle
        
        legacy_data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "embedding_matrix": self.embedding_matrix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(legacy_data, f)
        
        logger.info(f"Exported database to legacy format: {path}")
    
    def import_from_legacy_format(self, path: Union[str, Path]):
        """Import database from legacy pickle format."""
        import pickle
        
        try:
            with open(path, 'rb') as f:
                legacy_data = pickle.load(f)
            
            self.embeddings = legacy_data.get("embeddings", [])
            self.metadata = legacy_data.get("metadata", [])
            self.embedding_matrix = legacy_data.get("embedding_matrix", None)
            
            # Add checksums to imported data
            for i, (embedding, meta) in enumerate(zip(self.embeddings, self.metadata)):
                if 'checksum' not in meta:
                    meta['checksum'] = self._compute_checksum(embedding)
            
            logger.info(f"Imported {len(self.embeddings)} videos from legacy format")
            
        except Exception as e:
            raise DatabaseError(f"Failed to import legacy database: {str(e)}")
