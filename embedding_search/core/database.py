"""
Unified Parquet Database System for Vector Embeddings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Any
import logging
from datetime import datetime
import hashlib
import base64
import io
import os
import tempfile

from .config import VideoRetrievalConfig
from .exceptions import DatabaseError
from .visualizer import VideoResultsVisualizer

logger = logging.getLogger(__name__)


class ParquetVectorDatabase:
    """
    Unified Parquet-based vector database for both main and query embeddings.
    """
    
    def __init__(self, database_path: Union[str, Path], config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize Parquet vector database.
        
        Args:
            database_path: Path to the parquet database file
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.database_path = Path(database_path)
        self.df = None
        
        # Initialize thumbnail extractor
        self.thumbnail_extractor = VideoResultsVisualizer(thumbnail_size=self.config.thumbnail_size)
        
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()
        
    def load(self):
        """Load database from parquet file."""
        if self.database_path.exists():
            try:
                self.df = pd.read_parquet(self.database_path)
                # Set slice_id as index for fast lookups
                if 'slice_id' in self.df.columns and self.df.index.name != 'slice_id':
                    self.df = self.df.set_index('slice_id', drop=False)
                logger.info(f"Loaded {len(self.df)} embeddings from {self.database_path}")
            except Exception as e:
                logger.warning(f"Could not load database {self.database_path}: {e}")
                self._create_empty_database()
        else:
            self._create_empty_database()
    
    def _create_empty_database(self):
        """Create empty database with proper schema."""
        # Define all required columns with their types
        self.df = pd.DataFrame({
            'slice_id': pd.Series(dtype='str'),
            'video_path': pd.Series(dtype='str'),
            'embedding': pd.Series(dtype='object'),  # Will hold lists
            'embedding_dim': pd.Series(dtype='int'),
            'num_frames': pd.Series(dtype='int'),
            'file_hash': pd.Series(dtype='str'),
            'file_size': pd.Series(dtype='int'),
            'created_at': pd.Series(dtype='str'),
            'last_accessed': pd.Series(dtype='str'),
            'access_count': pd.Series(dtype='int'),
            'category': pd.Series(dtype='str'),
            'thumbnail': pd.Series(dtype='str'),
            'thumbnail_size': pd.Series(dtype='object')  # Will hold tuples
        })
        
        # Set slice_id as index
        self.df = self.df.set_index('slice_id', drop=False)
        
        logger.info(f"Created empty database with columns: {self.df.columns.tolist()}")
    
    @property
    def embedding_matrix(self) -> Optional[np.ndarray]:
        """Get embeddings as a numpy matrix for FAISS search."""
        if self.df is None or len(self.df) == 0:
            return None
        # Convert list of embeddings to numpy array
        embeddings = np.array(self.df['embedding'].tolist()).astype('float32')
        return embeddings
    
    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Get metadata list for compatibility with FAISS search."""
        if self.df is None or len(self.df) == 0:
            return []
        # Convert dataframe rows to metadata dictionaries
        metadata_list = []
        for _, row in self.df.iterrows():
            meta = row.to_dict()
            # Remove embedding from metadata to save memory
            meta.pop('embedding', None)
            metadata_list.append(meta)
        return metadata_list
    
    def add_embedding(self, slice_id: str, video_path: Union[str, Path], 
                     embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """
        Add or update an embedding in the database.
        
        Args:
            slice_id: Unique identifier for the slice (PRIMARY KEY)
            video_path: Full path to video file
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            video_path = Path(video_path)
            
            file_stat = video_path.stat() if video_path.exists() else None
            file_hash = self._get_file_hash(video_path) if video_path.exists() else None
            thumbnail_b64, thumbnail_size = self._extract_and_encode_thumbnail(video_path) if video_path.exists() else ("", (0, 0))
            
            row_data = {
                'slice_id': slice_id,
                'video_path': str(video_path.absolute()),
                'embedding': embedding.tolist(),
                'embedding_dim': embedding.shape[0],
                'num_frames': metadata.get('num_frames', self.config.num_frames) if metadata else self.config.num_frames,
                'file_hash': file_hash,
                'file_size': file_stat.st_size if file_stat else 0,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 1,
                'category': metadata.get('category', 'query') if metadata else 'query',
                'thumbnail': thumbnail_b64,
                'thumbnail_size': thumbnail_size
            }
            
            if metadata:
                for key, value in metadata.items():
                    if key not in row_data:
                        row_data[key] = value
            
            if self.df is None or len(self.df) == 0:
                self._create_empty_database()
            
            new_row = pd.DataFrame([row_data])
            
            if len(self.df) == 0:
                self.df = new_row.set_index('slice_id', drop=False)
            else:
                if slice_id in self.df.index:
                    # Update each column individually to avoid "equal len keys and value" error
                    for key, value in row_data.items():
                        if key in self.df.columns:
                            try:
                                self.df.at[slice_id, key] = value
                            except Exception as column_error:
                                logger.warning(f"Error updating column {key} for {slice_id}: {column_error}")
                else:
                    # Make sure new row has all the columns from the dataframe
                    for col in self.df.columns:
                        if col not in new_row.columns:
                            new_row[col] = None
                            
                    new_row = new_row.set_index('slice_id', drop=False)
                    self.df = pd.concat([self.df, new_row])
            
            logger.info(f"Added embedding for: {slice_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding embedding for {slice_id}: {e}")
            return False

    def get_embedding(self, slice_id: str) -> Optional[np.ndarray]:
        """
        Get embedding by slice_id.
        
        Args:
            slice_id: Unique identifier for the slice
            
        Returns:
            Embedding array or None if not found
        """
        if self.df is None or len(self.df) == 0 or slice_id not in self.df.index:
            return None
        
        try:
            row = self.df.loc[slice_id]
            
            # Check if embedding exists
            if 'embedding' not in row or row['embedding'] is None:
                logger.warning(f"No embedding data found for slice_id: {slice_id}")
                return None
                
            embedding = np.array(row['embedding'], dtype='float32')
            
            # Update access info safely
            try:
                if 'last_accessed' in self.df.columns:
                    self.df.at[slice_id, 'last_accessed'] = datetime.now().isoformat()
                
                if 'access_count' in self.df.columns:
                    current_count = 0
                    if not pd.isna(row.get('access_count')):
                        try:
                            current_count = int(row['access_count'])
                        except (ValueError, TypeError):
                            current_count = 0
                    self.df.at[slice_id, 'access_count'] = current_count + 1
            except Exception as update_error:
                logger.warning(f"Error updating access info for {slice_id}: {update_error}")
            
            return embedding
        except Exception as e:
            logger.error(f"Error retrieving embedding for {slice_id}: {e}")
            return None

    def get_thumbnail(self, slice_id: str) -> Optional[np.ndarray]:
        """
        Get thumbnail image by video name.
        
        Args:
            slice_id: Name of the video file
            
        Returns:
            Thumbnail image as numpy array or None if not found
        """
        if self.df is None or len(self.df) == 0 or slice_id not in self.df.index:
            return None
        
        try:
            row = self.df.loc[slice_id]
            thumbnail_b64 = row.get('thumbnail', '')
            
            if not thumbnail_b64:
                return None
            
            from PIL import Image
            
            thumbnail_bytes = base64.b64decode(thumbnail_b64)
            thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
            thumbnail_array = np.array(thumbnail_pil)
            
            return thumbnail_array
        except Exception as e:
            logger.error(f"Error retrieving thumbnail for {slice_id}: {e}")
            return None

    def get_thumbnail_base64(self, slice_id: str) -> Optional[str]:
        """
        Get thumbnail as base64 string by video name.
        
        Args:
            slice_id: Name of the video file
            
        Returns:
            Base64 encoded thumbnail or None if not found
        """
        if self.df is None or len(self.df) == 0 or slice_id not in self.df.index:
            return None
        
        try:
            row = self.df.loc[slice_id]
            return row.get('thumbnail', '')
        except Exception as e:
            logger.error(f"Error retrieving thumbnail base64 for {slice_id}: {e}")
            return None

    def list_videos(self) -> List[str]:
        """Get list of all slice_ids in the database."""
        if self.df is None or len(self.df) == 0:
            return []
        return self.df['slice_id'].tolist()
    
    def save(self):
        """Save database to parquet file."""
        if self.df is not None and len(self.df) > 0:
            try:
                df_to_save = self.df.reset_index(drop=True)
                df_to_save.to_parquet(self.database_path, index=False)
                logger.info(f"Saved {len(df_to_save)} embeddings to {self.database_path}")
            except Exception as e:
                logger.error(f"Error saving database: {e}")
                raise DatabaseError(f"Failed to save database: {e}")
        else:
            logger.info("No data to save")

    def _extract_and_encode_thumbnail(self, video_path: Path) -> tuple[str, tuple[int, int]]:
        """
        Extract thumbnail from video and encode as base64.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (base64_encoded_thumbnail, (width, height))
        """
        try:
            thumbnail_array = self.thumbnail_extractor.extract_thumbnail(video_path)
            
            if thumbnail_array is None:
                return "", (0, 0)
            
            from PIL import Image
            
            if isinstance(thumbnail_array, np.ndarray):
                thumbnail_pil = Image.fromarray(thumbnail_array)
            else:
                thumbnail_pil = thumbnail_array
            
            img_buffer = io.BytesIO()
            thumbnail_pil.save(img_buffer, format='JPEG', quality=95, optimize=True)
            thumbnail_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            width, height = thumbnail_pil.size
            
            logger.debug(f"Extracted thumbnail for {video_path.name}: {width}x{height}")
            return thumbnail_b64, (width, height)
        except Exception as e:
            logger.warning(f"Failed to extract thumbnail for {video_path}: {e}")
            return "", (0, 0)

    def _get_file_hash(self, file_path: Path) -> str:
        """Compute file hash for integrity checking."""
        if not file_path.exists():
            return ""
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.df is None or len(self.df) == 0:
            return {"total_embeddings": 0, "categories": {}}
            
        stats = {
            "total_embeddings": len(self.df),
            "categories": self.df['category'].value_counts().to_dict() if 'category' in self.df.columns else {},
            "embedding_dim": int(self.df['embedding_dim'].iloc[0]) if len(self.df) > 0 else 0,
            "database_size_mb": round(self.database_path.stat().st_size / (1024 * 1024), 2) if self.database_path.exists() else 0
        }
        
        return stats


class LakeFSParquetDatabase(ParquetVectorDatabase):
    """
    LakeFS-enabled Parquet database that stores embeddings in a LakeFS repository.
    """
    
    def __init__(self, database_path: Union[str, Path], config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize LakeFS Parquet vector database.
        
        Args:
            database_path: Path to the parquet database file (used as fallback/cache)
            config: Configuration object with LakeFS settings
        """
        self.config = config or VideoRetrievalConfig()
        
        if not self.config.use_lakefs:
            raise ValueError("LakeFS is not enabled in configuration. Set use_lakefs=True.")
        
        # Validate LakeFS configuration
        self._validate_lakefs_config()
        
        # Initialize LakeFS filesystem
        self._init_lakefs()
        
        # Set up paths
        self.local_database_path = Path(database_path)
        self.lakefs_path = f"{self.config.lakefs_repository}/{self.config.lakefs_branch}/{self.config.lakefs_embeddings_path}"
        
        # Initialize parent class with local path for caching
        super().__init__(database_path, config)
        
        # Try to load from LakeFS first
        self._load_from_lakefs()
    
    def _validate_lakefs_config(self):
        """Validate LakeFS configuration parameters."""
        required_params = ['lakefs_repository']
        for param in required_params:
            if not getattr(self.config, param, None):
                raise ValueError(f"LakeFS configuration missing: {param}")
        
        # Check for credentials - they should be in ~/.lakectl.yaml
        lakectl_path = Path.home() / '.lakectl.yaml'
        if not lakectl_path.exists():
            logger.warning("~/.lakectl.yaml not found. LakeFS credentials should be configured there.")
        else:
            logger.info("Using LakeFS credentials from ~/.lakectl.yaml")
    
    def _init_lakefs(self):
        """Initialize LakeFS filesystem connection."""
        try:
            # Import lakefs-spec
            from lakefs_spec import LakeFSFileSystem
            
            # Initialize filesystem (uses ~/.lakectl.yaml configuration)
            self.lakefs = LakeFSFileSystem()
            logger.info(f"Initialized LakeFS connection to repository: {self.config.lakefs_repository}")
            
        except ImportError:
            raise ImportError(
                "lakefs-spec is required for LakeFS integration. "
                "Install with: pip install lakefs-spec"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize LakeFS connection: {e}")
    
    def _load_from_lakefs(self):
        """Load database from LakeFS if available."""
        try:
            # Check if file exists in LakeFS
            if self.lakefs.exists(self.lakefs_path):
                logger.info(f"Loading database from LakeFS: {self.lakefs_path}")
                
                # Create temporary file to download to
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Download from LakeFS to temporary file
                    self.lakefs.get(self.lakefs_path, temp_path)
                    
                    # Load from temporary file
                    self.df = pd.read_parquet(temp_path)
                    
                    # Set slice_id as index for fast lookups
                    if 'slice_id' in self.df.columns and self.df.index.name != 'slice_id':
                        self.df = self.df.set_index('slice_id', drop=False)
                    
                    logger.info(f"Loaded {len(self.df)} embeddings from LakeFS")
                    
                    # Update local cache
                    self._save_local_cache()
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            else:
                logger.info(f"No existing database found in LakeFS at {self.lakefs_path}")
                # Fall back to local file if it exists
                if self.local_database_path.exists():
                    logger.info("Loading from local cache")
                    super().load()
                else:
                    self._create_empty_database()
                    
        except Exception as e:
            logger.warning(f"Failed to load from LakeFS: {e}")
            logger.info("Falling back to local database")
            super().load()
    
    def save(self):
        """Save database to both local cache and LakeFS."""
        if self.df is not None and len(self.df) > 0:
            try:
                # Save to local cache first
                self._save_local_cache()
                
                # Save to LakeFS
                self._save_to_lakefs()
                
                logger.info(f"Saved {len(self.df)} embeddings to both local cache and LakeFS")
                
            except Exception as e:
                logger.error(f"Error saving database: {e}")
                raise DatabaseError(f"Failed to save database: {e}")
        else:
            logger.info("No data to save")
    
    def _save_local_cache(self):
        """Save database to local cache file."""
        try:
            df_to_save = self.df.reset_index(drop=True)
            self.local_database_path.parent.mkdir(parents=True, exist_ok=True)
            df_to_save.to_parquet(self.local_database_path, index=False)
            logger.debug(f"Saved local cache to {self.local_database_path}")
        except Exception as e:
            logger.warning(f"Failed to save local cache: {e}")
    
    def _save_to_lakefs(self):
        """Save database to LakeFS."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save to temporary file
                df_to_save = self.df.reset_index(drop=True)
                df_to_save.to_parquet(temp_path, index=False)
                
                # Upload to LakeFS
                self.lakefs.put(temp_path, self.lakefs_path)
                logger.info(f"Uploaded database to LakeFS: {self.lakefs_path}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to save to LakeFS: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics with LakeFS information."""
        stats = super().get_statistics()
        
        # Add LakeFS-specific information
        stats.update({
            "storage_backend": "LakeFS",
            "lakefs_repository": self.config.lakefs_repository,
            "lakefs_branch": self.config.lakefs_branch,
            "lakefs_path": self.lakefs_path,
            "local_cache_path": str(self.local_database_path)
        })
        
        return stats


# UnifiedQueryManager removed - functionality moved to VideoSearchEngine for true unification
