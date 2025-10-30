"""
Configuration management for the ALFA 0.1 retrieval system.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import json
import os

@dataclass
class VideoRetrievalConfig:
    """Configuration for the ALFA 0.1 retrieval."""
    
    model_name: str = "models/nvidia_cosmos_embed_1"
    device: str = "cuda"
    batch_size: int = 4
    num_frames: int = 8
    # Unified embeddings and paths (single source of truth)
    embeddings_path: str = "data/unified_embeddings.parquet"  # All video embeddings (database + query)
    input_path: str = "data/unified_input_path.parquet"       # All video file paths
    use_safe_serialization: bool = True
    supported_formats: Tuple[str, ...] = ('.mp4', '.avi', '.mov')
    thumbnail_size: Tuple[int, int] = (480, 270)  # 16:9 aspect ratio, higher resolution
    resolution: Tuple[int, int] = (448, 448)
    default_top_k: int = 5
    similarity_threshold: float = 0.0
    # Threshold for filtering out low-quality results (e.g., 0.1 means results below 0.1 similarity are filtered)
    quality_threshold: float = 0.1
    # Default clip duration when span_end cannot be determined
    default_clip_duration: int = 20
    enable_caching: bool = True
    cache_size: int = 1000
    log_level: str = "INFO"
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def from_yaml(cls, path: str) -> 'VideoRetrievalConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> 'VideoRetrievalConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def validate(self):
        """Validate configuration values."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if self.quality_threshold < 0 or self.quality_threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        
        # Get project root (parent of core directory)
        project_root = Path(__file__).parent.parent
        
        
        # Check unified file paths if provided
        for file_path_attr in ['embeddings_path', 'input_path']:
            if hasattr(self, file_path_attr):
                file_path = getattr(self, file_path_attr)
                if file_path:
                    if not Path(file_path).is_absolute():
                        abs_path = project_root / file_path
                    else:
                        abs_path = Path(file_path)
                    
                    if not abs_path.exists():
                        print(f"Warning: {file_path_attr} does not exist: {abs_path}")


# Default configuration instance
DEFAULT_CONFIG = VideoRetrievalConfig()
