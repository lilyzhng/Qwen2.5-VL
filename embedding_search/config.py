"""
Configuration management for the video retrieval system.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class VideoRetrievalConfig:
    """Configuration for the video retrieval system."""
    
    # Model configuration
    model_name: str = "/Users/lilyzhang/Desktop/Qwen2.5-VL/cookbooks/nvidia_cosmos_embed_1"
    device: str = "cuda"
    batch_size: int = 4
    num_frames: int = 8
    
    # Database configuration
    database_path: str = "video_embeddings.pkl"
    use_safe_serialization: bool = True
    
    # Video processing
    supported_formats: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    thumbnail_size: Tuple[int, int] = (224, 224)
    resolution: Tuple[int, int] = (448, 448)  # Match model resolution
    
    # Search configuration
    default_top_k: int = 5
    similarity_threshold: float = 0.0
    
    # Paths
    video_database_dir: str = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/videos/video_database"
    user_input_dir: str = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/videos/user_input"
    
    # Performance
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Logging
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
        
        # Check if paths exist
        for path_attr in ['video_database_dir', 'user_input_dir']:
            path = getattr(self, path_attr)
            if path and not Path(path).exists():
                print(f"Warning: {path_attr} does not exist: {path}")


# Default configuration instance
DEFAULT_CONFIG = VideoRetrievalConfig()
