"""
Improved Video Embedding Extractor with batch processing and better error handling.
"""

import torch
import numpy as np
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Warning: decord not available. Video processing will be disabled.")
from transformers import AutoProcessor, AutoModel
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import logging
from functools import wraps

from base import EmbeddingModel, VideoProcessor
from config import VideoRetrievalConfig
from exceptions import (
    VideoNotFoundError, VideoLoadError, InvalidVideoFormatError,
    ModelLoadError, EmbeddingExtractionError
)

logger = logging.getLogger(__name__)


def validate_video_path(func):
    """Decorator to validate video paths."""
    @wraps(func)
    def wrapper(self, video_path: Union[str, Path], *args, **kwargs):
        path = Path(video_path)
        if not path.exists():
            raise VideoNotFoundError(f"Video not found: {path}")
        if not path.suffix.lower() in self.config.supported_formats:
            raise InvalidVideoFormatError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {self.config.supported_formats}"
            )
        return func(self, path, *args, **kwargs)
    return wrapper


class VideoFrameProcessor(VideoProcessor):
    """Handles video frame extraction and processing."""
    
    def __init__(self, config: VideoRetrievalConfig):
        self.config = config
    
    def load_frames(self, video_path: Path, num_frames: int = None) -> np.ndarray:
        """Load frames from a video file with error handling."""
        if not DECORD_AVAILABLE:
            raise VideoLoadError("decord package is not available. Please install it for video processing.")
            
        num_frames = num_frames or self.config.num_frames
        
        try:
            reader = decord.VideoReader(str(video_path))
            total_frames = len(reader)
            
            if total_frames == 0:
                raise VideoLoadError(f"Video has no frames: {video_path}")
            
            # Sample frames uniformly
            frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            frames = reader.get_batch(frame_ids).asnumpy()
            
            logger.debug(f"Loaded {num_frames} frames from {video_path.name}")
            return frames
            
        except Exception as e:
            if isinstance(e, VideoLoadError):
                raise
            raise VideoLoadError(f"Failed to load video {video_path}: {str(e)}")
    
    def validate_video(self, video_path: Path) -> bool:
        """Validate if a video file can be processed."""
        try:
            frames = self.load_frames(video_path, num_frames=1)
            return frames.shape[0] > 0
        except:
            return False


class CosmosVideoEmbedder(EmbeddingModel):
    """Extract video embeddings using NVIDIA Cosmos model with batch processing."""
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize the video embedder.
        
        Args:
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.video_processor = VideoFrameProcessor(self.config)
        
        # Set device
        self.device = self.config.device if torch.cuda.is_available() else "cpu"
        if self.device != self.config.device:
            logger.warning(f"CUDA not available, using CPU instead of {self.config.device}")
        
        logger.info(f"Initializing CosmosVideoEmbedder on {self.device}")
        
        # Load model and preprocessor
        try:
            # Load the model from local path
            self.model = AutoModel.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            ).to(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.preprocess = AutoProcessor.from_pretrained(
                self.config.model_name,
                resolution=self.config.resolution if hasattr(self.config, 'resolution') else (448, 448),
                trust_remote_code=True
            )
            
            # Cache embedding dimension
            self._embedding_dim = None
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.config.model_name}: {str(e)}")
    
    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._embedding_dim is None:
            # Extract a dummy embedding to get dimension
            dummy_frames = np.zeros((self.config.num_frames, 224, 224, 3), dtype=np.uint8)
            dummy_batch = np.transpose(np.expand_dims(dummy_frames, 0), (0, 1, 4, 2, 3))
            
            with torch.no_grad():
                inputs = self.preprocess(videos=dummy_batch).to(self.device)
                outputs = self.model.get_video_embeddings(**inputs)
                self._embedding_dim = outputs.visual_proj.shape[-1]
        
        return self._embedding_dim
    
    @validate_video_path
    def extract_video_embedding(self, video_path: Path) -> np.ndarray:
        """
        Extract embedding from a single video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Load video frames
            frames = self.video_processor.load_frames(video_path)
            
            # Prepare batch for model (BTCHW format)
            batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
            
            # Process video (following official implementation pattern)
            with torch.no_grad():
                video_inputs = self.preprocess(videos=batch).to(
                    self.device, 
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )
                video_out = self.model.get_video_embeddings(**video_inputs)
                
            # Extract normalized embedding
            embedding = video_out.visual_proj[0].cpu().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            if isinstance(e, (VideoNotFoundError, VideoLoadError, InvalidVideoFormatError)):
                raise
            raise EmbeddingExtractionError(f"Failed to extract embedding from {video_path}: {str(e)}")
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract embedding from text query.
        
        Args:
            text: Text query
            
        Returns:
            Normalized embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text query cannot be empty")
        
        try:
            with torch.no_grad():
                text_inputs = self.preprocess(text=[text]).to(
                    self.device,
                    dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                )
                text_out = self.model.get_text_embeddings(**text_inputs)
                
            embedding = text_out.text_proj[0].cpu().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to extract text embedding: {str(e)}")
    
    def extract_video_embeddings_batch(self, 
                                     video_paths: List[Path], 
                                     batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract embeddings from multiple videos using batch processing.
        
        Args:
            video_paths: List of video file paths
            batch_size: Batch size for processing (uses config if not specified)
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        batch_size = batch_size or self.config.batch_size
        embeddings_data = []
        failed_videos = []
        
        # Process videos in batches
        with tqdm(total=len(video_paths), desc="Extracting embeddings") as pbar:
            for i in range(0, len(video_paths), batch_size):
                batch_paths = video_paths[i:i+batch_size]
                batch_frames = []
                valid_paths = []
                
                # Load frames for batch
                for path in batch_paths:
                    try:
                        # Validate path
                        if not path.exists():
                            raise VideoNotFoundError(f"Video not found: {path}")
                        if not path.suffix.lower() in self.config.supported_formats:
                            raise InvalidVideoFormatError(f"Unsupported format: {path.suffix}")
                        
                        frames = self.video_processor.load_frames(path)
                        batch_frames.append(frames)
                        valid_paths.append(path)
                        
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")
                        failed_videos.append({"path": str(path), "error": str(e)})
                        pbar.update(1)
                        continue
                
                if not batch_frames:
                    continue
                
                try:
                    # Stack frames for batch processing
                    # Each video becomes TCHW, stack to BTCHW
                    batch_videos = []
                    for frames in batch_frames:
                        video_tensor = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
                        batch_videos.append(video_tensor[0])  # Remove batch dimension
                    
                    batch_tensor = np.stack(batch_videos, axis=0)
                    
                    # Process batch
                    with torch.no_grad():
                        inputs = self.preprocess(videos=batch_tensor).to(
                            self.device,
                            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                        )
                        outputs = self.model.get_video_embeddings(**inputs)
                    
                    # Extract embeddings
                    for j, path in enumerate(valid_paths):
                        embedding = outputs.visual_proj[j].cpu().numpy()
                        
                        # Normalize
                        embedding = embedding / np.linalg.norm(embedding)
                        
                        embeddings_data.append({
                            "video_path": str(path),
                            "embedding": embedding,
                            "embedding_dim": embedding.shape[0],
                            "num_frames": len(batch_frames[j])
                        })
                        
                    pbar.update(len(valid_paths))
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    for path in valid_paths:
                        failed_videos.append({"path": str(path), "error": f"Batch processing error: {str(e)}"})
                    pbar.update(len(valid_paths))
        
        if failed_videos:
            logger.warning(f"Failed to process {len(failed_videos)} videos")
            for fail in failed_videos[:5]:  # Show first 5 failures
                logger.warning(f"  - {fail['path']}: {fail['error']}")
            if len(failed_videos) > 5:
                logger.warning(f"  ... and {len(failed_videos) - 5} more")
        
        logger.info(f"Successfully processed {len(embeddings_data)} out of {len(video_paths)} videos")
        
        return embeddings_data
