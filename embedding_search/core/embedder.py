"""
Embedding Extractor with batch processing and error handling.
"""

import torch
import numpy as np
try:
    import decord
    DECORD_AVAILABLE = True
except (ImportError, RuntimeError):
    DECORD_AVAILABLE = False
    print("Warning: decord not available. Video processing will be disabled.")
from transformers import AutoProcessor, AutoModel
import cv2
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import logging
from functools import wraps
import os
import random

from .base import EmbeddingModel, VideoProcessor
from .config import VideoRetrievalConfig
from .exceptions import (
    VideoNotFoundError, VideoLoadError, InvalidVideoFormatError,
    ModelLoadError, EmbeddingExtractionError
)

logger = logging.getLogger(__name__)


def validate_input_path(func):
    """Decorator to validate video paths or frame folders."""
    @wraps(func)
    def wrapper(self, input_path: Union[str, Path], *args, **kwargs):
        path = Path(input_path)
        if not path.exists():
            raise VideoNotFoundError(f"Input path not found: {path}")
        
        # Check if it's a video file
        if path.is_file() and path.suffix.lower() in self.config.supported_formats:
            return func(self, path, *args, **kwargs)
        
        # Check if it's a frame folder
        elif path.is_dir():
            # Check if directory contains image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = [f for f in path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            if not image_files:
                raise InvalidVideoFormatError(
                    f"Frame folder contains no valid image files: {path}. "
                    f"Supported image formats: {image_extensions}"
                )
            return func(self, path, *args, **kwargs)
        
        else:
            raise InvalidVideoFormatError(
                f"Input must be either a video file {self.config.supported_formats} "
                f"or a directory containing image frames: {path}"
            )
    return wrapper


class VideoFrameProcessor(VideoProcessor):
    """Handles video frame extraction and processing."""
    
    def __init__(self, config: VideoRetrievalConfig):
        self.config = config
    
    def load_frames(self, video_path: Path, num_frames: int = None) -> np.ndarray:
        """Load frames from a video file with error handling and automatic resolution scaling."""
        num_frames = num_frames or self.config.num_frames
        
        # Use OpenCV as fallback if decord is not available
        if not DECORD_AVAILABLE:
            return self._load_frames_opencv(video_path, num_frames)
            
        try:
            reader = decord.VideoReader(str(video_path))
            total_frames = len(reader)
            
            if total_frames == 0:
                raise VideoLoadError(f"Video has no frames: {video_path}")
            
            # Sample frames uniformly
            frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
            frames = reader.get_batch(frame_ids).asnumpy()
            
            target_height, target_width = self.config.resolution
            
            # Resize frames if they don't match target resolution
            original_height, original_width = frames.shape[1], frames.shape[2]
            if original_height != target_height or original_width != target_width:
                resized_frames = []
                for frame in frames:
                    # cv2.resize expects (width, height)
                    resized_frame = cv2.resize(frame, (target_width, target_height))
                    resized_frames.append(resized_frame)
                frames = np.array(resized_frames)
                logger.debug(f"Resized frames from {original_width}x{original_height} to {target_width}x{target_height}")
            
            logger.debug(f"Loaded {num_frames} frames from {video_path.name}")
            return frames
            
        except Exception as e:
            if isinstance(e, VideoLoadError):
                raise
            raise VideoLoadError(f"Failed to load video {video_path}: {str(e)}")
    
    def _load_frames_opencv(self, video_path: Path, num_frames: int) -> np.ndarray:
        """Load frames using OpenCV as fallback."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoLoadError(f"Cannot open video: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise VideoLoadError(f"Video has no frames: {video_path}")
            
            # Sample frames uniformly
            frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            
            target_height, target_width = self.config.resolution
            
            for frame_id in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (target_width, target_height))
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise VideoLoadError(f"No frames could be read from video: {video_path}")
            
            frames = np.array(frames)
            logger.debug(f"Loaded {len(frames)} frames from {video_path.name} using OpenCV")
            return frames
            
        except Exception as e:
            if isinstance(e, VideoLoadError):
                raise
            raise VideoLoadError(f"Failed to load video with OpenCV {video_path}: {str(e)}")
    
    def load_frames_from_folder(self, folder_path: Path, num_frames: int = None) -> np.ndarray:
        """Load and select frames from a folder containing image files."""
        num_frames = num_frames or self.config.num_frames
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise VideoLoadError(f"No image files found in folder: {folder_path}")
        
        # Sort files to ensure consistent ordering
        image_files.sort()
        
        # Select frames uniformly if we have more frames than needed
        if len(image_files) >= num_frames:
            # Sample frames uniformly across the available frames
            indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
            selected_files = [image_files[i] for i in indices]
        else:
            # If we have fewer frames than needed, use all available frames
            # and repeat the last frame to reach the desired count
            selected_files = image_files[:]
            while len(selected_files) < num_frames:
                selected_files.append(image_files[-1])
            logger.warning(f"Folder {folder_path.name} has only {len(image_files)} frames, "
                          f"repeating last frame to reach {num_frames} frames")
        
        # Load and resize frames
        frames = []
        target_height, target_width = self.config.resolution
        
        for img_path in selected_files:
            try:
                # Load image using OpenCV
                img = cv2.imread(str(img_path))
                if img is None:
                    raise VideoLoadError(f"Failed to load image: {img_path}")
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target resolution
                img = cv2.resize(img, (target_width, target_height))
                frames.append(img)
                
            except Exception as e:
                raise VideoLoadError(f"Failed to process image {img_path}: {str(e)}")
        
        frames = np.array(frames)
        logger.debug(f"Loaded {len(frames)} frames from folder {folder_path.name}")
        return frames
    
    def validate_video(self, video_path: Path) -> bool:
        """Validate if a video file can be processed."""
        try:
            frames = self.load_frames(video_path, num_frames=1)
            return frames.shape[0] > 0
        except:
            return False
    
    def validate_input(self, input_path: Path) -> bool:
        """Validate if an input (video file or frame folder) can be processed."""
        try:
            if input_path.is_file():
                return self.validate_video(input_path)
            elif input_path.is_dir():
                frames = self.load_frames_from_folder(input_path, num_frames=1)
                return frames.shape[0] > 0
            return False
        except:
            return False


class CosmosVideoEmbedder(EmbeddingModel):
    """Extract video embeddings with batch processing."""
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize the video embedder.
        
        Args:
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.video_processor = VideoFrameProcessor(self.config)
        
        # Use CPU fallback when CUDA unavailable to prevent runtime errors
        self.device = self.config.device if torch.cuda.is_available() else "cpu"
        if self.device != self.config.device:
            logger.warning(f"CUDA not available, using CPU instead of {self.config.device}")
        
        # Use bfloat16 when available for better performance, fallback to float32
        if self.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            logger.info(f"Using bfloat16 precision on {self.device} (optimal for Cosmos model)")
        else:
            self.dtype = torch.float32
            logger.info(f"Using float32 precision on {self.device} (bfloat16 not supported)")
        
        logger.info(f"Initializing CosmosVideoEmbedder on {self.device} with dtype {self.dtype}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.config.model_name, 
                trust_remote_code=True
            ).to(self.device, dtype=self.dtype)
            
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
                inputs = self.preprocess(videos=dummy_batch).to(self.device, dtype=self.dtype)
                outputs = self.model.get_video_embeddings(**inputs)
                self._embedding_dim = outputs.visual_proj.shape[-1]
        
        return self._embedding_dim
    
    @validate_input_path
    def extract_video_embedding(self, input_path: Path) -> np.ndarray:
        """
        Extract embedding from a single video file or frame folder.
        
        Args:
            input_path: Path to the video file or frame folder
            
        Returns:
            Normalized embedding vector
        """
        try:
            # Determine if input is a video file or frame folder
            if input_path.is_file():
                frames = self.video_processor.load_frames(input_path)
            elif input_path.is_dir():
                frames = self.video_processor.load_frames_from_folder(input_path)
            else:
                raise InvalidVideoFormatError(f"Input path must be a file or directory: {input_path}")
            
            # Prepare batch for model (BTCHW format)
            batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
            
            # Disable gradients for inference to save memory and improve speed
            with torch.no_grad():
                video_inputs = self.preprocess(videos=batch).to(
                    self.device, 
                    dtype=self.dtype
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
            raise EmbeddingExtractionError(f"Failed to extract embedding from {input_path}: {str(e)}")
    
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
                    dtype=self.dtype
                )
                text_out = self.model.get_text_embeddings(**text_inputs)
                
            embedding = text_out.text_proj[0].cpu().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to extract text embedding: {str(e)}")
    
    def extract_video_embeddings_batch(self, 
                                     input_paths: List[Path], 
                                     batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Extract embeddings from multiple videos or frame folders using batch processing.
        
        Args:
            input_paths: List of video file paths or frame folder paths
            batch_size: Batch size for processing (uses config if not specified)
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        batch_size = batch_size or self.config.batch_size
        embeddings_data = []
        failed_inputs = []
        
        with tqdm(total=len(input_paths), desc="Extracting embeddings") as pbar:
            for i in range(0, len(input_paths), batch_size):
                batch_paths = input_paths[i:i+batch_size]
                batch_frames = []
                valid_paths = []
                
                for path in batch_paths:
                    try:
                        if not path.exists():
                            raise VideoNotFoundError(f"Input not found: {path}")
                        
                        # Load frames based on input type
                        if path.is_file():
                            if not path.suffix.lower() in self.config.supported_formats:
                                raise InvalidVideoFormatError(f"Unsupported format: {path.suffix}")
                            frames = self.video_processor.load_frames(path)
                        elif path.is_dir():
                            frames = self.video_processor.load_frames_from_folder(path)
                        else:
                            raise InvalidVideoFormatError(f"Input must be a file or directory: {path}")
                        
                        batch_frames.append(frames)
                        valid_paths.append(path)
                        
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")
                        failed_inputs.append({"path": str(path), "error": str(e)})
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
                    
                    with torch.no_grad():
                        inputs = self.preprocess(videos=batch_tensor).to(
                            self.device,
                            dtype=self.dtype
                        )
                        outputs = self.model.get_video_embeddings(**inputs)
                    
                    # Extract embeddings
                    for j, path in enumerate(valid_paths):
                        embedding = outputs.visual_proj[j].cpu().numpy()
                        
                        # Normalize
                        embedding = embedding / np.linalg.norm(embedding)
                        
                        # Determine input type for metadata
                        input_type = "video" if path.is_file() else "frame_folder"
                        
                        embeddings_data.append({
                            "input_path": str(path),
                            "input_type": input_type,
                            "embedding": embedding,
                            "embedding_dim": embedding.shape[0],
                            "num_frames": len(batch_frames[j])
                        })
                        
                    pbar.update(len(valid_paths))
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    for path in valid_paths:
                        failed_inputs.append({"path": str(path), "error": f"Batch processing error: {str(e)}"})
                    pbar.update(len(valid_paths))
        
        if failed_inputs:
            logger.warning(f"Failed to process {len(failed_inputs)} inputs")
            for fail in failed_inputs[:5]:  # Show first 5 failures
                logger.warning(f"  - {fail['path']}: {fail['error']}")
            if len(failed_inputs) > 5:
                logger.warning(f"  ... and {len(failed_inputs) - 5} more")
        
        logger.info(f"Successfully processed {len(embeddings_data)} out of {len(input_paths)} inputs")
        
        return embeddings_data
