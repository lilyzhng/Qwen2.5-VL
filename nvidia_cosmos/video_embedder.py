"""
Video Embedding Extractor using NVIDIA Cosmos-Embed1-448p model.
"""

import torch
import numpy as np
import decord
from transformers import AutoProcessor, AutoModel
from pathlib import Path
from typing import Union, List, Dict, Tuple
import json
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoEmbedder:
    """Extract video embeddings using NVIDIA Cosmos-Embed1-448p model."""
    
    def __init__(self, model_name: str = "nvidia/Cosmos-Embed1-448p", device: str = "cuda"):
        """
        Initialize the video embedder.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on (cuda or cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing VideoEmbedder on {self.device}")
        
        # Load model and preprocessor
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True
        ).to(self.device, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32)
        
        self.preprocess = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def load_video_frames(self, video_path: Union[str, Path], num_frames: int = 8) -> np.ndarray:
        """
        Load video frames from a video file.
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to extract
            
        Returns:
            numpy array of frames in shape (num_frames, height, width, channels)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Load video using decord
        reader = decord.VideoReader(str(video_path))
        total_frames = len(reader)
        
        # Sample frames uniformly
        frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        frames = reader.get_batch(frame_ids).asnumpy()
        
        logger.info(f"Loaded {num_frames} frames from {video_path.name}")
        return frames
    
    def extract_embedding(self, video_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Extract video embedding from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing the embedding and metadata
        """
        video_path = Path(video_path)
        
        # Load video frames
        frames = self.load_video_frames(video_path)
        
        # Prepare batch for model (BTCHW format)
        batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
        
        # Process video
        with torch.no_grad():
            video_inputs = self.preprocess(videos=batch).to(
                self.device, 
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            video_out = self.model.get_video_embeddings(**video_inputs)
            
        # Extract normalized embedding
        embedding = video_out.visual_proj[0].cpu().numpy()
        
        return {
            "video_path": str(video_path),
            "embedding": embedding,
            "embedding_dim": embedding.shape[0],
            "num_frames": len(frames)
        }
    
    def extract_embeddings_batch(self, video_paths: List[Union[str, Path]]) -> List[Dict[str, np.ndarray]]:
        """
        Extract embeddings for multiple videos.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            List of dictionaries containing embeddings and metadata
        """
        embeddings = []
        
        for video_path in tqdm(video_paths, desc="Extracting embeddings"):
            try:
                embedding_data = self.extract_embedding(video_path)
                embeddings.append(embedding_data)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue
        
        return embeddings
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract text embedding for text-to-video search.
        
        Args:
            text: Text query
            
        Returns:
            Text embedding as numpy array
        """
        with torch.no_grad():
            text_inputs = self.preprocess(text=[text]).to(
                self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
            text_out = self.model.get_text_embeddings(**text_inputs)
            
        return text_out.text_proj[0].cpu().numpy()
