#!/usr/bin/env python3
"""
Test script to verify the embedding_search implementation
"""

import sys
import logging
from pathlib import Path
import torch
import numpy as np

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from core.config import VideoRetrievalConfig
from core.embedder import CosmosVideoEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test if the model loads correctly from the local path."""
    logger.info("Testing model loading...")
    
    try:
        config = VideoRetrievalConfig()
        embedder = CosmosVideoEmbedder(config)
        logger.info("✓ Model loaded successfully")
        
        # Test getting embedding dimension
        dim = embedder.embedding_dim
        logger.info(f"✓ Embedding dimension: {dim}")
        
        return embedder
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return None


def test_text_embedding(embedder):
    """Test text embedding extraction."""
    logger.info("Testing text embedding extraction...")
    
    try:
        test_text = "a person riding a motorcycle in the night"
        embedding = embedder.extract_text_embedding(test_text)
        
        logger.info(f"✓ Text embedding shape: {embedding.shape}")
        logger.info(f"✓ Text embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return embedding
    except Exception as e:
        logger.error(f"✗ Text embedding failed: {e}")
        return None


def test_video_embedding(embedder):
    """Test video embedding extraction if videos are available."""
    logger.info("Testing video embedding extraction...")
    
    video_dir = project_root / "data" / "videos" / "video_database"
    if not video_dir.exists():
        logger.warning("✗ Video directory not found, skipping video test")
        return None
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning("✗ No video files found, skipping video test")
        return None
    
    try:
        test_video = video_files[0]
        logger.info(f"Testing with video: {test_video.name}")
        
        embedding = embedder.extract_video_embedding(test_video)
        
        logger.info(f"✓ Video embedding shape: {embedding.shape}")
        logger.info(f"✓ Video embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return embedding
    except Exception as e:
        logger.error(f"✗ Video embedding failed: {e}")
        return None


def test_similarity_calculation(text_embedding, video_embedding):
    """Test similarity calculation between text and video embeddings."""
    if text_embedding is None or video_embedding is None:
        logger.warning("✗ Skipping similarity test - missing embeddings")
        return
        
    logger.info("Testing similarity calculation...")
    
    try:
        # Calculate cosine similarity
        similarity = np.dot(text_embedding, video_embedding)
        logger.info(f"✓ Text-Video similarity: {similarity:.4f}")
        
        # Verify embeddings are normalized
        text_norm = np.linalg.norm(text_embedding)
        video_norm = np.linalg.norm(video_embedding)
        logger.info(f"✓ Text embedding norm: {text_norm:.4f}")
        logger.info(f"✓ Video embedding norm: {video_norm:.4f}")
        
        return similarity
    except Exception as e:
        logger.error(f"✗ Similarity calculation failed: {e}")
        return None


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("alpha 0.1 Implementation Test")
    logger.info("=" * 60)
    
    # Test model loading
    embedder = test_model_loading()
    if embedder is None:
        logger.error("Model loading failed. Cannot continue with tests.")
        return 1
    
    # Test text embedding
    text_embedding = test_text_embedding(embedder)
    
    # Test video embedding
    video_embedding = test_video_embedding(embedder)
    
    # Test similarity calculation
    test_similarity_calculation(text_embedding, video_embedding)
    
    logger.info("=" * 60)
    logger.info("Test completed!")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
