"""
PCA Feature Visualization Module.
Based on the official Cosmos-Embed1 demo for advanced feature analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any, List
import logging
from PIL import Image
import cv2

from .embedder import CosmosVideoEmbedder
from .config import VideoRetrievalConfig
from .exceptions import VideoLoadError, EmbeddingExtractionError

logger = logging.getLogger(__name__)


def get_robust_pca(
    features: torch.Tensor, 
    m: float = 2, 
    remove_first_component: bool = False, 
    skip: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Robust PCA for feature visualization.
    Based on official Cosmos-Embed1 demo implementation.
    
    Args:
        features: Feature tensor of shape (N, C)
        m: Robustness parameter for outlier removal
        remove_first_component: Whether to remove the first component
        skip: Number of components to skip
        
    Returns:
        Tuple of (reduction_matrix, color_min, color_max)
    """
    assert len(features.shape) == 2, "features should be (N, C)"
    
    reduction_mat = torch.pca_lowrank(features, q=3 + skip, niter=20)[2]
    reduction_mat = reduction_mat[:, skip:]
    colors = features @ reduction_mat
    
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size: Tuple[int, int],
    interpolation: str = "bicubic",
    return_pca_stats: bool = False,
    pca_stats: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    skip_components: int = 0,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Create PCA visualization map from feature maps.
    Based on official Cosmos-Embed1 demo implementation.
    
    Args:
        feature_map: Feature map tensor
        img_size: Target image size (height, width)
        interpolation: Interpolation method
        return_pca_stats: Whether to return PCA statistics
        pca_stats: Pre-computed PCA statistics
        skip_components: Number of components to skip
        
    Returns:
        PCA color map as numpy array, optionally with PCA statistics
    """
    if feature_map.shape[0] != 1:
        feature_map = feature_map[None]
    
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1]), skip=skip_components,
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color


class FeatureVisualizer:
    """
    Advanced feature visualization for Cosmos video embeddings.
    Provides PCA-based visualization of intermediate features.
    """
    
    def __init__(self, embedder: Optional[CosmosVideoEmbedder] = None, config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize feature visualizer.
        
        Args:
            embedder: Video embedder instance
            config: Configuration object
        """
        self.config = config or VideoRetrievalConfig()
        self.embedder = embedder or CosmosVideoEmbedder(self.config)
        
    def extract_dense_features(self, video_path: Union[str, Path]) -> torch.Tensor:
        """
        Extract dense per-frame features for visualization.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dense features tensor of shape (num_frames, height, width, channels)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoLoadError(f"Video not found: {video_path}")
        
        try:
            # Load and process frames
            frames = self.embedder.video_processor.load_frames(video_path)
            
            # Prepare batch for model (BTCHW format)
            batch = np.transpose(np.expand_dims(frames, 0), (0, 1, 4, 2, 3))
            
            with torch.no_grad():
                video_inputs = self.embedder.preprocess(videos=batch).to(
                    self.embedder.device, 
                    dtype=self.embedder.dtype
                )
                video_out = self.embedder.model.get_video_embeddings(**video_inputs)
                
                # Extract dense features if available
                if hasattr(video_out, 'visual_embs'):
                    dense_features = video_out.visual_embs[0]  # Remove batch dimension
                else:
                    logger.warning("Model doesn't provide visual_embs, using visual_proj instead")
                    # Fallback: reshape visual_proj to simulate dense features
                    visual_proj = video_out.visual_proj[0]  # Shape: (embedding_dim,)
                    # Create a dummy spatial structure
                    spatial_size = int(np.sqrt(visual_proj.shape[0] // frames.shape[0]))
                    if spatial_size * spatial_size * frames.shape[0] == visual_proj.shape[0]:
                        dense_features = visual_proj.reshape(frames.shape[0], spatial_size, spatial_size, -1)
                    else:
                        # If reshaping doesn't work, create a simple feature map
                        dense_features = visual_proj.unsqueeze(0).unsqueeze(0).expand(
                            frames.shape[0], 14, 14, -1  # Assume 14x14 spatial resolution
                        )
                
            return dense_features.to("cpu", dtype=torch.float32)
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to extract dense features from {video_path}: {str(e)}")
    
    def create_pca_visualization(
        self, 
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        num_keyframes: int = 30,
        interpolation: str = "bilinear"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create PCA visualization for a video.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save visualization
            num_keyframes: Number of keyframes for PCA computation
            interpolation: Interpolation method
            
        Returns:
            Tuple of (original_frames, pca_visualized_frames)
        """
        video_path = Path(video_path)
        
        try:
            # Extract dense features
            dense_features = self.extract_dense_features(video_path)
            logger.info(f"Extracted dense features: {dense_features.shape}")
            
            # Load original frames for comparison
            original_frames = self.embedder.video_processor.load_frames(video_path)
            
            # Compute PCA statistics from keyframes
            kf_stride = max(dense_features.shape[0] // num_keyframes, 1)
            sampled_features = dense_features[::kf_stride]
            pca_stats = get_robust_pca(sampled_features.flatten(0, 2))
            
            # Generate PCA visualizations for all frames
            pca_frames = []
            for i, (raw_frame, features) in enumerate(zip(original_frames, dense_features)):
                try:
                    pca_features = get_pca_map(
                        features, 
                        raw_frame.shape[:2], 
                        pca_stats=pca_stats, 
                        interpolation=interpolation
                    )
                    pca_features = np.floor(pca_features * 255.0).astype(np.uint8)
                    pca_frames.append(pca_features)
                except Exception as e:
                    logger.warning(f"Failed to create PCA map for frame {i}: {e}")
                    # Fallback: create a placeholder
                    pca_frames.append(np.zeros_like(raw_frame))
            
            pca_frames = np.stack(pca_frames)
            
            # Save visualization if output path provided
            if output_path:
                self._save_pca_video(original_frames, pca_frames, output_path)
            
            return original_frames, pca_frames
            
        except Exception as e:
            raise EmbeddingExtractionError(f"Failed to create PCA visualization: {str(e)}")
    
    def _save_pca_video(
        self, 
        original_frames: np.ndarray, 
        pca_frames: np.ndarray, 
        output_path: Union[str, Path]
    ):
        """
        Save PCA visualization as a side-by-side video.
        
        Args:
            original_frames: Original video frames
            pca_frames: PCA visualized frames
            output_path: Output video path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Combine original and PCA frames side by side
            combined_frames = []
            for orig, pca in zip(original_frames, pca_frames):
                combined = np.concatenate((orig, pca), axis=1)
                combined_frames.append(combined)
            
            combined_frames = np.stack(combined_frames)
            
            # Save using OpenCV
            height, width = combined_frames.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (width, height))
            
            for frame in combined_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"Saved PCA visualization to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save PCA video: {e}")
    
    def analyze_temporal_stability(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze temporal stability of features across frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with stability metrics
        """
        try:
            dense_features = self.extract_dense_features(video_path)
            
            # Flatten spatial dimensions
            features_flat = dense_features.flatten(1, 2)  # (frames, spatial*channels)
            
            # Calculate frame-to-frame similarities
            similarities = []
            for i in range(len(features_flat) - 1):
                f1 = features_flat[i].flatten()
                f2 = features_flat[i + 1].flatten()
                
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(f1, f2, dim=0)
                similarities.append(sim.item())
            
            similarities = np.array(similarities)
            
            return {
                "mean_temporal_similarity": float(similarities.mean()),
                "std_temporal_similarity": float(similarities.std()),
                "min_temporal_similarity": float(similarities.min()),
                "max_temporal_similarity": float(similarities.max()),
                "frame_similarities": similarities.tolist()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal stability: {e}")
            return {"error": str(e)}
