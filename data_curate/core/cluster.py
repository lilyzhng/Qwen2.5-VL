#!/usr/bin/env python3
"""
Clustering and dimensionality reduction module for embedding visualization.

This module provides functionality to:
1. Pre-reduce embeddings with PCA
2. Transform to 2D using UMAP
3. Cluster with DBSCAN
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
try:
    import umap.umap_ as umap
except ImportError:
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")
import json

logger = logging.getLogger(__name__)


class EmbeddingClusterer:
    """
    Handles dimensionality reduction and clustering of embeddings.
    
    Workflow:
    1. Pre-reduce high-dimensional embeddings with PCA
    2. Apply UMAP to get 2D coordinates
    3. Cluster with DBSCAN
    """
    
    def __init__(
        self,
        pca_components: int = 50,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = 'cosine',
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the clusterer with configurable parameters.
        
        Args:
            pca_components: Number of PCA components for pre-reduction
            umap_n_neighbors: Number of neighbors for UMAP
            umap_min_dist: Minimum distance for UMAP
            umap_metric: Metric for UMAP distance calculation
            dbscan_eps: Maximum distance between samples for DBSCAN
            dbscan_min_samples: Minimum samples in a neighborhood for DBSCAN
            random_state: Random seed for reproducibility
        """
        self.pca_components = pca_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.random_state = random_state
        
        # Initialize models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.umap_model = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            n_components=2,
            random_state=self.random_state
        )
        self.dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        
        # Storage for fitted data
        self.embeddings_2d = None
        self.cluster_labels = None
        
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the complete workflow: PCA -> UMAP -> DBSCAN.
        
        Args:
            embeddings: High-dimensional embeddings array (n_samples, n_features)
            
        Returns:
            Tuple of (2D coordinates, cluster labels)
        """
        logger.info(f"Processing {embeddings.shape[0]} embeddings with shape {embeddings.shape}")
        
        # Step 1: Standardize the embeddings
        logger.info("Standardizing embeddings...")
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Step 2: Pre-reduce with PCA
        logger.info(f"Applying PCA reduction to {self.pca_components} components...")
        # Handle case where we have fewer samples than components
        n_components = min(self.pca_components, embeddings.shape[0] - 1, embeddings.shape[1])
        if n_components != self.pca_components:
            logger.warning(f"Adjusting PCA components from {self.pca_components} to {n_components} due to data constraints")
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
        
        embeddings_pca = self.pca.fit_transform(embeddings_scaled)
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_variance:.2%}")
        
        # Step 3: Apply UMAP for 2D visualization
        logger.info("Applying UMAP for 2D projection...")
        self.embeddings_2d = self.umap_model.fit_transform(embeddings_pca)
        
        # Step 4: Cluster with DBSCAN
        logger.info("Clustering with DBSCAN...")
        self.cluster_labels = self.dbscan.fit_predict(self.embeddings_2d)
        
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        logger.info(f"Found {n_clusters} clusters and {n_noise} noise points")
        
        return self.embeddings_2d, self.cluster_labels
    
    def transform(self, new_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new embeddings using fitted models.
        
        Args:
            new_embeddings: New embeddings to transform
            
        Returns:
            Tuple of (2D coordinates, cluster labels)
        """
        # Standardize
        embeddings_scaled = self.scaler.transform(new_embeddings)
        
        # PCA transform
        embeddings_pca = self.pca.transform(embeddings_scaled)
        
        # UMAP transform
        embeddings_2d = self.umap_model.transform(embeddings_pca)
        
        # For clustering new points, we need to find nearest cluster
        # DBSCAN doesn't have a predict method, so we'll assign to nearest cluster center
        cluster_labels = self._assign_to_nearest_cluster(embeddings_2d)
        
        return embeddings_2d, cluster_labels
    
    def _assign_to_nearest_cluster(self, embeddings_2d: np.ndarray) -> np.ndarray:
        """
        Assign new points to nearest existing cluster.
        
        Args:
            embeddings_2d: 2D coordinates of new points
            
        Returns:
            Cluster labels for new points
        """
        if self.embeddings_2d is None or self.cluster_labels is None:
            raise ValueError("Model must be fitted before transforming new data")
        
        labels = []
        for point in embeddings_2d:
            # Find nearest point in original data
            distances = np.linalg.norm(self.embeddings_2d - point, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_label = self.cluster_labels[nearest_idx]
            
            # If nearest point is noise and distance is large, assign as noise
            if nearest_label == -1 and distances[nearest_idx] > self.dbscan_eps:
                labels.append(-1)
            else:
                labels.append(nearest_label)
        
        return np.array(labels)
    
    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about the clustering results.
        
        Returns:
            Dictionary with cluster statistics
        """
        if self.cluster_labels is None:
            return {}
        
        unique_labels = set(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[f"cluster_{label}"] = list(self.cluster_labels).count(label)
        
        return {
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "cluster_sizes": cluster_sizes,
            "pca_explained_variance": float(np.sum(self.pca.explained_variance_ratio_)) if hasattr(self.pca, 'explained_variance_ratio_') else None
        }
    
    def save_results(self, output_path: Union[str, Path], metadata: Optional[List[Dict]] = None) -> None:
        """
        Save clustering results to a JSON file.
        
        Args:
            output_path: Path to save results
            metadata: Optional metadata for each point (e.g., video names)
        """
        if self.embeddings_2d is None or self.cluster_labels is None:
            raise ValueError("No results to save. Run fit_transform first.")
        
        output_path = Path(output_path)
        
        results = []
        for i in range(len(self.embeddings_2d)):
            point_data = {
                "index": i,
                "x": float(self.embeddings_2d[i, 0]),
                "y": float(self.embeddings_2d[i, 1]),
                "cluster_id": int(self.cluster_labels[i])
            }
            
            # Add metadata if provided
            if metadata and i < len(metadata):
                point_data.update(metadata[i])
            
            results.append(point_data)
        
        # Add clustering statistics
        output_data = {
            "statistics": self.get_cluster_statistics(),
            "parameters": {
                "pca_components": self.pca_components,
                "umap_n_neighbors": self.umap_n_neighbors,
                "umap_min_dist": self.umap_min_dist,
                "umap_metric": self.umap_metric,
                "dbscan_eps": self.dbscan_eps,
                "dbscan_min_samples": self.dbscan_min_samples
            },
            "points": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved clustering results to {output_path}")
    
    @staticmethod
    def load_results(input_path: Union[str, Path]) -> Dict:
        """
        Load clustering results from a JSON file.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Dictionary with clustering results
        """
        input_path = Path(input_path)
        with open(input_path, 'r') as f:
            return json.load(f)


def process_embeddings_file(
    embeddings_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict:
    """
    Process embeddings from a numpy file and save clustering results.
    
    Args:
        embeddings_path: Path to .npy file containing embeddings
        output_path: Optional path to save results (defaults to same dir as input)
        **kwargs: Additional parameters for EmbeddingClusterer
        
    Returns:
        Dictionary with clustering results
    """
    embeddings_path = Path(embeddings_path)
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    # Initialize clusterer
    clusterer = EmbeddingClusterer(**kwargs)
    
    # Process embeddings
    coords_2d, cluster_labels = clusterer.fit_transform(embeddings)
    
    # Save results
    if output_path is None:
        output_path = embeddings_path.parent / f"{embeddings_path.stem}_clusters.json"
    
    clusterer.save_results(output_path)
    
    return {
        "coordinates_2d": coords_2d,
        "cluster_labels": cluster_labels,
        "statistics": clusterer.get_cluster_statistics()
    }


def visualize_clusters(
    coordinates_2d: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    point_names: Optional[List[str]] = None
) -> None:
    """
    Create a visualization of the clustering results.
    
    Args:
        coordinates_2d: 2D coordinates from UMAP
        cluster_labels: Cluster labels from DBSCAN
        save_path: Optional path to save the plot
        point_names: Optional names for each point
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        
        # Create color palette
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        colors = sns.color_palette('husl', n_clusters)
        
        # Plot each cluster
        for label, color in zip(range(n_clusters), colors):
            mask = cluster_labels == label
            plt.scatter(
                coordinates_2d[mask, 0],
                coordinates_2d[mask, 1],
                c=[color],
                label=f'Cluster {label}',
                alpha=0.6,
                s=50
            )
        
        # Plot noise points
        noise_mask = cluster_labels == -1
        if np.any(noise_mask):
            plt.scatter(
                coordinates_2d[noise_mask, 0],
                coordinates_2d[noise_mask, 1],
                c='gray',
                label='Noise',
                alpha=0.3,
                s=30
            )
        
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('Embedding Clusters (PCA -> UMAP -> DBSCAN)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available. Skipping visualization.")
