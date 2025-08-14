"""
Advanced visualization techniques for high-dimensional video embeddings.
Better alternatives to t-SNE for the NVIDIA Cosmos model.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Any
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ParametricProjection(nn.Module):
    """
    Neural network for parametric dimensionality reduction.
    Learns a mapping from high-dimensional embeddings to 2D/3D space.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 2, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        current_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False

try:
    import trimap
    TRIMAP_AVAILABLE = True
except ImportError:
    TRIMAP_AVAILABLE = False
    logger.warning("TriMAP not available. Install with: pip install trimap")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Parametric methods require torch.")


class AdvancedEmbeddingVisualizer:
    """
    Advanced visualization techniques for high-dimensional embeddings.
    
    Provides better alternatives to t-SNE:
    1. UMAP - Better preserves global structure
    2. PCA - Linear, interpretable
    3. Hierarchical clustering visualization
    4. Interactive 3D projections
    5. Similarity heatmaps
    """
    
    def __init__(self):
        self.projection_cache = {}
        
    def create_umap_projection(self, embeddings: np.ndarray, 
                             n_neighbors: int = 15,
                             min_dist: float = 0.1,
                             n_components: int = 2,
                             metric: str = 'cosine') -> np.ndarray:
        """
        Create UMAP projection - often better than t-SNE for embeddings.
        
        UMAP advantages:
        - Preserves both local and global structure
        - Faster than t-SNE
        - More stable results
        - Better for large datasets
        
        Args:
            embeddings: High-dimensional embeddings
            n_neighbors: Balance local vs global structure (5-50)
            min_dist: Minimum distance between points (0.0-1.0)
            n_components: Output dimensions (2 or 3)
            metric: Distance metric ('cosine', 'euclidean', etc.)
            
        Returns:
            Low-dimensional projection
        """
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to PCA")
            return self.create_pca_projection(embeddings, n_components)
        
        cache_key = f"umap_{n_neighbors}_{min_dist}_{n_components}_{metric}"
        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]
        
        logger.info(f"Computing UMAP projection with {n_components}D output")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=42
        )
        
        projection = reducer.fit_transform(embeddings)
        self.projection_cache[cache_key] = projection
        
        return projection
    
    def create_pca_projection(self, embeddings: np.ndarray, 
                            n_components: int = 2) -> np.ndarray:
        """
        Create PCA projection - linear and interpretable.
        
        PCA advantages:
        - Deterministic results
        - Fast computation
        - Interpretable components
        - Good for initial exploration
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Output dimensions
            
        Returns:
            Low-dimensional projection with explained variance info
        """
        cache_key = f"pca_{n_components}"
        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]
        
        logger.info(f"Computing PCA projection with {n_components}D output")
        
        pca = PCA(n_components=n_components, random_state=42)
        projection = pca.fit_transform(embeddings)
        
        # Store explained variance for interpretation
        self.explained_variance_ratio = pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        logger.info(f"PCA explained variance: {self.explained_variance_ratio[:5]}")
        logger.info(f"Cumulative variance: {self.cumulative_variance[:5]}")
        
        self.projection_cache[cache_key] = projection
        return projection
    
    def create_trimap_projection(self, embeddings: np.ndarray,
                               n_inliers: int = 10,
                               n_outliers: int = 5,
                               n_random: int = 5,
                               n_components: int = 2) -> np.ndarray:
        """
        Create TriMAP projection - often better than t-SNE for large datasets.
        
        TriMAP advantages:
        - Faster than t-SNE
        - Better preserves global structure
        - More robust to outliers
        - Scalable to large datasets
        
        Args:
            embeddings: High-dimensional embeddings
            n_inliers: Number of inlier points for triplet generation
            n_outliers: Number of outlier points for triplet generation  
            n_random: Number of random points for triplet generation
            n_components: Output dimensions
            
        Returns:
            Low-dimensional projection
        """
        if not TRIMAP_AVAILABLE:
            logger.warning("TriMAP not available, falling back to UMAP")
            return self.create_umap_projection(embeddings, n_components=n_components)
        
        cache_key = f"trimap_{n_inliers}_{n_outliers}_{n_random}_{n_components}"
        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]
        
        logger.info(f"Computing TriMAP projection with {n_components}D output")
        
        embedding = trimap.TRIMAP(
            n_inliers=n_inliers,
            n_outliers=n_outliers,
            n_random=n_random,
            n_dims=n_components,
            verbose=False
        )
        
        projection = embedding.fit_transform(embeddings)
        self.projection_cache[cache_key] = projection
        
        return projection
    
    def train_parametric_projection(self, embeddings: np.ndarray,
                                  method: str = 'parametric_umap',
                                  n_components: int = 2,
                                  epochs: int = 100,
                                  batch_size: int = 64,
                                  learning_rate: float = 0.001) -> 'ParametricProjection':
        """
        Train a parametric neural network projection.
        
        Args:
            embeddings: Training embeddings
            method: 'parametric_umap' or 'autoencoder'
            n_components: Output dimensions
            epochs: Training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Trained parametric model
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot train parametric projection")
            return None
        
        input_dim = embeddings.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create target projection using UMAP
        if method == 'parametric_umap':
            target_projection = self.create_umap_projection(embeddings, n_components=n_components)
        else:  # autoencoder
            target_projection = self.create_pca_projection(embeddings, n_components=n_components)
        
        # Create model
        model = ParametricProjection(input_dim, n_components).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Prepare data
        X_tensor = torch.FloatTensor(embeddings).to(device)
        y_tensor = torch.FloatTensor(target_projection).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        model.eval()
        return model
    
    def create_multiscale_visualization(self, embeddings: np.ndarray,
                                      metadata: List[Dict],
                                      scales: List[float] = None,
                                      method: str = 'umap') -> go.Figure:
        """
        Create multiscale visualization showing structure at different scales.
        
        Args:
            embeddings: High-dimensional embeddings
            metadata: Video metadata
            scales: List of scale parameters for UMAP min_dist
            method: Base projection method
            
        Returns:
            Multi-subplot figure showing different scales
        """
        if scales is None:
            scales = [0.01, 0.1, 0.5, 1.0]
        
        n_scales = len(scales)
        cols = min(2, n_scales)
        rows = (n_scales + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'Scale: {scale}' for scale in scales],
            specs=[[{'type': 'scatter'}] * cols for _ in range(rows)]
        )
        
        # Create projections at different scales
        for i, scale in enumerate(scales):
            row = i // cols + 1
            col = i % cols + 1
            
            if method == 'umap':
                projection = self.create_umap_projection(
                    embeddings, min_dist=scale, n_components=2
                )
            else:
                projection = self.create_pca_projection(embeddings, n_components=2)
            
            # Prepare data
            categories = [m.get('category', 'unknown') for m in metadata]
            video_names = [m.get('video_name', f'Video_{i}') for i, m in enumerate(metadata)]
            
            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=projection[:, 0],
                    y=projection[:, 1],
                    mode='markers',
                    text=video_names,
                    marker=dict(
                        color=[hash(cat) % 10 for cat in categories],
                        colorscale='Set3',
                        size=6,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    ),
                    name=f'Scale {scale}',
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Multiscale Embedding Visualization",
            height=300 * rows,
            template='plotly_white'
        )
        
        return fig
    
    def create_temporal_visualization(self, video_sequences: List[np.ndarray],
                                    timestamps: List[List[float]],
                                    method: str = 'force_directed') -> go.Figure:
        """
        Create temporal visualization for video sequences.
        
        Args:
            video_sequences: List of video embedding sequences
            timestamps: Corresponding timestamps for each sequence
            method: 'force_directed', 'trajectory', or 'flow'
            
        Returns:
            Temporal visualization figure
        """
        # Flatten all embeddings with sequence and time info
        all_embeddings = []
        all_metadata = []
        
        for seq_idx, (sequence, times) in enumerate(zip(video_sequences, timestamps)):
            for frame_idx, (embedding, timestamp) in enumerate(zip(sequence, times)):
                all_embeddings.append(embedding)
                all_metadata.append({
                    'sequence_id': seq_idx,
                    'frame_id': frame_idx,
                    'timestamp': timestamp,
                    'video_name': f'Sequence_{seq_idx}_Frame_{frame_idx}'
                })
        
        all_embeddings = np.array(all_embeddings)
        
        # Create 2D projection
        projection = self.create_umap_projection(all_embeddings, n_components=2)
        
        if method == 'trajectory':
            # Show trajectories as connected lines
            fig = go.Figure()
            
            # Group by sequence
            sequences = {}
            for i, meta in enumerate(all_metadata):
                seq_id = meta['sequence_id']
                if seq_id not in sequences:
                    sequences[seq_id] = {'x': [], 'y': [], 'times': [], 'frames': []}
                
                sequences[seq_id]['x'].append(projection[i, 0])
                sequences[seq_id]['y'].append(projection[i, 1])
                sequences[seq_id]['times'].append(meta['timestamp'])
                sequences[seq_id]['frames'].append(meta['frame_id'])
            
            # Add trajectory for each sequence
            for seq_id, seq_data in sequences.items():
                # Sort by timestamp
                sorted_indices = np.argsort(seq_data['times'])
                
                fig.add_trace(go.Scatter(
                    x=[seq_data['x'][i] for i in sorted_indices],
                    y=[seq_data['y'][i] for i in sorted_indices],
                    mode='lines+markers',
                    name=f'Sequence {seq_id}',
                    line=dict(width=2),
                    marker=dict(size=8, line=dict(width=1, color='white')),
                    hovertemplate=f'<b>Sequence {seq_id}</b><br>' +
                                 'Time: %{text}<br>' +
                                 'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
                    text=[f"{seq_data['times'][i]:.2f}s" for i in sorted_indices]
                ))
            
            fig.update_layout(
                title="Video Sequence Trajectories in Embedding Space",
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                height=600,
                template='plotly_white'
            )
            
        else:  # flow or force_directed
            # Create animated scatter plot
            import pandas as pd
            
            plot_data = pd.DataFrame({
                'x': projection[:, 0],
                'y': projection[:, 1],
                'sequence_id': [m['sequence_id'] for m in all_metadata],
                'frame_id': [m['frame_id'] for m in all_metadata],
                'timestamp': [m['timestamp'] for m in all_metadata],
                'video_name': [m['video_name'] for m in all_metadata]
            })
            
            fig = px.scatter(
                plot_data,
                x='x', y='y',
                animation_frame='frame_id',
                animation_group='sequence_id',
                color='sequence_id',
                hover_name='video_name',
                hover_data={'timestamp': ':.2f'},
                title="Temporal Video Embedding Evolution",
                labels={
                    'x': 'UMAP Dimension 1',
                    'y': 'UMAP Dimension 2'
                }
            )
            
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
            fig.update_layout(height=600, template='plotly_white')
        
        return fig
    
    def create_contrastive_visualization(self, anchor_embeddings: np.ndarray,
                                       positive_embeddings: np.ndarray,
                                       negative_embeddings: np.ndarray) -> go.Figure:
        """
        Visualize contrastive learning spaces.
        
        Args:
            anchor_embeddings: Anchor (query) embeddings
            positive_embeddings: Positive (similar) embeddings
            negative_embeddings: Negative (dissimilar) embeddings
            
        Returns:
            Contrastive learning visualization
        """
        # Combine all embeddings
        all_embeddings = np.vstack([
            anchor_embeddings, positive_embeddings, negative_embeddings
        ])
        
        # Create labels
        n_anchors = len(anchor_embeddings)
        n_positives = len(positive_embeddings)
        n_negatives = len(negative_embeddings)
        
        labels = (['Anchor'] * n_anchors + 
                 ['Positive'] * n_positives + 
                 ['Negative'] * n_negatives)
        
        # Create triplet IDs (assuming anchor[i] matches positive[i])
        triplet_ids = (list(range(n_anchors)) + 
                      list(range(min(n_anchors, n_positives))) + 
                      [-1] * n_negatives)  # -1 for unpaired negatives
        
        # Get 2D projection
        projection = self.create_umap_projection(all_embeddings, n_components=2)
        
        # Create interactive plot
        import pandas as pd
        plot_data = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'type': labels,
            'triplet_id': triplet_ids,
            'embedding_id': range(len(all_embeddings))
        })
        
        fig = px.scatter(
            plot_data,
            x='x', y='y',
            color='type',
            symbol='type',
            hover_data={'triplet_id': True, 'embedding_id': True},
            title="Contrastive Learning Embedding Space",
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2'
            },
            color_discrete_map={
                'Anchor': 'blue',
                'Positive': 'green', 
                'Negative': 'red'
            }
        )
        
        # Add lines connecting anchor-positive pairs
        for i in range(min(n_anchors, n_positives)):
            anchor_proj = projection[i]
            positive_proj = projection[n_anchors + i]
            
            fig.add_shape(
                type="line",
                x0=anchor_proj[0], y0=anchor_proj[1],
                x1=positive_proj[0], y1=positive_proj[1],
                line=dict(color="green", width=1, dash="dot"),
                opacity=0.6
            )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
        fig.update_layout(height=600, template='plotly_white')
        
        return fig
    
    def create_interactive_2d_plot(self, embeddings: np.ndarray,
                                 metadata: List[Dict],
                                 method: str = 'umap',
                                 color_by: str = 'category',
                                 **kwargs) -> go.Figure:
        """
        Create interactive 2D visualization with multiple methods.
        
        Args:
            embeddings: High-dimensional embeddings
            metadata: Video metadata for labeling
            method: 'umap', 'pca', or 'tsne'
            color_by: Metadata field to color by
            **kwargs: Additional parameters for projection method
            
        Returns:
            Interactive Plotly figure
        """
        # Get 2D projection
        if method == 'umap':
            projection = self.create_umap_projection(embeddings, n_components=2, **kwargs)
            title_suffix = "UMAP Projection"
        elif method == 'pca':
            projection = self.create_pca_projection(embeddings, n_components=2)
            variance_info = f" (Explained Variance: {self.cumulative_variance[1]:.1%})"
            title_suffix = f"PCA Projection{variance_info}"
        elif method == 'tsne':
            projection = self.create_tsne_projection(embeddings, n_components=2, **kwargs)
            title_suffix = "t-SNE Projection"
        elif method == 'trimap':
            projection = self.create_trimap_projection(embeddings, n_components=2, **kwargs)
            title_suffix = "TriMAP Projection"
        else:
            raise ValueError(f"Unknown method: {method}. Supported: umap, pca, tsne, trimap")
        
        # Create DataFrame for plotting
        import pandas as pd
        plot_data = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'video_name': [m.get('video_name', f'Video_{i}') for i, m in enumerate(metadata)],
            'category': [m.get('category', 'unknown') for m in metadata],
            'similarity_score': [m.get('similarity_score', 0) for m in metadata],
            'duration': [m.get('duration', 0) for m in metadata]
        })
        
        # Create interactive plot
        fig = px.scatter(
            plot_data,
            x='x', y='y',
            color=color_by,
            size='similarity_score',
            hover_name='video_name',
            hover_data={
                'category': True,
                'duration': True,
                'similarity_score': ':.3f',
                'x': ':.3f',
                'y': ':.3f'
            },
            title=f"Video Similarity Visualization - {title_suffix}",
            labels={
                'x': f'{method.upper()} Dimension 1',
                'y': f'{method.upper()} Dimension 2'
            }
        )
        
        # Enhance the plot
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='DarkSlateGrey'),
                sizemode='area',
                sizeref=2.*max(plot_data['similarity_score'])/(40.**2),
                sizemin=4
            )
        )
        
        fig.update_layout(
            height=600,
            hovermode='closest',
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_3d_plot(self, embeddings: np.ndarray,
                                 metadata: List[Dict],
                                 method: str = 'umap',
                                 color_by: str = 'category') -> go.Figure:
        """
        Create interactive 3D visualization - better for exploring clusters.
        
        3D advantages:
        - More space to separate clusters
        - Interactive rotation and zoom
        - Better preservation of neighborhood structure
        """
        # Get 3D projection
        if method == 'umap':
            projection = self.create_umap_projection(embeddings, n_components=3)
            title_suffix = "UMAP 3D Projection"
        elif method == 'pca':
            projection = self.create_pca_projection(embeddings, n_components=3)
            variance_info = f" (Explained Variance: {self.cumulative_variance[2]:.1%})"
            title_suffix = f"PCA 3D Projection{variance_info}"
        else:
            raise ValueError(f"3D visualization not supported for method: {method}")
        
        # Create DataFrame for plotting
        import pandas as pd
        plot_data = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'z': projection[:, 2],
            'video_name': [m.get('video_name', f'Video_{i}') for i, m in enumerate(metadata)],
            'category': [m.get('category', 'unknown') for m in metadata],
            'similarity_score': [m.get('similarity_score', 0) for m in metadata]
        })
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            plot_data,
            x='x', y='y', z='z',
            color=color_by,
            size='similarity_score',
            hover_name='video_name',
            title=title_suffix,
            labels={
                'x': f'{method.upper()} Dimension 1',
                'y': f'{method.upper()} Dimension 2',
                'z': f'{method.upper()} Dimension 3'
            }
        )
        
        fig.update_traces(
            marker=dict(
                line=dict(width=0.5, color='DarkSlateGrey'),
                sizemode='area',
                sizeref=2.*max(plot_data['similarity_score'])/(20.**2),
                sizemin=3
            )
        )
        
        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title=f'{method.upper()} Dimension 1',
                yaxis_title=f'{method.upper()} Dimension 2',
                zaxis_title=f'{method.upper()} Dimension 3'
            )
        )
        
        return fig
    
    def create_similarity_heatmap(self, embeddings: np.ndarray,
                                metadata: List[Dict],
                                max_videos: int = 50) -> go.Figure:
        """
        Create similarity heatmap - direct visualization of embedding similarities.
        
        Advantages:
        - No dimensionality reduction artifacts
        - Direct interpretation of similarities
        - Good for finding outliers
        """
        # Limit to manageable number of videos
        if len(embeddings) > max_videos:
            indices = np.random.choice(len(embeddings), max_videos, replace=False)
            embeddings = embeddings[indices]
            metadata = [metadata[i] for i in indices]
        
        # Compute similarity matrix
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Create labels
        labels = [m.get('video_name', f'Video_{i}')[:20] for i, m in enumerate(metadata)]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            colorbar=dict(title="Cosine Similarity"),
            hovertemplate='<b>%{y}</b><br>' +
                         '<b>%{x}</b><br>' +
                         'Similarity: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Video Similarity Heatmap ({len(embeddings)} videos)",
            width=800,
            height=800,
            xaxis=dict(side='bottom', tickangle=45),
            yaxis=dict(side='left')
        )
        
        return fig
    
    def create_cluster_analysis(self, embeddings: np.ndarray,
                              metadata: List[Dict],
                              n_clusters: int = 5,
                              method: str = 'kmeans') -> Tuple[go.Figure, Dict]:
        """
        Create cluster analysis visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            metadata: Video metadata
            n_clusters: Number of clusters
            method: 'kmeans', 'dbscan', or 'gmm'
            
        Returns:
            Plotly figure and cluster information
        """
        if not CLUSTERING_AVAILABLE:
            logger.warning("Clustering not available")
            return None, {}
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
            cluster_labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        elif method == 'gmm':
            clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Get 2D projection for visualization
        projection = self.create_umap_projection(embeddings, n_components=2)
        
        # Create DataFrame
        import pandas as pd
        plot_data = pd.DataFrame({
            'x': projection[:, 0],
            'y': projection[:, 1],
            'cluster': cluster_labels.astype(str),
            'video_name': [m.get('video_name', f'Video_{i}') for i, m in enumerate(metadata)],
            'category': [m.get('category', 'unknown') for m in metadata]
        })
        
        # Create cluster plot
        fig = px.scatter(
            plot_data,
            x='x', y='y',
            color='cluster',
            hover_name='video_name',
            hover_data={'category': True},
            title=f"Video Clustering - {method.upper()} ({n_clusters} clusters)",
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2'
            }
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
        fig.update_layout(height=600, template='plotly_white')
        
        # Analyze clusters
        cluster_info = {}
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # DBSCAN noise
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_metadata = [metadata[i] for i in range(len(metadata)) if cluster_mask[i]]
            
            # Analyze cluster composition
            categories = [m.get('category', 'unknown') for m in cluster_metadata]
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            cluster_info[f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'categories': category_counts,
                'videos': [m.get('video_name', '') for m in cluster_metadata][:5]  # Top 5
            }
        
        return fig, cluster_info
    
    def create_tsne_projection(self, embeddings: np.ndarray,
                             n_components: int = 2,
                             perplexity: float = 30.0,
                             learning_rate: float = 200.0) -> np.ndarray:
        """
        Create t-SNE projection (kept for comparison).
        
        Note: Generally UMAP is preferred, but t-SNE can be useful for:
        - Small datasets
        - When you want to emphasize local structure
        """
        cache_key = f"tsne_{n_components}_{perplexity}_{learning_rate}"
        if cache_key in self.projection_cache:
            return self.projection_cache[cache_key]
        
        logger.info(f"Computing t-SNE projection (this may take a while...)")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42,
            init='pca'
        )
        
        projection = tsne.fit_transform(embeddings)
        self.projection_cache[cache_key] = projection
        
        return projection
    
    def create_comparison_plot(self, embeddings: np.ndarray,
                             metadata: List[Dict]) -> go.Figure:
        """
        Create side-by-side comparison of different methods.
        """
        # Get projections
        umap_proj = self.create_umap_projection(embeddings, n_components=2)
        pca_proj = self.create_pca_projection(embeddings, n_components=2)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['UMAP Projection', 'PCA Projection'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Prepare data
        categories = [m.get('category', 'unknown') for m in metadata]
        video_names = [m.get('video_name', f'Video_{i}') for i, m in enumerate(metadata)]
        
        # Add UMAP trace
        fig.add_trace(
            go.Scatter(
                x=umap_proj[:, 0],
                y=umap_proj[:, 1],
                mode='markers',
                text=video_names,
                marker=dict(
                    color=[hash(cat) for cat in categories],
                    colorscale='Set3',
                    size=8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                name='UMAP',
                hovertemplate='<b>%{text}</b><br>Category: %{marker.color}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add PCA trace
        fig.add_trace(
            go.Scatter(
                x=pca_proj[:, 0],
                y=pca_proj[:, 1],
                mode='markers',
                text=video_names,
                marker=dict(
                    color=[hash(cat) for cat in categories],
                    colorscale='Set3',
                    size=8,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                name='PCA',
                hovertemplate='<b>%{text}</b><br>Category: %{marker.color}<extra></extra>'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Dimensionality Reduction Comparison",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def benchmark_methods(self, embeddings: np.ndarray,
                         metadata: List[Dict],
                         methods: List[str] = None) -> Dict[str, Dict]:
        """
        Benchmark different visualization methods for performance and quality.
        
        Args:
            embeddings: High-dimensional embeddings
            metadata: Video metadata
            methods: List of methods to benchmark
            
        Returns:
            Benchmark results dictionary
        """
        if methods is None:
            methods = ['pca', 'umap']
            if TRIMAP_AVAILABLE:
                methods.append('trimap')
        
        import time
        from sklearn.metrics import silhouette_score
        
        results = {}
        
        for method in methods:
            logger.info(f"Benchmarking {method.upper()}...")
            
            start_time = time.time()
            
            try:
                if method == 'umap':
                    projection = self.create_umap_projection(embeddings, n_components=2)
                elif method == 'pca':
                    projection = self.create_pca_projection(embeddings, n_components=2)
                elif method == 'trimap':
                    projection = self.create_trimap_projection(embeddings, n_components=2)
                elif method == 'tsne':
                    projection = self.create_tsne_projection(embeddings, n_components=2)
                else:
                    continue
                
                computation_time = time.time() - start_time
                
                # Calculate clustering quality if categories available
                categories = [m.get('category', 'unknown') for m in metadata]
                unique_categories = list(set(categories))
                
                if len(unique_categories) > 1:
                    category_labels = [unique_categories.index(cat) for cat in categories]
                    silhouette = silhouette_score(projection, category_labels)
                else:
                    silhouette = None
                
                results[method] = {
                    'computation_time': computation_time,
                    'silhouette_score': silhouette,
                    'projection_shape': projection.shape,
                    'successful': True
                }
                
            except Exception as e:
                logger.error(f"Error benchmarking {method}: {e}")
                results[method] = {
                    'computation_time': None,
                    'silhouette_score': None,
                    'projection_shape': None,
                    'successful': False,
                    'error': str(e)
                }
        
        return results
    
    def create_benchmark_report(self, benchmark_results: Dict[str, Dict]) -> str:
        """Create a formatted benchmark report."""
        report = "\n🔬 VISUALIZATION METHODS BENCHMARK REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Sort by computation time
        successful_results = {k: v for k, v in benchmark_results.items() if v['successful']}
        sorted_methods = sorted(successful_results.items(), key=lambda x: x[1]['computation_time'])
        
        report += "📊 Performance Ranking (by speed):\n"
        for i, (method, results) in enumerate(sorted_methods, 1):
            time_str = f"{results['computation_time']:.3f}s"
            silhouette_str = f"{results['silhouette_score']:.3f}" if results['silhouette_score'] else "N/A"
            report += f"  {i}. {method.upper():8} | Time: {time_str:8} | Quality: {silhouette_str}\n"
        
        report += "\n🎯 Recommendations:\n"
        
        # Find fastest and best quality
        fastest = min(successful_results.items(), key=lambda x: x[1]['computation_time'])
        
        if successful_results:
            best_quality = max(
                [(k, v) for k, v in successful_results.items() if v['silhouette_score'] is not None],
                key=lambda x: x[1]['silhouette_score'],
                default=(None, None)
            )
            
            report += f"  🚀 Fastest: {fastest[0].upper()} ({fastest[1]['computation_time']:.3f}s)\n"
            if best_quality[0]:
                report += f"  🎯 Best Quality: {best_quality[0].upper()} (silhouette: {best_quality[1]['silhouette_score']:.3f})\n"
        
        # Failed methods
        failed_methods = [k for k, v in benchmark_results.items() if not v['successful']]
        if failed_methods:
            report += f"\n❌ Failed Methods: {', '.join(failed_methods)}\n"
        
        return report


def demonstrate_advanced_visualizations():
    """Demonstrate all advanced visualization techniques."""
    # Generate mock data
    np.random.seed(42)
    n_videos = 100
    embedding_dim = 768
    
    # Create embeddings with some structure
    embeddings = []
    metadata = []
    categories = ['car2cyclist', 'car2pedestrian', 'car2car', 'traffic', 'highway']
    
    for i in range(n_videos):
        category = np.random.choice(categories)
        
        # Create category-specific embeddings
        base_embedding = np.random.normal(0, 0.1, embedding_dim)
        category_bias = np.zeros(embedding_dim)
        category_idx = categories.index(category)
        category_bias[category_idx*100:(category_idx+1)*100] = np.random.normal(0.5, 0.2, 100)
        
        embedding = base_embedding + category_bias
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        embeddings.append(embedding)
        metadata.append({
            'video_name': f'{category}_{i:03d}.mp4',
            'category': category,
            'similarity_score': np.random.uniform(0.3, 0.95),
            'duration': np.random.uniform(1.5, 8.0)
        })
    
    embeddings = np.array(embeddings)
    
    # Create visualizer
    visualizer = AdvancedEmbeddingVisualizer()
    
    print("🎨 ADVANCED EMBEDDING VISUALIZATIONS DEMO")
    print("=" * 50)
    
    # 1. UMAP vs PCA comparison
    print("\n1. Creating UMAP vs PCA comparison...")
    comparison_fig = visualizer.create_comparison_plot(embeddings, metadata)
    comparison_fig.write_html("comparison_umap_pca.html")
    print("   📊 Saved: comparison_umap_pca.html")
    
    # 2. Interactive 3D UMAP
    print("\n2. Creating 3D UMAP visualization...")
    umap_3d_fig = visualizer.create_interactive_3d_plot(embeddings, metadata, method='umap')
    umap_3d_fig.write_html("umap_3d_visualization.html")
    print("   📊 Saved: umap_3d_visualization.html")
    
    # 3. Similarity heatmap
    print("\n3. Creating similarity heatmap...")
    heatmap_fig = visualizer.create_similarity_heatmap(embeddings, metadata, max_videos=30)
    heatmap_fig.write_html("similarity_heatmap.html")
    print("   📊 Saved: similarity_heatmap.html")
    
    # 4. Cluster analysis
    print("\n4. Creating cluster analysis...")
    cluster_fig, cluster_info = visualizer.create_cluster_analysis(
        embeddings, metadata, n_clusters=5, method='kmeans'
    )
    if cluster_fig:
        cluster_fig.write_html("cluster_analysis.html")
        print("   📊 Saved: cluster_analysis.html")
        
        print("\n   📊 Cluster Analysis Results:")
        for cluster_id, info in cluster_info.items():
            print(f"     {cluster_id}: {info['size']} videos")
            print(f"       Categories: {info['categories']}")
    
    # 5. Advanced method demonstrations
    methods = ['umap', 'pca']
    if TRIMAP_AVAILABLE:
        methods.append('trimap')
    if UMAP_AVAILABLE:  # Only add t-SNE if we have other methods working
        methods.append('tsne')
    
    for method in methods:
        print(f"\n5.{methods.index(method)+1} Creating {method.upper()} visualization...")
        fig = visualizer.create_interactive_2d_plot(
            embeddings, metadata, method=method, color_by='category'
        )
        fig.write_html(f"{method}_visualization.html")
        print(f"   📊 Saved: {method}_visualization.html")
    
    # 6. Multiscale visualization
    print("\n6. Creating multiscale visualization...")
    multiscale_fig = visualizer.create_multiscale_visualization(embeddings, metadata)
    multiscale_fig.write_html("multiscale_visualization.html")
    print("   📊 Saved: multiscale_visualization.html")
    
    # 7. Temporal visualization (mock sequence data)
    print("\n7. Creating temporal visualization...")
    # Create mock video sequences
    n_sequences = 3
    frames_per_sequence = 10
    video_sequences = []
    timestamps = []
    
    for seq_id in range(n_sequences):
        sequence = []
        times = []
        for frame in range(frames_per_sequence):
            # Create correlated embeddings within sequence
            base_embedding = embeddings[seq_id * 5 + frame % 5]  # Reuse some embeddings
            noise = np.random.normal(0, 0.1, embedding_dim)
            sequence_embedding = base_embedding + noise
            sequence_embedding = sequence_embedding / np.linalg.norm(sequence_embedding)
            
            sequence.append(sequence_embedding)
            times.append(frame * 0.5)  # 0.5 second intervals
        
        video_sequences.append(np.array(sequence))
        timestamps.append(times)
    
    temporal_fig = visualizer.create_temporal_visualization(
        video_sequences, timestamps, method='trajectory'
    )
    temporal_fig.write_html("temporal_visualization.html")
    print("   📊 Saved: temporal_visualization.html")
    
    # 8. Contrastive learning visualization
    print("\n8. Creating contrastive learning visualization...")
    # Create mock contrastive data
    n_triplets = 20
    anchor_embeddings = embeddings[:n_triplets]
    
    # Positives: add small noise to anchors
    positive_embeddings = anchor_embeddings + np.random.normal(0, 0.1, anchor_embeddings.shape)
    positive_embeddings = positive_embeddings / np.linalg.norm(positive_embeddings, axis=1, keepdims=True)
    
    # Negatives: random embeddings from different categories
    negative_embeddings = embeddings[50:70]  # Different subset
    
    contrastive_fig = visualizer.create_contrastive_visualization(
        anchor_embeddings, positive_embeddings, negative_embeddings
    )
    contrastive_fig.write_html("contrastive_visualization.html")
    print("   📊 Saved: contrastive_visualization.html")
    
    # 9. Parametric projection training (if PyTorch available)
    if TORCH_AVAILABLE:
        print("\n9. Training parametric projection...")
        parametric_model = visualizer.train_parametric_projection(
            embeddings[:50], epochs=20  # Small subset for demo
        )
        if parametric_model:
            print("   🧠 Parametric model trained successfully!")
            
            # Test on new data
            new_projection = parametric_model(
                torch.FloatTensor(embeddings[50:60])
            ).detach().numpy()
            print(f"   📊 Projected {len(new_projection)} new embeddings")
    
    # 10. Benchmark all methods
    print("\n10. Benchmarking visualization methods...")
    benchmark_results = visualizer.benchmark_methods(
        embeddings[:50], metadata[:50]  # Smaller subset for faster benchmarking
    )
    
    benchmark_report = visualizer.create_benchmark_report(benchmark_results)
    print(benchmark_report)
    
    # Save benchmark results
    with open("benchmark_report.txt", "w") as f:
        f.write(benchmark_report)
    print("   📊 Saved: benchmark_report.txt")
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETED!")
    print("=" * 50)
    
    print("\n📁 Generated Files:")
    files = [
        "comparison_umap_pca.html",
        "umap_3d_visualization.html", 
        "similarity_heatmap.html",
        "cluster_analysis.html",
        "multiscale_visualization.html",
        "temporal_visualization.html",
        "contrastive_visualization.html",
        "benchmark_report.txt"
    ] + [f"{method}_visualization.html" for method in methods]
    
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file}")
    
    print("\n🎯 Advanced Recommendations for NVIDIA Cosmos:")
    print("  🌟 UMAP: Best general-purpose replacement for t-SNE")
    print("  🚀 PCA: Fastest for exploration and baseline comparisons")
    print("  🔥 TriMAP: Better than t-SNE for large datasets")
    print("  🧠 Parametric: Use for production systems with new data")
    print("  📈 Multiscale: Understand structure at different granularities")
    print("  🎬 Temporal: Essential for video sequence analysis")
    print("  🎯 Contrastive: Debug and improve model training")
    print("  📊 Direct Similarity: Most accurate for detailed analysis")
    
    print("\n💡 Implementation Strategy:")
    print("  1. Start with PCA for quick baseline")
    print("  2. Use UMAP for main interactive visualizations")
    print("  3. Apply temporal methods for video sequences")
    print("  4. Train parametric models for production deployment")
    print("  5. Benchmark methods on your specific data")


if __name__ == "__main__":
    demonstrate_advanced_visualizations()
