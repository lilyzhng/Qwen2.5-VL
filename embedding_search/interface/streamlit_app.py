#!/usr/bin/env python3
"""
ALFA 0.1 - Similarity Search Interface
"""

import os
# Fix OpenMP library issue  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
from datetime import datetime
from typing import Optional, List, Dict
import logging
from PIL import Image
from core.search import VideoSearchEngine
from core.visualizer import VideoResultsVisualizer
from core.config import VideoRetrievalConfig
from core.exceptions import NoResultsError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectedVideo:
    """Track selected video with timestamp."""
    
    def __init__(self, idx: int = -1):
        self.idx = int(idx)
        self.timestamp = datetime.now()
    
    def __eq__(self, other):
        if isinstance(other, SelectedVideo):
            return self.idx == other.idx
        return self.idx == int(other)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def is_valid(self) -> bool:
        return self.idx >= 0


@st.cache_resource
def load_search_engine() -> VideoSearchEngine:
    """Load and cache the search engine."""
    try:
        config = VideoRetrievalConfig()
        search_engine = VideoSearchEngine(config=config)
        
        # Database loads automatically in ParquetVectorDatabase.__init__
        # No need to explicitly load
                
        return search_engine
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        raise


@st.cache_data
def load_database_info(_engine: VideoSearchEngine) -> Dict:
    """Load and cache database information."""
    try:
        return _engine.get_statistics()
    except Exception as e:
        return {
            "num_inputs": 0,
            "categories": 0,
            "embedding_dim": 768,
            "search_backend": "FAISS",
            "error": str(e)
        }


def get_all_videos_from_database(search_engine) -> List[Dict]:
    """Get all videos from the database for visualization."""
    try:
        # Try to get data directly from the parquet file for better cluster info
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        unified_embeddings_path = project_root / "data" / "unified_embeddings.parquet"
        
        if unified_embeddings_path.exists():
            import pandas as pd
            df = pd.read_parquet(unified_embeddings_path)
            all_videos = []
            for _, row in df.iterrows():
                video_info = {
                    'slice_id': row['slice_id'],
                    'video_path': row.get('video_path', ''),
                    'similarity_score': 0.1,  # Default low similarity
                    'rank': 1000,  # High rank for non-search results
                    'metadata': row.to_dict()
                }
                all_videos.append(video_info)
            return all_videos
        
        # Fallback to original method
        if hasattr(search_engine, 'database') and hasattr(search_engine.database, 'metadata'):
            all_videos = []
            for i, metadata in enumerate(search_engine.database.metadata):
                video_info = {
                    'slice_id': metadata.get('slice_id', f"Video {i+1}"),
                    'video_path': metadata.get('video_path', ''),
                    'similarity_score': 0.1,  # Default low similarity
                    'rank': i + 1000,  # High rank for non-search results
                    'metadata': metadata
                }
                all_videos.append(video_info)
            return all_videos
    except Exception as e:
        logger.warning(f"Could not get all videos from database: {e}")
    return []


def create_embedding_visualization(results: List[Dict], viz_method: str = "umap", selected_idx: Optional[int] = None, query_info: Optional[Dict] = None, top_k: Optional[int] = None, all_videos: Optional[List[Dict]] = None, analysis_type: str = "embedding_similarity", **kwargs) -> go.Figure:
    """Create advanced embedding visualization with multiple methods."""
    if not results:
        return go.Figure()
    
    # Load the actual cluster data from the main embeddings parquet file
    try:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        unified_embeddings_path = project_root / "data" / "unified_embeddings.parquet"
        
        if unified_embeddings_path.exists():
            cluster_df = pd.read_parquet(unified_embeddings_path)
            # Create a mapping from slice_id to coordinates and cluster
            coord_mapping = {
                row['slice_id']: {
                    'x': row['x'],
                    'y': row['y'],
                    'cluster_id': row['cluster_id']
                }
                for _, row in cluster_df.iterrows()
                if 'x' in cluster_df.columns and 'y' in cluster_df.columns
            }
            has_real_coords = len(coord_mapping) > 0
        else:
            coord_mapping = {}
            has_real_coords = False
    except Exception as e:
        logger.warning(f"Could not load cluster data: {e}")
        coord_mapping = {}
        has_real_coords = False
    
    # Use all videos from database if provided, otherwise use just search results
    if all_videos:
        # Create a set of search result slice_ids for quick lookup
        search_result_ids = {r['slice_id'] for r in results}
        
        # Create dataframe for all videos
        all_video_data = []
        for i, video in enumerate(all_videos):
            is_search_result = video['slice_id'] in search_result_ids
            
            # Find the corresponding search result if this is one
            if is_search_result:
                matching_result = next((r for r in results if r['slice_id'] == video['slice_id']), None)
                similarity = matching_result['similarity_score'] if matching_result else 0.1
                rank = matching_result['rank'] if matching_result else 1000
            else:
                similarity = 0.1
                rank = 1000 + i
                
            all_video_data.append({
                'slice_id': video['slice_id'],
                'similarity': similarity,
                'rank': rank,
                'category': video.get('metadata', {}).get('category', 'unknown'),
                'idx': i,
                'is_search_result': is_search_result
            })
        
        df = pd.DataFrame(all_video_data)
    else:
        # Just use search results if no database videos provided
        df = pd.DataFrame([
            {
                'slice_id': r['slice_id'],
                'similarity': r['similarity_score'],
                'rank': r['rank'],
                'category': getattr(r, 'category', 'unknown'),
                'idx': i,
                'is_search_result': True
            }
            for i, r in enumerate(results)
        ])
    
    # Add real coordinates and cluster info if available
    if has_real_coords:
        df['x'] = df['slice_id'].map(lambda sid: coord_mapping.get(sid, {}).get('x', np.nan))
        df['y'] = df['slice_id'].map(lambda sid: coord_mapping.get(sid, {}).get('y', np.nan))
        df['cluster_id'] = df['slice_id'].map(lambda sid: coord_mapping.get(sid, {}).get('cluster_id', -1))
        
        # For any missing coordinates (e.g., query videos), we'll generate them later
        has_coords = ~df['x'].isna()
    else:
        has_coords = pd.Series([False] * len(df))
    
    # Generate coordinates based on visualization method
    np.random.seed(42)
    
    # Calculate query position - place it at the center of the coordinate space
    if has_real_coords and has_coords.any() and viz_method in ["umap"]:
        # Get the center of all points (not just search results)
        all_points_with_coords = df[has_coords]
        if len(all_points_with_coords) > 0:
            # Place query at the center of all data points
            query_x = all_points_with_coords['x'].mean()
            query_y = all_points_with_coords['y'].mean()
            query_z = 0
        else:
            query_x, query_y, query_z = 0, 0, 0
    else:
        query_x, query_y, query_z = 0, 0, 0
    
    if viz_method == "umap":
        # Use real coordinates if available, otherwise fall back to synthetic
        if not has_real_coords or not has_coords.any():
            # Position vectors based on distance from query (higher similarity = closer)
            distances = (1 - df['similarity']) * 8  # Convert similarity to distance
            angles = np.random.uniform(0, 2*np.pi, len(df))
            df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.5
            df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.5
        else:
            # For any points without coordinates (like query videos), place them near similar points
            missing_coords = df['x'].isna()
            if missing_coords.any():
                # Place missing points based on similarity to existing points
                for idx in df[missing_coords].index:
                    # Find most similar point with coordinates
                    point_similarity = df.loc[idx, 'similarity']
                    distances = (1 - point_similarity) * 2
                    angle = np.random.uniform(0, 2*np.pi)
                    df.loc[idx, 'x'] = query_x + distances * np.cos(angle)
                    df.loc[idx, 'y'] = query_y + distances * np.sin(angle)
        title = "2D Embedding Space - UMAP"
        x_title, y_title = "UMAP Dimension 1", "UMAP Dimension 2"
    
    elif viz_method == "3d_umap":
        if not has_real_coords or not has_coords.any():
            distances = (1 - df['similarity']) * 6
            # Generate random 3D directions
            theta = np.random.uniform(0, 2*np.pi, len(df))  # azimuthal angle
            phi = np.random.uniform(0, np.pi, len(df))      # polar angle
            df['x'] = query_x + distances * np.sin(phi) * np.cos(theta) + np.random.randn(len(df)) * 0.3
            df['y'] = query_y + distances * np.sin(phi) * np.sin(theta) + np.random.randn(len(df)) * 0.3
            df['z'] = query_z + distances * np.cos(phi) + np.random.randn(len(df)) * 0.3
        else:
            # Use real 2D coordinates and add a synthetic z based on cluster
            missing_coords = df['x'].isna()
            if missing_coords.any():
                for idx in df[missing_coords].index:
                    point_similarity = df.loc[idx, 'similarity']
                    distances = (1 - point_similarity) * 2
                    angle = np.random.uniform(0, 2*np.pi)
                    df.loc[idx, 'x'] = query_x + distances * np.cos(angle)
                    df.loc[idx, 'y'] = query_y + distances * np.sin(angle)
            # Add z coordinate based on cluster with some variation
            if 'cluster_id' in df.columns:
                df['z'] = df['cluster_id'] * 0.5 + np.random.randn(len(df)) * 0.1
            else:
                df['z'] = query_z + (1 - df['similarity']) * np.random.randn(len(df)) * 0.3
        title = "3D Embedding Space - UMAP"
    else:  # similarity heatmap
        # For heatmap, only show search results to make it meaningful
        if 'is_search_result' in df.columns:
            df_heatmap = df[df['is_search_result'] == True].reset_index(drop=True)
        else:
            df_heatmap = df.copy()
            
        # Limit to reasonable number of videos for readability
        if len(df_heatmap) > 20:
            df_heatmap = df_heatmap.head(20)
            
        n = len(df_heatmap)
        
        if n < 2:
            # Not enough data for a meaningful heatmap
            fig = go.Figure()
            fig.add_annotation(
                text="Need at least 2 search results for similarity matrix",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Video Similarity Matrix",
                height=400,
                showlegend=False
            )
            return fig
        
        # Create similarity matrix based on actual similarity scores
        similarity_matrix = np.zeros((n, n))
        
        # Fill the matrix with meaningful similarity values
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Self-similarity is always 1
                else:
                    # Create similarity based on how close their similarity scores are
                    sim_i = df_heatmap.iloc[i]['similarity']
                    sim_j = df_heatmap.iloc[j]['similarity']
                    rank_i = df_heatmap.iloc[i]['rank']
                    rank_j = df_heatmap.iloc[j]['rank']
                    
                    # Combine similarity score difference and rank difference
                    sim_diff = abs(sim_i - sim_j)
                    rank_diff = abs(rank_i - rank_j)
                    
                    # Videos with similar scores and close ranks are more similar
                    # Normalize by the maximum possible differences
                    max_sim_diff = max(df_heatmap['similarity']) - min(df_heatmap['similarity'])
                    max_rank_diff = max(df_heatmap['rank']) - min(df_heatmap['rank'])
                    
                    if max_sim_diff > 0:
                        norm_sim_diff = sim_diff / max_sim_diff
                    else:
                        norm_sim_diff = 0
                        
                    if max_rank_diff > 0:
                        norm_rank_diff = rank_diff / max_rank_diff
                    else:
                        norm_rank_diff = 0
                    
                    # Combine both factors (lower differences = higher similarity)
                    combined_diff = (norm_sim_diff + norm_rank_diff) / 2
                    similarity_matrix[i, j] = max(0.1, 1.0 - combined_diff)
        
        # Create shorter, more readable labels
        labels = []
        for _, row in df_heatmap.iterrows():
            slice_id = row['slice_id']
            rank = row['rank']
            # Show rank and shortened slice_id
            if len(slice_id) > 12:
                label = f"#{rank}: {slice_id[:12]}..."
            else:
                label = f"#{rank}: {slice_id}"
            labels.append(label)
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',  # Green-Yellow scheme for consistency
            showscale=True,
            zmin=0,  # Set minimum value for color scale
            zmax=1,  # Set maximum value for color scale
            colorbar=dict(
                title="Similarity",
                title_side="right",
                x=1.02,
                len=0.8,
                thickness=15,
                title_font=dict(size=12),
                tickfont=dict(size=10),
                outlinewidth=0,
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
            ),
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: <b>%{z:.3f}</b><extra></extra>',
            text=similarity_matrix,
            texttemplate='%{z:.2f}',
            textfont=dict(size=8, color='white'),
        ))
        fig.update_layout(
            title="Video Similarity Matrix (Yellow = More Similar, Purple = Less Similar)",
            height=max(400, n * 25 + 150),  # Dynamic height based on number of videos
            width=None,  # Let it be responsive to container width
            title_x=0.5,
            title_font=dict(size=14),
            margin=dict(l=150, r=80, t=80, b=150),  # More space for labels
            plot_bgcolor='white',
            xaxis=dict(
                tickfont=dict(size=9), 
                tickangle=-45,
                side='bottom',
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                tickfont=dict(size=9),
                showgrid=False,
                zeroline=False,
                autorange='reversed'  # Reverse y-axis so first result is at top
            ),
            showlegend=False,  # No legend needed for heatmap
            font=dict(family="Arial, sans-serif")
        )
        return fig
    
    if viz_method == "3d_umap":
        # Split data into search results and other videos
        if 'is_search_result' in df.columns:
            df_search = df[df['is_search_result'] == True]
            df_other = df[df['is_search_result'] == False]
            
            # Separate traces: grayed out background videos vs highlighted search results
            traces = []
            if len(df_other) > 0:
                if 'cluster_id' in df_other.columns and has_real_coords and analysis_type == "umap_dbscan_clusters":
                    # Show clusters for non-search results in 3D too
                    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85929E']
                    
                    # Add each cluster separately
                    for cluster_id in sorted(df_other['cluster_id'].unique()):
                        if cluster_id >= 0:  # Skip noise points
                            cluster_data = df_other[df_other['cluster_id'] == cluster_id]
                            traces.append(go.Scatter3d(
                                x=cluster_data['x'],
                                y=cluster_data['y'], 
                                z=cluster_data['z'],
                                mode='markers',
                                marker=dict(
                                    size=8,  # Smaller size for non-results
                                    color=cluster_colors[cluster_id % len(cluster_colors)],
                                    opacity=0.3,
                                    line=dict(width=0)
                                ),
                                text=cluster_data['slice_id'],
                                hovertemplate='<b>%{text}</b><br>Cluster: %{customdata}<br>Database Video<extra></extra>',
                                customdata=cluster_data['cluster_id'],
                                name=f'Cluster {cluster_id}',
                                showlegend=True
                            ))
                    
                    # Add noise points if any
                    noise_data = df_other[df_other['cluster_id'] == -1]
                    if len(noise_data) > 0:
                        traces.append(go.Scatter3d(
                            x=noise_data['x'],
                            y=noise_data['y'], 
                            z=noise_data['z'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color='lightgray',
                                opacity=0.2
                            ),
                            text=noise_data['slice_id'],
                            hovertemplate='<b>%{text}</b><br>Noise Point<extra></extra>',
                            name='Noise',
                            showlegend=True
                        ))
                else:
                    # Original gray markers if no cluster info
                    traces.append(go.Scatter3d(
                        x=df_other['x'],
                        y=df_other['y'], 
                        z=df_other['z'],
                        mode='markers',
                        marker=dict(
                            size=8,  # Smaller size for non-results
                            color='lightgray',
                            opacity=0.3
                        ),
                        text=df_other['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Database Video<extra></extra>',
                        name='Videos',
                        showlegend=True
                    ))
            
            if len(df_search) > 0:
                # Check if we have cluster information
                if 'cluster_id' in df_search.columns and has_real_coords and analysis_type == "umap_dbscan_clusters":
                    # Use cluster-based coloring
                    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85929E']
                    traces.append(go.Scatter3d(
                        x=df_search['x'],
                        y=df_search['y'], 
                        z=df_search['z'],
                        mode='markers',
                        marker=dict(
                            size=df_search['similarity'] * 15 + 10,
                            color=df_search['cluster_id'],
                            colorscale=[[i/(len(cluster_colors)-1), cluster_colors[i]] for i in range(len(cluster_colors))],
                            showscale=True,
                            colorbar=dict(
                                title="Cluster",
                                title_side="right",
                                x=1.01,
                                len=0.6,
                                thickness=10,
                                title_font=dict(size=10),
                                tickfont=dict(size=9),
                                outlinewidth=0,
                                tick0=0,
                                dtick=1
                            ),
                            line=dict(
                                color='white',
                                width=1
                            )
                        ),
                        text=df_search['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<br>Cluster: %{customdata[2]}<extra></extra>',
                        customdata=df_search[['rank', 'similarity', 'cluster_id']].values,
                        name='Search Results',
                        showlegend=True
                    ))
                else:
                    # Original similarity-based coloring
                    traces.append(go.Scatter3d(
                        x=df_search['x'],
                        y=df_search['y'], 
                        z=df_search['z'],
                        mode='markers',
                        marker=dict(
                            size=df_search['similarity'] * 15 + 10,
                            color=df_search['similarity'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Similarity",
                                title_side="right",
                                x=1.01,
                                len=0.6,
                                thickness=10,
                                title_font=dict(size=10),
                                tickfont=dict(size=9),
                                outlinewidth=0
                            )
                        ),
                        text=df_search['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                        customdata=df_search[['rank', 'similarity']].values,
                        name='Search Results',
                        showlegend=True
                    ))
            
            fig = go.Figure(data=traces)
        else:
            # Original behavior for backward compatibility
            fig = go.Figure(data=go.Scatter3d(
                x=df['x'],
                y=df['y'], 
                z=df['z'],
                mode='markers',
                marker=dict(
                    size=df['similarity'] * 15 + 5,
                    color=df['similarity'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Similarity",
                        title_side="right",
                        x=1.01,
                        len=0.6,
                        thickness=10,
                        title_font=dict(size=10),
                        tickfont=dict(size=9),
                        outlinewidth=0
                    )
                ),
                text=df['slice_id'],
                hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                customdata=df[['rank', 'similarity']].values
            ))
        
        # Highlight top K results with colored circles: red for selected, green for others
        if top_k is not None and top_k > 0:
            if 'is_search_result' in df.columns:
                search_results = df[df['is_search_result'] == True]
                top_k_points = search_results.head(top_k)
            else:
                top_k_points = df.head(top_k)  # Get top K results
            
            # Create colors array - red for selected, green for others
            circle_colors = []
            for i, row in top_k_points.iterrows():
                if selected_idx is not None and i == selected_idx:
                    circle_colors.append('#FF0000')  # Red for selected
                else:
                    circle_colors.append('#00FF00')  # Green for others
            
            fig.add_trace(
                go.Scatter3d(
                    x=top_k_points['x'],
                    y=top_k_points['y'],
                    z=top_k_points['z'],
                    mode='markers',
                    marker=dict(
                        size=top_k_points['similarity'] * 15 + 20,  # Slightly larger than base points
                        color='rgba(0,0,0,0)',  # Transparent fill
                        line=dict(width=6, color=circle_colors)  # Thicker circles with dynamic colors
                    ),
                    name=f'Top {top_k} Results',
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<br>Top Result<extra></extra>',
                    text=top_k_points['slice_id'],
                    customdata=top_k_points[['rank', 'similarity']].values
                )
            )
        
        # Add query vector point for 3D (using diamond symbol)
        if query_info:
            # Use fixed query coordinates at center (same as used for positioning)
            query_x, query_y, query_z = 0, 0, 0
            
            fig.add_trace(
                go.Scatter3d(
                    x=[query_x],
                    y=[query_y],
                    z=[query_z],
                    mode='markers',
                    marker=dict(size=25, color='gold', symbol='diamond', line=dict(width=3, color='orange')),
                    name='Query',
                    text=[query_info.get('display_text', 'Query')],
                    hovertemplate='<b>Query: %{text}</b><extra></extra>',
                    showlegend=True
                )
            )
        
        fig.add_trace(
            go.Scatter3d(
                x=[-10, 10], y=[0, 0], z=[0, 0],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[-10, 10], z=[0, 0],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[-10, 10],
                mode='lines',
                line=dict(color='rgba(128,128,128,0.5)', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
                )
            )
        
        fig.update_layout(
            height=350,  # Consistent height with other plots
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3",
                bgcolor='#f8fafc',  # Match tip section background
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)  # Better default view
                ),
                aspectmode='cube'  # Maintain aspect ratio
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                x=-0.13,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10),
                itemsizing="constant",
                itemwidth=30
            ),
            margin=dict(l=200, r=40, t=50, b=40)  # Extra left margin for legend
        )
    else:
        # 2D scatter plot with enhanced interactivity
        if 'is_search_result' in df.columns:
            df_search = df[df['is_search_result'] == True]
            df_other = df[df['is_search_result'] == False]
            
            # Create figure with traces
            fig = go.Figure()
            
            # Add non-search-result videos (grayed out but show clusters if available)
            if len(df_other) > 0:
                if 'cluster_id' in df_other.columns and has_real_coords and analysis_type == "umap_dbscan_clusters":
                    # Show clusters for non-search results too, but with lower opacity
                    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85929E']
                    
                    # Add each cluster separately for better legend control
                    for cluster_id in sorted(df_other['cluster_id'].unique()):
                        if cluster_id >= 0:  # Skip noise points
                            cluster_data = df_other[df_other['cluster_id'] == cluster_id]
                            fig.add_trace(go.Scatter(
                                x=cluster_data['x'],
                                y=cluster_data['y'],
                                mode='markers',
                                marker=dict(
                                    size=8,  # Smaller size for non-results
                                    color=cluster_colors[cluster_id % len(cluster_colors)],
                                    opacity=0.3,
                                    line=dict(width=0)
                                ),
                                text=cluster_data['slice_id'],
                                hovertemplate='<b>%{text}</b><br>Cluster: %{customdata}<br>Database Video<extra></extra>',
                                customdata=cluster_data['cluster_id'],
                                name=f'Cluster {cluster_id}',
                                showlegend=True
                            ))
                    
                    # Add noise points if any
                    noise_data = df_other[df_other['cluster_id'] == -1]
                    if len(noise_data) > 0:
                        fig.add_trace(go.Scatter(
                            x=noise_data['x'],
                            y=noise_data['y'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color='lightgray',
                                opacity=0.2
                            ),
                            text=noise_data['slice_id'],
                            hovertemplate='<b>%{text}</b><br>Noise Point<extra></extra>',
                            name='Noise',
                            showlegend=True
                        ))
                else:
                    # Original gray markers if no cluster info
                    fig.add_trace(go.Scatter(
                        x=df_other['x'],
                        y=df_other['y'],
                        mode='markers',
                        marker=dict(
                            size=8,  # Smaller size for non-results
                            color='lightgray',
                            opacity=0.3
                        ),
                        text=df_other['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Database Video<extra></extra>',
                        name='Videos',
                        showlegend=True
                    ))
            
            # Add search results (colorful)
            if len(df_search) > 0:
                # Check if we have cluster information
                if 'cluster_id' in df_search.columns and has_real_coords and analysis_type == "umap_dbscan_clusters":
                    # Use cluster-based coloring
                    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85929E']
                    fig.add_trace(go.Scatter(
                        x=df_search['x'],
                        y=df_search['y'],
                        mode='markers',
                        marker=dict(
                            size=df_search['similarity'] * 15 + 10,
                            color=df_search['cluster_id'],
                            colorscale=[[i/(len(cluster_colors)-1), cluster_colors[i]] for i in range(len(cluster_colors))],
                            showscale=True,
                            colorbar=dict(
                                title="Cluster",
                                title_side="right",
                                x=1.01,
                                len=0.6,
                                thickness=10,
                                title_font=dict(size=10),
                                tickfont=dict(size=9),
                                outlinewidth=0,
                                tick0=0,
                                dtick=1
                            ),
                            line=dict(
                                color='white',
                                width=1
                            )
                        ),
                        text=df_search['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<br>Cluster: %{customdata[2]}<extra></extra>',
                        customdata=df_search[['rank', 'similarity', 'cluster_id']].values,
                        name='Search Results',
                        showlegend=True
                    ))
                else:
                    # Original similarity-based coloring
                    fig.add_trace(go.Scatter(
                        x=df_search['x'],
                        y=df_search['y'],
                        mode='markers',
                        marker=dict(
                            size=df_search['similarity'] * 15 + 10,
                            color=df_search['similarity'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(
                                title="Similarity",
                                title_side="right",
                                x=1.01,
                                len=0.6,
                                thickness=10,
                                title_font=dict(size=10),
                                tickfont=dict(size=9),
                                outlinewidth=0
                            )
                        ),
                        text=df_search['slice_id'],
                        hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                        customdata=df_search[['rank', 'similarity']].values,
                        name='Search Results',
                        showlegend=True
                    ))
        else:
            # Original behavior for backward compatibility
            fig = px.scatter(
                df,
                x='x',
                y='y',
                size='similarity',
                color='similarity',
                hover_name='slice_id',
                hover_data=['rank', 'similarity'],
                color_continuous_scale='Viridis',
                title=title
            )
            
            # Enhance hover template for better information display
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                hovertext=df['slice_id'],
                customdata=df[['rank', 'similarity']].values
            )
        
        # Add green/red circle highlights for top K results (2D)
        if top_k is not None and top_k > 0:
            # Get top K search results only
            if 'is_search_result' in df.columns:
                search_results = df[df['is_search_result'] == True]
                top_k_points = search_results.head(top_k)
            else:
                top_k_points = df.head(top_k)  # Get top K results
            
            # Create colors array - red for selected, green for others
            circle_colors = []
            for i, row in top_k_points.iterrows():
                if selected_idx is not None and i == selected_idx:
                    circle_colors.append('#FF0000')  # Red for selected
                else:
                    circle_colors.append('#00FF00')  # Green for others
            
            fig.add_trace(
                go.Scatter(
                    x=top_k_points['x'],
                    y=top_k_points['y'],
                    mode='markers',
                    marker=dict(
                        size=top_k_points['similarity'] * 15 + 20,  # Slightly larger than base points
                        color='rgba(0,0,0,0)',  # Transparent fill
                        line=dict(width=3, color=circle_colors)  # Dynamic colors based on selection
                    ),
                    name=f'Top {top_k} Results',
                    showlegend=True,
                    hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<br>Top Result<extra></extra>',
                    text=top_k_points['slice_id'],
                    customdata=top_k_points[['rank', 'similarity']].values
                )
            )
        
        # Note: Selected point highlighting is now handled by red circles in the top K results above
        
        # Add query vector point for 2D (using star symbol)
        if query_info:
            # Use fixed query coordinates at center (same as used for positioning)
            query_x, query_y = 0, 0
            
            fig.add_trace(
                go.Scatter(
                    x=[query_x],
                    y=[query_y],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='gold',
                        symbol='star',
                        line=dict(width=3, color='orange')
                    ),
                    name='Query',
                    text=[query_info.get('display_text', 'Query')],
                    hovertemplate='<b>Query: %{text}</b><extra></extra>',
                    showlegend=True
                )
            )
        
        # Add cross coordinates at origin (x=0, y=0)
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=1)
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,0.5)", line_width=1)
        
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title,
            plot_bgcolor='#f8fafc',  # Match tip section background
            coloraxis_colorbar=dict(
                title="Similarity",
                title_side="right",
                x=1.01,
                len=0.6,
                thickness=10,
                title_font=dict(size=10),
                tickfont=dict(size=9)
            )
        )
    
    # Update common layout properties
    fig.update_layout(
        height=350,
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#1e293b'}
        },
        dragmode="zoom",
        showlegend=True,
        legend=dict(
            orientation="v",
            x=-0.13,
            y=1,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=10),
            itemsizing="constant",
            itemwidth=30
        ),
        margin=dict(l=200, r=60, t=60, b=60)  # Extra left margin for legend
    )
    
    return fig





def get_gif_path_from_result(video_info: Dict) -> Optional[str]:
    """
    Get GIF file path from search result.
    
    Args:
        video_info: Video information dictionary from search results
        
    Returns:
        GIF file path or None if not available
    """
    # First check if gif_file is directly available in video_info
    gif_path = video_info.get('gif_file', '')
    if gif_path and os.path.exists(gif_path):
        return gif_path
    
    # Check in metadata
    metadata = video_info.get('metadata', {})
    gif_path = metadata.get('gif_file', '')
    if gif_path and os.path.exists(gif_path):
        return gif_path
    
    return None


def get_input_gif_path(slice_id: str, search_engine) -> Optional[str]:
    """
    Get GIF file path for input video by slice_id from unified embeddings parquet file.
    
    Args:
        slice_id: The slice ID of the input video
        search_engine: The search engine instance (not used in this implementation)
        
    Returns:
        GIF file path or None if not available
    """
    try:
        # Load unified embeddings parquet file to get gif_file path
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        unified_embeddings_path = project_root / "data" / "unified_embeddings.parquet"
        
        if unified_embeddings_path.exists():
            logger.info(f"Loading unified embeddings from: {unified_embeddings_path}")
            df = pd.read_parquet(unified_embeddings_path)
            
            # Look for the slice_id in the dataframe
            matching_rows = df[df['slice_id'] == slice_id]
            if not matching_rows.empty:
                gif_path = matching_rows.iloc[0].get('gif_file', '')
                if gif_path and os.path.exists(gif_path):
                    logger.info(f"Found GIF in unified embeddings: {gif_path}")
                    return gif_path
                elif gif_path:
                    logger.warning(f"GIF path exists in database but file not found: {gif_path}")
                else:
                    logger.warning(f"No gif_file entry for slice_id: {slice_id}")
            else:
                logger.warning(f"slice_id not found in unified embeddings: {slice_id}")
        else:
            logger.warning(f"Unified embeddings file not found: {unified_embeddings_path}")
        
        # Note: With unified embeddings, GIF paths should be in the main unified database
        # No need for separate query database fallback
        
        logger.warning(f"No GIF found for slice_id: {slice_id}")
        
    except Exception as e:
        logger.warning(f"Error getting input GIF path for {slice_id}: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    return None


def get_thumbnail_from_result(video_info: Dict) -> Optional[str]:
    """
    Get thumbnail base64 string from search result.
    Uses stored thumbnail if available, otherwise extracts on-the-fly as fallback.
    
    Args:
        video_info: Video information dictionary from search results
        
    Returns:
        Base64 encoded thumbnail string or None if not available
    """
    slice_id = video_info.get('slice_id', 'Unknown')
    
    thumbnail_b64 = video_info.get('thumbnail', '')
    if thumbnail_b64:
        return thumbnail_b64
    
    # Fallback to on-the-fly extraction (legacy behavior)
    logger.info(f"Extracting thumbnail on-the-fly for {slice_id}")
    
    video_path = video_info.get('video_path', '')
    if not video_path:
        return None
        
    try:
        from pathlib import Path
        full_path = Path(video_path)
        
        if not full_path.exists():
            return None
            
        # Use VideoResultsVisualizer to extract thumbnail with config size
        from core.config import VideoRetrievalConfig
        config = VideoRetrievalConfig()
        visualizer = VideoResultsVisualizer(thumbnail_size=config.thumbnail_size)
        thumbnail = visualizer.extract_thumbnail(full_path)
        
        if thumbnail is None:
            return None
            
        # Convert numpy array to PIL if needed
        if isinstance(thumbnail, np.ndarray):
            thumbnail_pil = Image.fromarray(thumbnail)
        else:
            thumbnail_pil = thumbnail
        
        # Convert to base64 with high quality
        img_buffer = io.BytesIO()
        thumbnail_pil.save(img_buffer, format='JPEG', quality=95, optimize=True)
        result = base64.b64encode(img_buffer.getvalue()).decode()
        
        return result
        
    except Exception as e:
        logger.warning(f"Failed to extract thumbnail for {video_path}: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="ALFA 0.1 - Similarity Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    

    st.markdown("""
    <style>
        /* Force sidebar to always be visible */
        .css-1d391kg {
            width: 21rem !important;
            min-width: 21rem !important;
        }
        

        .css-14xtw13.e8zbici0 {
            display: none !important;
        }
        

        button[data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    

    st.markdown("""
    <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global styling */
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            color: white;
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 25px 50px rgba(99, 102, 241, 0.25);
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.05"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }
        
        .main-header h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #ffffff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
            position: relative;
            z-index: 1;
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        /* Sidebar styling */
        .sidebar .element-container {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        /* Section titles */
        .section-title {
            font-size: 1.4rem;
            color: #1e293b;
            margin-bottom: 1rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            position: relative;
            padding-bottom: 0.75rem;
        }
        
        .section-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 50px;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            border-radius: 2px;
        }
        
        /* Stats grid */
        .stats-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            margin: 0.75rem 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 0.75rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid rgba(226, 232, 240, 0.5);
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-1px);
        }
        
        .stat-value {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }
        
        .stat-label {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        
        /* Featured video styling */
        .featured-video {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            border: 1px solid rgba(226, 232, 240, 0.5);
        }
        

        /* Video cards */
        .video-card {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            margin-bottom: 0.75rem;
        }
        
        .video-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        
        .video-card.selected {
            border-color: #6366f1;
            box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
        }
        
        /* Search input styling */
        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }
        
        /* Button styling - Only style actual buttons, not containers */
        .stButton > button,
        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.2) !important;
            width: 100% !important;
            height: auto !important;
            min-height: 2.5rem !important;
            display: block !important;
            box-sizing: border-box !important;
        }
        
        /* Ensure button containers take full width but don't get button styling */
        .stButton,
        div[data-testid="stButton"] {
            width: 100% !important;
        }
        
        .stButton > button:hover,
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        }
        
        /* Secondary button */
        .stButton > button[kind="secondary"],
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
            box-shadow: 0 4px 8px rgba(100, 116, 139, 0.2) !important;
        }
        
        .stButton > button[kind="secondary"]:hover,
        div[data-testid="stButton"] > button[kind="secondary"]:hover {
            box-shadow: 0 8px 20px rgba(100, 116, 139, 0.4) !important;
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div {
            border-radius: 12px;
            border: 2px solid #e2e8f0;
        }
        
        /* Success/info messages */
        .stSuccess {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 12px;
            border: none;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border-radius: 12px;
            border: none;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            border: 1px solid rgba(226, 232, 240, 0.5);
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
            transform: translateY(-1px);
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white !important;
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.25);
        }
        
        /* Visualization tabs special styling */
        .stTabs [data-baseweb="tab-panel"] {
            padding: 1rem 0;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 12px;
            border: 1px solid rgba(226, 232, 240, 0.5);
            font-weight: 600;
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
        }
        
        /* Progress bar styling */
        .stProgress .st-bo {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
        }
        
        /* Metric styling */
        .metric-container {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid rgba(226, 232, 240, 0.5);
            text-align: center;
        }
        
        /* Remove default streamlit spacing */
        .element-container {
            margin-bottom: 0 !important;
        }
        
        .stButton > button {
            margin-bottom: 0 !important;
        }
        
        /* Results column styling */
        .results-column {
            padding: 0;
            margin: 0;
        }
        
        .results-column .element-container {
            margin-bottom: 0.25rem !important;
        }
        
        /* Smaller buttons for top K results */
        .top-k-button button {
            padding: 0.3rem 0.8rem !important;
            font-size: 0.8rem !important;
            min-height: 2rem !important;
        }
        
        /* Custom scrollbar styling for horizontal scroll */
        .horizontal-scroll-container::-webkit-scrollbar {
            height: 8px;
        }
        
        .horizontal-scroll-container::-webkit-scrollbar-track {
            background: #f1f5f9;
            border-radius: 4px;
        }
        
        .horizontal-scroll-container::-webkit-scrollbar-thumb {
            background: #6366f1;
            border-radius: 4px;
        }
        
        .horizontal-scroll-container::-webkit-scrollbar-thumb:hover {
            background: #4f46e5;
        }
        

        
        /* Style buttons in results column */
        .stColumn:last-child .stButton > button {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid rgba(226, 232, 240, 0.5);
            border-radius: 8px;
            padding: 0.4rem 0.6rem;
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
            transition: all 0.3s ease;
            text-align: left;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .stColumn:last-child .stButton > button:hover {
            background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Make result cards more interactive */
        .stColumn:last-child .video-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        /* Remove bottom margins from main container */
        .main .block-container {
            padding-bottom: 0 !important;
            margin-bottom: 0 !important;
        }
        
        /* Remove bottom margins from expanders */
        .streamlit-expanderHeader {
            margin-bottom: 0 !important;
        }
        
        .streamlit-expanderContent {
            padding-bottom: 0 !important;
            margin-bottom: 0 !important;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 2rem;
            }
            .main-header p {
                font-size: 1rem;
            }
            .stats-container {
                grid-template-columns: 1fr;
            }
            .video-card {
                padding: 0.75rem;
            }
            .featured-video {
                padding: 1.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    

    if 'text_selection' not in st.session_state:
        st.session_state.text_selection = SelectedVideo(-1)
    if 'click_selection' not in st.session_state:
        st.session_state.click_selection = SelectedVideo(-1)
    if 'text_query' not in st.session_state:
        st.session_state.text_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []

    if 'input_video_info' not in st.session_state:
        st.session_state.input_video_info = None
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.3  # Lower default for better text search compatibility
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'text_weight_alpha' not in st.session_state:
        st.session_state.text_weight_alpha = 0.5  # Default alpha for joint search
    
    top_k = st.session_state.top_k
    similarity_threshold = st.session_state.similarity_threshold
    # UMAP and DBSCAN are pre-computed and stored in the parquet file

    st.markdown("""
    <div class="main-header">
        <h1>ALFA 0.1 - Embedding Search</h1>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        with st.spinner("Loading search engine..."):
            try:
                search_engine = load_search_engine()
                db_info = load_database_info(search_engine)
            except Exception as e:
                st.error(f"❌ Failed to load search engine: {e}")
                st.error("Please check the logs for more details.")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        

        st.markdown('<div class="section-title">🔍 Settings</div>', unsafe_allow_html=True)
        

        try:
            from pathlib import Path

            project_root = Path(__file__).parent.parent
            video_dir = project_root / "data" / "videos" / "user_input"

            try:
                available_videos = search_engine.get_query_videos_list()
                if not available_videos and video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
                    available_videos = [f.name for f in sorted(video_files)]
            except:
                available_videos = []
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
                    available_videos = [f.name for f in sorted(video_files)]
            
            if available_videos:
                selected_video = st.selectbox(
                    "Select Video",
                    available_videos,
                    help="Select a video file (⚡ = pre-computed embedding)",
                    label_visibility="collapsed"
                )
                
                if st.button("🎥 Search by Video", use_container_width=True, help="Smart search: uses pre-computed embeddings when available, falls back to real-time processing"):
                    with st.spinner("Searching for similar videos..."):
                        try:
                            # Update the search engine's similarity threshold
                            search_engine.config.similarity_threshold = st.session_state.similarity_threshold
                            results = search_engine.search_by_filename(selected_video, top_k=st.session_state.top_k)
                            st.session_state.search_results = results
                            st.session_state.text_query = f"Similar to: {selected_video}"
                            st.session_state.click_selection = SelectedVideo(0)
                            
                            # Store input video info with thumbnail from query database
                            video_path = str(video_dir / selected_video)
                            input_thumbnail = None
                            try:
                                # Get thumbnail from unified database
                                input_thumbnail = search_engine.database.get_thumbnail_base64(selected_video)
                            except:
                                pass
                            
                            st.session_state.input_video_info = {
                                'slice_id': selected_video,
                                'video_path': video_path,
                                'type': 'video',
                                'thumbnail': input_thumbnail or ''
                            }
                            
                            try:
                                # With unified embeddings, all embeddings are in the main database
                                cached_embedding = search_engine.database.get_embedding(selected_video)
                                if cached_embedding is not None:
                                    st.success(f"Found {len(results)} similar videos! ⚡ (Used pre-computed embedding)")
                                else:
                                    st.success(f"Found {len(results)} similar videos! (Processed in real-time)")
                            except:
                                st.success(f"Found {len(results)} similar videos!")
                                
                        except Exception as e:
                            st.error(f"Video search failed: {e}")
            else:
                st.info("No video files found. Use CLI to build query database: `python main.py build-query`")
        except Exception as e:
            st.error(f"Error: {e}")

        text_query = st.text_input(
            "Text Query",
            value=st.session_state.text_query,
            placeholder="car approaching cyclist",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Search by Text", use_container_width=True):
            if text_query:
                with st.spinner("Searching..."):
                    try:
                        # Update the search engine's similarity threshold
                        search_engine.config.similarity_threshold = st.session_state.similarity_threshold
                        results = search_engine.search_by_text(text_query, top_k=st.session_state.top_k)
                        st.session_state.search_results = results
                        st.session_state.text_query = text_query
                        st.session_state.text_selection = SelectedVideo(0)
                        
                        # Store text query as input
                        st.session_state.input_video_info = {
                            'slice_id': text_query,
                            'type': 'text'
                        }
                        
                        st.success(f"Found {len(results)} results!")
                    except NoResultsError:
                        st.warning(f"No results found above similarity threshold ({st.session_state.similarity_threshold:.2f})")
                        if st.session_state.similarity_threshold > 0.3:
                            st.info("💡 **Tip:** Text searches typically have lower similarity scores. Try reducing the similarity threshold to 0.1-0.3 for better results.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")

        # Alpha slider for joint search
        st.session_state.text_weight_alpha = st.slider(
            "Text Weight (α)", 
            0.0, 
            1.0, 
            st.session_state.text_weight_alpha, 
            0.1,
            help="Weight for text in joint search: 0.0 = video only, 1.0 = text only, 0.5 = balanced"
        )

        # Joint search button
        if st.button("🔗 Joint Search (Text + Video)", use_container_width=True):
            if text_query and available_videos and selected_video:
                with st.spinner("Performing joint search..."):
                    try:
                        # Update the search engine's similarity threshold
                        search_engine.config.similarity_threshold = st.session_state.similarity_threshold
                        results = search_engine.search_by_joint(
                            text_query, 
                            selected_video, 
                            alpha=st.session_state.text_weight_alpha,
                            top_k=st.session_state.top_k
                        )
                        st.session_state.search_results = results
                        st.session_state.text_query = f"Joint: \"{text_query}\" + {selected_video}"
                        st.session_state.text_selection = SelectedVideo(0)
                        
                        # Store joint search info
                        video_path = str(video_dir / selected_video)
                        input_thumbnail = None
                        try:
                            # Get thumbnail from unified database
                            input_thumbnail = search_engine.database.get_thumbnail_base64(selected_video)
                        except:
                            pass
                        
                        st.session_state.input_video_info = {
                            'slice_id': f"Joint: \"{text_query}\" + {selected_video}",
                            'video_path': video_path,
                            'type': 'joint',
                            'text_query': text_query,
                            'video_slice_id': selected_video,
                            'alpha': st.session_state.text_weight_alpha,
                            'thumbnail': input_thumbnail or ''
                        }
                        
                        st.success(f"Found {len(results)} results! (α={st.session_state.text_weight_alpha:.1f}: {st.session_state.text_weight_alpha*100:.0f}% text, {(1-st.session_state.text_weight_alpha)*100:.0f}% video)")
                    except NoResultsError:
                        st.warning(f"No results found above similarity threshold ({st.session_state.similarity_threshold:.2f})")
                        if st.session_state.similarity_threshold > 0.3:
                            st.info("💡 **Tip:** Try reducing the similarity threshold for better results.")
                    except Exception as e:
                        st.error(f"Joint search failed: {e}")
            else:
                missing_items = []
                if not text_query:
                    missing_items.append("text query")
                if not available_videos or not selected_video:
                    missing_items.append("video selection")
                st.warning(f"Please provide: {', '.join(missing_items)}")

        st.session_state.top_k = st.slider("Top-K Results", 1, 15, st.session_state.top_k)

        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold", 
            0.0, 
            1.0, 
            st.session_state.similarity_threshold, 
            0.05,  # Smaller step size for finer control
            help="Lower values show more results. Text searches typically need lower thresholds (0.1-0.3) than video searches (0.3-0.5)."
        )

        analysis_type = st.selectbox(
            "Analysis Type",
            options=["embedding_similarity", "umap_dbscan_clusters"],
            format_func=lambda x: {
                "embedding_similarity": "Embedding Similarity",
                "umap_dbscan_clusters": "UMAP + DBSCAN Clusters"
            }[x],
            index=0,
            label_visibility="collapsed"
        )

        # Note: UMAP projections are pre-computed and stored in the parquet file

        st.markdown('<div class="section-title">📊 Database Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{db_info.get('num_inputs', 10000)}</div>
                <div class="stat-label">Database Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('categories', 0)}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('embedding_dim', 768)}</div>
                <div class="stat-label">Embedding Dim</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('search_backend', 'FAISS')}</div>
                <div class="stat-label">Search Backend</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

    st.markdown('<div class="section-title">Embedding Visualization</div>', unsafe_allow_html=True)
    


    

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["2D View", "3D View", "Heatmap"])
    
    with viz_tab1:
        if st.session_state.search_results:
            selected_idx = None
            if st.session_state.text_selection.is_valid() and st.session_state.click_selection.is_valid():
                if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                    selected_idx = st.session_state.text_selection.idx
                else:
                    selected_idx = st.session_state.click_selection.idx
            elif st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            

            query_info = {
                'display_text': st.session_state.text_query if st.session_state.text_query else "No query"
            }
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "umap",  # Always use UMAP for 2D scatter plot
                selected_idx,
                query_info=query_info,
                top_k=st.session_state.top_k,
                all_videos=all_videos,
                analysis_type=analysis_type
            )
            
            plot_selection = st.plotly_chart(
                fig, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=f"plot_2d_{selected_idx}"
            )
            

            if plot_selection and plot_selection.get("selection", {}).get("point_indices"):
                clicked_idx = plot_selection["selection"]["point_indices"][0]
                if clicked_idx != st.session_state.click_selection.idx:
                    st.session_state.click_selection = SelectedVideo(clicked_idx)
                    st.rerun()
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab2:
        if st.session_state.search_results:            
            # Interactive 3D plot - use explicit timestamp comparison
            selected_idx = None
            # Always use the most recent selection
            if st.session_state.text_selection.is_valid() and st.session_state.click_selection.is_valid():
                if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                    selected_idx = st.session_state.text_selection.idx
                else:
                    selected_idx = st.session_state.click_selection.idx
            elif st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            
            # Create 3D visualization with query info
            query_info = {
                'display_text': st.session_state.text_query if st.session_state.text_query else "No query"
            }
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "3d_umap", 
                selected_idx,
                query_info=query_info,
                top_k=st.session_state.top_k,
                all_videos=all_videos,
                analysis_type=analysis_type
            )
            
            plot_selection_3d = st.plotly_chart(
                fig, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=f"plot_3d_{selected_idx}"
            )
            
            # Handle 3D plot clicks
            if plot_selection_3d and plot_selection_3d.get("selection", {}).get("point_indices"):
                clicked_idx = plot_selection_3d["selection"]["point_indices"][0]
                if clicked_idx != st.session_state.click_selection.idx:
                    st.session_state.click_selection = SelectedVideo(clicked_idx)
                    st.rerun()
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab3:
        if st.session_state.search_results:
            # Create heatmap visualization
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "similarity", 
                None,
                all_videos=all_videos,
                analysis_type=analysis_type
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{len(st.session_state.search_results)}")
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    


    if st.session_state.search_results:

        current_selection = max(
            st.session_state.text_selection.idx if st.session_state.text_selection.is_valid() else -1,
            st.session_state.click_selection.idx if st.session_state.click_selection.is_valid() else -1
        )
        if current_selection == -1:
            current_selection = 0

        # Display results with improved styling
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">Top K Results</h3>
            <p style="color: #64748b; font-size: 0.9rem; margin-top: 0.5rem; font-style: italic;">
                💡 Click on points in the embedding visualization above to select and view different results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a fixed-width horizontal scrollable container for top K results
        total_results = len(st.session_state.search_results)
        
        # Use native Streamlit columns for guaranteed horizontal layout
        # Show up to 5 results in columns, then add a note about scrolling through embedding visualization
        num_display_results = min(5, len(st.session_state.search_results))
        
        if num_display_results > 0:
            result_columns = st.columns(num_display_results)
            
            for i in range(num_display_results):
                video = st.session_state.search_results[i]
                is_selected = (current_selection == i)
                
                with result_columns[i]:
                    # Get thumbnail data
                    thumbnail_data = get_thumbnail_from_result(video)
                    
                    # Style variables
                    border_style = "border: 3px solid #6366f1;" if is_selected else "border: 2px solid #e2e8f0;"
                    overlay_bg = "rgba(99,102,241,0.9)" if is_selected else "rgba(0,0,0,0.7)"
                    overlay_shadow = "box-shadow: 0 2px 8px rgba(99,102,241,0.4);" if is_selected else ""
                    checkmark = " ✓" if is_selected else ""
                    
                    if thumbnail_data:
                        try:
                            # Process image
                            thumbnail_bytes = base64.b64decode(thumbnail_data)
                            thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                            
                            # Resize to fixed height
                            fixed_height = 200
                            original_width, original_height = thumbnail_pil.size
                            aspect_ratio = original_width / original_height
                            calculated_width = int(fixed_height * aspect_ratio)
                            
                            thumbnail_pil_resized = thumbnail_pil.resize((calculated_width, fixed_height), Image.Resampling.LANCZOS)
                            
                            # Convert to base64
                            img_buffer = io.BytesIO()
                            thumbnail_pil_resized.save(img_buffer, format='JPEG', quality=95)
                            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                            
                            # Display image with overlay
                            st.markdown(f"""
                            <div style="position: relative; margin-bottom: 0.5rem;">
                                <img src="data:image/jpeg;base64,{img_b64}" 
                                     style="width: 100%; height: 200px; border-radius: 8px; object-fit: cover; {border_style}">
                                <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                     background: {overlay_bg}; color: white; padding: 4px 8px; border-radius: 4px; 
                                     font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); {overlay_shadow}">
                                    Rank {i+1}{checkmark}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            logger.warning(f"Failed to display thumbnail for {video['slice_id']}: {e}")
                            # Fallback placeholder
                            st.markdown(f"""
                            <div style="position: relative; margin-bottom: 0.5rem;">
                                <div style="width: 100%; height: 200px; background: #6366f1; border-radius: 8px; {border_style}
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                    🎬
                                </div>
                                <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                     background: {overlay_bg}; color: white; padding: 4px 8px; border-radius: 4px; 
                                     font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); {overlay_shadow}">
                                    Rank {i+1}{checkmark}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Colored placeholder
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                        color = colors[i % len(colors)]
                        st.markdown(f"""
                        <div style="position: relative; margin-bottom: 0.5rem;">
                            <div style="width: 100%; height: 200px; background: {color}; border-radius: 8px; {border_style}
                                 display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                🎬
                            </div>
                            <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                 background: {overlay_bg}; color: white; padding: 4px 8px; border-radius: 4px; 
                                 font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); {overlay_shadow}">
                                Rank {i+1}{checkmark}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Video info - single row with ID and score
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0.75rem;">
                        <div style="font-size: 0.85rem; color: #1e293b;">
                            <span style="font-weight: 600;">{video['slice_id'][:8]}</span>
                            <span style="color: #6366f1; font-weight: 600; margin-left: 0.5rem;">
                                Score: {video['similarity_score']:.3f}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add navigation hint if there are many results
        if total_results > 5:
            st.markdown(f"""
            <div style="text-align: center; margin-top: 0.5rem;">
                <small style="color: #64748b; font-style: italic;">
                    Showing top 5 of {total_results} results. Click points in the embedding visualization above to view other results.
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        # Display input and selected result side by side with improved styling
        if current_selection is not None and current_selection < len(st.session_state.search_results):
            st.markdown("""
            <hr style="margin: 2rem 0; border: none; border-top: 2px solid #e2e8f0; opacity: 0.6;">
            """, unsafe_allow_html=True)
            
            featured_video = st.session_state.search_results[current_selection]
            
            # Create two equal columns for side-by-side comparison with better spacing
            input_preview_col, selected_preview_col = st.columns(2, gap="large")
            
            with input_preview_col:
                # Display input based on type
                if st.session_state.input_video_info:
                    if st.session_state.input_video_info['type'] == 'joint':
                        # Joint search display
                        joint_text = st.session_state.input_video_info.get('text_query', '')
                        joint_video = st.session_state.input_video_info.get('video_slice_id', '')
                        joint_alpha = st.session_state.input_video_info.get('alpha', 0.5)
                        
                        # Try to get GIF for joint video first
                        joint_gif_path = get_input_gif_path(joint_video, search_engine)
                        if joint_gif_path:
                            try:
                                st.image(joint_gif_path, use_container_width=True)
                            except Exception as e:
                                logger.warning(f"Failed to display joint input GIF: {e}")
                                joint_gif_path = None
                        
                        # If no GIF or GIF failed, fall back to thumbnail
                        if not joint_gif_path:
                            input_thumbnail = st.session_state.input_video_info.get('thumbnail', '')
                            
                            if input_thumbnail:
                                try:
                                    thumbnail_bytes = base64.b64decode(input_thumbnail)
                                    thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                                    thumbnail_array = np.array(thumbnail_pil)
                                    
                                    st.image(thumbnail_array, use_container_width=True)
                                except Exception as e:
                                    logger.warning(f"Failed to display joint input thumbnail: {e}")
                                    st.markdown("""
                                    <div style="width: 100%; aspect-ratio: 16/9; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%); border-radius: 8px;
                                         display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                        🔗
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                # Placeholder for joint search
                                st.markdown("""
                                <div style="width: 100%; aspect-ratio: 16/9; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%); border-radius: 8px;
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                    🔗
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Joint search info with alpha display
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 0.75rem;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">
                                Joint Search
                            </div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.25rem;">
                                Text: "{joint_text}"
                            </div>
                            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.25rem;">
                                Video: {joint_video}
                            </div>
                            <div style="font-size: 0.8rem; color: #6366f1; font-weight: 600;">
                                α = {joint_alpha:.1f} ({joint_alpha*100:.0f}% text, {(1-joint_alpha)*100:.0f}% video)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif st.session_state.input_video_info['type'] == 'video':
                        # Try to get GIF for input video first
                        input_slice_id = st.session_state.input_video_info.get('slice_id', '')
                        input_video_path = st.session_state.input_video_info.get('video_path', '')
                        
                        # First try to display GIF
                        input_gif_path = get_input_gif_path(input_slice_id, search_engine)
                        if input_gif_path:
                            try:
                                st.image(input_gif_path, use_container_width=True)
                            except Exception as e:
                                logger.warning(f"Failed to display input GIF: {e}")
                                # Fall back to thumbnail
                                input_gif_path = None
                        
                        # If no GIF or GIF failed, fall back to thumbnail
                        if not input_gif_path:
                            # Get thumbnail from stored info or query database
                            input_thumbnail = st.session_state.input_video_info.get('thumbnail', '')
                            
                            if not input_thumbnail:
                                try:
                                    # Use the unified database to get pre-stored thumbnail
                                    input_thumbnail = search_engine.database.get_thumbnail_base64(input_slice_id)
                                    logger.info(f"Input thumbnail from unified DB: {'Success' if input_thumbnail else 'Not found'}")
                                    
                                    # Fallback to on-the-fly extraction if not in database
                                    if not input_thumbnail and input_video_path and Path(input_video_path).exists():
                                        logger.info(f"Falling back to on-the-fly extraction for {input_slice_id}")
                                        input_video_info = {
                                            'video_path': input_video_path,
                                            'slice_id': input_slice_id,
                                            'thumbnail': ''  # Force on-the-fly extraction
                                        }
                                        input_thumbnail = get_thumbnail_from_result(input_video_info)
                                        logger.info(f"On-the-fly extraction result: {'Success' if input_thumbnail else 'Failed'}")
                                except Exception as e:
                                    logger.error(f"Failed to get input thumbnail: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                            
                            # Display input thumbnail
                            if input_thumbnail:
                                try:
                                    thumbnail_bytes = base64.b64decode(input_thumbnail)
                                    thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                                    thumbnail_array = np.array(thumbnail_pil)
                                    
                                    st.image(thumbnail_array, use_container_width=True)
                                except Exception as e:
                                    logger.warning(f"Failed to display input thumbnail: {e}")
                                    st.markdown("""
                                    <div style="width: 100%; aspect-ratio: 16/9; background: #1e293b; border-radius: 8px;
                                         display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                        🎬
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                # Placeholder for input video
                                st.markdown("""
                                <div style="width: 100%; aspect-ratio: 16/9; background: #1e293b; border-radius: 8px;
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                    🎬
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Input video info with consistent styling
                        st.markdown(f"""
                        <div style="text-align: center; margin-top: 0rem;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">
                                Input: {input_slice_id}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Text query
                        st.markdown(f"""
                        <div style="width: 100%; aspect-ratio: 16/9; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                             border-radius: 8px; display: flex; align-items: center; justify-content: center; padding: 20px;">
                            <div style="color: white; text-align: center;">
                                <div style="font-size: 2.5rem; margin-bottom: 12px;">🔍</div>
                                <div style="font-size: 1rem; font-weight: 500; word-wrap: break-word; line-height: 1.4;">
                                    "{st.session_state.input_video_info['slice_id']}"
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Text query info
                        st.markdown("""
                        <div style="text-align: center; margin-top: 0rem;">
                            <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">
                                Text Query
                            </div>
                            <div style="font-size: 0.8rem; color: #64748b;">
                                Search Input
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with selected_preview_col:
                # Display selected video GIF
                gif_path = get_gif_path_from_result(featured_video)
                if gif_path:
                    try:
                        # Display GIF using st.image which supports GIF files
                        st.image(gif_path, use_container_width=True)
                    except Exception as e:
                        logger.warning(f"Failed to display GIF: {e}")
                        # Fallback to thumbnail if GIF fails
                        thumbnail_data = get_thumbnail_from_result(featured_video)
                        if thumbnail_data:
                            try:
                                thumbnail_bytes = base64.b64decode(thumbnail_data)
                                thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                                thumbnail_array = np.array(thumbnail_pil)
                                
                                st.image(thumbnail_array, use_container_width=True)
                            except Exception as thumb_e:
                                logger.warning(f"Failed to display thumbnail fallback: {thumb_e}")
                                st.markdown("""
                                <div style="width: 100%; aspect-ratio: 16/9; background: #10b981; border-radius: 8px;
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                    🎬
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="width: 100%; aspect-ratio: 16/9; background: #10b981; border-radius: 8px;
                                 display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                🎬
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    # Fallback to thumbnail if no GIF available
                    thumbnail_data = get_thumbnail_from_result(featured_video)
                    if thumbnail_data:
                        try:
                            thumbnail_bytes = base64.b64decode(thumbnail_data)
                            thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                            thumbnail_array = np.array(thumbnail_pil)
                            
                            st.image(thumbnail_array, use_container_width=True)
                        except Exception as e:
                            logger.warning(f"Failed to display thumbnail: {e}")
                            st.markdown("""
                            <div style="width: 100%; aspect-ratio: 16/9; background: #10b981; border-radius: 8px;
                                 display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                🎬
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="width: 100%; aspect-ratio: 16/9; background: #10b981; border-radius: 8px;
                             display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                            🎬
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display selected video info - only slice_id
                st.markdown(f"""
                <div style="text-align: center; margin-top: 0rem;">
                    <div style="font-size: 0.95rem; font-weight: 600; color: #1e293b;">
                        Selected Result: {featured_video['slice_id']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        

        with st.expander("📊 Detailed Results Table"):
            results_df = pd.DataFrame([
                {
                    'Rank': r['rank'],
                    'Video Name': r['slice_id'],
                    'Similarity Score': f"{r['similarity_score']:.4f}",
                    'Video Path': r['video_path']
                }
                for r in st.session_state.search_results
            ])
            
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

if __name__ == "__main__":
    main()
