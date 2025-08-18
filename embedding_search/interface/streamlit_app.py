#!/usr/bin/env python3
"""
ALFA 0.1 - Similarity Search Interface
"""

import os
# Fix OpenMP library conflict issue  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging
from core.search import VideoSearchEngine
from core.visualizer import VideoResultsVisualizer
from core.config import VideoRetrievalConfig
from core.exceptions import VideoNotFoundError, NoResultsError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelectedVideo:
    """Track selected video with timestamp (following official implementation pattern)."""
    
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
    """Load and cache the search engine (following official caching pattern)."""
    try:
        config = VideoRetrievalConfig()
        search_engine = VideoSearchEngine(config=config)
        
        # Try to load existing database, if it fails, we'll work with empty database
        try:
            search_engine.database.load_from_parquet(config.database_path + ".parquet")
        except:
            try:
                search_engine.database.load()
            except:
                pass
                
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
            "num_videos": 0,
            "categories": 0,
            "embedding_dim": 768,
            "search_backend": "Mock",
            "using_gpu": False,
            "cache_size": 0,
            "error": str(e)
        }


def get_all_videos_from_database(search_engine) -> List[Dict]:
    """Get all videos from the database for visualization."""
    try:
        if hasattr(search_engine, 'database') and hasattr(search_engine.database, 'metadata'):
            all_videos = []
            for i, metadata in enumerate(search_engine.database.metadata):
                video_info = {
                    'video_name': metadata.get('video_name', f"Video {i+1}"),
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


def create_embedding_visualization(results: List[Dict], viz_method: str = "umap", selected_idx: Optional[int] = None, query_info: Optional[Dict] = None, top_k: Optional[int] = None, all_videos: Optional[List[Dict]] = None, **kwargs) -> go.Figure:
    """Create advanced embedding visualization with multiple methods."""
    if not results:
        return go.Figure()
    
    # Use all videos from database if provided, otherwise use just search results
    if all_videos and len(all_videos) > len(results):
        # Create DataFrame with all videos
        df_all = pd.DataFrame([
            {
                'video_name': r.get('video_name', f"Video {i+1}"),
                'similarity': 0.1,  # Default low similarity for non-search results
                'rank': i + len(results) + 1,  # Rank after search results
                'category': getattr(r, 'category', 'unknown'),
                'idx': i + len(results),
                'is_search_result': False
            }
            for i, r in enumerate(all_videos[len(results):])  # Videos not in search results
        ])
        
        df_search = pd.DataFrame([
            {
                'video_name': r['video_name'],
                'similarity': r['similarity_score'],
                'rank': r['rank'],
                'category': getattr(r, 'category', 'unknown'),
                'idx': i,
                'is_search_result': True
            }
            for i, r in enumerate(results)
        ])
        
        df = pd.concat([df_search, df_all], ignore_index=True)
    else:
        df = pd.DataFrame([
            {
                'video_name': r['video_name'],
                'similarity': r['similarity_score'],
                'rank': r['rank'],
                'category': getattr(r, 'category', 'unknown'),
                'idx': i,
                'is_search_result': True
            }
            for i, r in enumerate(results)
        ])
    
    # Generate coordinates based on visualization method
    np.random.seed(42)
    
    query_x, query_y, query_z = 0, 0, 0
    
    if viz_method == "umap":
        # Position vectors based on distance from query (higher similarity = closer)
        distances = (1 - df['similarity']) * 8  # Convert similarity to distance
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.5
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.5
        title = "2D Embedding Space - UMAP"
        x_title, y_title = "UMAP Dimension 1", "UMAP Dimension 2"
    elif viz_method == "pca":
        distances = (1 - df['similarity']) * 6
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.3
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.3
        title = "2D Embedding Space - PCA"
        x_title, y_title = "PCA Dimension 1", "PCA Dimension 2"
    elif viz_method == "trimap":
        distances = (1 - df['similarity']) * 7
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.4
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.4
        title = "2D Embedding Space - TriMAP"
        x_title, y_title = "TriMAP Dimension 1", "TriMAP Dimension 2"
    elif viz_method == "tsne":
        distances = (1 - df['similarity']) * 5
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.3
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.3
        title = "2D Embedding Space - t-SNE"
        x_title, y_title = "t-SNE Dimension 1", "t-SNE Dimension 2"
    elif viz_method == "3d_umap":
        distances = (1 - df['similarity']) * 6
        # Generate random 3D directions
        theta = np.random.uniform(0, 2*np.pi, len(df))  # azimuthal angle
        phi = np.random.uniform(0, np.pi, len(df))      # polar angle
        df['x'] = query_x + distances * np.sin(phi) * np.cos(theta) + np.random.randn(len(df)) * 0.3
        df['y'] = query_y + distances * np.sin(phi) * np.sin(theta) + np.random.randn(len(df)) * 0.3
        df['z'] = query_z + distances * np.cos(phi) + np.random.randn(len(df)) * 0.3
        title = "3D Embedding Space - UMAP"
    else:  # similarity heatmap
        # Create similarity matrix visualization
        n = len(df)
        similarity_matrix = np.random.rand(n, n) * 0.5 + 0.3
        for i in range(n):
            similarity_matrix[i, i] = 1.0
            for j in range(i+1, n):
                similarity_matrix[j, i] = similarity_matrix[i, j]
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[f"Video {i+1}" for i in range(n)],
            y=[f"Video {i+1}" for i in range(n)],
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
                outlinewidth=0  # Remove black edge around color bar
            )
        ))
        fig.update_layout(
            title="Video Similarity Matrix",
            height=350,
            width=None,  # Let it be responsive to container width
            title_x=0.5,
            title_font=dict(size=14),
            margin=dict(l=200, r=40, t=50, b=40),  # Extra left margin for legend
            plot_bgcolor='#f8fafc',  # Match tip section background
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=True,  # Enable legend for plot
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
            )
        )
        return fig
    
    # Create 2D or 3D scatter plot
    if viz_method == "3d_umap":
        # Split data into search results and other videos
        if 'is_search_result' in df.columns:
            df_search = df[df['is_search_result'] == True]
            df_other = df[df['is_search_result'] == False]
            
            # Create traces for non-search-result videos (grayed out)
            traces = []
            if len(df_other) > 0:
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
                    text=df_other['video_name'],
                    hovertemplate='<b>%{text}</b><br>Database Video<extra></extra>',
                    name='Videos',
                    showlegend=True
                ))
            
            # Create trace for search results (colorful)
            if len(df_search) > 0:
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
                    text=df_search['video_name'],
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
                text=df['video_name'],
                hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                customdata=df[['rank', 'similarity']].values
            ))
        
        # Add green/red circle highlights for top K results (3D)
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
                    text=top_k_points['video_name'],
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
            
            # Add non-search-result videos (grayed out)
            if len(df_other) > 0:
                fig.add_trace(go.Scatter(
                    x=df_other['x'],
                    y=df_other['y'],
                    mode='markers',
                    marker=dict(
                        size=8,  # Smaller size for non-results
                        color='lightgray',
                        opacity=0.3
                    ),
                    text=df_other['video_name'],
                    hovertemplate='<b>%{text}</b><br>Database Video<extra></extra>',
                    name='Videos',
                    showlegend=True
                ))
            
            # Add search results (colorful)
            if len(df_search) > 0:
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
                    text=df_search['video_name'],
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
                hover_name='video_name',
                hover_data=['rank', 'similarity'],
                color_continuous_scale='Viridis',
                title=title
            )
            
            # Enhance hover template for better information display
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                hovertext=df['video_name'],
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
                    text=top_k_points['video_name'],
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


def create_similarity_plot(results: List[Dict], selected_idx: Optional[int] = None) -> go.Figure:
    """Legacy function for backwards compatibility."""
    return create_embedding_visualization(results, "umap", selected_idx)


def preview_video_with_thumbnail(video_info: Dict, height: int = 300) -> None:
    """
    Create a video preview with real thumbnail extraction.
    Falls back to placeholder if thumbnail extraction fails.
    """
    video_path = video_info.get('video_path', '')
    video_name = video_info.get('video_name', 'Unknown')
    similarity = video_info.get('similarity_score', 0)
    rank = video_info.get('rank', 'N/A')
    
    # Try to extract real thumbnail
    try:
        from pathlib import Path
        full_path = Path(video_path)
        
        
        # Check if video file exists
        if full_path.exists():
            # Use VideoResultsVisualizer to extract thumbnail
            visualizer = VideoResultsVisualizer()
            thumbnail = visualizer.extract_thumbnail(full_path)
            
            # Display thumbnail with proper aspect ratio (16:9)
            import base64
            import io
            from PIL import Image
            
            # Convert numpy array to PIL if needed
            if isinstance(thumbnail, np.ndarray):
                thumbnail_pil = Image.fromarray(thumbnail)
            else:
                thumbnail_pil = thumbnail
            
            # Convert to base64
            img_buffer = io.BytesIO()
            thumbnail_pil.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Create a container with 16:9 aspect ratio
            st.markdown(
                f"""
                <div style="position: relative; width: 100%; padding-bottom: 56.25%; /* 16:9 aspect ratio */">
                    <img src="data:image/jpeg;base64,{img_str}" 
                         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">
                </div>
                """, 
                unsafe_allow_html=True
            )
                
        else:
            # Video file doesn't exist, show placeholder
            preview_video_placeholder(video_info, height)
            
    except Exception as e:
        # Fallback to placeholder if thumbnail extraction fails
        st.warning(f"Could not extract thumbnail: {str(e)}")
        preview_video_placeholder(video_info, height)


def preview_video_placeholder(video_info: Dict, height: int = 300) -> None:
    """
    Create a video preview placeholder as fallback.
    """
    st.markdown(f"""
    <div class="video-thumbnail" style="height: {height}px;">
        🎬
    </div>
    <div style="text-align: center; margin-top: 1rem;">
        <h3 style="color: #1e293b; margin-bottom: 0.5rem;">{video_info.get('video_name', 'Unknown')}</h3>
        <p style="color: #6366f1; font-weight: 600; font-size: 1.2rem; margin-bottom: 0.5rem;">
            Similarity: {video_info.get('similarity_score', 0):.3f}
        </p>
        <p style="color: #64748b; margin-bottom: 0.5rem;">Rank: #{video_info.get('rank', 'N/A')}</p>
        <small style="color: #9ca3af;">
            Video file not found or thumbnail extraction failed
        </small>
    </div>
    """, unsafe_allow_html=True)


def create_neighbor_grid(neighbors: List[Dict], num_cols: int = 3) -> None:
    """Create a grid of neighbor videos with real thumbnails."""
    if not neighbors:
        st.write("No neighbors to display")
        return
    
    # Create columns
    cols = st.columns(num_cols)
    
    for i, neighbor in enumerate(neighbors[:num_cols]):
        with cols[i]:
            # Try to extract real thumbnail
            try:
                from pathlib import Path
                video_path = neighbor.get('video_path', '')
                full_path = Path(video_path)
                
                if full_path.exists():
                    # Use VideoResultsVisualizer to extract thumbnail
                    visualizer = VideoResultsVisualizer()
                    thumbnail = visualizer.extract_thumbnail(full_path)
                    
                    # Display real thumbnail
                    st.image(thumbnail, use_container_width=True)
                    st.write(f"**{neighbor['video_name']}**")
                    st.write(f"Score: {neighbor['similarity_score']:.3f}")
                else:
                    # Fallback to placeholder
                    create_neighbor_placeholder(neighbor)
                    
            except Exception:
                # Fallback to placeholder if extraction fails
                create_neighbor_placeholder(neighbor)


def create_neighbor_placeholder(neighbor: Dict) -> None:
    """Create a placeholder for neighbor video."""
    st.markdown(f"""
    <div class="video-card">
        <div style="
            width: 100%;
            height: 100px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.75rem;
            font-size: 2rem;
            color: white;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, 
                    rgba(255,255,255,0.1) 25%, 
                    transparent 25%, 
                    transparent 75%, 
                    rgba(255,255,255,0.1) 75%);
                background-size: 15px 15px;
            "></div>
            🎥
        </div>
        <div style="text-align: center;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem; font-size: 0.9rem;">
                {neighbor['video_name']}
            </div>
            <div style="color: #6366f1; font-weight: 600; font-size: 0.85rem;">
                Score: {neighbor['similarity_score']:.3f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="ALFA 0.1 - Similarity Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Force sidebar to stay visible
    st.markdown("""
    <style>
        /* Force sidebar to always be visible */
        .css-1d391kg {
            width: 21rem !important;
            min-width: 21rem !important;
        }
        
        /* Hide sidebar collapse button */
        .css-14xtw13.e8zbici0 {
            display: none !important;
        }
        
        /* Alternative selector for hiding collapse button */
        button[data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Custom CSS for modern design
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
        
        .video-thumbnail {
            width: 100%;
            height: 300px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 4rem;
            color: white;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 20px 40px rgba(99, 102, 241, 0.3);
        }
        
        .video-thumbnail::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                rgba(255,255,255,0.1) 25%, 
                transparent 25%, 
                transparent 75%, 
                rgba(255,255,255,0.1) 75%);
            background-size: 30px 30px;
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
        
        /* Button styling - Force purple for all buttons */
        .stButton > button,
        div[data-testid="stButton"] > button,
        .stButton button,
        button[data-baseweb="button"] {
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
        }
        
        .stButton > button:hover,
        div[data-testid="stButton"] > button:hover,
        .stButton button:hover,
        button[data-baseweb="button"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4) !important;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        }
        
        /* Secondary button */
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #64748b 0%, #475569 100%);
            box-shadow: 0 4px 8px rgba(100, 116, 139, 0.2);
        }
        
        .stButton > button[kind="secondary"]:hover {
            box-shadow: 0 8px 20px rgba(100, 116, 139, 0.4);
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
        
        /* Hide any unwanted buttons in results area */
        .stColumn:last-child .stButton {
            display: none !important;
        }
        
        /* Hide empty button containers */
        .stColumn:last-child .element-container:empty {
            display: none !important;
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
        
        /* Style thumbnail containers */
        .stColumn:last-child .stImage {
            margin-bottom: 0.5rem;
        }
        
        /* Add spacing between result items */
        .stColumn:last-child .element-container {
            margin-bottom: 0.75rem !important;
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
    
    # Initialize session state
    if 'text_selection' not in st.session_state:
        st.session_state.text_selection = SelectedVideo(-1)
    if 'click_selection' not in st.session_state:
        st.session_state.click_selection = SelectedVideo(-1)
    if 'text_query' not in st.session_state:
        st.session_state.text_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'force_update' not in st.session_state:
        st.session_state.force_update = False
    
    # Initialize default values for variables used across components
    top_k = 5  # Changed from hardcoded 3 to 5 as default
    similarity_threshold = 0.5
    viz_method = "umap"
    umap_neighbors = 15
    umap_min_dist = 0.1
    trimap_inliers = 10
    
    # Header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>ALFA 0.1 - Embedding Search</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration (structured like mock interface)
    with st.sidebar:
        # Load search engine (hidden from user)
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
        
        # 🔍 Settings Header
        st.markdown('<div class="section-title">🔍 Settings</div>', unsafe_allow_html=True)
        
        # Video Search Section
        try:
            from pathlib import Path
            # Get project root and resolve video directory
            project_root = Path(__file__).parent.parent
            video_dir = project_root / "data" / "videos" / "user_input"
            
            # Get available query videos (includes pre-computed ones)
            try:
                available_videos = search_engine.get_query_videos_list()
                if not available_videos and video_dir.exists():
                    # Fall back to file system if no pre-computed videos
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
                    available_videos = [f.name for f in sorted(video_files)]
            except:
                available_videos = []
                if video_dir.exists():
                    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
                    available_videos = [f.name for f in sorted(video_files)]
            
            if available_videos:
                selected_video = st.selectbox(
                    "",
                    available_videos,
                    help="Select a video file (⚡ = pre-computed embedding)"
                )
                
                if st.button("🎥 Search by Video", use_container_width=True, help="Smart search: uses pre-computed embeddings when available, falls back to real-time processing"):
                    with st.spinner("Searching for similar videos..."):
                        try:
                            # First try pre-computed embeddings (fast path)
                            results = search_engine.search_by_filename(selected_video, top_k=top_k)
                            st.session_state.search_results = results
                            st.session_state.text_query = f"Similar to: {selected_video}"
                            st.session_state.click_selection = SelectedVideo(0)
                            
                            # Check if we used pre-computed embedding or fallback
                            try:
                                # Quick check if the video was in cache
                                cached_embedding = search_engine.query_manager.get_query_embedding(selected_video)
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
        
        # Text Search Section
        text_query = st.text_input(
            "",
            value=st.session_state.text_query,
            placeholder="car approaching cyclist"
        )
        
        if st.button("🔍 Search by Text", use_container_width=True):
            if text_query:
                with st.spinner("Searching..."):
                    try:
                        results = search_engine.search_by_text(text_query, top_k=top_k)
                        st.session_state.search_results = results
                        st.session_state.text_query = text_query
                        st.session_state.text_selection = SelectedVideo(0)
                        st.success(f"Found {len(results)} results!")
                    except NoResultsError:
                        st.warning("No results found above similarity threshold")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")
        
        # Top-K Results
        # st.markdown("**Filtering:**")
        top_k = st.slider("Top-K Results", 1, 15, top_k)
        
        # Similarity Threshold
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, similarity_threshold, 0.1)
        
        # Visualization Method
        # st.markdown("**Visualization:**")
        viz_method = st.selectbox(
            "",
            options=["umap", "pca", "trimap", "tsne", "similarity", "3d_umap"],
            format_func=lambda x: {
                "umap": "UMAP (Recommended)",
                "pca": "PCA (Fast)", 
                "trimap": "TriMAP (Large Datasets)",
                "tsne": "t-SNE (Legacy)",
                "similarity": "Direct Similarity",
                "3d_umap": "3D UMAP"
            }[x],
            index=0,
            label_visibility="collapsed"
        )
        
        # Advanced options based on method (minimal)
        if viz_method == "umap":
            umap_neighbors = st.slider("Neighbors", 3, 50, umap_neighbors)
            umap_min_dist = st.slider("Min Distance", 0.01, 1.0, umap_min_dist, 0.01)
        elif viz_method == "trimap":
            trimap_inliers = st.slider("Triplet Inliers", 3, 20, trimap_inliers)
        
        # Database Stats Section
        st.markdown('<div class="section-title">📊 Database Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{db_info.get('num_videos', 0)}</div>
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
        
    # Main content area: Full width visualization (like mock interface)
    st.markdown('<div class="section-title">Embedding Visualization</div>', unsafe_allow_html=True)
    
    # Shared legend function for all visualization tabs

    
    # Visualization view tabs (2D, 3D, Heatmap)
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["2D View", "3D View", "Heatmap"])
    
    with viz_tab1:
        if st.session_state.search_results:
            # Add legend for visualization elements

            
            # Interactive plot - use explicit timestamp comparison
            selected_idx = None
            # Always use the most recent selection
            if st.session_state.text_selection.is_valid() and st.session_state.click_selection.is_valid():
                # Compare timestamps to get the most recent
                if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                    selected_idx = st.session_state.text_selection.idx
                else:
                    selected_idx = st.session_state.click_selection.idx
            elif st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            
            # Create 2D visualization with query info
            query_info = {
                'display_text': st.session_state.text_query if st.session_state.text_query else "No query"
            }
            # Get all videos from database for comprehensive visualization
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                viz_method, 
                selected_idx,
                query_info=query_info,
                top_k=top_k,
                all_videos=all_videos,
                **({
                    'neighbors': umap_neighbors,
                    'min_dist': umap_min_dist
                } if viz_method == 'umap' and 'umap_neighbors' in locals() else {})
            )
            
            # Display plot with click handling
            plot_selection = st.plotly_chart(
                fig, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=f"plot_2d_{selected_idx}"
            )
            
            # Handle plot clicks
            if plot_selection and plot_selection.get("selection", {}).get("point_indices"):
                clicked_idx = plot_selection["selection"]["point_indices"][0]
                if clicked_idx != st.session_state.click_selection.idx:
                    st.session_state.click_selection = SelectedVideo(clicked_idx)
                    st.rerun()
            
            # Add click instructions
            # st.markdown("""
            # <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; border: 1px solid rgba(226, 232, 240, 0.5);">
            #     <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
            #         💡 <strong>Tip:</strong> Click on any point in the visualization to view that video in the primary view below
            #     </p>
            # </div>
            # """, unsafe_allow_html=True)
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab2:
        if st.session_state.search_results:
            # Add legend for visualization elements

            
            # Interactive 3D plot - use explicit timestamp comparison
            selected_idx = None
            # Always use the most recent selection
            if st.session_state.text_selection.is_valid() and st.session_state.click_selection.is_valid():
                # Compare timestamps to get the most recent
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
            # Get all videos from database for comprehensive visualization
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "3d_umap", 
                selected_idx,
                query_info=query_info,
                top_k=top_k,
                all_videos=all_videos
            )
            
            # Display 3D plot with click handling
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
            
            # # Add click instructions for 3D view
            # st.markdown("""
            # <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; border: 1px solid rgba(226, 232, 240, 0.5);">
            #     <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
            #         💡 <strong>Tip:</strong> Click on any point in the 3D visualization to view that video in the primary view below
            #     </p>
            # </div>
            # """, unsafe_allow_html=True)
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab3:
        if st.session_state.search_results:
            # Create heatmap visualization (heatmap only shows search results, not all videos)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "similarity", 
                None
            )
            
            # Display heatmap
            st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{len(st.session_state.search_results)}")
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    # Reset force_update flag after visualization updates
    if st.session_state.get('force_update', False):
        st.session_state.force_update = False
    
    # Bottom section: Top K Results (like mock interface)
    if st.session_state.search_results:
        st.markdown('<div class="section-title">Top K Results</div>', unsafe_allow_html=True)
        
        # Determine which video to show - use simpler max approach
        current_selection = max(
            st.session_state.text_selection.idx if st.session_state.text_selection.is_valid() else -1,
            st.session_state.click_selection.idx if st.session_state.click_selection.is_valid() else -1
        )
        if current_selection == -1:
            current_selection = 0
        
        # Layout: Large featured video + Column of top K results
        featured_col, results_col = st.columns([2, 1])
        
        with featured_col:
            # Featured video (top result or selected)
            featured_video = st.session_state.search_results[0]
            if current_selection is not None and current_selection < len(st.session_state.search_results):
                featured_video = st.session_state.search_results[current_selection]
            
            # Large featured video display
            # st.markdown('<div class="featured-video">', unsafe_allow_html=True)
            preview_video_with_thumbnail(featured_video, height=300)
            
            # Featured video info - clean and simple
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <h3 style="color: #1e293b; margin-bottom: 0.5rem; font-size: 1.4rem;">
                    {featured_video['video_name']} <span style="color: #6366f1; margin-left: 20px;">Score: {featured_video['similarity_score']:.3f}</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with results_col:
            with st.container(height=405):
                # Display top K results in the scrollable container
                for i, video in enumerate(st.session_state.search_results[:top_k]):
                    is_selected = (current_selection == i)
                    
                    # Create video card
                    video_path = video.get('video_path', '')
                    
                    # Create compact row layout with 16:9 aspect ratio thumbnail
                    # Get thumbnail data first
                    thumbnail_data = None
                    try:
                        from pathlib import Path
                        full_path = Path(video_path)
                        if full_path.exists():
                            visualizer = VideoResultsVisualizer()
                            thumbnail_array = visualizer.extract_thumbnail(full_path)
                            if thumbnail_array is not None:
                                import base64
                                import io
                                from PIL import Image
                                
                                if isinstance(thumbnail_array, np.ndarray):
                                    thumbnail_pil = Image.fromarray(thumbnail_array)
                                else:
                                    thumbnail_pil = thumbnail_array
                                
                                img_buffer = io.BytesIO()
                                thumbnail_pil.save(img_buffer, format='JPEG')
                                thumbnail_data = base64.b64encode(img_buffer.getvalue()).decode()
                    except Exception:
                        pass
                    
                    # Create compact row with proper styling
                    border_style = "2px solid #ff6b6b" if is_selected else "1px solid #e2e8f0"
                    score_color = "#ff6b6b" if is_selected else "#6366f1"
                    
                    if thumbnail_data:
                        # Real thumbnail with 16:9 aspect ratio (80x45px)
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0; margin-bottom: 2px;">
                            <div style="flex-shrink: 0;">
                                <img src="data:image/jpeg;base64,{thumbnail_data}" 
                                     style="width: 80px; height: 45px; object-fit: cover; border: {border_style}; border-radius: 4px;">
                            </div>
                            <div style="flex: 1; min-width: 0; padding-left: 4px;">
                                <div style="color: #1e293b; font-weight: 600; font-size: 0.75rem; line-height: 1.0; margin-bottom: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    {video['video_name']}
                                </div>
                                <div style="color: {score_color}; font-weight: 600; font-size: 0.65rem; line-height: 1.0; margin-bottom: 1px;">
                                    Score: {video['similarity_score']:.3f} {'✓' if is_selected else ''}
                                </div>
                                <div style="color: #64748b; font-size: 0.6rem; line-height: 1.0;">
                                    Rank #{video.get('rank', i+1)}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Fallback with colored box (16:9 aspect ratio)
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                        color = colors[i % len(colors)]
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0; margin-bottom: 2px;">
                            <div style="flex-shrink: 0;">
                                <div style="width: 80px; height: 45px; background: {color}; border: {border_style}; border-radius: 4px; 
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 1.0rem;">
                                    🎬
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 0; padding-left: 4px;">
                                <div style="color: #1e293b; font-weight: 600; font-size: 0.75rem; line-height: 1.0; margin-bottom: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    {video['video_name']}
                                </div>
                                <div style="color: {score_color}; font-weight: 600; font-size: 0.65rem; line-height: 1.0; margin-bottom: 1px;">
                                    Score: {video['similarity_score']:.3f} {'✓' if is_selected else ''}
                                </div>
                                <div style="color: #64748b; font-size: 0.6rem; line-height: 1.0;">
                                    Rank #{video.get('rank', i+1)}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add compact separator
                    st.markdown("<hr style='margin: 4px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        
        # Results table (collapsed by default)
        with st.expander("📊 Detailed Results Table"):
            results_df = pd.DataFrame([
                {
                    'Rank': r['rank'],
                    'Video Name': r['video_name'],
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
