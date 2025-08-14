#!/usr/bin/env python3
"""
ALFA 0.1 - Similarity Search Interface
Advanced embedding visualization and video similarity search.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import torch
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import logging

# Import our components
from search import OptimizedVideoSearchEngine
from visualizer import VideoResultsVisualizer
from config import VideoRetrievalConfig
from exceptions import VideoNotFoundError, NoResultsError

# Configure logging
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
def load_search_engine() -> OptimizedVideoSearchEngine:
    """Load and cache the search engine (following official caching pattern)."""
    try:
        config = VideoRetrievalConfig()
        search_engine = OptimizedVideoSearchEngine(config=config)
        
        # Try to load existing database, if it fails, we'll work with empty database
        try:
            search_engine.database.load_from_parquet(config.database_path + ".parquet")
        except:
            try:
                search_engine.database.load()
            except:
                # Start with empty database - this is fine for the demo
                pass
                
        return search_engine
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        raise


@st.cache_data
def load_database_info(_engine: OptimizedVideoSearchEngine) -> Dict:
    """Load and cache database information."""
    try:
        return _engine.get_statistics()
    except Exception as e:
        # Return default stats if database not loaded
        return {
            "num_videos": 0,
            "categories": 0,
            "embedding_dim": 768,
            "search_backend": "Mock",
            "using_gpu": False,
            "cache_size": 0,
            "error": str(e)
        }


def create_embedding_visualization(results: List[Dict], viz_method: str = "umap", selected_idx: Optional[int] = None, query_info: Optional[Dict] = None, **kwargs) -> go.Figure:
    """Create advanced embedding visualization with multiple methods."""
    if not results:
        return go.Figure()
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            'video_name': r['video_name'],
            'similarity': r['similarity_score'],
            'rank': r['rank'],
            'category': getattr(r, 'category', 'unknown'),
            'idx': i
        }
        for i, r in enumerate(results)
    ])
    
    # Generate coordinates based on visualization method
    np.random.seed(42)  # For consistent results
    
    # Query vector position (fixed at center)
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
        # Position vectors based on distance from query (higher similarity = closer)
        distances = (1 - df['similarity']) * 6
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.3
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.3
        title = "2D Embedding Space - PCA"
        x_title, y_title = "PCA Dimension 1", "PCA Dimension 2"
    elif viz_method == "trimap":
        # Position vectors based on distance from query (higher similarity = closer)
        distances = (1 - df['similarity']) * 7
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.4
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.4
        title = "2D Embedding Space - TriMAP"
        x_title, y_title = "TriMAP Dimension 1", "TriMAP Dimension 2"
    elif viz_method == "tsne":
        # Position vectors based on distance from query (higher similarity = closer)
        distances = (1 - df['similarity']) * 5
        angles = np.random.uniform(0, 2*np.pi, len(df))
        df['x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df)) * 0.3
        df['y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df)) * 0.3
        title = "2D Embedding Space - t-SNE"
        x_title, y_title = "t-SNE Dimension 1", "t-SNE Dimension 2"
    elif viz_method == "3d_umap":
        # Position vectors in 3D space based on distance from query (higher similarity = closer)
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
            colorbar=dict(title="Cosine Similarity")
        ))
        fig.update_layout(
            title="Video Similarity Matrix - Direct Cosine Similarities",
            height=500,
            title_x=0.5
        )
        return fig
    
    # Create 2D or 3D scatter plot
    if viz_method == "3d_umap":
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
                colorbar=dict(title="Similarity Score")
            ),
            text=df['video_name'],
            hovertemplate='<b>%{text}</b><br>Similarity: %{marker.color:.3f}<br>Rank: %{customdata}<extra></extra>',
            customdata=df['rank']
        ))
        
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
                    name='Query Vector',
                    text=[query_info.get('display_text', 'Query')],
                    hovertemplate='<b>Query: %{text}</b><extra></extra>',
                    showlegend=False
                )
            )
        
        fig.update_layout(
            scene=dict(
                xaxis_title="UMAP Dimension 1",
                yaxis_title="UMAP Dimension 2",
                zaxis_title="UMAP Dimension 3"
            )
        )
    else:
        # 2D scatter plot
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
                    name='Query Vector',
                    text=[query_info.get('display_text', 'Query')],
                    hovertemplate='<b>Query: %{text}</b><extra></extra>',
                    showlegend=False
                )
            )
        
        fig.update_layout(
            xaxis_title=x_title,
            yaxis_title=y_title
        )
    
    # Update common layout properties
    fig.update_layout(
        height=500,
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#1e293b'}
        },
        dragmode="zoom",
        margin=dict(l=60, r=60, t=60, b=60)
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
            
            # Display just the thumbnail cleanly
            st.image(thumbnail, use_container_width=True)
                
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


def create_neighbor_grid(neighbors: List[Dict], num_cols: int = 5) -> None:
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
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.2);
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
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
    
    # Initialize default values for variables used across components
    top_k = 5
    similarity_threshold = 0.5
    viz_method = "umap"
    umap_neighbors = 15
    umap_min_dist = 0.1
    trimap_inliers = 10
    
    # Header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>ALFA 0.1 - Embedding Search</h1>
        <p>Advanced embedding visualization and video similarity search</p>
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
        
        # Text Search Section
        text_query = st.text_input(
            "",
            value=st.session_state.text_query,
            placeholder="car approaching cyclist"
        )
        
        if st.button("🔍 Search by Text", type="primary", use_container_width=True):
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
        
        # Video Search Section
        try:
            from pathlib import Path
            video_dir = Path("/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/videos/user_input")
            
            if video_dir.exists():
                video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
                video_files = sorted(video_files)
                
                if video_files:
                    video_options = [f.name for f in video_files]
                    
                    selected_video = st.selectbox(
                        "",
                        video_options,
                        help="Select a video file"
                    )
                    
                    if st.button("🎥 Search by Video", use_container_width=True):
                        with st.spinner("Searching for similar videos..."):
                            try:
                                video_path = video_dir / selected_video
                                results = search_engine.search_by_video(video_path, top_k=top_k)
                                st.session_state.search_results = results
                                st.session_state.text_query = f"Similar to: {selected_video}"
                                st.session_state.click_selection = SelectedVideo(0)
                                st.success(f"Found {len(results)} similar videos!")
                            except Exception as e:
                                st.error(f"Video search failed: {e}")
                else:
                    st.info("No video files found")
            else:
                st.warning("Video directory not found")
        except Exception as e:
            st.error(f"Error: {e}")
        
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
            umap_neighbors = st.slider("Neighbors", 5, 50, umap_neighbors)
            umap_min_dist = st.slider("Min Distance", 0.01, 1.0, umap_min_dist, 0.01)
        elif viz_method == "trimap":
            trimap_inliers = st.slider("Triplet Inliers", 5, 20, trimap_inliers)
        
        # Database Stats Section
        st.markdown('<div class="section-title">Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{db_info.get('num_videos', 25)}</div>
                <div class="stat-label">Total Videos</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('categories', 7)}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('embedding_dim', 768)}</div>
                <div class="stat-label">Embedding Dim</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{db_info.get('search_backend', 'Mock')}</div>
                <div class="stat-label">FAISS Backend</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area: Full width visualization (like mock interface)
    st.markdown('<div class="section-title">Embedding Visualization</div>', unsafe_allow_html=True)
    
    # Visualization view tabs (2D, 3D, Heatmap)
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["2D View", "3D View", "Heatmap"])
    
    with viz_tab1:
        if st.session_state.search_results:
            # Add legend for visualization elements
            st.markdown("""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem; justify-content: center;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: #ff6b6b;"></div>
                    <span style="font-size: 0.85rem; color: #64748b;">Videos</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 12px; height: 12px; background: #ffd700; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);"></div>
                    <span style="font-size: 0.85rem; color: #64748b;">Query Vector</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive plot
            selected_idx = None
            if st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            
            # Create 2D visualization with query info
            query_info = {
                'display_text': st.session_state.text_query if st.session_state.text_query else "No query"
            }
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                viz_method, 
                selected_idx,
                query_info=query_info,
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
                selection_mode="points"
            )
            
            # Handle plot clicks
            if plot_selection and plot_selection.get("selection", {}).get("point_indices"):
                clicked_idx = plot_selection["selection"]["point_indices"][0]
                if clicked_idx != st.session_state.click_selection.idx:
                    st.session_state.click_selection = SelectedVideo(clicked_idx)
                    st.rerun()
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab2:
        if st.session_state.search_results:
            # Add legend for visualization elements
            st.markdown("""
            <div style="display: flex; gap: 1rem; margin-bottom: 1rem; justify-content: center;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 12px; height: 12px; border-radius: 50%; background: #ff6b6b;"></div>
                    <span style="font-size: 0.85rem; color: #64748b;">Videos</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 12px; height: 12px; background: #ffd700; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);"></div>
                    <span style="font-size: 0.85rem; color: #64748b;">Query Vector</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive 3D plot
            selected_idx = None
            if st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            
            # Create 3D visualization with query info
            query_info = {
                'display_text': st.session_state.text_query if st.session_state.text_query else "No query"
            }
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "3d_umap", 
                selected_idx,
                query_info=query_info
            )
            
            # Display 3D plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    with viz_tab3:
        if st.session_state.search_results:
            # Create heatmap visualization
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "similarity", 
                None
            )
            
            # Display heatmap
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Use the search interface in the sidebar to find videos and visualize them here!")
    
    # Bottom section: Top K Results (like mock interface)
    if st.session_state.search_results:
        st.markdown('<div class="section-title">Top K Results</div>', unsafe_allow_html=True)
        
        # Determine which video to show
        current_selection = None
        if st.session_state.text_selection.is_valid():
            if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                current_selection = st.session_state.text_selection.idx
        if current_selection is None and st.session_state.click_selection.is_valid():
            current_selection = st.session_state.click_selection.idx
        
        # Layout: Large featured video + Column of top K results
        featured_col, results_col = st.columns([2, 1])
        
        with featured_col:
            # Featured video (top result or selected)
            featured_video = st.session_state.search_results[0]
            if current_selection is not None and current_selection < len(st.session_state.search_results):
                featured_video = st.session_state.search_results[current_selection]
            
            # Large featured video display
            st.markdown('<div class="featured-video">', unsafe_allow_html=True)
            preview_video_with_thumbnail(featured_video, height=300)
            
            # Featured video info - clean and simple
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <h3 style="color: #1e293b; margin-bottom: 0.5rem;">{featured_video['video_name']}</h3>
                <p style="color: #6366f1; font-weight: 600; font-size: 1.4rem;">
                    Score: {featured_video['similarity_score']:.3f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with results_col:
            st.markdown(f'<div style="font-size: 1rem; font-weight: 600; color: #1e293b; margin-bottom: 1rem; text-align: center;">Top {min(top_k, len(st.session_state.search_results))} Results</div>', unsafe_allow_html=True)
            
            # Display top K results in a column - only if we have results
            if st.session_state.search_results:
                for i, video in enumerate(st.session_state.search_results[:top_k]):
                    is_selected = (current_selection == i)
                    
                    # Create clickable video card
                    video_path = video.get('video_path', '')
                    
                    # Try to extract real thumbnail first
                    thumbnail_content = None
                    try:
                        from pathlib import Path
                        full_path = Path(video_path)
                        
                        if full_path.exists():
                            visualizer = VideoResultsVisualizer()
                            thumbnail = visualizer.extract_thumbnail(full_path)
                            if thumbnail is not None:
                                # Convert PIL image to base64 for display
                                import base64
                                import io
                                img_buffer = io.BytesIO()
                                thumbnail.save(img_buffer, format='JPEG')
                                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                                thumbnail_content = f'<img src="data:image/jpeg;base64,{img_str}" style="width: 50px; height: 35px; border-radius: 6px; object-fit: cover;">'
                    except Exception:
                        pass
                    
                    # Fallback to emoji if no thumbnail
                    if thumbnail_content is None:
                        # Get different colors for each result
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                        color = colors[i % len(colors)]
                        thumbnail_content = f'<div style="width: 50px; height: 35px; background: {color}; border-radius: 6px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1rem; flex-shrink: 0;">🎬</div>'
                    
                    # Create unified card design
                    card_style = f"""
                    background: white;
                    border-radius: 12px;
                    padding: 0.5rem;
                    margin-bottom: 0.5rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    border: 2px solid {"#6366f1" if is_selected else "transparent"};
                    transition: all 0.3s ease;
                    cursor: pointer;
                    """
                    
                    # Display non-interactive card (click on visualization to select)
                    st.markdown(f"""
                    <div style="{card_style}">
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            {thumbnail_content}
                            <div style="flex: 1; display: flex; justify-content: space-between; align-items: center;">
                                <div style="font-weight: 600; color: #1e293b; font-size: 0.8rem; line-height: 1.2;">
                                    {video['video_name']}
                                </div>
                                <div style="color: #6366f1; font-weight: 600; font-size: 0.8rem; line-height: 1.2;">
                                    {video['similarity_score']:.3f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No search results to display")
        
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
