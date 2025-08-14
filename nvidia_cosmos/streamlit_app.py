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
from video_search_optimized import OptimizedVideoSearchEngine
from video_visualizer import VideoResultsVisualizer
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


@st.cache_data
def load_search_engine() -> OptimizedVideoSearchEngine:
    """Load and cache the search engine (following official caching pattern)."""
    config = VideoRetrievalConfig()
    return OptimizedVideoSearchEngine(config=config)


@st.cache_data
def load_database_info(engine: OptimizedVideoSearchEngine) -> Dict:
    """Load and cache database information."""
    return engine.get_statistics()


def create_embedding_visualization(results: List[Dict], viz_method: str = "umap", selected_idx: Optional[int] = None, **kwargs) -> go.Figure:
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
    
    if viz_method == "umap":
        # Mock UMAP projection with better clustering
        df['x'] = np.random.randn(len(df)) * 3 + df['similarity'] * 5
        df['y'] = np.random.randn(len(df)) * 3 + df['similarity'] * 3
        title = "2D Embedding Space - UMAP"
        x_title, y_title = "UMAP Dimension 1", "UMAP Dimension 2"
    elif viz_method == "pca":
        # Mock PCA projection (more spread out)
        df['x'] = np.random.randn(len(df)) * 2 + df['similarity'] * 4
        df['y'] = np.random.randn(len(df)) * 2 + df['similarity'] * 2
        title = "2D Embedding Space - PCA"
        x_title, y_title = "PCA Dimension 1", "PCA Dimension 2"
    elif viz_method == "trimap":
        # Mock TriMAP projection
        df['x'] = np.random.randn(len(df)) * 4 + df['similarity'] * 3
        df['y'] = np.random.randn(len(df)) * 4 + df['similarity'] * 4
        title = "2D Embedding Space - TriMAP"
        x_title, y_title = "TriMAP Dimension 1", "TriMAP Dimension 2"
    elif viz_method == "tsne":
        # Mock t-SNE projection
        df['x'] = np.random.randn(len(df)) * 2.5 + df['similarity'] * 2
        df['y'] = np.random.randn(len(df)) * 2.5 + df['similarity'] * 3
        title = "2D Embedding Space - t-SNE"
        x_title, y_title = "t-SNE Dimension 1", "t-SNE Dimension 2"
    elif viz_method == "3d_umap":
        # Mock 3D UMAP
        df['x'] = np.random.randn(len(df)) * 3 + df['similarity'] * 4
        df['y'] = np.random.randn(len(df)) * 3 + df['similarity'] * 3
        df['z'] = np.random.randn(len(df)) * 2 + df['similarity'] * 2
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
        
        # Add selected point highlight for 3D
        if selected_idx is not None and selected_idx < len(df):
            selected_row = df.iloc[selected_idx]
            fig.add_trace(
                go.Scatter3d(
                    x=[selected_row['x']],
                    y=[selected_row['y']],
                    z=[selected_row['z']],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='cross'),
                    name='Selected',
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
        
        # Add selected point highlight for 2D
        if selected_idx is not None and selected_idx < len(df):
            selected_row = df.iloc[selected_idx]
            fig.add_trace(
                go.Scatter(
                    x=[selected_row['x']],
                    y=[selected_row['y']],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star',
                        line=dict(width=3, color='darkred')
                    ),
                    name='Selected',
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
        title_x=0.5,
        dragmode="zoom",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_similarity_plot(results: List[Dict], selected_idx: Optional[int] = None) -> go.Figure:
    """Legacy function for backwards compatibility."""
    return create_embedding_visualization(results, "umap", selected_idx)


def preview_video_placeholder(video_info: Dict, height: int = 300) -> None:
    """
    Create a video preview placeholder (since we can't embed actual videos easily).
    In a real implementation, this would show actual video content.
    """
    st.markdown(f"""
    <div style="
        border: 2px solid #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        height: {height}px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background: linear-gradient(45deg, #f0f0f0, #e0e0e0);
    ">
        <h3>🎬 Video Preview</h3>
        <p><strong>{video_info.get('video_name', 'Unknown')}</strong></p>
        <p>Similarity: {video_info.get('similarity_score', 0):.3f}</p>
        <p>Rank: #{video_info.get('rank', 'N/A')}</p>
        <small style="color: #666;">
            In a real implementation, this would show the actual video player
        </small>
    </div>
    """, unsafe_allow_html=True)


def create_neighbor_grid(neighbors: List[Dict], num_cols: int = 5) -> None:
    """Create a grid of neighbor videos."""
    if not neighbors:
        st.write("No neighbors to display")
        return
    
    # Create columns
    cols = st.columns(num_cols)
    
    for i, neighbor in enumerate(neighbors[:num_cols]):
        with cols[i]:
            # Create mini preview
            st.markdown(f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                margin: 5px 0;
                background: #f9f9f9;
            ">
                <div style="
                    width: 100%;
                    height: 100px;
                    background: linear-gradient(45deg, #e0e0e0, #d0d0d0);
                    border-radius: 3px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 10px;
                ">
                    🎥
                </div>
                <small><strong>{neighbor['video_name']}</strong></small><br>
                <small>Score: {neighbor['similarity_score']:.3f}</small>
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
    
    # Initialize session state
    if 'text_selection' not in st.session_state:
        st.session_state.text_selection = SelectedVideo(-1)
    if 'click_selection' not in st.session_state:
        st.session_state.click_selection = SelectedVideo(-1)
    if 'text_query' not in st.session_state:
        st.session_state.text_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Header
    st.title("ALFA 0.1 - Similarity Search")
    st.markdown("*Advanced embedding visualization and video similarity search*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔍 Settings")
        
        # Load search engine
        with st.spinner("Loading search engine..."):
            try:
                search_engine = load_search_engine()
                db_info = load_database_info(search_engine)
                st.success("✅ Search engine loaded!")
            except Exception as e:
                st.error(f"❌ Failed to load search engine: {e}")
                st.stop()
        
        # Search configuration
        st.subheader("🎯 Search Settings")
        top_k = st.slider("Top-K Results", 1, 15, 5)
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.1)
        
        # Visualization method
        st.subheader("📊 Visualization Method")
        viz_method = st.selectbox(
            "Select method:",
            options=["umap", "pca", "trimap", "tsne", "similarity", "3d_umap"],
            format_func=lambda x: {
                "umap": "🌟 UMAP (Recommended)",
                "pca": "🚀 PCA (Fast)", 
                "trimap": "🔥 TriMAP (Large Datasets)",
                "tsne": "⚠️ t-SNE (Legacy)",
                "similarity": "📊 Direct Similarity",
                "3d_umap": "🎯 3D UMAP"
            }[x],
            index=0
        )
        
        # Advanced options based on method
        if viz_method == "umap":
            st.write("**UMAP Parameters:**")
            umap_neighbors = st.slider("Neighbors", 5, 50, 15)
            umap_min_dist = st.slider("Min Distance", 0.01, 1.0, 0.1, 0.01)
        elif viz_method == "trimap":
            st.write("**TriMAP Parameters:**")
            trimap_inliers = st.slider("Triplet Inliers", 5, 20, 10)
        
        # Database info
        st.subheader("📊 Database Stats")
        st.metric("Total Videos", db_info.get('num_videos', 25))
        st.metric("Categories", db_info.get('categories', 7))
        st.metric("Embedding Dim", db_info.get('embedding_dim', 768))
        st.metric("Backend", db_info.get('search_backend', 'Mock'))
        
        # Export options
        st.subheader("💾 Export")
        if st.button("Export Results"):
            if st.session_state.search_results:
                # Create export data
                export_data = {
                    'query': st.session_state.text_query,
                    'results': st.session_state.search_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Convert to JSON for download
                import json
                json_str = json.dumps(export_data, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No results to export")
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    # Left column: Embedding visualization
    with col1:
        st.header("📊 Embedding Visualization")
        
        # Search methods
        search_tab1, search_tab2 = st.tabs(["📝 Text Search", "🎥 Video Search"])
        
        with search_tab1:
            # Text search
            text_query = st.text_input(
                "Enter your search query:",
                value=st.session_state.text_query,
                placeholder="e.g., 'car approaching cyclist'"
            )
            
            if st.button("🔍 Search by Text", type="primary"):
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
        
        with search_tab2:
            # Video search
            uploaded_file = st.file_uploader(
                "Upload a video for similarity search",
                type=['mp4', 'avi', 'mov'],
                help="Upload a video to find similar videos in the database"
            )
            
            if uploaded_file and st.button("🎥 Search by Video"):
                # Save uploaded file temporarily
                temp_path = Path(f"/tmp/{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Processing video..."):
                    try:
                        results = search_engine.search_by_video(temp_path, top_k=top_k)
                        st.session_state.search_results = results
                        st.session_state.text_query = f"Video: {uploaded_file.name}"
                        st.session_state.click_selection = SelectedVideo(0)
                        st.success(f"Found {len(results)} similar videos!")
                    except Exception as e:
                        st.error(f"Video search failed: {e}")
                    finally:
                        # Clean up
                        if temp_path.exists():
                            temp_path.unlink()
        
        # Results visualization
        if st.session_state.search_results:
            st.subheader("📊 Search Results")
            
            # Interactive plot
            selected_idx = None
            if st.session_state.text_selection.is_valid():
                selected_idx = st.session_state.text_selection.idx
            elif st.session_state.click_selection.is_valid():
                selected_idx = st.session_state.click_selection.idx
            
            # Create visualization with selected method
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                viz_method, 
                selected_idx,
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
    
    # Right column: Top K Results  
    with col2:
        st.header("📊 Top K Results")
        
        # Determine which video to show
        current_selection = None
        if st.session_state.text_selection.is_valid():
            if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                current_selection = st.session_state.text_selection.idx
        if current_selection is None and st.session_state.click_selection.is_valid():
            current_selection = st.session_state.click_selection.idx
        
        if st.session_state.search_results:
            # Featured video (top result)
            featured_video = st.session_state.search_results[0]
            
            # Use current selection or default to first result
            display_video = featured_video
            if current_selection is not None and current_selection < len(st.session_state.search_results):
                display_video = st.session_state.search_results[current_selection]
            
            # Featured video display
            st.subheader("🏆 Featured Video")
            
            # Main video preview
            preview_video_placeholder(display_video, height=250)
            
            # Featured video details
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Rank", display_video['rank'])
                st.metric("Similarity", f"{display_video['similarity_score']:.3f}")
            with col_b:
                st.write(f"**File:** {display_video['video_name']}")
                st.write(f"**Category:** {getattr(display_video, 'category', 'Unknown')}")
            
            # Top K results list
            st.subheader(f"🎯 Top {min(top_k, len(st.session_state.search_results))} Results")
            
            # Display top K results in a compact list
            for i, video in enumerate(st.session_state.search_results[:top_k]):
                is_selected = (current_selection == i)
                
                # Create a container for each video result
                with st.container():
                    cols = st.columns([1, 3, 1])
                    
                    with cols[0]:
                        # Video thumbnail placeholder
                        st.markdown(f"""
                        <div style="
                            width: 60px; 
                            height: 40px; 
                            background: linear-gradient(45deg, #6366f1, #8b5cf6); 
                            border-radius: 8px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            color: white; 
                            font-size: 16px;
                            {'border: 2px solid #3498db;' if is_selected else ''}
                        ">🎬</div>
                        """, unsafe_allow_html=True)
                    
                    with cols[1]:
                        # Video info
                        st.write(f"**{video['video_name']}**")
                        st.write(f"Score: {video['similarity_score']:.3f}")
                    
                    with cols[2]:
                        # Rank
                        st.write(f"#{video['rank']}")
                    
                    # Make it clickable (in real implementation)
                    if st.button(f"Select", key=f"select_{i}", help=f"Select {video['video_name']}"):
                        st.session_state.click_selection = SelectedVideo(i)
                        st.rerun()
                    
                    # Add separator
                    if i < min(top_k, len(st.session_state.search_results)) - 1:
                        st.divider()
        else:
            st.info("👆 Use the search interface to find videos, then click on a result to preview it here!")
    
    # Bottom section: Neighbor videos
    if st.session_state.search_results:
        st.header("🎬 Similar Videos")
        
        # Show top results as neighbors
        neighbors = st.session_state.search_results[:5]
        create_neighbor_grid(neighbors, num_cols=5)
        
        # Results table
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
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 Powered by **NVIDIA Cosmos-Embed1** | "
        "🔧 Built with **Streamlit** | "
        "⚡ Optimized with **FAISS**"
    )


if __name__ == "__main__":
    main()
