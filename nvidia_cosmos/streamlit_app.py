#!/usr/bin/env python3
"""
Streamlit-based interactive search interface inspired by the official NVIDIA implementation.
Reference: https://huggingface.co/spaces/nvidia/Cosmos-Embed1/blob/main/src/streamlit_app.py
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


def create_similarity_plot(results: List[Dict], selected_idx: Optional[int] = None) -> go.Figure:
    """
    Create an interactive similarity plot inspired by the official implementation.
    """
    if not results:
        return go.Figure()
    
    # Create DataFrame for plotting
    df = pd.DataFrame([
        {
            'video_name': r['video_name'],
            'similarity': r['similarity_score'],
            'rank': r['rank'],
            'x': np.random.uniform(-5, 5),  # Mock x coordinate (would use t-SNE in real implementation)
            'y': np.random.uniform(-5, 5),  # Mock y coordinate
            'idx': i
        }
        for i, r in enumerate(results)
    ])
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x', y='y',
        size='similarity',
        color='similarity',
        hover_name='video_name',
        hover_data=['rank', 'similarity'],
        title="Video Similarity Space (Interactive)",
        color_continuous_scale="Viridis"
    )
    
    # Highlight selected point
    if selected_idx is not None and selected_idx < len(df):
        fig.add_trace(
            go.Scatter(
                x=[df.iloc[selected_idx]['x']],
                y=[df.iloc[selected_idx]['y']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='circle-open',
                    line=dict(width=3, color='red')
                ),
                name='Selected',
                showlegend=False
            )
        )
    
    # Update layout
    fig.update_layout(
        dragmode="zoom",
        xaxis_title="Embedding Dimension 1 (t-SNE)",
        yaxis_title="Embedding Dimension 2 (t-SNE)",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


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
        page_title="NVIDIA Cosmos Video Search",
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
    st.title("🔍 NVIDIA Cosmos Video Search")
    st.markdown("*Interactive video retrieval using Cosmos-Embed1*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load search engine
        with st.spinner("Loading search engine..."):
            try:
                search_engine = load_search_engine()
                db_info = load_database_info(search_engine)
                st.success("✅ Search engine loaded!")
            except Exception as e:
                st.error(f"❌ Failed to load search engine: {e}")
                st.stop()
        
        # Database info
        st.subheader("📊 Database Info")
        st.metric("Total Videos", db_info.get('num_videos', 0))
        st.metric("Embedding Dim", db_info.get('embedding_dim', 0))
        st.metric("Search Backend", db_info.get('search_backend', 'Unknown'))
        
        # Search configuration
        st.subheader("🔧 Search Settings")
        top_k = st.slider("Number of results", 1, 20, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.1)
        
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
    
    # Left column: Search and visualization
    with col1:
        st.header("🔍 Search Interface")
        
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
            
            if uploaded_file and st.button("🔍 Search by Video"):
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
            
            fig = create_similarity_plot(st.session_state.search_results, selected_idx)
            
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
    
    # Right column: Video preview and details
    with col2:
        st.header("🎬 Video Preview")
        
        # Determine which video to show
        current_selection = None
        if st.session_state.text_selection.is_valid():
            if st.session_state.text_selection.timestamp > st.session_state.click_selection.timestamp:
                current_selection = st.session_state.text_selection.idx
        if current_selection is None and st.session_state.click_selection.is_valid():
            current_selection = st.session_state.click_selection.idx
        
        if current_selection is not None and st.session_state.search_results:
            if current_selection < len(st.session_state.search_results):
                selected_video = st.session_state.search_results[current_selection]
                
                # Main video preview
                preview_video_placeholder(selected_video, height=300)
                
                # Video details
                st.subheader("📋 Details")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Rank", selected_video['rank'])
                    st.metric("Similarity", f"{selected_video['similarity_score']:.3f}")
                with col_b:
                    st.write(f"**File:** {selected_video['video_name']}")
                    if selected_video.get('metadata'):
                        metadata = selected_video['metadata']
                        st.write(f"**Added:** {metadata.get('added_at', 'Unknown')}")
                
                # Action buttons
                col_x, col_y = st.columns(2)
                with col_x:
                    if st.button("📥 Export Video Info"):
                        st.json(selected_video)
                with col_y:
                    if st.button("🔍 Find Similar"):
                        # Search for videos similar to this one
                        with st.spinner("Finding similar videos..."):
                            try:
                                similar = search_engine.search_by_video(
                                    selected_video['video_path'], 
                                    top_k=top_k
                                )
                                st.session_state.search_results = similar
                                st.session_state.text_query = f"Similar to: {selected_video['video_name']}"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to find similar videos: {e}")
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
