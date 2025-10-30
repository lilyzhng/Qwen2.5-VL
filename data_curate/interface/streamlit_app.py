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
import plotly.graph_objects as go
import base64
import io
from datetime import datetime
from typing import Optional, List, Dict
import logging
import traceback
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


def create_embedding_visualization(results: List[Dict], viz_method: str = "umap", selected_idx: Optional[int] = None, query_info: Optional[Dict] = None, top_k: Optional[int] = None, all_videos: Optional[List[Dict]] = None, **kwargs) -> go.Figure:
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
    
    # Limit results to top 10 for visualization
    top_10_results = results[:10] if len(results) >= 10 else results
    
    # Use all videos from database if provided, but prioritize top 10 search results
    if all_videos:
        # Create a set of top 10 search result slice_ids for quick lookup
        top_10_search_result_ids = {r['slice_id'] for r in top_10_results}
        
        # Create dataframe prioritizing top 10 search results
        all_video_data = []
        
        # First, add the top 10 search results
        for i, result in enumerate(top_10_results):
            all_video_data.append({
                'slice_id': result['slice_id'],
                'similarity': result['similarity_score'],
                'rank': result['rank'],
                'category': getattr(result, 'category', 'unknown'),
                'idx': i,
                'is_search_result': True
            })
        
        # Then add some background videos from database (limited to avoid clutter)
        background_videos = [v for v in all_videos if v['slice_id'] not in top_10_search_result_ids]
        # Limit background videos to 20 for performance
        background_videos = background_videos[:20]
        
        for i, video in enumerate(background_videos):
            all_video_data.append({
                'slice_id': video['slice_id'],
                'similarity': 0.1,  # Low similarity for background
                'rank': 1000 + i,
                'category': video.get('metadata', {}).get('category', 'unknown'),
                'idx': len(top_10_results) + i,
                'is_search_result': False
            })
        
        df = pd.DataFrame(all_video_data)
    else:
        # Just use top 10 search results if no database videos provided
        df = pd.DataFrame([
            {
                'slice_id': r['slice_id'],
                'similarity': r['similarity_score'],
                'rank': r['rank'],
                'category': getattr(r, 'category', 'unknown'),
                'idx': i,
                'is_search_result': True
            }
            for i, r in enumerate(top_10_results)
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
            # Position vectors based on distance from query (higher similarity = closer to center)
            # Use a more refined distance calculation for better positioning
            df_search = df[df['is_search_result'] == True] if 'is_search_result' in df.columns else df
            df_background = df[df['is_search_result'] == False] if 'is_search_result' in df.columns else pd.DataFrame()
            
            # For search results, use similarity-based positioning with better scaling
            if len(df_search) > 0:
                # Normalize similarities to create better distance distribution
                max_sim = df_search['similarity'].max()
                min_sim = df_search['similarity'].min()
                if max_sim > min_sim:
                    normalized_sim = (df_search['similarity'] - min_sim) / (max_sim - min_sim)
                else:
                    normalized_sim = df_search['similarity']
                
                # Convert to distance: higher similarity = smaller distance (closer to center)
                distances = (1 - normalized_sim) * 6 + 0.5  # Range from 0.5 to 6.5
                
                # Create concentric rings based on similarity ranking
                angles = np.random.uniform(0, 2*np.pi, len(df_search))
                
                # Add slight noise for better visual separation
                noise_factor = 0.3
                df.loc[df_search.index, 'x'] = query_x + distances * np.cos(angles) + np.random.randn(len(df_search)) * noise_factor
                df.loc[df_search.index, 'y'] = query_y + distances * np.sin(angles) + np.random.randn(len(df_search)) * noise_factor
            
            # For background videos, place them further away
            if len(df_background) > 0:
                bg_distances = np.random.uniform(8, 12, len(df_background))  # Further from center
                bg_angles = np.random.uniform(0, 2*np.pi, len(df_background))
                df.loc[df_background.index, 'x'] = query_x + bg_distances * np.cos(bg_angles) + np.random.randn(len(df_background)) * 0.8
                df.loc[df_background.index, 'y'] = query_y + bg_distances * np.sin(bg_angles) + np.random.randn(len(df_background)) * 0.8
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
        title = "2D Embedding Space"
        x_title, y_title = "UMAP Dimension 1", "UMAP Dimension 2"
    
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
            title_x=0.5,  # Center the title horizontally
            title_y=0.95,  # Position title near the top
            title_xanchor='center',  # Ensure title is centered on the x position
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
                text=df_other['slice_id'],
                hovertemplate='<b>%{text}</b><br>Database Video<extra></extra>',
                name='Videos',
                showlegend=True
            ))
            
        # Add search results (colorful)
        if len(df_search) > 0:
            # Use similarity-based coloring
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
            fig = go.Figure(data=go.Scatter(
                x=df['x'],
                y=df['y'],
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
            fig.update_layout(title=title)
            
            # Enhance hover template for better information display
            fig.update_traces(
                hovertemplate='<b>%{hovertext}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<extra></extra>',
                hovertext=df['slice_id'],
                customdata=df[['rank', 'similarity']].values
            )
        
        # Add green/red circle highlights for top 10 results (2D)
        if 'is_search_result' in df.columns:
            search_results = df[df['is_search_result'] == True]
            # Since we're already limiting to top 10, show all search results as highlighted
            top_10_points = search_results.head(10)
            
            if len(top_10_points) > 0:
                # Create colors array - red for selected, green for others
                circle_colors = []
                for i, (idx, row) in enumerate(top_10_points.iterrows()):
                    if selected_idx is not None and i == selected_idx:
                        circle_colors.append('#FF0000')  # Red for selected
                    else:
                        circle_colors.append('#00FF00')  # Green for others
                
                fig.add_trace(
                    go.Scatter(
                        x=top_10_points['x'],
                        y=top_10_points['y'],
                        mode='markers',
                        marker=dict(
                            size=top_10_points['similarity'] * 15 + 20,  # Slightly larger than base points
                            color='rgba(0,0,0,0)',  # Transparent fill
                            line=dict(width=3, color=circle_colors)  # Dynamic colors based on selection
                        ),
                        name=f'Top 10 Results',
                        showlegend=True,
                        hovertemplate='<b>%{text}</b><br>Rank: #%{customdata[0]}<br>Score: %{customdata[1]:.3f}<br>Top Result<extra></extra>',
                        text=top_10_points['slice_id'],
                        customdata=top_10_points[['rank', 'similarity']].values
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
    # First check if gif_path is directly available in video_info
    gif_path = video_info.get('gif_path', '')
    if gif_path and os.path.exists(gif_path):
        return gif_path
    
    # Check in metadata
    metadata = video_info.get('metadata', {})
    gif_path = metadata.get('gif_path', '')
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
        # Load unified embeddings parquet file to get gif_path path
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        unified_embeddings_path = project_root / "data" / "unified_embeddings.parquet"
        
        if unified_embeddings_path.exists():
            logger.info(f"Loading unified embeddings from: {unified_embeddings_path}")
            df = pd.read_parquet(unified_embeddings_path)
            
            # Look for the slice_id in the dataframe
            matching_rows = df[df['slice_id'] == slice_id]
            if not matching_rows.empty:
                gif_path = matching_rows.iloc[0].get('gif_path', '')
                if gif_path and os.path.exists(gif_path):
                    logger.info(f"Found GIF in unified embeddings: {gif_path}")
                    return gif_path
                elif gif_path:
                    logger.warning(f"GIF path exists in database but file not found: {gif_path}")
                else:
                    logger.warning(f"No gif_path entry for slice_id: {slice_id}")
            else:
                logger.warning(f"slice_id not found in unified embeddings: {slice_id}")
        else:
            logger.warning(f"Unified embeddings file not found: {unified_embeddings_path}")
        
        # Note: With unified embeddings, GIF paths should be in the main unified database
        # No need for separate query database fallback
        
        logger.warning(f"No GIF found for slice_id: {slice_id}")
        
    except Exception as e:
        logger.warning(f"Error getting input GIF path for {slice_id}: {e}")
        logger.warning(traceback.format_exc())
    
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
        st.session_state.similarity_threshold = 0.1  # Lower default for better text search compatibility
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    
    top_k = st.session_state.top_k
    similarity_threshold = st.session_state.similarity_threshold

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
                st.code(traceback.format_exc())
                st.stop()
        

        st.markdown('<div class="section-title">🔍 Settings</div>', unsafe_allow_html=True)
        
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
                        if st.session_state.similarity_threshold > 0.1:
                            st.info("💡 **Tip:** Text searches typically have lower similarity scores. Try reducing the similarity threshold to 0.05-0.15 for better results.")
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")

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

        st.session_state.top_k = st.slider("Top-K Results", 1, 10, st.session_state.top_k)

        st.session_state.similarity_threshold = st.slider(
            "Similarity Threshold", 
            0.0, 
            1.0, 
            st.session_state.similarity_threshold, 
            0.05,  # Smaller step size for finer control
            help="Lower values show more results. Text searches typically need lower thresholds (0.05-0.2) than video searches (0.2-0.5)."
        )

        st.markdown('<div class="section-title">📊 Database Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{db_info.get('num_inputs', 10000)}</div>
                <div class="stat-label">Database Videos</div>
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
    
    viz_tab1, viz_tab2 = st.tabs(["2D View", "Heatmap"])
    
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
                all_videos=all_videos
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
            # Create heatmap visualization
            all_videos = get_all_videos_from_database(search_engine)
            fig = create_embedding_visualization(
                st.session_state.search_results, 
                "similarity", 
                None,
                all_videos=all_videos
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
            <h3 style="color: #1e293b; font-size: 1.5rem; font-weight: 700; margin: 0;">Search Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a 2x3 grid layout
        total_results = len(st.session_state.search_results)
        
        # Row 1: Input + Top 2 results
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        
        # Row 2: Top 3-5 results
        row2_col1, row2_col2, row2_col3 = st.columns(3)
        
        all_columns = [row1_col1, row1_col2, row1_col3, row2_col1, row2_col2, row2_col3]
        
        # Position 0: Input query/video (top-left)
        with all_columns[0]:
            if st.session_state.input_video_info:
                if st.session_state.input_video_info['type'] == 'video':
                    # Display input video
                    input_slice_id = st.session_state.input_video_info.get('slice_id', '')
                    input_video_path = st.session_state.input_video_info.get('video_path', '')
                    
                    # Try to get GIF for input video first
                    input_gif_path = get_input_gif_path(input_slice_id, search_engine)
                    if input_gif_path:
                        try:
                            # Display GIF with overlay but preserve aspect ratio
                            st.markdown(f"""
                            <div style="position: relative; margin-bottom: 0.5rem;">
                                <img src="{input_gif_path}" 
                                     style="width: 100%; height: auto; border-radius: 8px; border: 3px solid #f59e0b;">
                                <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                     background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                                     font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                                    INPUT
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            logger.warning(f"Failed to display input GIF: {e}")
                            input_gif_path = None
                    
                    # If no GIF or GIF failed, fall back to thumbnail
                    if not input_gif_path:
                        input_thumbnail = st.session_state.input_video_info.get('thumbnail', '')
                        
                        if not input_thumbnail:
                            try:
                                input_thumbnail = search_engine.database.get_thumbnail_base64(input_slice_id)
                            except Exception as e:
                                logger.error(f"Failed to get input thumbnail: {e}")
                        
                        # Display input thumbnail or placeholder
                        if input_thumbnail:
                            try:
                                thumbnail_bytes = base64.b64decode(input_thumbnail)
                                thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                                
                                # Convert to base64 without resizing to preserve aspect ratio
                                img_buffer = io.BytesIO()
                                thumbnail_pil.save(img_buffer, format='JPEG', quality=95)
                                img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                                
                                st.markdown(f"""
                                <div style="position: relative; margin-bottom: 0.5rem;">
                                    <img src="data:image/jpeg;base64,{img_b64}" 
                                         style="width: 100%; height: auto; border-radius: 8px; border: 3px solid #f59e0b;">
                                    <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                         background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                                         font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                                        INPUT
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            except Exception as e:
                                logger.warning(f"Failed to display input thumbnail: {e}")
                                st.markdown("""
                                <div style="position: relative; margin-bottom: 0.5rem;">
                                    <div style="width: 100%; height: 200px; background: #f59e0b; border-radius: 8px; border: 3px solid #f59e0b;
                                         display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                        🎬
                                    </div>
                                    <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                         background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                                         font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                                        INPUT
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Placeholder for input video
                            st.markdown("""
                            <div style="position: relative; margin-bottom: 0.5rem;">
                                <div style="width: 100%; height: 200px; background: #f59e0b; border-radius: 8px; border: 3px solid #f59e0b;
                                     display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                                    🎬
                                </div>
                                <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                     background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                                     font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                                    INPUT
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Input video info
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 0.75rem;">
                        <div style="font-size: 0.85rem; color: #1e293b;">
                            <span style="font-weight: 600; color: #f59e0b;">{input_slice_id[:12]}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # Text query display
                    st.markdown(f"""
                    <div style="position: relative; margin-bottom: 0.5rem;">
                        <div style="width: 100%; aspect-ratio: 16/9; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                             border-radius: 8px; border: 3px solid #f59e0b; display: flex; align-items: center; justify-content: center; padding: 20px;">
                            <div style="color: white; text-align: center;">
                                <div style="font-size: 2rem; margin-bottom: 8px;">🔍</div>
                                <div style="font-size: 0.9rem; font-weight: 500; word-wrap: break-word; line-height: 1.3;">
                                    "{st.session_state.input_video_info['slice_id'][:50]}{'...' if len(st.session_state.input_video_info['slice_id']) > 50 else ''}"
                                </div>
                            </div>
                        </div>
                        <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                             background: rgba(245,158,11,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                             font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                            INPUT
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Text query info
                    st.markdown("""
                    <div style="text-align: center; margin-top: 0.75rem;">
                        <div style="font-size: 0.85rem; color: #1e293b;">
                            <span style="font-weight: 600; color: #f59e0b;">Text Query</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # No input available
                st.markdown("""
                <div style="position: relative; margin-bottom: 0.5rem;">
                    <div style="width: 100%; height: 200px; background: #9ca3af; border-radius: 8px; border: 3px solid #9ca3af;
                         display: flex; align-items: center; justify-content: center; color: white; font-size: 2rem;">
                        ❓
                    </div>
                    <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                         background: rgba(156,163,175,0.9); color: white; padding: 4px 8px; border-radius: 4px; 
                         font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px);">
                        NO INPUT
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="text-align: center; margin-top: 0.75rem;">
                    <div style="font-size: 0.85rem; color: #6b7280;">
                        <span style="font-weight: 600;">No Input</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
                # Positions 1-5: Top 5 retrieved results
        num_results_to_show = min(5, total_results)
        for i in range(num_results_to_show):
            video = st.session_state.search_results[i]
            is_selected = (current_selection == i)
            
            with all_columns[i + 1]:  # +1 because position 0 is input
                # Try to get GIF first, then fall back to thumbnail
                gif_path = get_gif_path_from_result(video)
                thumbnail_data = video.get('thumbnail', '')
                
                # Style variables
                border_style = "border: 3px solid #6366f1;" if is_selected else "border: 2px solid #e2e8f0;"
                overlay_bg = "rgba(99,102,241,0.9)" if is_selected else "rgba(0,0,0,0.7)"
                overlay_shadow = "box-shadow: 0 2px 8px rgba(99,102,241,0.4);" if is_selected else ""
                checkmark = " ✓" if is_selected else ""
                
                # Prioritize GIF over thumbnail
                if gif_path:
                    try:
                        # Display GIF with original aspect ratio
                        st.markdown(f"""
                        <div style="position: relative; margin-bottom: 0.5rem;">
                            <img src="{gif_path}" 
                                 style="width: 100%; height: auto; border-radius: 8px; {border_style}">
                            <div style="position: absolute; bottom: 8px; left: 50%; transform: translateX(-50%); 
                                 background: {overlay_bg}; color: white; padding: 4px 8px; border-radius: 4px; 
                                 font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); {overlay_shadow}">
                                Rank {i+1}{checkmark}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        logger.warning(f"Failed to display GIF for {video['slice_id']}: {e}")
                        gif_path = None  # Fall back to thumbnail
                
                # Fall back to thumbnail if no GIF or GIF failed
                if not gif_path and thumbnail_data:
                    try:
                        # Process image without resizing to preserve aspect ratio
                        thumbnail_bytes = base64.b64decode(thumbnail_data)
                        thumbnail_pil = Image.open(io.BytesIO(thumbnail_bytes))
                        
                        # Convert to base64 without resizing
                        img_buffer = io.BytesIO()
                        thumbnail_pil.save(img_buffer, format='JPEG', quality=95)
                        img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        # Display image with overlay and preserve aspect ratio
                        st.markdown(f"""
                        <div style="position: relative; margin-bottom: 0.5rem;">
                            <img src="data:image/jpeg;base64,{img_b64}" 
                                 style="width: 100%; height: auto; border-radius: 8px; {border_style}">
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
                        gif_path = None
                        thumbnail_data = None
                
                # If no GIF and no thumbnail, show colored placeholder
                if not gif_path and not thumbnail_data:
                    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                    color = colors[i % len(colors)]
                    st.markdown(f"""
                    <div style="position: relative; margin-bottom: 0.5rem;">
                        <div style="width: 100%; aspect-ratio: 16/9; background: {color}; border-radius: 8px; {border_style}
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
            <div style="text-align: center; margin-top: 1rem;">
                <small style="color: #64748b; font-style: italic;">
                    Showing input + top 5 of {total_results} results in 2x3 grid. Click points in the embedding visualization above to view other results.
                </small>
                </div>
                """, unsafe_allow_html=True)
        

        with st.expander("📊 Detailed Results Table"):
            config = VideoRetrievalConfig()
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
