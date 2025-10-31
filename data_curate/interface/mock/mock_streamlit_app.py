#!/usr/bin/env python3
"""
Mock Streamlit interface that works without CUDA, videos, or heavy dependencies.
This shows the exact interface design without requiring any special hardware.

Run with: streamlit run mock_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random
import json
import time


# Mock data generation
@st.cache_data
def generate_mock_video_data():
    """Generate mock video database."""
    categories = ["car2cyclist", "car2pedestrian", "car2car", "car2motorcycle", 
                 "traffic_intersection", "highway_merge", "parking_lot"]
    
    videos = []
    for i in range(25):
        category = random.choice(categories)
        videos.append({
            "id": i,
            "video_name": f"{category}_{i:02d}.mp4",
            "category": category,
            "duration": round(random.uniform(1.5, 8.0), 1),
            "resolution": random.choice(["720p", "1080p", "480p"]),
            "similarity_to_query": random.uniform(0.3, 0.95),
            "x_coord": random.uniform(-5, 5),  # Mock t-SNE coordinates
            "y_coord": random.uniform(-5, 5),
            "file_size_mb": round(random.uniform(5, 45), 1),
            "added_date": f"2024-01-{random.randint(1, 30):02d}"
        })
    
    return pd.DataFrame(videos)


@st.cache_data
def mock_search(query_type, query_value, top_k=5):
    """Mock search function that simulates real search behavior."""
    df = generate_mock_video_data()
    
    if query_type == "text":
        # Simple keyword matching
        query_lower = query_value.lower()
        scores = []
        
        for _, video in df.iterrows():
            score = 0.1  # Base score
            
            # Check category match
            if any(word in video["category"] for word in query_lower.split()):
                score += 0.6
            
            # Check name match
            if any(word in video["video_name"].lower() for word in query_lower.split()):
                score += 0.3
            
            # Add randomness
            score += random.uniform(-0.1, 0.2)
            score = max(0.1, min(1.0, score))
            
            scores.append(score)
        
        df["similarity_score"] = scores
        
    else:  # video search
        # Simulate finding similar videos based on category
        query_category = random.choice(df["category"].unique())
        
        scores = []
        for _, video in df.iterrows():
            if video["category"] == query_category:
                score = random.uniform(0.7, 0.95)
            else:
                score = random.uniform(0.2, 0.6)
            scores.append(score)
        
        df["similarity_score"] = scores
    
    # Sort by similarity and return top-k
    df = df.sort_values("similarity_score", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def create_similarity_plot(df, selected_idx=None):
    """Create interactive similarity plot."""
    if df.empty:
        return go.Figure()
    
    fig = px.scatter(
        df,
        x='x_coord',
        y='y_coord',
        size='similarity_score',
        color='similarity_score',
        hover_name='video_name',
        hover_data={
            'category': True,
            'duration': True,
            'similarity_score': ':.3f',
            'x_coord': False,
            'y_coord': False
        },
        title="Video Similarity Space (Mock t-SNE Projection)",
        color_continuous_scale="Viridis",
        size_max=20
    )
    
    # Highlight selected point
    if selected_idx is not None and selected_idx < len(df):
        selected_video = df.iloc[selected_idx]
        fig.add_trace(
            go.Scatter(
                x=[selected_video['x_coord']],
                y=[selected_video['y_coord']],
                mode='markers',
                marker=dict(
                    size=25,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                name='Selected Video',
                showlegend=True
            )
        )
    
    fig.update_layout(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        height=400,
        showlegend=True
    )
    
    return fig


def create_video_thumbnail_mock(video_name, width=150, height=100):
    """Create a mock video thumbnail."""
    # Generate a colorful placeholder
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    color = random.choice(colors)
    
    st.markdown(f"""
    <div style="
        width: {width}px;
        height: {height}px;
        background: linear-gradient(45deg, {color}, {color}88);
        border: 2px solid #ddd;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 24px; margin-bottom: 5px;">🎬</div>
        <div style="font-size: 10px; text-align: center; padding: 0 5px; color: #333;">
            {video_name[:20]}{'...' if len(video_name) > 20 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Page config
    st.set_page_config(
        page_title="alpha 0.1 Mock Interface",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = pd.DataFrame()
    if 'selected_video_idx' not in st.session_state:
        st.session_state.selected_video_idx = None
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    # Header
    st.title("🔍 alpha 0.1 Video Search - Mock Interface")
    st.markdown("*Demonstration interface that works without CUDA or video files*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mock database stats
        st.subheader("📊 Mock Database")
        mock_df = generate_mock_video_data()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Videos", len(mock_df))
            st.metric("Categories", mock_df['category'].nunique())
        with col2:
            st.metric("Avg Duration", f"{mock_df['duration'].mean():.1f}s")
            st.metric("Total Size", f"{mock_df['file_size_mb'].sum():.0f}MB")
        
        # Search settings
        st.subheader("🔧 Search Settings")
        top_k = st.slider("Number of results", 1, 15, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.0, 0.1)
        
        # System info
        st.subheader("💻 System Info")
        st.info("✅ Mock Mode Active\n🚫 No CUDA Required\n📱 Pure Python Interface")
        
        # Search history
        if st.session_state.search_history:
            st.subheader("📝 Search History")
            for i, search in enumerate(st.session_state.search_history[-3:]):
                st.text(f"{i+1}. {search['query'][:20]}...")
    
    # Main content
    col1, col2 = st.columns([1.2, 0.8])
    
    # Left column: Search and visualization
    with col1:
        st.header("🔍 Search Interface")
        
        # Search tabs
        tab1, tab2 = st.tabs(["📝 Text Search", "🎥 Video Search"])
        
        with tab1:
            st.subheader("Text-to-Video Search")
            
            # Predefined queries for easy testing
            example_queries = [
                "car approaching cyclist",
                "pedestrian crossing street", 
                "traffic intersection",
                "highway merge",
                "parking lot maneuver"
            ]
            
            selected_example = st.selectbox("Try an example query:", 
                                          [""] + example_queries)
            
            text_query = st.text_input(
                "Enter your search query:",
                value=selected_example if selected_example else "",
                placeholder="e.g., 'car approaching cyclist'"
            )
            
            col_search, col_clear = st.columns([3, 1])
            
            with col_search:
                if st.button("🔍 Search by Text", type="primary", use_container_width=True):
                    if text_query:
                        with st.spinner("Searching..."):
                            time.sleep(0.5)  # Simulate processing
                            results = mock_search("text", text_query, top_k)
                            st.session_state.search_results = results
                            st.session_state.selected_video_idx = 0
                            
                            # Add to history
                            st.session_state.search_history.append({
                                "query": text_query,
                                "type": "text",
                                "timestamp": datetime.now().isoformat(),
                                "results_count": len(results)
                            })
                            
                        st.success(f"Found {len(results)} results!")
                    else:
                        st.warning("Please enter a search query")
            
            with col_clear:
                if st.button("🗑️ Clear"):
                    st.session_state.search_results = pd.DataFrame()
                    st.session_state.selected_video_idx = None
        
        with tab2:
            st.subheader("Video-to-Video Search")
            
            # Mock video upload
            st.info("📁 In the real interface, you would upload a video file here")
            
            # Simulate selecting a video from database
            if not mock_df.empty:
                selected_video = st.selectbox(
                    "Select a video from database to find similar ones:",
                    options=mock_df['video_name'].tolist(),
                    index=0
                )
                
                if st.button("🔍 Find Similar Videos", type="primary"):
                    with st.spinner("Processing video..."):
                        time.sleep(0.8)  # Simulate video processing
                        results = mock_search("video", selected_video, top_k)
                        st.session_state.search_results = results
                        st.session_state.selected_video_idx = 0
                        
                        # Add to history
                        st.session_state.search_history.append({
                            "query": f"Similar to: {selected_video}",
                            "type": "video",
                            "timestamp": datetime.now().isoformat(),
                            "results_count": len(results)
                        })
                        
                    st.success(f"Found {len(results)} similar videos!")
        
        # Results visualization
        if not st.session_state.search_results.empty:
            st.subheader("📊 Search Results Visualization")
            
            # Interactive plot
            fig = create_similarity_plot(
                st.session_state.search_results, 
                st.session_state.selected_video_idx
            )
            
            # Handle plot selection
            selected_points = st.plotly_chart(
                fig, 
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="similarity_plot"
            )
            
            # Handle point selection
            if selected_points and selected_points.get("selection", {}).get("point_indices"):
                point_idx = selected_points["selection"]["point_indices"][0]
                if point_idx != st.session_state.selected_video_idx:
                    st.session_state.selected_video_idx = point_idx
                    st.rerun()
    
    # Right column: Video preview and details
    with col2:
        st.header("🎬 Video Preview")
        
        if not st.session_state.search_results.empty and st.session_state.selected_video_idx is not None:
            if st.session_state.selected_video_idx < len(st.session_state.search_results):
                selected_video = st.session_state.search_results.iloc[st.session_state.selected_video_idx]
                
                # Video preview
                st.subheader(f"🎥 {selected_video['video_name']}")
                
                # Mock video player
                create_video_thumbnail_mock(selected_video['video_name'], width=250, height=180)
                
                # Video details
                st.subheader("📋 Details")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Similarity", f"{selected_video['similarity_score']:.3f}")
                    st.metric("Duration", f"{selected_video['duration']}s")
                with col_b:
                    st.metric("Resolution", selected_video['resolution'])
                    st.metric("Size", f"{selected_video['file_size_mb']}MB")
                
                # Category and metadata
                st.markdown(f"**Category:** {selected_video['category']}")
                st.markdown(f"**Added:** {selected_video['added_date']}")
                
                # Action buttons
                col_x, col_y = st.columns(2)
                with col_x:
                    if st.button("🔍 Find Similar", use_container_width=True):
                        # Search for videos similar to this one
                        with st.spinner("Finding similar videos..."):
                            time.sleep(0.5)
                            results = mock_search("video", selected_video['video_name'], top_k)
                            st.session_state.search_results = results
                            st.session_state.selected_video_idx = 0
                        st.success("Updated results!")
                        st.rerun()
                
                with col_y:
                    if st.button("📥 Export Info", use_container_width=True):
                        # Export video information
                        export_data = {
                            "video_info": selected_video.to_dict(),
                            "export_timestamp": datetime.now().isoformat()
                        }
                        
                        st.download_button(
                            label="📄 Download JSON",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"video_info_{selected_video['video_name']}.json",
                            mime="application/json",
                            use_container_width=True
                        )
        else:
            st.info("👆 Perform a search and click on a result to preview it here!")
    
    # Bottom section: Similar videos grid
    if not st.session_state.search_results.empty:
        st.header("🎬 Search Results Grid")
        
        # Show results in a grid
        results_df = st.session_state.search_results
        
        # Filter by threshold
        filtered_results = results_df[results_df['similarity_score'] >= similarity_threshold]
        
        if not filtered_results.empty:
            # Create grid layout
            cols = st.columns(min(5, len(filtered_results)))
            
            for i, (_, video) in enumerate(filtered_results.iterrows()):
                if i < len(cols):
                    with cols[i]:
                        # Create clickable video card
                        if st.button(
                            f"🎬 {video['video_name'][:15]}...\nScore: {video['similarity_score']:.3f}",
                            key=f"video_card_{i}",
                            use_container_width=True
                        ):
                            st.session_state.selected_video_idx = i
                            st.rerun()
                        
                        # Mini thumbnail
                        create_video_thumbnail_mock(video['video_name'], width=120, height=80)
            
            # Results table
            with st.expander("📊 Detailed Results Table"):
                display_df = filtered_results[['video_name', 'category', 'duration', 
                                              'similarity_score', 'resolution']].copy()
                display_df['similarity_score'] = display_df['similarity_score'].round(4)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.warning(f"No results above similarity threshold {similarity_threshold}")
    
    # Footer
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("🚀 **Powered by alpha 0.1-Embed1**")
    with col_footer2:
        st.markdown("🔧 **Built with Streamlit**")
    with col_footer3:
        st.markdown("⚡ **Mock Interface - No CUDA Required**")
    
    # Export session data
    if st.session_state.search_history:
        with st.sidebar:
            st.subheader("💾 Export Session")
            if st.button("Download Session Data"):
                session_data = {
                    "session_id": f"mock_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "search_history": st.session_state.search_history,
                    "current_results": st.session_state.search_results.to_dict('records') if not st.session_state.search_results.empty else [],
                    "export_timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📁 Download Session JSON",
                    data=json.dumps(session_data, indent=2),
                    file_name=f"session_data_{datetime.now().strftime('%H%M%S')}.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
