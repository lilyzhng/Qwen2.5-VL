#!/usr/bin/env python3
"""
Streamlit App for Recall Evaluation Analysis and Triaging.

This app provides interactive visualization and analysis of recall evaluation results
to help identify why certain scenarios have low recall performance.
"""

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
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import base64
from io import BytesIO
from PIL import Image

from core.evaluate import (
    run_recall_evaluation,
    run_text_to_video_evaluation,
    run_video_to_video_evaluation,
    GroundTruthProcessor
)
from core.search import VideoSearchEngine
from core.config import VideoRetrievalConfig


# Page configuration
st.set_page_config(
    page_title="Recall Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-metric {
        border-left-color: #51cf66;
    }
    .warning-metric {
        border-left-color: #ffd43b;
    }
    .error-metric {
        border-left-color: #ff6b6b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_ground_truth_data():
    """Load and cache ground truth data."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if not annotation_path.exists():
        st.error(f"Annotation file not found: {annotation_path}")
        return None
    
    processor = GroundTruthProcessor(str(annotation_path))
    return processor


@st.cache_data
def load_evaluation_results():
    """Load and cache evaluation results."""
    try:
        with st.spinner("Running comprehensive recall evaluation..."):
            results = run_recall_evaluation()
        return results
    except Exception as e:
        st.error(f"Failed to load evaluation results: {e}")
        return None


@st.cache_data
def load_keyword_evaluation(keywords: List[str], mode: str):
    """Load keyword-specific evaluation results."""
    try:
        with st.spinner(f"Evaluating {mode} recall for keywords: {', '.join(keywords)}..."):
            if mode == "text":
                return run_text_to_video_evaluation(keywords=keywords)
            elif mode == "video":
                return run_video_to_video_evaluation(keywords=keywords)
            else:
                return {
                    "text": run_text_to_video_evaluation(keywords=keywords),
                    "video": run_video_to_video_evaluation(keywords=keywords)
                }
    except Exception as e:
        st.error(f"Failed to evaluate keywords: {e}")
        return None


def create_recall_metrics_chart(results: Dict[str, Any]):
    """Create a chart showing recall metrics comparison."""
    metrics_data = []
    
    # Video-to-video metrics
    if 'video_to_video' in results and 'average_recalls' in results['video_to_video']:
        v2v_recalls = results['video_to_video']['average_recalls']
        for metric, value in v2v_recalls.items():
            k_value = int(metric.split('@')[1])
            metrics_data.append({
                'Mode': 'Video-to-Video',
                'K': k_value,
                'Recall': value,
                'Metric': metric
            })
    
    # Text-to-video metrics
    if 'text_to_video' in results and 'average_recalls' in results['text_to_video']:
        t2v_recalls = results['text_to_video']['average_recalls']
        for metric, value in t2v_recalls.items():
            k_value = int(metric.split('@')[1])
            metrics_data.append({
                'Mode': 'Text-to-Video',
                'K': k_value,
                'Recall': value,
                'Metric': metric
            })
    
    if not metrics_data:
        return None
    
    df = pd.DataFrame(metrics_data)
    
    fig = px.bar(
        df, 
        x='K', 
        y='Recall', 
        color='Mode',
        title='Recall Performance Comparison',
        labels={'Recall': 'Recall Score', 'K': 'Top-K'},
        text='Recall'
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis=dict(range=[0, 1]),
        xaxis=dict(type='category'),
        height=400
    )
    
    return fig


def create_category_heatmap(results: Dict[str, Any]):
    """Create a heatmap of category-specific recall performance."""
    if 'category_specific' not in results:
        return None
    
    category_data = []
    
    for group_name, group_data in results['category_specific'].items():
        for category, cat_data in group_data.items():
            if 'average_recalls' in cat_data:
                recalls = cat_data['average_recalls']
                video_count = cat_data.get('video_count', 0)
                
                for metric, value in recalls.items():
                    k_value = int(metric.split('@')[1])
                    category_data.append({
                        'Group': group_name.title(),
                        'Category': category,
                        'K': k_value,
                        'Recall': value,
                        'Video Count': video_count
                    })
    
    if not category_data:
        return None
    
    df = pd.DataFrame(category_data)
    
    # Create pivot table for heatmap
    pivot_df = df.pivot_table(
        values='Recall', 
        index='Category', 
        columns='K', 
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot_df,
        title='Category-Specific Recall Heatmap',
        labels=dict(x="Top-K", y="Category", color="Recall"),
        aspect="auto",
        color_continuous_scale="RdYlGn"
    )
    
    fig.update_layout(height=600)
    
    return fig


def display_detailed_results(results: Dict[str, Any], result_type: str):
    """Display detailed results for analysis."""
    if result_type not in results:
        st.warning(f"No {result_type} results available")
        return
    
    data = results[result_type]
    
    if 'detailed_results' not in data:
        st.warning(f"No detailed results available for {result_type}")
        return
    
    detailed_results = data['detailed_results']
    
    # Convert to DataFrame for better display
    df_data = []
    for result in detailed_results:
        if result_type == 'video_to_video':
            query_id = result['query_video']
            query_info = f"Keywords: {', '.join(result['query_keywords'])}"
        else:  # text_to_video
            query_id = result['query_text']
            query_info = f"Relevant videos: {result['relevant_count']}"
        
        recalls = result['recalls']
        retrieved_ids = result.get('retrieved_ids', [])
        
        df_data.append({
            'Query': query_id,
            'Query Info': query_info,
            'Relevant Count': result['relevant_count'],
            'Recall@1': recalls.get(1, recalls.get('recall@1', 0)),
            'Recall@3': recalls.get(3, recalls.get('recall@3', 0)),
            'Recall@5': recalls.get(5, recalls.get('recall@5', 0)),
            'Top Retrieved': ', '.join(retrieved_ids[:3]) if retrieved_ids else 'None'
        })
    
    df = pd.DataFrame(df_data)
    
    # Sort by Recall@1 to identify problematic queries
    df = df.sort_values('Recall@1', ascending=True)
    
    st.subheader(f"📊 {result_type.replace('_', '-').title()} Detailed Results")
    
    # Show summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_r1 = df['Recall@1'].mean()
        st.metric("Avg Recall@1", f"{avg_r1:.3f}")
    
    with col2:
        avg_r3 = df['Recall@3'].mean()
        st.metric("Avg Recall@3", f"{avg_r3:.3f}")
    
    with col3:
        avg_r5 = df['Recall@5'].mean()
        st.metric("Avg Recall@5", f"{avg_r5:.3f}")
    
    with col4:
        zero_recall = (df['Recall@1'] == 0).sum()
        st.metric("Zero Recall@1", f"{zero_recall}/{len(df)}")
    
    # Show the detailed table
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )
    
    # Highlight problematic queries
    st.subheader("🚨 Low Recall Queries (Recall@1 = 0)")
    low_recall_df = df[df['Recall@1'] == 0]
    
    if len(low_recall_df) > 0:
        st.dataframe(low_recall_df, use_container_width=True)
        
        # Analysis suggestions
        st.subheader("💡 Analysis Suggestions")
        st.write("**Potential reasons for low recall:**")
        st.write("- Keywords may be too specific or rare")
        st.write("- Visual similarity might not match semantic similarity")
        st.write("- Embedding model may not capture the relevant features")
        st.write("- Ground truth annotations might need refinement")
        
    else:
        st.success("✅ All queries have non-zero Recall@1!")


def display_video_analysis(ground_truth: GroundTruthProcessor):
    """Display video-level analysis and triaging tools."""
    st.subheader("🎬 Video Analysis & Triaging")
    
    # Video selection
    all_videos = list(ground_truth.video_to_keywords.keys())
    selected_video = st.selectbox("Select a video to analyze:", all_videos)
    
    if selected_video:
        video_info = ground_truth.get_video_info(selected_video)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Video Information:**")
            st.write(f"**ID:** {selected_video}")
            st.write(f"**Keywords:** {', '.join(video_info['keywords'])}")
            st.write(f"**Path:** {video_info['video_path']}")
            
            # Show relevant videos
            relevant_videos = ground_truth.get_relevant_videos(selected_video, include_self=False)
            st.write(f"**Relevant Videos ({len(relevant_videos)}):**")
            for vid in list(relevant_videos)[:5]:  # Show first 5
                st.write(f"- {vid}")
            if len(relevant_videos) > 5:
                st.write(f"... and {len(relevant_videos) - 5} more")
        
        with col2:
            # Try to display GIF if available
            gif_path = Path(video_info['gif_path'])
            if gif_path.exists():
                st.write("**Video Preview:**")
                try:
                    st.image(str(gif_path), caption=selected_video, width=400)
                except Exception as e:
                    st.write(f"Could not display GIF: {e}")
            else:
                st.write("**Video Preview:** Not available")
        
        # Run search for this specific video
        if st.button(f"🔍 Search Similar to {selected_video}"):
            try:
                config = VideoRetrievalConfig()
                search_engine = VideoSearchEngine(config=config)
                
                with st.spinner("Searching for similar videos..."):
                    search_results = search_engine.search_by_video(
                        video_info['video_path'],
                        top_k=10,
                        use_cache=True
                    )
                
                st.write("**Search Results:**")
                results_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Video ID': result['slice_id'],
                        'Similarity': f"{result['similarity']:.3f}",
                        'Is Relevant': result['slice_id'] in relevant_videos
                    }
                    for i, result in enumerate(search_results)
                ])
                
                # Color code relevant vs irrelevant
                def highlight_relevant(row):
                    if row['Is Relevant']:
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    results_df.style.apply(highlight_relevant, axis=1),
                    use_container_width=True
                )
                
                # Calculate recall for this query
                retrieved_ids = [r['slice_id'] for r in search_results]
                recall_1 = len(set(retrieved_ids[:1]).intersection(relevant_videos)) / len(relevant_videos) if relevant_videos else 0
                recall_3 = len(set(retrieved_ids[:3]).intersection(relevant_videos)) / len(relevant_videos) if relevant_videos else 0
                recall_5 = len(set(retrieved_ids[:5]).intersection(relevant_videos)) / len(relevant_videos) if relevant_videos else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recall@1", f"{recall_1:.3f}")
                with col2:
                    st.metric("Recall@3", f"{recall_3:.3f}")
                with col3:
                    st.metric("Recall@5", f"{recall_5:.3f}")
                
            except Exception as e:
                st.error(f"Search failed: {e}")


def main():
    """Main Streamlit app."""
    st.title("🎯 Recall Analysis Dashboard")
    st.markdown("Interactive analysis of video search recall performance")
    
    # Sidebar for navigation and controls
    st.sidebar.title("🔧 Controls")
    
    # Load ground truth data
    ground_truth = load_ground_truth_data()
    if ground_truth is None:
        st.error("Failed to load ground truth data. Please check the annotation file.")
        return
    
    # Navigation
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["📊 Overall Performance", "🏷️ Keyword Analysis", "🎬 Video Triaging", "📈 Custom Evaluation"]
    )
    
    if analysis_mode == "📊 Overall Performance":
        st.header("📊 Overall Recall Performance")
        
        # Load comprehensive results
        results = load_evaluation_results()
        if results is None:
            return
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        
        if 'video_to_video' in results and 'average_recalls' in results['video_to_video']:
            v2v_r1 = results['video_to_video']['average_recalls'].get('recall@1', 0)
            with col1:
                st.metric("Video-to-Video R@1", f"{v2v_r1:.3f}")
        
        if 'text_to_video' in results and 'average_recalls' in results['text_to_video']:
            t2v_r1 = results['text_to_video']['average_recalls'].get('recall@1', 0)
            with col2:
                st.metric("Text-to-Video R@1", f"{t2v_r1:.3f}")
        
        if 'evaluation_config' in results:
            total_videos = results['evaluation_config']['total_annotated_videos']
            with col3:
                st.metric("Total Videos", total_videos)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_chart = create_recall_metrics_chart(results)
            if metrics_chart:
                st.plotly_chart(metrics_chart, use_container_width=True)
        
        with col2:
            category_heatmap = create_category_heatmap(results)
            if category_heatmap:
                st.plotly_chart(category_heatmap, use_container_width=True)
        
        # Detailed results tabs
        tab1, tab2 = st.tabs(["Video-to-Video Details", "Text-to-Video Details"])
        
        with tab1:
            display_detailed_results(results, 'video_to_video')
        
        with tab2:
            display_detailed_results(results, 'text_to_video')
    
    elif analysis_mode == "🏷️ Keyword Analysis":
        st.header("🏷️ Keyword-Specific Analysis")
        
        # Keyword selection
        all_keywords = ground_truth.get_all_keywords()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_keywords = st.multiselect(
                "Select keywords to analyze:",
                all_keywords,
                default=['urban', 'car2pedestrian']
            )
        
        with col2:
            eval_mode = st.selectbox(
                "Evaluation Mode:",
                ["both", "text", "video"]
            )
        
        if selected_keywords:
            # Load keyword-specific results
            keyword_results = load_keyword_evaluation(selected_keywords, eval_mode)
            
            if keyword_results:
                if eval_mode == "both":
                    # Display both modes
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Text-to-Video")
                        if 'text' in keyword_results:
                            text_results = keyword_results['text']
                            if 'average_recalls' in text_results:
                                for metric, value in text_results['average_recalls'].items():
                                    st.metric(metric, f"{value:.3f}")
                    
                    with col2:
                        st.subheader("🎬 Video-to-Video")
                        if 'video' in keyword_results:
                            video_results = keyword_results['video']
                            if 'average_recalls' in video_results:
                                for metric, value in video_results['average_recalls'].items():
                                    st.metric(metric, f"{value:.3f}")
                
                else:
                    # Display single mode results
                    if 'average_recalls' in keyword_results:
                        st.subheader(f"📊 {eval_mode.title()}-to-Video Results")
                        
                        cols = st.columns(len(keyword_results['average_recalls']))
                        for i, (metric, value) in enumerate(keyword_results['average_recalls'].items()):
                            with cols[i]:
                                st.metric(metric, f"{value:.3f}")
                    
                    # Show keyword breakdown if available
                    if 'keyword_breakdown' in keyword_results:
                        st.subheader("🏷️ Per-Keyword Breakdown")
                        
                        breakdown_data = []
                        for keyword, data in keyword_results['keyword_breakdown'].items():
                            recalls = data['recalls']
                            breakdown_data.append({
                                'Keyword': keyword,
                                'Relevant Videos': data['relevant_count'],
                                'Recall@1': recalls.get('recall@1', 0),
                                'Recall@3': recalls.get('recall@3', 0),
                                'Recall@5': recalls.get('recall@5', 0)
                            })
                        
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True)
                        
                        # Create chart
                        fig = px.bar(
                            breakdown_df,
                            x='Keyword',
                            y=['Recall@1', 'Recall@3', 'Recall@5'],
                            title='Per-Keyword Recall Performance',
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_mode == "🎬 Video Triaging":
        display_video_analysis(ground_truth)
    
    elif analysis_mode == "📈 Custom Evaluation":
        st.header("📈 Custom Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            custom_keywords = st.text_input(
                "Enter keywords (comma-separated):",
                placeholder="urban, highway, car2pedestrian"
            )
        
        with col2:
            custom_k_values = st.text_input(
                "Enter K values (comma-separated):",
                value="1,3,5",
                placeholder="1,3,5,10"
            )
        
        if st.button("🚀 Run Custom Evaluation"):
            if custom_keywords:
                keywords = [k.strip() for k in custom_keywords.split(',')]
                k_values = [int(k.strip()) for k in custom_k_values.split(',')]
                
                try:
                    with st.spinner("Running custom evaluation..."):
                        text_results = run_text_to_video_evaluation(keywords=keywords, k_values=k_values)
                        video_results = run_video_to_video_evaluation(keywords=keywords, k_values=k_values)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Text-to-Video Results")
                        for metric, value in text_results['average_recalls'].items():
                            st.metric(metric, f"{value:.3f}")
                    
                    with col2:
                        st.subheader("🎬 Video-to-Video Results")
                        for metric, value in video_results['average_recalls'].items():
                            st.metric(metric, f"{value:.3f}")
                
                except Exception as e:
                    st.error(f"Custom evaluation failed: {e}")
            else:
                st.warning("Please enter keywords to evaluate")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Dataset Info**")
    st.sidebar.write(f"Total Videos: {len(ground_truth.video_to_keywords)}")
    st.sidebar.write(f"Total Keywords: {len(ground_truth.keyword_to_videos)}")


if __name__ == "__main__":
    main()
