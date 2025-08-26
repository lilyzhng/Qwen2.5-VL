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


class KeywordMapper:
    """Maps query text to fixed keywords using KeyBERT with preprocessing."""
    
    def __init__(self, fixed_keywords: List[str]):
        self.fixed_keywords = fixed_keywords
        self.kw_model = None
        self.keyword_variations = {}
        self._initialize_model()
        self._build_keyword_variations()
    
    def _initialize_model(self):
        """Initialize the KeyBERT model."""
        try:
            from keybert import KeyBERT
            self.kw_model = KeyBERT('all-MiniLM-L6-v2')
        except ImportError:
            st.warning("⚠️ KeyBERT not installed. Install with: pip install keybert")
            self.kw_model = None
        except Exception as e:
            st.warning(f"⚠️ Could not load KeyBERT model: {e}")
            self.kw_model = None
    
    def _build_keyword_variations(self):
        """Create variations of fixed keywords for better matching."""
        import re
        
        for kw in self.fixed_keywords:
            variations = [
                kw,
                kw.replace('_', ' '),  # turning_left -> turning left
                kw.replace('-', ' '),  # real-time -> real time
                ' '.join(kw.split('_')),  # handle multi_word_keywords
                re.sub(r'(\d+)', r' \1 ', kw).strip(),  # car2pedestrian -> car 2 pedestrian
            ]
            # Add variations without numbers for car2pedestrian -> car pedestrian
            if any(char.isdigit() for char in kw):
                clean_version = re.sub(r'\d+', '', kw).replace('_', ' ').strip()
                if clean_version:
                    variations.append(clean_version)
            
            for v in variations:
                if v.strip():  # Only add non-empty variations
                    self.keyword_variations[v.lower().strip()] = kw
    
    def preprocess_query(self, query: str) -> str:
        """Remove punctuation and extra spaces."""
        import re
        import string
        
        query = query.lower()
        query = re.sub(f'[{re.escape(string.punctuation)}]', ' ', query)
        query = ' '.join(query.split())
        return query
    
    def extract_and_map(self, query: str, threshold: float = 0.5) -> List[tuple]:
        """
        Extract and map keywords from query to fixed vocabulary.
        
        Args:
            query: Input text query
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (keyword, similarity_score) tuples, sorted by similarity
        """
        if self.kw_model is None:
            return self._simple_word_matching(query)
        
        try:
            # Preprocess query
            clean_query = self.preprocess_query(query)
            
            # Extract keywords/keyphrases using KeyBERT
            extracted_keywords = self.kw_model.extract_keywords(
                clean_query,
                keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
                stop_words='english',
                top_n=15,  # Increased to get more candidates
                use_mmr=True,  # Use Maximal Marginal Relevance for diversity
                diversity=0.5  # Diversity parameter for MMR
            )
            
            # Add manual phrase detection for common patterns
            import re
            manual_phrases = []
            
            # Detect common traffic/driving patterns
            traffic_patterns = [
                # Turning patterns
                (r'\b(turning|turn)\s+(left|right)\b', lambda m: f"{m[0]} {m[1]}"),
                (r'\b(left|right)\s+(turn|turning)\b', lambda m: f"{m[1]} {m[0]}"),
                
                # Car interaction patterns  
                (r'\bcar\s*2?\s*(pedestrian|cyclist|motorcyclist|car)\b', lambda m: f"car2{m[0]}"),
                (r'\b(pedestrian|cyclist|motorcyclist)\s+crossing\b', lambda m: f"{m[0]} crossing"),
                
                # Lane patterns
                (r'\blane\s+(merge|change)\b', lambda m: f"lane_{m[0]}"),
                (r'\b(merge|change)\s+lane\b', lambda m: f"lane_{m[0]}"),
                
                # Traffic light patterns
                (r'\btraffic\s+light\b', lambda m: "traffic_light"),
            ]
            
            for pattern, formatter in traffic_patterns:
                matches = re.findall(pattern, clean_query.lower())
                for match in matches:
                    try:
                        if isinstance(match, tuple):
                            phrase = formatter(match)
                        else:
                            phrase = formatter([match])
                        # Add with high score since it's a direct pattern match
                        manual_phrases.append((phrase, 0.85))
                    except Exception:
                        continue
            
            # Combine KeyBERT results with manual phrases
            all_extracted = list(extracted_keywords) + manual_phrases
            extracted_keywords = all_extracted
            
            mapped_keywords = {}
            
            for keyword, score in extracted_keywords:
                keyword_lower = keyword.lower().strip()
                
                # Direct match check first
                if keyword_lower in self.keyword_variations:
                    fixed_kw = self.keyword_variations[keyword_lower]
                    if fixed_kw not in mapped_keywords or score > mapped_keywords[fixed_kw]:
                        mapped_keywords[fixed_kw] = score
                else:
                    # Use KeyBERT to find similar keywords from fixed list
                    try:
                        similarities = self.kw_model.extract_keywords(
                            keyword,
                            candidates=self.fixed_keywords,
                            top_n=3,
                            use_mmr=True,  # Use Maximal Marginal Relevance
                            diversity=0.5
                        )
                        
                        for similar_kw, sim_score in similarities:
                            # Combine the extraction score with similarity score
                            combined_score = score * sim_score
                            if combined_score >= threshold:
                                if similar_kw not in mapped_keywords or combined_score > mapped_keywords[similar_kw]:
                                    mapped_keywords[similar_kw] = combined_score
                    except Exception as e:
                        # Skip this keyword if similarity calculation fails
                        continue
            
            # Convert to list of tuples and sort by score
            result = [(kw, score) for kw, score in mapped_keywords.items()]
            return sorted(result, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            st.warning(f"KeyBERT mapping failed: {e}. Using fallback method.")
            return self._simple_word_matching(query)
    
    def _simple_word_matching(self, query_text: str) -> List[tuple]:
        """Fallback method using simple word overlap matching."""
        query_words = set(self.preprocess_query(query_text).split())
        matched = []
        
        for keyword in self.fixed_keywords:
            # Handle keywords with underscores and numbers
            keyword_words = set(keyword.lower().replace('_', ' ').replace('2', ' ').split())
            overlap = len(query_words.intersection(keyword_words))
            if overlap > 0:
                # Simple overlap score
                score = overlap / len(keyword_words)
                matched.append((keyword, score))
        
        return sorted(matched, key=lambda x: x[1], reverse=True)


def _get_keyword_match_status(video_id: str, primary_keywords: List[str], ground_truth: GroundTruthProcessor) -> str:
    """
    Determine if a video matches the keyword criteria.
    Returns ✅ Yes only if ALL N keywords match.
    """
    if not primary_keywords:
        return '❌ No keywords'
    
    video_keywords = set(ground_truth.get_video_info(video_id)['keywords'])
    matched_keywords = [kw for kw in primary_keywords if kw in video_keywords]
    
    # Require ALL keywords to match for ✅ Yes
    if len(matched_keywords) == len(primary_keywords):
        return f'✅ Yes ({len(matched_keywords)}/{len(primary_keywords)})'
    else:
        return f'❌ Partial ({len(matched_keywords)}/{len(primary_keywords)})'


def extract_keybert_keywords_from_query(query_text: str, ground_truth: GroundTruthProcessor, 
                                       similarity_threshold: float = 0.5) -> tuple:
    """
    Extract semantically relevant keywords from query text using KeyBERT.
    
    Returns:
        (keybert_keywords, relevant_videos)
    """
    # Get all available keywords from ground truth
    all_keywords = ground_truth.get_all_keywords()
    
    # Initialize KeyBERT mapper
    mapper = KeywordMapper(all_keywords)
    
    # Extract and map semantic keywords
    keybert_keywords = mapper.extract_and_map(query_text, similarity_threshold)
    
    # Find relevant videos using KeyBERT keywords
    # FIXED: Use intersection logic - videos must match ALL keywords to be considered relevant
    relevant_videos = None
    for keyword, score in keybert_keywords:
        videos_for_keyword = ground_truth.get_relevant_videos_for_text(keyword)
        if relevant_videos is None:
            # First keyword - initialize with its videos
            relevant_videos = videos_for_keyword.copy()
        else:
            # Subsequent keywords - keep only videos that also have this keyword
            relevant_videos = relevant_videos.intersection(videos_for_keyword)
    
    # If no keywords found, return empty set
    if relevant_videos is None:
        relevant_videos = set()
    
    return keybert_keywords, relevant_videos


# Page configuration
st.set_page_config(
    page_title="Recall Analysis Dashboard",
    page_icon="",
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
def load_ground_truth_data(_annotation_file_mtime=None):
    """Load and cache ground truth data. Cache invalidates when file is modified."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if not annotation_path.exists():
        st.error(f"Annotation file not found: {annotation_path}")
        return None
    
    processor = GroundTruthProcessor(str(annotation_path))
    return processor

def get_ground_truth_data():
    """Get ground truth data with automatic cache invalidation on file changes."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if annotation_path.exists():
        import os
        file_mtime = os.path.getmtime(annotation_path)
        return load_ground_truth_data(_annotation_file_mtime=file_mtime)
    return None


@st.cache_data
def load_evaluation_results(_annotation_file_mtime=None, quality_threshold=0.0):
    """Load and cache evaluation results. Cache invalidates when annotation file changes."""
    try:
        with st.spinner("Running comprehensive recall evaluation..."):
            results = run_recall_evaluation(quality_threshold=quality_threshold)
        return results
    except Exception as e:
        st.error(f"Failed to load evaluation results: {e}")
        return None

def get_evaluation_results(quality_threshold=0.0):
    """Get evaluation results with automatic cache invalidation on annotation file changes."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if annotation_path.exists():
        import os
        file_mtime = os.path.getmtime(annotation_path)
        return load_evaluation_results(_annotation_file_mtime=file_mtime, quality_threshold=quality_threshold)
    return None


@st.cache_data
def load_keyword_evaluation(keywords: List[str], mode: str, _annotation_file_mtime=None, quality_threshold=0.0):
    """Load keyword-specific evaluation results. Cache invalidates when annotation file changes."""
    try:
        with st.spinner(f"Evaluating {mode} recall for keywords: {', '.join(keywords)}..."):
            if mode == "text":
                return run_text_to_video_evaluation(keywords=keywords, quality_threshold=quality_threshold)
            elif mode == "video":
                return run_video_to_video_evaluation(keywords=keywords)
            else:
                return {
                    "text": run_text_to_video_evaluation(keywords=keywords, quality_threshold=quality_threshold),
                    "video": run_video_to_video_evaluation(keywords=keywords)
                }
    except Exception as e:
        st.error(f"Failed to evaluate keywords: {e}")
        return None

def get_keyword_evaluation(keywords: List[str], mode: str, quality_threshold=0.0):
    """Get keyword evaluation results with automatic cache invalidation."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if annotation_path.exists():
        import os
        file_mtime = os.path.getmtime(annotation_path)
        return load_keyword_evaluation(keywords, mode, _annotation_file_mtime=file_mtime, quality_threshold=quality_threshold)
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


def display_text_search_analysis(ground_truth: GroundTruthProcessor, quality_threshold: float = 0.15):
    """Display Text Search Triage and triaging tools."""
    st.subheader("📝 Text Search Triage")
    
    # Text query input
    text_query = st.text_input(
        "Enter a text query to analyze:",
        placeholder="e.g., car turning left at intersection, pedestrian crossing street",
        help="Enter a descriptive text query to search for similar videos"
    )
    
    # Set default search parameters
    top_k = 10  # Search for more results to ensure we have 5 after quality filtering
    similarity_threshold = 0.0
    semantic_threshold = 0.5
    
    if text_query and st.button(f"🔍 Search for: '{text_query}'", key=f"text_search_{hash(text_query)}"):
            try:
                config = VideoRetrievalConfig()
                search_engine = VideoSearchEngine(config=config)
                
                with st.spinner("Searching for videos matching your text query..."):
                    search_results = search_engine.search_by_text(
                        text_query,
                        top_k=top_k
                    )
                
                # Filter by similarity threshold if specified
                if similarity_threshold > 0:
                    search_results = [r for r in search_results if r.get('similarity_score', r.get('similarity', 0)) >= similarity_threshold]
                
                # Apply quality threshold filtering
                original_count = len(search_results)
                search_results = [r for r in search_results if r.get('similarity_score', r.get('similarity', 0)) >= quality_threshold]
                filtered_count = original_count - len(search_results)
                
                # Check if top result has very low similarity
                if search_results:
                    top_similarity = search_results[0].get('similarity_score', search_results[0].get('similarity', 0))
                    if top_similarity < quality_threshold:
                        st.error(f"❌ **No Results Found**")
                        st.write(f"Top similarity score ({top_similarity:.3f}) is below quality threshold ({quality_threshold:.3f})")
                        st.write("This suggests the database doesn't contain relevant videos for this query.")
                        st.write("**Suggestions:**")
                        st.write("- Try a different query with more common terms")
                        st.write("- Lower the quality threshold")
                        st.write("- The database may be too small for this specific query")
                        return
                
                if not search_results:
                    st.error(f"❌ **No Results Found**")
                    st.write(f"No results found for query: '{text_query}' with similarity >= {quality_threshold:.3f}")
                    if filtered_count > 0:
                        st.write(f"({filtered_count} results were filtered out due to low quality)")
                    st.write("**Suggestions:**")
                    st.write("- Try lowering the quality threshold")
                    st.write("- Use simpler or more common terms")
                    st.write("- The database may be too small for this specific query")
                    return
                
                if filtered_count > 0:
                    st.info(f"ℹ️ Filtered out {filtered_count} low-quality results (similarity < {quality_threshold:.3f})")
                
                # Extract KeyBERT keywords
                keybert_keywords, relevant_videos = extract_keybert_keywords_from_query(
                    text_query, ground_truth, semantic_threshold
                )
                
                # Use KeyBERT keywords only
                if keybert_keywords:
                    primary_keywords = [kw for kw, score in keybert_keywords]
                else:
                    primary_keywords = []
                
                # Calculate recall metrics (Standard: Recall@K = relevant_found_in_top_K / K)
                retrieved_ids = [r['slice_id'] for r in search_results]
                
                # Calculate TP (True Positives) for each K
                tp_1 = len(set(retrieved_ids[:1]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
                tp_3 = len(set(retrieved_ids[:3]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
                tp_5 = len(set(retrieved_ids[:5]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
                
                # Standard Recall@K = TP / K
                recall_1 = tp_1 / 1 if retrieved_ids else 0
                recall_3 = tp_3 / 3 if retrieved_ids else 0
                recall_5 = tp_5 / 5 if retrieved_ids else 0
                
                # Track qualifying results count
                qualifying_results = len(retrieved_ids)
                
                # Display search performance metrics
                st.subheader("Text Search Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Top-K", len(search_results))
                with col2:
                    st.metric("Recall@1", f"{recall_1:.3f}")
                    if qualifying_results < 1:
                        st.caption(f"TP: {tp_1}, FP: {1 - tp_1}")
                        st.caption(f"Qualifying results: {qualifying_results} < 1")
                    else:
                        fp_1 = 1 - tp_1
                        st.caption(f"TP: {tp_1}, FP: {fp_1}")
                with col3:
                    st.metric("Recall@3", f"{recall_3:.3f}")
                    if qualifying_results < 3:
                        fp_3 = qualifying_results - tp_3  # FP = qualifying_results - TP when results < K
                        st.caption(f"TP: {tp_3}, FP: {fp_3}")
                        st.caption(f"Qualifying results: {qualifying_results} < 3")
                    else:
                        fp_3 = 3 - tp_3
                        st.caption(f"TP: {tp_3}, FP: {fp_3}")
                with col4:
                    st.metric("Recall@5", f"{recall_5:.3f}")
                    if qualifying_results < 5:
                        fp_5 = qualifying_results - tp_5  # FP = qualifying_results - TP when results < K
                        st.caption(f"TP: {tp_5}, FP: {fp_5}")
                        st.caption(f"Qualifying results: {qualifying_results} < 5")
                    else:
                        fp_5 = 5 - tp_5
                        st.caption(f"TP: {tp_5}, FP: {fp_5}")
                
                
                # Check alignment between search results and keyword matching
                high_sim_results = [r for r in search_results[:5] if r.get('similarity_score', r.get('similarity', 0)) > 0.7]
                high_sim_relevant = [r for r in high_sim_results if r['slice_id'] in relevant_videos]
                
                if high_sim_results:
                    semantic_precision = len(high_sim_relevant) / len(high_sim_results)
                    
                    if semantic_precision < 0.5:
                        st.warning(f"⚠️ **Low Alignment** ({semantic_precision:.1%}): Search results don't match keyword expectations")
                    else:
                        st.success(f"✅ **Good Alignment** ({semantic_precision:.1%}): Search results match keyword expectations")
                
                # Display text query vs top results comparison
                st.subheader("Query vs Retrieved Results")
                
                # Create columns for layout: Query + Top 5 results
                cols = st.columns(6)
                
                # Column 1: Text Query
                with cols[0]:
                    st.markdown("**Input Query**")
                    st.markdown(f"*\"{text_query}\"*")
                    
                    # Show KeyBERT keywords
                    st.markdown("**Input Keywords (KeyBERT)**")
                    if keybert_keywords:
                        for keyword, score in keybert_keywords:
                            st.markdown(f"- {keyword} ({score:.2f})")
                    else:
                        st.markdown("❌ No keywords found")
                
                # Columns 2-6: Top 5 results
                for i in range(5):
                    with cols[i + 1]:
                        if i < len(search_results):
                            result = search_results[i]
                            result_id = result['slice_id']
                            similarity = result.get('similarity_score', result.get('similarity', result.get('score', 0.0)))
                            is_relevant = result_id in relevant_videos
                            
                            # Success/failure indicator
                            status_icon = "✅" if is_relevant else "❌"
                            
                            st.markdown(f"**{status_icon} RANK #{i+1}**")
                            st.markdown(f"**{result_id}**")
                            st.markdown(f"*Similarity: {similarity:.3f}*")
                            
                            # Get result video info and display GIF
                            try:
                                result_info = ground_truth.get_video_info(result_id)
                                result_gif_path = Path(result_info['gif_path'])
                                
                                # Display result GIF
                                if result_gif_path.exists():
                                    try:
                                        st.image(str(result_gif_path), width=180, use_container_width=True)
                                    except Exception as e:
                                        st.write("❌ GIF not available")
                                else:
                                    st.write("❌ GIF not available")
                                
                                # Show result keywords with matching indicators
                                st.markdown("**Video Keywords:**")
                                result_keywords = result_info['keywords']
                                
                                for keyword in result_keywords:
                                    # Check if keyword matches KeyBERT keywords
                                    is_keybert_match = keyword in primary_keywords
                                    
                                    if is_keybert_match:
                                        st.markdown(f"✅ {keyword}")  # Match
                                    else:
                                        st.markdown(f"🔸 {keyword}")  # No match
                                
                            except Exception as e:
                                st.write(f"❌ Error: {e}")
                        else:
                            st.write("No result")
                

                
                # Detailed results table
                st.subheader("Detailed Text Search Results")
                results_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Video ID': result['slice_id'],
                        'Similarity': f"{result.get('similarity_score', result.get('similarity', 0.0)):.3f}",
                        'Status': _get_keyword_match_status(result['slice_id'], primary_keywords, ground_truth),
                        'Video Keywords': ', '.join(ground_truth.get_video_info(result['slice_id'])['keywords'][:3]) + 
                                        ('...' if len(ground_truth.get_video_info(result['slice_id'])['keywords']) > 3 else '')
                    }
                    for i, result in enumerate(search_results[:5])
                ])
                
                # Color code the dataframe
                def highlight_text_results(row):
                    if '✅' in row['Keyword Match']:
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    results_df.style.apply(highlight_text_results, axis=1),
                    use_container_width=True
                )
                
                # Improvement suggestions in expandable section
                with st.expander("💡 Improvement Suggestions"):
                    if recall_1 == 0 or recall_3 < 0.3:
                        st.markdown("**Why might performance be low?**")
                        if recall_1 == 0:
                            st.write("• Query might be too specific or use uncommon terminology")
                            st.write("• Try breaking down complex queries into simpler terms")
                        if recall_3 < 0.3:
                            st.write("• Query might describe concepts not well represented in the dataset")
                            st.write("• Consider using synonyms or alternative phrasings")
                    
                    # Suggest related keywords from the dataset
                    st.markdown("**Tips for better queries:**")
                    st.write("• Use concrete actions: 'car turning left' vs 'vehicle maneuvering'")
                    st.write("• Include spatial relationships: 'pedestrian crossing street'")
                    st.write("• Keep queries concise but descriptive")
                
            except Exception as e:
                st.error(f"Text search failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def display_video_analysis(ground_truth: GroundTruthProcessor, quality_threshold: float = 0.15):
    """Display video-level analysis and triaging tools."""
    st.subheader("🎬 Video Search Triage")
    
    # Video selection
    all_videos = list(ground_truth.video_to_keywords.keys())
    selected_video = st.selectbox(
        "Select a video to analyze:",
        all_videos,
        help="Choose a video to search for similar videos in the database"
    )
    
    if selected_video and st.button(f"🔍 Search for: '{selected_video}'", key=f"search_{selected_video}"):
        try:
            video_info = ground_truth.get_video_info(selected_video)
            relevant_videos = ground_truth.get_relevant_videos(selected_video, include_self=False)
            
            config = VideoRetrievalConfig()
            search_engine = VideoSearchEngine(config=config)
            
            with st.spinner("Searching for similar videos..."):
                search_results = search_engine.search_by_video(
                    video_info['video_path'],
                    top_k=10,  # Search for more results to ensure we have enough after filtering
                    use_cache=True
                )
            
            # Apply quality threshold filtering
            original_count = len(search_results)
            search_results = [r for r in search_results if r.get('similarity_score', r.get('similarity', 0)) >= quality_threshold]
            filtered_count = original_count - len(search_results)
            
            # Check if top result has very low similarity
            if search_results:
                top_similarity = search_results[0].get('similarity_score', search_results[0].get('similarity', 0))
                if top_similarity < quality_threshold:
                    st.error(f"❌ **No Results Found**")
                    st.write(f"Top similarity score ({top_similarity:.3f}) is below quality threshold ({quality_threshold:.3f})")
                    st.write("This suggests the database doesn't contain similar videos for this query.")
                    st.write("**Suggestions:**")
                    st.write("- Try a different video")
                    st.write("- Lower the quality threshold")
                    st.write("- The database may be too small for this specific video")
                    return
            
            if not search_results:
                st.error(f"❌ **No Results Found**")
                st.write(f"No results found for video: '{selected_video}' with similarity >= {quality_threshold:.3f}")
                if filtered_count > 0:
                    st.write(f"({filtered_count} results were filtered out due to low quality)")
                st.write("**Suggestions:**")
                st.write("- Try lowering the quality threshold")
                st.write("- Try a different video")
                st.write("- The database may be too small for this specific video")
                return
            
            if filtered_count > 0:
                st.info(f"ℹ️ Filtered out {filtered_count} low-quality results (similarity < {quality_threshold:.3f})")
            
            # Calculate recall metrics (Standard: Recall@K = relevant_found_in_top_K / K)
            retrieved_ids = [r['slice_id'] for r in search_results]
            
            # Calculate TP (True Positives) for each K
            tp_1 = len(set(retrieved_ids[:1]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
            tp_3 = len(set(retrieved_ids[:3]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
            tp_5 = len(set(retrieved_ids[:5]).intersection(relevant_videos)) if len(retrieved_ids) >= 1 else 0
            
            # Standard Recall@K = TP / K
            recall_1 = tp_1 / 1 if retrieved_ids else 0
            recall_3 = tp_3 / 3 if retrieved_ids else 0
            recall_5 = tp_5 / 5 if retrieved_ids else 0
            
            # Track qualifying results count
            qualifying_results_video = len(retrieved_ids)
            
            # Display search performance metrics
            st.subheader("Video Search Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Top-K", len(search_results))
            with col2:
                st.metric("Recall@1", f"{recall_1:.3f}")
                if qualifying_results_video < 1:
                    st.caption(f"TP: {tp_1}, FP: {1 - tp_1}")
                    st.caption(f"Qualifying results: {qualifying_results_video} < 1")
                else:
                    fp_1 = 1 - tp_1
                    st.caption(f"TP: {tp_1}, FP: {fp_1}")
            with col3:
                st.metric("Recall@3", f"{recall_3:.3f}")
                if qualifying_results_video < 3:
                    fp_3 = qualifying_results_video - tp_3  # FP = qualifying_results - TP when results < K
                    st.caption(f"TP: {tp_3}, FP: {fp_3}")
                    st.caption(f"Qualifying results: {qualifying_results_video} < 3")
                else:
                    fp_3 = 3 - tp_3
                    st.caption(f"TP: {tp_3}, FP: {fp_3}")
            with col4:
                st.metric("Recall@5", f"{recall_5:.3f}")
                if qualifying_results_video < 5:
                    fp_5 = qualifying_results_video - tp_5  # FP = qualifying_results - TP when results < K
                    st.caption(f"TP: {tp_5}, FP: {fp_5}")
                    st.caption(f"Qualifying results: {qualifying_results_video} < 5")
                else:
                    fp_5 = 5 - tp_5
                    st.caption(f"TP: {tp_5}, FP: {fp_5}")
            
            # Display side-by-side comparison with GIFs
            st.subheader("Input vs Retrieved Results")
            
            # Create 6 columns: Input + Top 5 results
            cols = st.columns(6)
            
            # Column 1: Input video
            with cols[0]:
                st.markdown("**INPUT**")
                st.markdown(f"**{selected_video}**")
                
                # Display input video GIF
                query_gif_path = Path(video_info['gif_path'])
                if query_gif_path.exists():
                    try:
                        st.image(str(query_gif_path), width=180, use_container_width=True)
                    except Exception as e:
                        st.write("❌ GIF not available")
                else:
                    st.write("❌ GIF not available")
                
                # Show input keywords
                st.markdown("**Keywords:**")
                for keyword in video_info['keywords']:
                    st.markdown(f"🏷️ {keyword}")
            
            # Columns 2-6: Top 5 results
            for i in range(5):
                with cols[i + 1]:
                    if i < len(search_results):
                        result = search_results[i]
                        result_id = result['slice_id']
                        # Handle different possible similarity key names
                        similarity = result.get('similarity_score', result.get('similarity', result.get('score', 0.0)))
                        is_relevant = result_id in relevant_videos
                        
                        # Success/failure indicator
                        status_icon = "✅" if is_relevant else "❌"
                        
                        st.markdown(f"**{status_icon} RANK #{i+1}**")
                        st.markdown(f"**{result_id}**")
                        st.markdown(f"*Similarity: {similarity:.3f}*")
                        
                        # Get result video info and display GIF
                        try:
                            result_info = ground_truth.get_video_info(result_id)
                            result_gif_path = Path(result_info['gif_path'])
                            
                            # Display result GIF
                            if result_gif_path.exists():
                                try:
                                    st.image(str(result_gif_path), width=180, use_container_width=True)
                                except Exception as e:
                                    st.write("❌ GIF not available")
                            else:
                                st.write("❌ GIF not available")
                            
                            # Show result keywords with matching indicators
                            st.markdown("**Keywords:**")
                            result_keywords = result_info['keywords']
                            query_keywords_set = set(video_info['keywords'])
                            
                            for keyword in result_keywords:
                                if keyword in query_keywords_set:
                                    st.markdown(f"✅ {keyword}")  # Matching keyword
                                else:
                                    st.markdown(f"🔸 {keyword}")  # Non-matching keyword
                            
                        except Exception as e:
                            st.write(f"❌ Error: {e}")
                    else:
                        st.write("No result")
            
            
            # Detailed results table
            st.subheader("📋 Detailed Search Results")
            results_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Video ID': result['slice_id'],
                    'Similarity': f"{result.get('similarity_score', result.get('similarity', 0.0)):.3f}",
                    'Status': _get_keyword_match_status(result['slice_id'], list(video_info['keywords']), ground_truth),
                    'Video Keywords': ', '.join(ground_truth.get_video_info(result['slice_id'])['keywords'][:3]) + 
                                    ('...' if len(ground_truth.get_video_info(result['slice_id'])['keywords']) > 3 else '')
                }
                for i, result in enumerate(search_results[:5])
            ])
            
            # Color code the dataframe
            def highlight_results(row):
                if '✅' in row['Status']:
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            st.dataframe(
                results_df.style.apply(highlight_results, axis=1),
                use_container_width=True
            )
            
            # Improvement suggestions in expandable section
            with st.expander("💡 Improvement Suggestions"):
                if recall_1 == 0 or recall_3 < 0.3:
                    st.markdown("**Why might performance be low?**")
                    if recall_1 == 0:
                        st.write("• The embedding model may not capture the semantic meaning of the keywords")
                        st.write("• Visual features might not align with the keyword annotations")
                        st.write("• The query video might be an outlier in its category")
                    if recall_3 < 0.3:
                        st.write("• Limited training data for this type of scenario")
                        st.write("• The embedding space might not cluster similar scenarios well")
                        st.write("• Consider data augmentation or fine-tuning the model")
                
                # Show what was expected vs what was retrieved
                expected_but_missing = relevant_videos - set(retrieved_ids[:5])
                if expected_but_missing:
                    st.markdown("**Expected but missing in top-5:**")
                    for vid in list(expected_but_missing)[:3]:
                        st.write(f"• {vid}")
                
                st.markdown("**Tips for better results:**")
                st.write("• Review keyword annotations for accuracy")
                st.write("• Consider visual similarity vs semantic similarity alignment")
                st.write("• Check if query video represents typical examples of its keywords")
        
        except Exception as e:
            st.error(f"Search failed: {e}")
            import traceback
            st.code(traceback.format_exc())


def main():
    """Main Streamlit app."""
    st.title("Recall Analysis Dashboard")

    # Show recall formula info
    st.info("""
    **Recall@K = relevant_found_in_top_K / K** 
    
    This means:
    - **Recall@1 = 100%** when the top result is relevant
    - **Recall@3 = 33%** when 1 out of 3 top results is relevant
    - **Recall@5 = 40%** when 2 out of 5 top results are relevant
    
    Note: When qualifying results < K, this is indicated in the recall details.
    """)
    
    # Sidebar for navigation and controls
    st.sidebar.title("Controls")
    
    # Quality threshold control
    st.sidebar.subheader("Quality Settings")
    global_quality_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.15, 
        step=0.01,
        help="Filter out results below this similarity score globally. Higher values = stricter filtering."
    )
    
    # Cache management
    if st.sidebar.button("🔄 Clear Cache & Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    
    st.sidebar.markdown("---")
    
    # Load ground truth data
    ground_truth = get_ground_truth_data()
    if ground_truth is None:
        st.error("Failed to load ground truth data. Please check the annotation file.")
        return
    
    # Navigation
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["📊 Overall Performance", "🏷️ Keyword Analysis", "🎬 Video Search Triage", "📝 Text Search Triage", "📈 Custom Evaluation"]
    )
    
    if analysis_mode == "📊 Overall Performance":
        st.header("📊 Overall Recall Performance")
        
        # Load comprehensive results
        results = get_evaluation_results(quality_threshold=global_quality_threshold)
        if results is None:
            return
        
        # Display quality threshold info if active
        if global_quality_threshold > 0.0:
            st.info(f"🔍 **Quality Filtering Active**: Results below {global_quality_threshold:.2f} similarity are excluded from evaluation")
            if 'text_to_video' in results and 'filtered_queries' in results['text_to_video']:
                filtered_count = results['text_to_video']['filtered_queries']
                if filtered_count > 0:
                    st.warning(f"⚠️ {filtered_count} text queries were filtered out due to low quality")
        
        # Display key metrics with fixed calculation context
        st.markdown("### 📊 Key Performance Metrics (Fixed Formula)")
        col1, col2, col3 = st.columns(3)
        
        if 'video_to_video' in results and 'average_recalls' in results['video_to_video']:
            v2v_r1 = results['video_to_video']['average_recalls'].get('recall@1', 0)
            with col1:
                st.metric("Video-to-Video R@1", f"{v2v_r1:.3f}", 
                         help="% of queries where top result is relevant")
        
        if 'text_to_video' in results and 'average_recalls' in results['text_to_video']:
            t2v_r1 = results['text_to_video']['average_recalls'].get('recall@1', 0)
            with col2:
                st.metric("Text-to-Video R@1", f"{t2v_r1:.3f}",
                         help="% of text queries where top result is relevant")
        
        if 'evaluation_config' in results:
            total_videos = results['evaluation_config']['total_annotated_videos']
            with col3:
                st.metric("Total Videos", total_videos,
                         help="Total annotated videos in ground truth")
        
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
            keyword_results = get_keyword_evaluation(selected_keywords, eval_mode, quality_threshold=global_quality_threshold)
            
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
    
    elif analysis_mode == "🎬 Video Search Triage":
        display_video_analysis(ground_truth, quality_threshold=global_quality_threshold)
    
    elif analysis_mode == "📝 Text Search Triage":
        display_text_search_analysis(ground_truth, quality_threshold=global_quality_threshold)
    
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
                        text_results = run_text_to_video_evaluation(keywords=keywords, k_values=k_values, quality_threshold=global_quality_threshold)
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
