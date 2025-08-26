#!/usr/bin/env python3
"""
Streamlit app for video annotation with GIF preview.
Allows manual annotation of videos with ground truth categories for recall evaluation.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import base64

# Page config
st.set_page_config(
    page_title="Video Annotation Tool",
    page_icon="🎬",
    layout="wide"
)

def load_annotation_data():
    """Load the annotation template data and filter for videos needing annotation."""
    template_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/video_annotation.csv"
    
    if not os.path.exists(template_file):
        st.error(f"Template file not found: {template_file}")
        st.stop()
    
    df = pd.read_csv(template_file)
    
    # Filter for videos with empty keywords (needing annotation)
    empty_keywords_mask = df['keywords'].isna() | (df['keywords'] == '') | (df['keywords'].str.strip() == '')
    videos_needing_annotation = df[empty_keywords_mask].copy()
    
    # Store original indices for saving back to the full dataset
    videos_needing_annotation['original_index'] = df[empty_keywords_mask].index
    
    return videos_needing_annotation, df

def save_annotations(df, output_file):
    """Save annotations to CSV file."""
    try:
        df.to_csv(output_file, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def display_gif(gif_path):
    """Display GIF in Streamlit."""
    if not os.path.exists(gif_path):
        st.warning(f"GIF not found: {gif_path}")
        return False
    
    try:
        # Read and encode GIF
        with open(gif_path, "rb") as f:
            gif_bytes = f.read()
        
        gif_base64 = base64.b64encode(gif_bytes).decode()
        
        # Display GIF with HTML
        gif_html = f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/gif;base64,{gif_base64}" 
                 style="width: 100%; max-width: 800px; height: auto; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        </div>
        """
        st.markdown(gif_html, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.error(f"Error displaying GIF: {e}")
        return False

def main():
    st.title("🎬 Video Annotation Tool")
    # Load data
    videos_to_annotate, full_df = load_annotation_data()
    
    # Check if there are videos needing annotation
    if len(videos_to_annotate) == 0:
        st.success("🎉 All videos have been annotated!")
        st.info("No videos with empty keywords found. All videos in the dataset have been annotated.")
        
        # Show summary of annotated videos
        st.subheader("📊 Dataset Summary")
        total_videos = len(full_df)
        annotated_videos = len(full_df[full_df['keywords'].notna() & (full_df['keywords'] != '')])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Videos", total_videos)
        with col2:
            st.metric("Annotated Videos", annotated_videos)
        with col3:
            completion_pct = (annotated_videos / total_videos) * 100 if total_videos > 0 else 0
            st.metric("Completion", f"{completion_pct:.1f}%")
        
        # Show keyword distribution
        if annotated_videos > 0:
            st.subheader("🏷️ Keyword Distribution")
            all_keywords = []
            for keywords_str in full_df['keywords'].dropna():
                if keywords_str.strip():
                    keywords = [k.strip() for k in keywords_str.split(',')]
                    all_keywords.extend(keywords)
            
            if all_keywords:
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                
                # Display as a table
                keyword_df = pd.DataFrame([
                    {"Keyword": keyword, "Count": count} 
                    for keyword, count in keyword_counts.most_common()
                ])
                st.dataframe(keyword_df, use_container_width=True)
        
        return
    
    st.info(f"📝 Found {len(videos_to_annotate)} videos needing annotation (with empty keywords)")
    df = videos_to_annotate
    
    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    
    # Keywords for annotation
    available_keywords = [
        "highway",
        "parking", 
        "urban",
        "intersection",
        "cyclist",
        "pedestrian",
        "motorcyclist",
        "lane_merge",
        "car2cyclist",
        "car2pedestrian",
        "car2motorcyclist",
        "car2car",
        "turning_left",
        "turning_right",
        "merging",
        "traffic_light",
        "crosswalk",
        "bridge",
        "tunnel",
        "construction",
        "rain",
        "night",
        "daytime",
        "busy_traffic",
        "light_traffic",
        "residential",
        "commercial",
        "freeway",
        "parked_bicycle",
        "other"
    ]
    
    # Progress bar
    progress = st.session_state.current_index / len(df)
    st.progress(progress)
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(df)} videos")
    
    # Current video info
    current_video = df.iloc[st.session_state.current_index]
    
    # Layout: GIF on left, controls on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📹 {current_video['slice_id']}")
        
        # Display GIF
        gif_path = current_video['gif_path']
        if pd.notna(gif_path) and gif_path:
            display_gif(gif_path)
        else:
            st.warning("No GIF available for this video")
            st.write(f"Video path: {current_video['video_path']}")
    
    with col2:
        # Header with save button
        col_header, col_save = st.columns([2, 1])
        
        with col_header:
            st.subheader("🏷️ Annotation")
        
        with col_save:
            output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/video_annotation.csv"
            if st.button("💾 Save", type="primary", help="Save all annotations to CSV"):
                # Prepare the full dataframe for saving
                save_df = full_df.copy()
                
                # Update annotations for videos that were annotated
                for idx in range(len(df)):
                    keywords = st.session_state.annotations.get(idx, [])
                    if isinstance(keywords, str):
                        keywords = [keywords] if keywords else []
                    
                    keywords_str = ", ".join(keywords) if keywords else ""
                    
                    # Get the original index in the full dataset
                    original_idx = df.iloc[idx]['original_index']
                    
                    # Update the full dataframe
                    save_df.loc[original_idx, 'keywords'] = keywords_str
                
                if save_annotations(save_df, output_file):
                    st.success(f"✅ Saved annotations for {len([k for k in st.session_state.annotations.values() if k])} videos!")
                    # Refresh the page to reload data
                    st.rerun()
                else:
                    st.error("❌ Save failed")
        
        # Video info
        st.write(f"**Slice ID:** {current_video['slice_id']}")
        
        # Get folder name for context
        video_path = current_video['video_path']
        folder_name = Path(video_path).parent.name if pd.notna(video_path) else "Unknown"
        
        # Keywords selection
        current_keywords = st.session_state.annotations.get(st.session_state.current_index, [])
        if isinstance(current_keywords, str):
            # Handle legacy single category format
            current_keywords = [current_keywords] if current_keywords else []
        
        selected_keywords = st.multiselect(
            "Select keywords (choose multiple):",
            available_keywords,
            default=current_keywords,
            key=f"keywords_{st.session_state.current_index}",
            help="Select all keywords that describe this video"
        )
        
        # Save current annotation
        st.session_state.annotations[st.session_state.current_index] = selected_keywords
        
        # Navigation buttons
        st.markdown("---")
        
        col_prev, col_next = st.columns(2)
        
        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0):
                st.session_state.current_index -= 1
                st.rerun()
        
        with col_next:
            if st.button("➡️ Next", disabled=st.session_state.current_index >= len(df) - 1):
                st.session_state.current_index += 1
                st.rerun()
        
        # Jump to specific video
        st.markdown("---")
        jump_to = st.number_input(
            "Jump to video #:",
            min_value=1,
            max_value=len(df),
            value=st.session_state.current_index + 1
        )
        
        if st.button("🔄 Jump"):
            st.session_state.current_index = jump_to - 1
            st.rerun()
    
    # Summary section
    st.markdown("---")
    st.subheader("📊 Annotation Summary")
    
    # Count annotations and keywords
    keyword_counts = {}
    annotated_videos = 0
    
    for idx, keywords in st.session_state.annotations.items():
        if isinstance(idx, int):  # Only count video annotations
            if isinstance(keywords, str):
                keywords = [keywords] if keywords else []
            
            if keywords:  # Video has at least one keyword
                annotated_videos += 1
                for keyword in keywords:
                    if keyword in keyword_counts:
                        keyword_counts[keyword] += 1
                    else:
                        keyword_counts[keyword] = 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Annotated", annotated_videos)
    
    with col2:
        st.metric("Remaining", len(df) - annotated_videos)
    
    with col3:
        completion_pct = (annotated_videos / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Completion", f"{completion_pct:.1f}%")
    
    if keyword_counts:
        # Keyword breakdown
        st.write("**Keyword Usage:**")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {keyword}: {count}")
    
    # Load previous annotations (moved to bottom)
    st.markdown("---")
    
    if st.button("📥 Load Previous Annotations"):
        output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/video_annotation.csv"
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                
                # Load existing annotations into session state for videos currently being annotated
                for current_idx in range(len(df)):
                    original_idx = df.iloc[current_idx]['original_index']
                    
                    if original_idx < len(existing_df):
                        row = existing_df.iloc[original_idx]
                        
                        # Try to load keywords first, fallback to category for backward compatibility
                        if pd.notna(row.get('keywords', '')) and row.get('keywords', '').strip():
                            keywords_str = row['keywords']
                            keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                            st.session_state.annotations[current_idx] = keywords
                        elif pd.notna(row.get('category', '')):
                            # Backward compatibility with old category format
                            st.session_state.annotations[current_idx] = [row['category']]
                
                st.success("✅ Previous annotations loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading annotations: {e}")
        else:
            st.warning("No previous annotations found")
    
    # Instructions
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use this annotation tool:
        
        **This tool automatically loads only videos with empty keywords that need annotation.**
        
        1. **View the GIF** on the left to understand the video content
        2. **Select multiple keywords** that describe the video content
        3. **Navigate** using Previous/Next buttons or jump to specific videos
        4. **Save regularly** to preserve your work
        5. **When all videos are annotated**, the tool will show a completion summary
        
        ### Available Keywords:
        1. **Environment:** highway, urban, intersection, residential, commercial, freeway, bridge, tunnel, construction
        2. **Interactions:** cyclist, pedestrian, motorcyclist, lane_merge, crosswalk
        3. **Maneuvers:** turning_left, turning_right, merging, parking
        4. **Traffic:** traffic_light, busy_traffic, light_traffic
        5. **Conditions:** rain, night, daytime
        6. **Other:** other
        
        ### Tips:
        1. **Select multiple keywords** that apply to each video
        2. **Be descriptive** - choose all relevant keywords
        3. **The tool filters automatically** - only shows videos needing annotation
        
        ### For Recall Evaluation:
        Videos with the same keyword will be used as ground truth for measuring recall@5.
        When you search for one video, other videos with the same keyword should appear in the top-5 results.
        """)

if __name__ == "__main__":
    main()
