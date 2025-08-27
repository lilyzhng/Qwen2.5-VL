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
    template_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/video_annotation.csv"
    
    if not os.path.exists(template_file):
        st.error(f"Template file not found: {template_file}")
        st.stop()
    
    df = pd.read_csv(template_file)
    
    # Filter for videos with empty semantic group annotations (needing annotation)
    # Check if any of the semantic group columns are empty
    semantic_columns = ['object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']
    
    # Convert semantic columns to string to handle NaN values properly
    for col in semantic_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # A video needs annotation if ANY semantic group columns are empty
    empty_mask = False
    for col in semantic_columns:
        if col in df.columns:
            col_empty = df[col].isna() | (df[col] == '') | (df[col] == 'nan') | (df[col].str.strip() == '')
            empty_mask = empty_mask | col_empty
    
    empty_annotations_mask = empty_mask
    videos_needing_annotation = df[empty_annotations_mask].copy()
    
    # Store original indices for saving back to the full dataset
    videos_needing_annotation['original_index'] = df[empty_annotations_mask].index
    
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
        st.info("No videos with incomplete semantic annotations found. All videos in the dataset have been fully annotated.")
        
        # Show summary of annotated videos
        st.subheader("📊 Dataset Summary")
        total_videos = len(full_df)
        
        # Count videos that have annotations in ALL semantic groups (fully annotated)
        semantic_columns = ['object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']
        fully_annotated_mask = True
        for col in semantic_columns:
            if col in full_df.columns:
                col_annotated = full_df[col].notna() & (full_df[col] != '') & (full_df[col] != 'nan') & (full_df[col].str.strip() != '')
                fully_annotated_mask = fully_annotated_mask & col_annotated
        
        annotated_videos = len(full_df[fully_annotated_mask])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Videos", total_videos)
        with col2:
            st.metric("Annotated Videos", annotated_videos)
        with col3:
            completion_pct = (annotated_videos / total_videos) * 100 if total_videos > 0 else 0
            st.metric("Completion", f"{completion_pct:.1f}%")
        
        # Show keyword distribution by semantic group
        if annotated_videos > 0:
            st.subheader("🏷️ Keyword Distribution by Semantic Group")
            
            for group_name in semantic_columns:
                if group_name in full_df.columns:
                    group_display_name = group_name.replace('_', ' ').title()
                    st.write(f"**{group_display_name}:**")
                    
                    all_keywords = []
                    for keywords_str in full_df[group_name].dropna():
                        if str(keywords_str).strip() and str(keywords_str) != 'nan':
                            keywords = [k.strip() for k in str(keywords_str).split(',') if k.strip()]
                            all_keywords.extend(keywords)
                    
                    if all_keywords:
                        from collections import Counter
                        keyword_counts = Counter(all_keywords)
                        
                        # Display as a simple list
                        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                            st.write(f"  - {keyword}: {count}")
                    else:
                        st.write("  - No annotations yet")
                    st.write("")  # Add spacing
        
        return
    
    st.info(f"📝 Found {len(videos_to_annotate)} videos needing annotation (with incomplete semantic annotations)")
    df = videos_to_annotate
    
    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'annotations' not in st.session_state:
        st.session_state.annotations = {}
    
    # Auto-load existing annotations on first run
    if 'annotations_loaded' not in st.session_state:
        st.session_state.annotations_loaded = False
        
    if not st.session_state.annotations_loaded:
        # Load existing annotations automatically
        output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/video_annotation.csv"
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                
                # Load existing annotations into session state for videos currently being annotated
                for current_idx in range(len(df)):
                    original_idx = df.iloc[current_idx]['original_index']
                    
                    if original_idx < len(existing_df):
                        row = existing_df.iloc[original_idx]
                        
                        # Load semantic group annotations
                        annotations = {}
                        for group_name in ['object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']:
                            if group_name in row and pd.notna(row[group_name]) and str(row[group_name]).strip() and str(row[group_name]) != 'nan':
                                keywords_str = str(row[group_name])
                                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                                annotations[group_name] = keywords
                        
                        # Only save if there are any annotations
                        if any(annotations.values()):
                            st.session_state.annotations[current_idx] = annotations
                
                st.session_state.annotations_loaded = True
            except Exception as e:
                st.warning(f"Could not auto-load existing annotations: {e}")
    
    # Semantic groups with their respective keywords
    semantic_groups = {
        'object_type': [
            "small vehicle", "large vehicle", "bollard", "stationary object", 
            "pedestrian", "motorcyclist", "bicyclist", "other", "unknown"
        ],
        'actor_behavior': [
            "entering ego path", "stationary", "traveling in same direction", 
            "traveling in opposite direction", "straight crossing path", 
            "oncoming turn across path"
        ],
        'spatial_relation': [
            "corridor front", "corridor behind", "left adjacent", "right adjacent",
            "left adjacent front", "left adjacent behind", "right adjacent front", 
            "right adjacent behind", "left split", "right split", "left split front",
            "left split behind", "right split front", "right split behind"
        ],
        'ego_behavior': [
            "ego turning", "proceeding straight", "ego lane change", "ego stationary"
        ],
        'scene_type': [
            "test track", "parking lot/depot", "intersection", "non-intersection", 
            "crosswalk", "highway", "urban",  "bridge/tunnel", "curved road", "positive road grade", 
            "negative road grade", "street parked vehicle", 
            "vulnerable road user present", "nighttime", "daytime", 
            "rainy", "sunny", "overcast", "other"
        ]
    }
    
    # Ensure current_index is within valid bounds
    if st.session_state.current_index >= len(df):
        st.session_state.current_index = len(df) - 1 if len(df) > 0 else 0
    if st.session_state.current_index < 0:
        st.session_state.current_index = 0
    
    # Progress bar
    progress = st.session_state.current_index / len(df) if len(df) > 0 else 0
    st.progress(progress)
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(df)} videos")
    
    # Safety check for empty dataframe
    if len(df) == 0:
        st.error("No videos available for annotation.")
        return
    
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
            output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/video_annotation.csv"
            if st.button("💾 Save", type="primary", help="Save all annotations to CSV"):
                # Prepare the full dataframe for saving
                save_df = full_df.copy()
                
                # Update annotations for videos that were annotated
                for idx in range(len(df)):
                    annotations = st.session_state.annotations.get(idx, {})
                    if not isinstance(annotations, dict):
                        annotations = {}
                    
                    # Get the original index in the full dataset
                    original_idx = df.iloc[idx]['original_index']
                    
                    # Only update semantic group columns that have new annotations
                    for group_name in ['object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']:
                        if group_name in annotations:  # Only update if this group was annotated
                            group_keywords = annotations.get(group_name, [])
                            if isinstance(group_keywords, str):
                                group_keywords = [group_keywords] if group_keywords else []
                            
                            keywords_str = ", ".join(group_keywords) if group_keywords else ""
                            save_df.loc[original_idx, group_name] = keywords_str
                            # Note: We don't touch columns that weren't annotated, preserving existing data
                
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
        st.write(f"**Source:** {folder_name}")
        
        # Semantic group selections
        st.subheader("🏷️ Semantic Annotation")
        
        # Initialize current annotations for this video
        current_annotations = st.session_state.annotations.get(st.session_state.current_index, {})
        if not isinstance(current_annotations, dict):
            current_annotations = {}
        
        # Create selection widgets for each semantic group
        selected_annotations = {}
        
        for group_name, keywords in semantic_groups.items():
            group_display_name = group_name.replace('_', ' ').title()
            
            # Get current selection for this group
            current_selection = current_annotations.get(group_name, [])
            if isinstance(current_selection, str):
                current_selection = [current_selection] if current_selection else []
            
            # Create multiselect for this semantic group
            selected = st.multiselect(
                f"🔹 {group_display_name}:",
                keywords,
                default=current_selection,
                key=f"{group_name}_{st.session_state.current_index}",
                help=f"Select {group_display_name.lower()} keywords that apply to this video"
            )
            
            selected_annotations[group_name] = selected
        
        # Save current annotations
        st.session_state.annotations[st.session_state.current_index] = selected_annotations
        
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
    
    # Count annotations and keywords by semantic group
    group_keyword_counts = {group: {} for group in semantic_groups.keys()}
    annotated_videos = 0
    
    for idx, annotations in st.session_state.annotations.items():
        if isinstance(idx, int):  # Only count video annotations
            if not isinstance(annotations, dict):
                continue
                
            # Check if video has any annotations
            has_annotations = False
            for group_name, keywords in annotations.items():
                if isinstance(keywords, str):
                    keywords = [keywords] if keywords else []
                if keywords:
                    has_annotations = True
                    for keyword in keywords:
                        if keyword in group_keyword_counts[group_name]:
                            group_keyword_counts[group_name][keyword] += 1
                        else:
                            group_keyword_counts[group_name][keyword] = 1
            
            if has_annotations:
                annotated_videos += 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Annotated", annotated_videos)
    
    with col2:
        st.metric("Remaining", len(df) - annotated_videos)
    
    with col3:
        completion_pct = (annotated_videos / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Completion", f"{completion_pct:.1f}%")
    
    # Display keyword usage by semantic group
    for group_name, keyword_counts in group_keyword_counts.items():
        if keyword_counts:
            group_display_name = group_name.replace('_', ' ').title()
            st.write(f"**{group_display_name} Usage:**")
            for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"  - {keyword}: {count}")
    
    # Load previous annotations (moved to bottom)
    st.markdown("---")
    
    if st.button("📥 Load Previous Annotations"):
        output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/video_annotation.csv"
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                
                # Load existing annotations into session state for videos currently being annotated
                for current_idx in range(len(df)):
                    original_idx = df.iloc[current_idx]['original_index']
                    
                    if original_idx < len(existing_df):
                        row = existing_df.iloc[original_idx]
                        
                        # Load semantic group annotations
                        annotations = {}
                        for group_name in ['object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']:
                            if group_name in row and pd.notna(row[group_name]) and str(row[group_name]).strip() and str(row[group_name]) != 'nan':
                                keywords_str = str(row[group_name])
                                keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                                annotations[group_name] = keywords
                            # Don't set empty lists for missing annotations - let them remain unset
                        
                        # Only save if there are any annotations
                        if any(annotations.values()):
                            st.session_state.annotations[current_idx] = annotations
                
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
        
        **This tool automatically loads only videos with incomplete semantic annotations that need annotation.**
        
        1. **View the GIF** on the left to understand the video content
        2. **Select multiple keywords** that describe the video content
        3. **Navigate** using Previous/Next buttons or jump to specific videos
        4. **Save regularly** to preserve your work
        5. **When all videos are annotated**, the tool will show a completion summary
        
        ### Available Keywords (Organized by Semantic Groups):
        
        1. **Object Type:** What objects/actors are present in the scene
           - Vehicles: small vehicle, large vehicle
           - People: pedestrian, motorcyclist, bicyclist
           - Objects: bollard, stationary object, other, unknown
        
        2. **Actor Behavior:** How other actors are moving/behaving
           - entering ego path, stationary, traveling in same direction
           - traveling in opposite direction, straight crossing path, oncoming turn across path
        
        3. **Spatial Relation:** Where objects are positioned relative to ego vehicle
           - corridor front, corridor behind, left/right adjacent (front/behind)
           - left/right split (front/behind)
        
        4. **Ego Behavior:** What the ego vehicle is doing
           - ego turning, proceeding straight, ego lane change
        
        5. **Scene Type:** Environmental and contextual information
           - Location: test track, parking lot/depot, intersection, non-intersection, crosswalk, highway
           - Road: curved road, positive/negative road grade, street parked vehicle
           - Conditions: nighttime, daytime, rainy, sunny, overcast
           - Special: vulnerable road user present, other
        
        ### Tips:
        1. **Select keywords from multiple groups** - try to describe the scene comprehensively
        2. **Object Type**: Always identify what objects/actors are present
        3. **Actor Behavior**: Describe how other actors are moving relative to ego vehicle
        4. **Spatial Relation**: Note where objects are positioned around ego vehicle
        5. **Ego Behavior**: Describe what the ego vehicle is doing
        6. **Scene Type**: Include environmental context (location, conditions, etc.)
        7. **The tool filters automatically** - only shows videos needing annotation
        
        ### For Recall Evaluation:
        Videos with the same semantic annotations will be used as ground truth for measuring recall@5.
        When you search for one video, other videos with similar semantic annotations should appear in the top-5 results.
        """)

if __name__ == "__main__":
    main()
