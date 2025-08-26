#!/usr/bin/env python3
"""
Script to create a CSV annotation template for manual scene description annotation.
This will be used to create ground truth for recall evaluation.
"""

import pandas as pd
import os
from pathlib import Path

def create_annotation_csv():
    """Create CSV template for manual annotation of video scenes."""
    
    # File paths
    input_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/main_input_mini.parquet"
    output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/video_annotation.csv"
    
    print("📋 Creating annotation CSV template...")
    print(f"Reading from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file does not exist: {input_file}")
        return
    
    try:
        # Load the parquet file
        df = pd.read_parquet(input_file)
        
        print(f"✅ Loaded parquet file with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample data to understand structure
        print(f"\nSample data:")
        print(df.head())
        
        # Create annotation template
        annotation_df = pd.DataFrame()
        
        # Extract filename from sensor_video_file path
        if 'sensor_video_file' in df.columns:
            annotation_df['filename'] = df['sensor_video_file'].apply(lambda x: Path(x).name)
        else:
            print("❌ 'sensor_video_file' column not found")
            return
        
        # Add slice_id for reference
        if 'slice_id' in df.columns:
            annotation_df['slice_id'] = df['slice_id']
        
        # Add full video path for reference
        annotation_df['video_path'] = df['sensor_video_file']
        
        # Add GIF path if available (for easier viewing during annotation)
        if 'gif_file' in df.columns:
            annotation_df['gif_path'] = df['gif_file']
        
        # Add empty scene_description column for manual annotation
        annotation_df['scene_description'] = ''
        
        # Add category column for grouping similar scenes
        annotation_df['category'] = ''
        
        # Add relevance_group column for recall evaluation
        # Videos with same relevance_group should be retrieved together
        annotation_df['relevance_group'] = ''
        
        # Add notes column for additional observations
        annotation_df['notes'] = ''
        
        # Sort by filename for easier annotation
        annotation_df = annotation_df.sort_values('filename').reset_index(drop=True)
        
        # Save to CSV
        annotation_df.to_csv(output_file, index=False)
        
        print(f"✅ Created annotation template: {output_file}")
        print(f"📊 Template contains {len(annotation_df)} videos to annotate")
        
        # Show sample of the template
        print(f"\n📋 Sample annotation template:")
        print(annotation_df[['filename', 'scene_description', 'category', 'relevance_group']].head(10))
        
        # Print instructions
        print_annotation_instructions(output_file)
        
        return annotation_df
        
    except Exception as e:
        print(f"❌ Error creating annotation CSV: {e}")
        return None

def print_annotation_instructions(csv_file):
    """Print instructions for manual annotation."""
    
    print("\n" + "="*80)
    print("📝 ANNOTATION INSTRUCTIONS")
    print("="*80)
    
    print(f"""
🎯 GOAL: Create ground truth for recall@5 evaluation

📋 HOW TO ANNOTATE:

1. SCENE_DESCRIPTION:
   - Describe what happens in the video (e.g., "car approaching cyclist")
   - Be specific about interactions, movements, and objects
   - Examples:
     * "car turning left while cyclist goes straight"
     * "pedestrian crossing street as car approaches"
     * "two cars merging in traffic"

2. CATEGORY:
   - Group similar types of interactions
   - Examples: "car_cyclist", "car_pedestrian", "car_car", "traffic_light"

3. RELEVANCE_GROUP:
   - Assign same group ID to videos that should be retrieved together
   - For recall evaluation: if you search for one video in a group,
     the other videos in that group should appear in top-5 results
   - Examples:
     * All "car approaching cyclist" videos → group "cyclist_approach"
     * All "pedestrian crossing" videos → group "ped_crossing"

4. NOTES:
   - Any additional observations
   - Weather conditions, time of day, location details

📁 FILE LOCATION: {csv_file}

🔍 FOR RECALL EVALUATION:
   - We'll test if searching for one video returns other videos 
     from the same relevance_group in the top-5 results
   - The more precise your relevance_groups, the better the evaluation

💡 TIP: You can view the GIF files (if available) for easier annotation
    """)
    
    print("="*80)

def create_sample_annotations():
    """Create a few sample annotations to show the expected format."""
    
    sample_data = {
        'filename': [
            'car2cyclist_1.mp4',
            'car2cyclist_2.mp4', 
            'car2ped_1.mp4',
            'car2car_1.mp4'
        ],
        'scene_description': [
            'car approaching cyclist from behind on road',
            'car passing cyclist on narrow street',
            'car stopping for pedestrian crossing street',
            'two cars merging in traffic lane'
        ],
        'category': [
            'car_cyclist',
            'car_cyclist',
            'car_pedestrian', 
            'car_car'
        ],
        'relevance_group': [
            'cyclist_interaction',
            'cyclist_interaction',
            'pedestrian_crossing',
            'car_merging'
        ],
        'notes': [
            'daytime, urban setting',
            'narrow street, close proximity',
            'crosswalk visible, car yields',
            'highway merge, multiple lanes'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation_example.csv"
    sample_df.to_csv(sample_file, index=False)
    
    print(f"\n📋 Created sample annotation file: {sample_file}")
    print("Use this as a reference for annotation format:")
    print(sample_df.to_string(index=False))

if __name__ == "__main__":
    # Create the annotation template
    result = create_annotation_csv()
    
    if result is not None:
        # Create sample annotations for reference
        create_sample_annotations()
        
        print(f"\n🎉 Ready for annotation!")
        print(f"📝 Edit the CSV file to add your scene descriptions")
        print(f"🔍 This will be used for recall@5 evaluation")
