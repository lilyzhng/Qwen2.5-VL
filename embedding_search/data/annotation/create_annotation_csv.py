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
    input_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/unified_input_path.parquet"
    output_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/video_annotation.csv"
    
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
        
        # Add slice_id as the primary identifier
        if 'slice_id' in df.columns:
            annotation_df['slice_id'] = df['slice_id']
        else:
            print("❌ 'slice_id' column not found")
            return
        
        # Add full video path for reference
        annotation_df['video_path'] = df['sensor_video_file']
        
        # Add GIF path if available (for easier viewing during annotation)
        if 'gif_file' in df.columns:
            annotation_df['gif_path'] = df['gif_file']
        
        # Add separate columns for each semantic group
        annotation_df['object_type'] = ''
        annotation_df['actor_behavior'] = ''
        annotation_df['spatial_relation'] = ''
        annotation_df['ego_behavior'] = ''
        annotation_df['scene_type'] = ''
        
        # Sort by slice_id for easier annotation
        annotation_df = annotation_df.sort_values('slice_id').reset_index(drop=True)
        
        # Save to CSV
        annotation_df.to_csv(output_file, index=False)
        
        print(f"✅ Created annotation template: {output_file}")
        print(f"📊 Template contains {len(annotation_df)} videos to annotate")
        
        # Show sample of the template
        print(f"\n📋 Sample annotation template:")
        print(annotation_df[['slice_id', 'object_type', 'actor_behavior', 'spatial_relation', 'ego_behavior', 'scene_type']].head(5))
        
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
🎯 GOAL: Create ground truth for recall@5 evaluation using semantic keywords

📋 HOW TO ANNOTATE:

1. SEMANTIC GROUP COLUMNS:
   Fill in each semantic group column with comma-separated keywords from that group:

   🚗 OBJECT TYPE: What objects/actors are present
   - Vehicles: small vehicle, large vehicle
   - People: pedestrian, motorcyclist, bicyclist  
   - Objects: bollard, stationary object, other, unknown

   🏃 ACTOR BEHAVIOR: How other actors are moving/behaving
   - entering ego path, stationary, traveling in same direction
   - traveling in opposite direction, straight crossing path, oncoming turn across path

   📍 SPATIAL RELATION: Where objects are positioned relative to ego vehicle
   - corridor front, corridor behind, left/right adjacent (front/behind)
   - left/right split (front/behind)

   🚙 EGO BEHAVIOR: What the ego vehicle is doing
   - ego turning, proceeding straight, ego lane change

   🌍 SCENE TYPE: Environmental and contextual information
   - Location: test track, parking lot/depot, intersection, non-intersection, crosswalk, highway, urban, bridge/tunnel
   - Road: curved road, positive/negative road grade, street parked vehicle
   - Conditions: nighttime, daytime, rainy, sunny, overcast
   - Special: vulnerable road user present, other

2. SLICE_ID:
   - Unique identifier for each video clip
   - Used as the primary key for annotation

📁 FILE LOCATION: {csv_file}

🔍 FOR RECALL EVALUATION:
   - Videos with the same keywords will be used as ground truth
   - When searching for one video, other videos with shared keywords should appear in top-5 results
   - Use multiple keywords from different semantic groups for comprehensive annotation

💡 TIPS:
   - Select keywords from multiple semantic groups
   - Be specific and descriptive
   - Use GIF files (if available) for easier viewing during annotation

📋 EXAMPLE ANNOTATION:
   object_type: "pedestrian"
   actor_behavior: "entering ego path"
   spatial_relation: "corridor front"
   ego_behavior: "proceeding straight"
   scene_type: "intersection, daytime"
    """)
    
    print("="*80)

def create_sample_annotations():
    """Create a few sample annotations to show the expected format."""
    
    sample_data = {
        'slice_id': [
            'car2cyclist_1.mp4',
            'car2cyclist_2.mp4',
            'car2ped_1.mp4', 
            'car2car_1.mp4'
        ],
        'video_path': [
            '/path/to/car2cyclist_1.mp4',
            '/path/to/car2cyclist_2.mp4',
            '/path/to/car2ped_1.mp4',
            '/path/to/car2car_1.mp4'
        ],
        'object_type': [
            'bicyclist',
            'bicyclist',
            'pedestrian',
            'small vehicle'
        ],
        'actor_behavior': [
            'traveling in same direction',
            'entering ego path',
            'straight crossing path',
            'traveling in same direction'
        ],
        'spatial_relation': [
            'corridor front',
            'left adjacent',
            'corridor front',
            'right adjacent'
        ],
        'ego_behavior': [
            'proceeding straight',
            'ego lane change',
            'proceeding straight',
            'proceeding straight'
        ],
        'scene_type': [
            'non-intersection, daytime',
            'intersection, daytime',
            'crosswalk, daytime',
            'highway, daytime'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = "/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation/annotation_example.csv"
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
