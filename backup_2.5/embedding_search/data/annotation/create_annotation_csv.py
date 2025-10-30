#!/usr/bin/env python3
"""
Script to create a CSV annotation template for manual scene description annotation.
This reads from unified_embeddings.parquet and creates unified_embedding_annotation.csv
for ground truth recall evaluation.
"""

import pandas as pd
import os
from pathlib import Path

# Get project root directory
def get_project_root():
    """Get the project root directory (embedding_search folder)."""
    current_dir = Path(__file__).parent.absolute()
    # Go up from data/annotation to project root
    return current_dir.parent.parent

def create_annotation_csv():
    """Create CSV and parquet templates for manual annotation of video scenes from unified embeddings."""
    
    # File paths
    project_root = get_project_root()
    input_file = project_root / "data" / "unified_embeddings.parquet"
    output_csv_file = project_root / "data" / "annotation" / "unified_annotation.csv"
    
    print("📋 Creating annotation CSV and parquet templates...")
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
        
        # Create annotation template starting with all columns from unified_embeddings.parquet
        annotation_df = df.copy()
        
        # Verify slice_id exists
        if 'slice_id' not in df.columns:
            print("❌ 'slice_id' column not found")
            return
        
        # Add separate columns for each semantic group (for manual annotation)
        annotation_df['pv_object_type'] = ''
        annotation_df['pv_actor_behavior'] = ''
        annotation_df['pv_spatial_relation'] = ''
        annotation_df['ego_behavior'] = ''
        annotation_df['scene_type'] = ''
        
        # Sort by slice_id for easier annotation
        annotation_df = annotation_df.sort_values('slice_id').reset_index(drop=True)
        
        # Save to both CSV (for easy editing) and parquet (for later use)
        # For CSV: Remove embedding and thumbnail columns as they can't be saved to CSV
        csv_df = annotation_df.copy()
        if 'embedding' in csv_df.columns:
            csv_df = csv_df.drop('embedding', axis=1)
            print("ℹ️  Removed 'embedding' column from CSV (not suitable for CSV format)")
        if 'thumbnail' in csv_df.columns:
            csv_df = csv_df.drop('thumbnail', axis=1)
            print("ℹ️  Removed 'thumbnail' column from CSV (binary data not suitable for CSV)")
        
        csv_df.to_csv(output_csv_file, index=False)
        
        print(f"✅ Created annotation template CSV: {output_csv_file}")
        print(f"📊 Template contains {len(annotation_df)} videos to annotate")
        
        # Show sample of the template (key columns)
        print(f"\n📋 Sample annotation template (key columns):")
        display_cols = ['slice_id', 'video_path', 'gif_path', 'pv_object_type', 'pv_actor_behavior', 'pv_spatial_relation', 'ego_behavior', 'scene_type']
        available_display_cols = [col for col in display_cols if col in annotation_df.columns]
        print(annotation_df[available_display_cols].head(3))
        
        print(f"\n📊 All columns in the annotation file:")
        print(f"Total columns: {len(annotation_df.columns)}")
        print(f"Columns: {list(annotation_df.columns)}")
        
        # Print instructions
        print_annotation_instructions(output_csv_file)
        
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
   pv_object_type: "pedestrian"
   pv_actor_behavior: "entering ego path"
   pv_spatial_relation: "corridor front"
   ego_behavior: "proceeding straight"
   scene_type: "intersection, daytime"
    """)
    
    print("="*80)

if __name__ == "__main__":
    # Create the annotation template
    result = create_annotation_csv()
    
    if result is not None:
        
        print(f"\n🎉 Ready for annotation!")
        print(f"📝 Edit the CSV file to add your scene descriptions")
        print(f"🔍 This will be used for recall@5 evaluation")
