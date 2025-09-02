#!/bin/bash
# Complete Video Processing Workflow
# Converts frame zip files to MP4 segments, generates GIFs, and launches annotation app
#
# Usage: bash complete_workflow.sh [FPS]
# Example: bash complete_workflow.sh 10.0  # Use 10 FPS instead of default 30

set -e  # Exit on any error

# Configuration
FPS=${1:-30.0}  # Default to 30.0 FPS if not provided

echo "🎬 Starting Complete Video Processing Workflow..."
echo "=================================================="
echo "📋 Configuration:"
echo "  - FPS: $FPS"
echo ""

# Check if input file exists
INPUT_PARQUET="data/unified_input_path.parquet"
if [ ! -f "$INPUT_PARQUET" ]; then
    echo "❌ Error: Input parquet file not found: $INPUT_PARQUET"
    echo "Please ensure the file exists with sensor_frame_zip column"
    exit 1
fi

# Step 1: Generate MP4 segments from frame zip
echo ""
echo "📹 Step 1: Converting frames to MP4 segments..."
echo "Input: $INPUT_PARQUET"
echo "Output Directory: data/video_segments/"
echo ""

python -c "
import sys
sys.path.append('.')
from core.preprocessor import preprocess_clips_cli

print('Processing frame zip files to MP4 segments...')
processed_df = preprocess_clips_cli(
    'data/unified_input_path.parquet',
    'data/unified_input_path.parquet',  # Update same file
    target_duration=20.0,
    create_video_segments=True,
    output_video_dir='data/video_segments',
    fps=$FPS,  # Configurable FPS for video generation from frames
    resolution=(448, 448)  # Height=448 (width scales proportionally) for embedding model compatibility
)
print('✅ Step 1 completed: MP4 segments generated')
"

# Check if video files were created
if [ ! -d "data/video_segments" ] || [ -z "$(ls -A data/video_segments)" ]; then
    echo "❌ Error: No video segments were created"
    exit 1
fi

echo "📊 Generated video segments:"
ls -lh data/video_segments/*.mp4

# Step 2: Generate GIFs from video segments
echo ""
echo "🎨 Step 2: Creating GIF previews..."
echo "Input: data/unified_input_path.parquet"
echo "Output Directory: data/gifs/"
echo ""

python data/annotation/create_gifs.py \
    --input-parquet data/unified_input_path.parquet \
    --gif-dir data/gifs \
    --overwrite

echo "✅ Step 2 completed: GIF previews generated"

# Check if GIF files were created
if [ ! -d "data/gifs" ]; then
    echo "❌ Error: GIF directory was not created"
    exit 1
fi

echo "📊 Generated GIF files:"
find data/gifs -name "*87tB8RTtetg_short*.gif" -exec ls -lh {} \;

# Step 3: Prepare annotation app
echo ""
echo "🚀 Step 3: Preparing annotation app..."

# Update annotation app to use the processed data
echo "Updating annotation app configuration..."
python -c "
import os
import sys

# Check if the final parquet file exists
final_parquet = 'data/unified_input_path.parquet'
if not os.path.exists(final_parquet):
    print(f'❌ Error: Final parquet file not found: {final_parquet}')
    sys.exit(1)

print(f'✅ Final data file ready: {final_parquet}')

# Show sample data
import pandas as pd
df = pd.read_parquet(final_parquet)
print(f'📊 Dataset contains {len(df)} video segments')
print('Sample entries:')
for i, row in df.head(3).iterrows():
    print(f'  - {row[\"slice_id\"]}: {row[\"span_start\"]}s-{row[\"span_end\"]}s')
"

echo ""
echo "🎉 Workflow completed successfully!"
echo "=================================================="
echo ""
echo "📁 Generated Files:"
echo "  - Video segments: data/video_segments/"
echo "  - GIF previews: data/gifs/"  
echo "  - Updated dataset: data/unified_input_path.parquet (now contains video and gif paths)"
echo ""
echo "🚀 Next Steps:"
echo "  1. Launch annotation app: bash data/annotation/launch_annotation_app.sh"
echo "  2. Access app at: http://localhost:8502"
echo "  3. Annotate video segments with ground truth categories"
echo "  4. Annotations will be saved to: data/annotation/unified_annotation.csv"
echo ""
echo "📖 For more details, see: WORKFLOW_GUIDE.md"
