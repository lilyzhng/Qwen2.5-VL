# Video Processing Workflow Guide

This guide describes the complete workflow for processing video frames into segments, generating GIFs, and preparing data for annotation.

## Overview

The workflow consists of three main steps:
1. **Frame-to-Video Conversion**: Convert frame zip files to 20-second MP4 segments
2. **GIF Generation**: Create GIF previews from video segments  
3. **Annotation Setup**: Prepare data for the annotation app

## Prerequisites

- FFmpeg installed and available in PATH
- Python environment with required dependencies
- Frame zip files with the format: `data/filename.zip#subfolder/frames`

## Step 1: Generate MP4 Clips from Frame Zip

Convert frame zip files to 20-second MP4 video segments.

### Input Format
Your parquet file should contain a `sensor_frame_zip` column with paths like:
```
data/87tB8RTtetg_short.zip#87tB8RTtetg_short/camera_front_wide
```

### Command
```bash
# Activate environment
conda activate qwen_env

# Run preprocessor with dedicated output directory
python -c "
import sys
sys.path.append('.')
from core.preprocessor import preprocess_clips_cli

processed_df = preprocess_clips_cli(
    'data/unified_input_path.parquet',                    # Input parquet with sensor_frame_zip
    'data/unified_input_path.parquet',                    # Update same file with sensor_video_file
    target_duration=20.0,                                 # 20-second segments
    create_video_segments=True,                           # Actually create video files
    output_video_dir='data/video_segments'                # Dedicated output directory
)
"
```

### Alternative CLI Usage
```bash
python core/preprocessor.py data/unified_input_path.parquet \
    --output data/unified_input_path.parquet \
    --target-duration 20.0 \
    --output-video-dir data/video_segments
```

### Output
- **Video files**: `data/video_segments/filename_0000_0020.mp4`, `filename_0020_0040.mp4`, etc.
- **Updated parquet**: Contains `sensor_video_file` column pointing to generated MP4 files
- **Statistics**: Duration distribution and segment count

## Step 2: Generate GIFs from Video Segments

Create GIF previews from the generated video segments for easier annotation.

### Command
```bash
python data/annotation/create_gifs.py \
    --input-parquet data/unified_input_path.parquet \
    --gif-dir data/gifs \
    --overwrite
```

### Parameters
- `--input-parquet`: Parquet file with `sensor_video_file` column
- `--gif-dir`: Directory to save generated GIFs
- `--overwrite`: Overwrite existing GIF files
- `--gif-width 320`: GIF width in pixels (default: 320)
- `--gif-fps 5`: GIF frame rate (default: 5)

### Output
- **GIF files**: `data/gifs/filename_0000_0020.gif`, etc.
- **Updated parquet**: Same file now includes `gif_file` column

## Step 3: Launch Annotation App

Use the annotation app to manually annotate video segments with ground truth categories.

### Setup
Launch the annotation app:
```bash
# The app will automatically use: data/unified_input_path.parquet
bash data/annotation/launch_annotation_app.sh
```

### Access
- **URL**: http://localhost:8502
- **Data**: Video segments with GIF previews
- **Annotations**: Saved to `data/annotation/video_annotation.csv`

## Complete Workflow Script

Here's a complete script that runs all three steps:

```bash
#!/bin/bash
# complete_workflow.sh

echo "🎬 Starting Video Processing Workflow..."

# Step 1: Generate MP4 segments from frame zip
echo "📹 Step 1: Converting frames to MP4 segments..."
conda activate qwen_env
python -c "
import sys
sys.path.append('.')
from core.preprocessor import preprocess_clips_cli

processed_df = preprocess_clips_cli(
    'data/unified_input_path.parquet',
    'data/unified_input_path.parquet',  # Update same file
    target_duration=20.0,
    create_video_segments=True,
    output_video_dir='data/video_segments'
)
print('✅ Step 1 completed: MP4 segments generated')
"

# Step 2: Generate GIFs from video segments
echo "🎨 Step 2: Creating GIF previews..."
python data/annotation/create_gifs.py \
    --input-parquet data/unified_input_path.parquet \
    --gif-dir data/gifs \
    --overwrite

echo "✅ Step 2 completed: GIF previews generated"

# Step 3: Launch annotation app
echo "🚀 Step 3: Launching annotation app..."
echo "📝 Access the app at: http://localhost:8502"
bash data/annotation/launch_annotation_app.sh
```

## File Structure After Processing

```
data/
├── unified_input_path.parquet                    # Single file updated throughout workflow
│                                                 # (starts with sensor_frame_zip, ends with video+gif paths)
├── video_segments/                              # Generated MP4 files
│   ├── 87tB8RTtetg_short_0000_0020.mp4
│   ├── 87tB8RTtetg_short_0020_0040.mp4
│   ├── 87tB8RTtetg_short_0040_0060.mp4
│   └── 87tB8RTtetg_short_0060_0069.mp4
├── gifs/                                        # Generated GIF files
│   ├── 87tB8RTtetg_short_0000_0020.gif
│   ├── 87tB8RTtetg_short_0020_0040.gif
│   ├── 87tB8RTtetg_short_0040_0060.gif
│   └── 87tB8RTtetg_short_0060_0069.gif
└── annotation/
    └── video_annotation.csv                     # Manual annotations
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   ```

2. **Frame zip format**
   - Ensure zip contains frames in subfolder: `zipfile.zip#subfolder/frames`
   - Supported formats: jpg, jpeg, png, bmp, tiff, tif

3. **Memory issues with large videos**
   - Process smaller batches
   - Use `--no-video-segments` for metadata-only processing first

4. **Path issues in annotation app**
   - Ensure all paths in parquet are absolute paths
   - Check that video/gif files exist at specified locations

### Performance Tips

- **Parallel processing**: Process multiple videos simultaneously
- **SSD storage**: Use fast storage for temporary frame extraction
- **Batch size**: Adjust frame batch size for memory optimization
- **Quality settings**: Adjust CRF value (23 default) for size/quality balance

## Advanced Configuration

### Custom Preprocessor Settings

```python
from core.preprocessor import ClipPreprocessor

# Custom configuration
preprocessor = ClipPreprocessor(
    target_duration=15.0,           # 15-second segments
    min_segment_duration=5.0,       # Minimum final segment
    fps=25.0,                       # 25 FPS for frame timing
    output_video_dir='custom/path'  # Custom output directory
)
```

### Custom GIF Settings

```bash
python data/annotation/create_gifs.py \
    --input-parquet data/input.parquet \
    --gif-dir data/custom_gifs \
    --gif-width 480 \
    --gif-fps 10 \
    --gif-duration 15
```
