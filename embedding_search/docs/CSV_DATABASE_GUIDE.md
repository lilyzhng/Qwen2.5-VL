# Video Index Database Guide

## Overview

The embedding search system supports building databases from video index files (Parquet or CSV), which provides better control over which videos to include and ensures consistent absolute paths.

## Index File Format

The index file (Parquet or CSV) must have the following columns:
- `video_name`: Display name of the video (e.g., "car2cyclist_1.mp4")
- `sensor_video_file`: Absolute path to the video file
- `category`: Optional category for organizing videos (e.g., "video_database", "user_input")

Example:
```csv
video_name,sensor_video_file,category
car2car_1.mp4,/path/to/videos/car2car_1.mp4,video_database
car2cyclist_1.mp4,/path/to/videos/car2cyclist_1.mp4,video_database
```

## Default Index File

The system uses `data/video_index.parquet` by default. This file is automatically loaded when building the database. Both Parquet and CSV formats are supported.

## Creating a Video Index CSV

```python
import pandas as pd
from pathlib import Path

# Scan for video files
video_dir = Path('data/videos/video_database')
video_files = []
for ext in ['.mp4', '.avi', '.mov']:
    video_files.extend(video_dir.glob(f'*{ext}'))

# Create DataFrame
data = []
for video_file in sorted(video_files):
    data.append({
        'video_name': video_file.name,
        'sensor_video_file': str(video_file.absolute()),
        'category': 'video_database'
    })

# Save to Parquet (recommended) or CSV
df = pd.DataFrame(data)
df.to_parquet('data/video_index.parquet', index=False)
# Or: df.to_csv('data/video_index.csv', index=False)
```

## Building Database with Index File

The database will automatically use the index file if configured:

```bash
python scripts/main.py build --force
```

The system will:
1. Load video paths from the index file (Parquet or CSV)
2. Extract embeddings for each video
3. Store absolute paths in the database
4. Enable proper video preview in the UI

## Benefits

- **Consistent Paths**: Always uses absolute paths, avoiding path resolution issues
- **Selective Processing**: Only process videos listed in the index file
- **Metadata Support**: Can include additional columns for future features
- **Efficient Storage**: Parquet format is more efficient than CSV for large datasets
- **Portable**: Index files can be version controlled and shared
