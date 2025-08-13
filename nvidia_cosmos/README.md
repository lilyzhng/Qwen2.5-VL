# NVIDIA Cosmos Video Retrieval System

A video retrieval system using NVIDIA Cosmos-Embed1-448p model for video embedding search. This system allows you to find similar videos in a database by providing either a query video or text description.

## Features

- **Video-to-Video Search**: Find similar videos by providing a query video
- **Text-to-Video Search**: Search videos using natural language descriptions
- **Efficient Embedding Storage**: Pre-compute and store video embeddings for fast retrieval
- **Visualization**: Generate visual comparisons of search results
- **Export Functionality**: Export search results with videos and metadata

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Build the Video Database

First, build the embeddings database from your video collection:

```bash
python main.py build --video-dir /Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database
```

### 2. Run a Demo

Run the demo to see the system in action:

```bash
python main.py demo
```

### 3. Search by Video

Search for videos similar to a query video:

```bash
python main.py search --query-video /Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/user_input/car2cyclist_2.mp4 --top-k 5 --visualize
```

### 4. Search by Text

Search for videos using a text description:

```bash
python main.py search --query-text "car approaching cyclist" --top-k 5 --visualize
```

## Usage Examples

### Building the Database

```bash
# Build database from scratch
python main.py build --video-dir /path/to/videos --force-rebuild

# Update existing database with new videos
python main.py build --video-dir /path/to/videos
```

### Searching Videos

```bash
# Video-to-video search with visualization
python main.py search --query-video /path/to/query.mp4 --top-k 5 --visualize

# Text-to-video search
python main.py search --query-text "car overtaking on highway" --top-k 3

# Export search results
python main.py search --query-video /path/to/query.mp4 --export-dir ./results
```

### Database Information

```bash
# Show database statistics
python main.py info
```

## Python API Usage

```python
from video_search import VideoSearchEngine

# Initialize search engine
search_engine = VideoSearchEngine()

# Build database
search_engine.build_database("/path/to/video/database")

# Search by video
results = search_engine.search_by_video("/path/to/query/video.mp4", top_k=5)

# Search by text
results = search_engine.search_by_text("car approaching pedestrian", top_k=5)

# Print results
for result in results:
    print(f"Rank {result['rank']}: {result['video_name']} (Score: {result['similarity_score']:.4f})")
```

## Project Structure

```
nvidia_cosmos/
├── main.py                 # Main CLI interface
├── video_embedder.py       # Video embedding extraction using Cosmos model
├── video_database.py       # Database management for embeddings
├── video_search.py         # Search engine implementation
├── video_visualizer.py     # Results visualization
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── videos/
    ├── user_input/        # Query videos
    └── video_database/    # Video collection for database
```

## How It Works

1. **Embedding Extraction**: The system uses NVIDIA Cosmos-Embed1-448p to extract 768-dimensional embeddings from videos
2. **Database Storage**: Video embeddings are computed once and stored in a pickle file for efficient retrieval
3. **Similarity Search**: Cosine similarity is used to find the most similar videos to a query
4. **Multi-modal Search**: Supports both video-to-video and text-to-video search using the same embedding space

## Notes

- The model runs best on CUDA-enabled GPUs with bfloat16 precision
- First run will download the model from HuggingFace (~4.8GB)
- Video processing extracts 8 frames uniformly sampled across the video
- The system supports various video formats: mp4, avi, mov, mkv, webm

## License

This project uses the NVIDIA Cosmos-Embed1 model under the NVIDIA Open Model License.
