# ALFA 0.1 Embedding Search

This system enables efficient similarity search across video databases using both video-to-video and text-to-video queries.

**Features:**
- Interactive video similarity search
- Text-to-video search
- Real-time embedding visualization

## Codebase Structure

```
embedding_search/
├── core/                          # Core system components
│   ├── config.py                  # Configuration management
│   ├── config.yaml                # Default configuration
│   ├── search.py                  # Main search engine
│   ├── embedder.py                # Video/text embedding extraction
│   ├── database.py                # Unified Parquet storage system
│   ├── faiss_backend.py           # FAISS-based search backend
│   ├── visualizer.py              # Result visualization
├── interface/                     # User interfaces
│   ├── streamlit_app.py           # Web interface (ALFA 0.1)
│   ├── main.py                    # Main CLI interface
│   ├── launch_streamlit.sh        # Streamlit launcher
│   └── mock/                      # Mock interfaces for testing
├── benchmarks/                    # Performance testing
│   ├── performance_test.py        # System benchmarks
│   ├── inference_benchmark.py     # Model inference benchmarks
│   └── example_benchmark.py       # Benchmark examples
├── data/                          # Data storage
│   ├── main_embeddings.parquet    # Main video database embeddings
│   ├── query_embeddings.parquet   # Query video embeddings
│   ├── main_file_path.parquet     # Main video file paths
│   ├── query_file_path.parquet    # Query video file paths
│   └── videos/                    # Video files
│       ├── video_database/        # Reference video collection
│       └── user_input/             # Query videos
├── tests/                         # Unit tests
└── requirements.txt               # Python dependencies
```

## Quick Start
```
# Install dependencies
pip install -r requirements.txt
```

## Commands
### 1. Generate Main Embeddings Database
Build the main video database from your reference video collection:

```bash
# Using video file list
python interface/main.py build-main --data-path-file data/main_file_path.parquet
```
### 2. Generate Query Embeddings Database
Build the query video database for fast similarity search:

```bash
# Using query file list
python interface/main.py build-query --data-path-file data/query_file_path.parquet
```

### 3. Launch Streamlit App
ALFA 0.1 -  Similarity Search Interface:

```bash
./interface/launch_streamlit.sh
```

The app will be available at: `http://localhost:8501`

### 4. Performance Benchmarks (Optional)

#### Model Inference Benchmark
Test embedding model performance:

```bash
python benchmarks/inference_benchmark.py --video-dir data/videos/video_database/ --max-videos 5
```

## Search Commands

### Search by Video File
```bash
python interface/main.py search --query-video data/videos/user_input/car2cyclist_2.mp4 --top-k 5
```

### Search by Pre-computed Query
```bash
python interface/main.py search --query-filename car2cyclist_2.mp4 --top-k 5
```

### Search by Text Description
```bash
python interface/main.py search --query-text "car approaching cyclist" --top-k 5
```

### Search with Visualization
```bash
python interface/main.py search --query-video data/videos/user_input/car2cyclist_2.mp4 --visualize
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```
