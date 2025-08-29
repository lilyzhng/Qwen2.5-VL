# ALFA 0.1 Embedding Search

This system enables efficient similarity search across video databases using both video-to-video and text-to-video queries.

**Features:**
- Interactive video similarity search
- Text-to-video search
- Real-time embedding visualization
- Comprehensive recall evaluation framework

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
│   ├── evaluate.py                # Recall evaluation framework
├── interface/                     # User interfaces
│   ├── streamlit_app.py           # Web interface (ALFA 0.1)
│   ├── recall_analysis_app.py     # Recall analysis dashboard
│   ├── main.py                    # Main CLI interface
│   ├── launch_streamlit.sh        # Main app launcher
│   ├── launch_recall_analysis.sh  # Recall analysis launcher
├── data/                          # Data storage
│   ├── unified_embeddings.parquet    # video database embeddings
│   ├── unified_input_path.parquet     # video file paths
│   ├── annotation/                # Ground truth annotations
│   │   ├── video_annotation.csv   # Annotated video clips with keywords
├── tests/                         # Unit tests
│   ├── test_evaluate.py           # Evaluation framework tests
├── benchmarks/                    # Performance testing
│   ├── inference_benchmark.py     # Model inference benchmarks
├── run_recall_evaluation.py       # Recall evaluation runner script
├── run_keyword_evaluation.py      # Keyword-specific evaluation script
└── requirements.txt               # Python dependencies
```

## Quick Start
```
# Install dependencies
pip install -r requirements.txt
```

## Storage Backends

The system supports two storage backends for embeddings:

### Local Storage (Default)
Stores embeddings in local Parquet files:
- `data/unified_embeddings.parquet` - Video embeddings
- `data/unified_input_path.parquet` - Video file paths

### LakeFS Storage
Store embeddings in a LakeFS repository for versioned data management.

#### Setup LakeFS
1. Install LakeFS dependencies (already included in requirements.txt):
   ```bash
   pip install lakefs-spec
   ```

2. Configure LakeFS credentials in `~/.lakectl.yaml`:
   ```yaml
   # Create or update ~/.lakectl.yaml
   credentials:
       access_key_id: your-access-key-id
       secret_access_key: your-secret-access-key
   server:
       endpoint_url: http://localhost:8000/api/v1
   ```

3. Update configuration to enable LakeFS:
   ```yaml
   # In core/config.yaml or your custom config
   use_lakefs: true
   lakefs_repository: "your-repo-name"
   lakefs_branch: "main"
   lakefs_embeddings_path: "data/unified_embeddings.parquet"
   lakefs_input_path: "data/unified_input_path.parquet"
   # Credentials are automatically read from ~/.lakectl.yaml
   ```

#### LakeFS Features
- **Automatic Sync**: Embeddings are saved to both local cache and LakeFS
- **Fallback**: Falls back to local storage if LakeFS is unavailable
- **Version Control**: Leverage LakeFS branching and versioning for embeddings
- **Collaboration**: Share embeddings across teams via LakeFS repository

### Pushing Parquet Files to LakeFS

Once LakeFS is configured, the system automatically manages parquet files for you:

```bash
# Build embeddings - automatically saves to LakeFS
python interface/main.py build --input-path data/unified_input_path.parquet

# Any search operations also trigger automatic saves to LakeFS
python interface/main.py search --query-video data/videos/user_input/car2cyclist_2.mp4 --top-k 5
```

You can also manually push existing embeddings:

```python
from core.config import VideoRetrievalConfig
from core.search import VideoSearchEngine

# Initialize with LakeFS enabled
config = VideoRetrievalConfig()
config.use_lakefs = True
config.lakefs_repository = "embedding-search"

# Create search engine (loads existing embeddings)
search_engine = VideoSearchEngine(config=config)

# Manually save to LakeFS
search_engine.database.save()
print("✅ Embeddings pushed to LakeFS!")
```

#### Verify Upload
Check your files in LakeFS web UI at http://localhost:8000 or programmatically:

```python
from lakefs_spec import LakeFSFileSystem

fs = LakeFSFileSystem()
repo_path = "embedding-search/main/data/unified_embeddings.parquet"

if fs.exists(repo_path):
    info = fs.info(repo_path)
    print(f"✅ File uploaded: {info['size'] / (1024*1024):.2f} MB")
else:
    print("❌ File not found in LakeFS")
```

## Commands
### 1. Generate Unified Embeddings Database
Build the unified video database for fast similarity search:

```bash
# Build from unified parquet input file (required)
python interface/main.py build --input-path data/unified_input_path.parquet
```

**Requirements:**
- Input must be a parquet file with columns: `slice_id`, `sensor_video_file`, `category`, `gif_file`
- All video files referenced in the parquet must exist on disk

### 2. Launch Streamlit Apps

#### Main Search Interface (ALFA 0.1)
```bash
./interface/launch_streamlit.sh
```
Available at: `http://localhost:8501`

#### Recall Analysis Dashboard
Interactive dashboard for analyzing recall performance and triaging low-performing scenarios:
```bash
./interface/launch_recall_analysis.sh
```
Available at: `http://localhost:8502`


### 4. Performance Benchmarks (Optional)

#### Model Inference Benchmark
Test embedding model performance:

```bash
python benchmarks/inference_benchmark.py --video-dir data/videos/video_database/ --max-videos 5

python benchmarks/cosmos_cpu_benchmark.py --max-videos 2 --output my_results.json
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

## Recall Evaluation

The system includes a comprehensive recall evaluation framework to measure search performance using annotated ground truth data.

### Quick Evaluation
Run the complete recall evaluation:
```bash
python run_recall_evaluation.py
```

This will evaluate:
- **Video-to-Video Recall**: Using each annotated video as a query
- **Text-to-Video Recall**: Using keywords as text queries  
- **Category-Specific Recall**: Performance for specific object types, behaviors, spatial relations, and scene types

### Keyword-Specific Evaluation
Evaluate specific keywords or categories separately:

#### List Available Keywords
```bash
python run_keyword_evaluation.py --list-keywords
```

#### Text-to-Video Evaluation for Specific Keywords
```bash
# Evaluate urban and highway scenarios
python run_keyword_evaluation.py --mode text --keywords urban highway

# Evaluate specific actor behaviors
python run_keyword_evaluation.py --mode text --keywords car2pedestrian car2cyclist
```

#### Video-to-Video Evaluation for Specific Keywords
```bash
# Evaluate videos with intersection scenarios
python run_keyword_evaluation.py --mode video --keywords intersection crosswalk

# Evaluate night driving videos
python run_keyword_evaluation.py --mode video --keywords night
```

#### Combined Evaluation
```bash
# Run both text-to-video and video-to-video for specific keywords
python run_keyword_evaluation.py --mode both --keywords car2pedestrian urban
```

#### Custom K Values
```bash
# Evaluate with custom Recall@K values
python run_keyword_evaluation.py --keywords urban --k-values 1 2 3 5 10
```

### Evaluation Metrics
The framework measures **Recall@K** for K=1,3,5:
- **Recall@1**: Percentage of relevant videos found in top-1 result
- **Recall@3**: Percentage of relevant videos found in top-3 results
- **Recall@5**: Percentage of relevant videos found in top-5 results

### Ground Truth Data
The evaluation uses annotated data from:
```
data/annotation/video_annotation.csv
```

This file contains 29 annotated video clips with keywords describing:
- **Interaction types**: car2pedestrian, car2cyclist, car2motorcyclist, car2car
- **Environment types**: urban, highway, intersection, crosswalk
- **Conditions**: night, daytime, rain, parking, tunnel
- **Actions**: turning_left, turning_right, lane_merge

### Programmatic Usage
```python
from core.evaluate import run_recall_evaluation, print_recall_results

# Run evaluation
results = run_recall_evaluation()

# Print formatted results
print_recall_results(results)

# Access specific metrics
video_to_video_recall = results['video_to_video']['average_recalls']
text_to_video_recall = results['text_to_video']['average_recalls']
category_specific = results['category_specific']
```

### Performance Optimization
The evaluation framework uses **pre-computed embeddings** from `unified_embeddings.parquet` for fast execution, avoiding the need to extract embeddings on-the-fly during evaluation.

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run evaluation framework tests:
```bash
python tests/test_evaluate.py
```

