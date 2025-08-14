# Installation Guide for NVIDIA Cosmos Embed 1 Implementation

## Prerequisites

1. **Python Virtual Environment**: Use the existing `qwen_venv` in the project root
2. **NVIDIA Cosmos Embed 1 Model**: Located in `cookbooks/nvidia_cosmos_embed_1/`

## Basic Installation (Text Embeddings Only)

### Step 1: Activate Virtual Environment
```bash
cd /Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search
source ../qwen_venv/bin/activate
```

### Step 2: Install Core Dependencies
```bash
pip install torch torchvision transformers>=4.51.3 einops numpy
```

### Step 3: Test Basic Functionality
```bash
python -c "
from config import VideoRetrievalConfig
from embedder import CosmosVideoEmbedder

config = VideoRetrievalConfig()
embedder = CosmosVideoEmbedder(config)

# Test text embedding
text = 'a person riding a motorcycle'
embedding = embedder.extract_text_embedding(text)
print(f'✓ Text embedding successful! Shape: {embedding.shape}')
"
```

## Full Installation (Text + Video Embeddings)

### Additional Dependencies

For video processing capabilities, you'll need:

```bash
# Install decord for video processing
# Note: This might require conda or special installation
pip install decord==0.6.0

# Install FAISS for optimized similarity search
pip install faiss-cpu>=1.7.4  # For CPU
# OR
pip install faiss-gpu>=1.7.4  # For GPU (if CUDA available)

# Install other optional dependencies
pip install scikit-learn pandas matplotlib tqdm opencv-python
```

### Complete Dependencies Install
```bash
pip install -r requirements.txt
```

## Verification

### Test Text Embeddings
```python
from config import VideoRetrievalConfig
from embedder import CosmosVideoEmbedder

config = VideoRetrievalConfig()
embedder = CosmosVideoEmbedder(config)

# Extract text embedding
text_embedding = embedder.extract_text_embedding("a car driving at night")
print(f"Text embedding shape: {text_embedding.shape}")
print(f"Embedding norm: {text_embedding.dot(text_embedding):.4f}")  # Should be ~1.0
```

### Test Video Embeddings (if decord available)
```python
from pathlib import Path

# Test with a video file
video_path = Path("videos/video_database/car2car_1.mp4")
if video_path.exists():
    video_embedding = embedder.extract_video_embedding(video_path)
    print(f"Video embedding shape: {video_embedding.shape}")
```

### Test Similarity Search (if FAISS available)
```python
from search import OptimizedVideoSearchEngine

search_engine = OptimizedVideoSearchEngine(config=config)

# Build database from video directory
search_engine.build_database("videos/video_database")

# Search by text
results = search_engine.search_by_text("car approaching cyclist")
for result in results[:3]:
    print(f"Video: {result['video_name']}, Score: {result['similarity_score']:.4f}")
```

## Troubleshooting

### Common Issues

1. **"No module named 'decord'"**
   - Video processing will be disabled, but text embeddings still work
   - Install decord using conda: `conda install -c conda-forge decord`

2. **"No module named 'faiss'"**
   - Search optimization will fall back to numpy implementation
   - Install FAISS: `pip install faiss-cpu`

3. **CUDA not available**
   - Model will automatically fall back to CPU processing
   - This is normal on systems without NVIDIA GPUs

4. **Model loading errors**
   - Ensure the model path is correct in `config.py`
   - Verify transformers version >= 4.51.3

### Environment Issues

If you encounter Python environment issues:

```bash
# Create a new virtual environment
python3 -m venv embedding_search_env
source embedding_search_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision transformers>=4.51.3 einops numpy
```

## Configuration

### Model Configuration
Edit `config.py` to customize:

```python
@dataclass
class VideoRetrievalConfig:
    # Model configuration
    model_name: str = "/path/to/nvidia_cosmos_embed_1"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 4
    num_frames: int = 8
    resolution: Tuple[int, int] = (448, 448)
    
    # Database paths
    video_database_dir: str = "/path/to/video/database"
    user_input_dir: str = "/path/to/user/input"
```

### Performance Optimization

For better performance:

1. **Use GPU**: Set `device: "cuda"` if available
2. **Install FAISS**: For faster similarity search
3. **Use SSD storage**: For faster video loading
4. **Adjust batch_size**: Increase if you have more memory

## Usage Examples

### Command Line Interface
```bash
# Build database
python main.py build --video-dir videos/video_database

# Search by text
python main.py search --query-text "car approaching cyclist" --top-k 5

# Search by video
python main.py search --query-video videos/user_input/sample.mp4

# Show database info
python main.py info
```

### Python API
```python
from config import VideoRetrievalConfig
from search import OptimizedVideoSearchEngine

# Initialize
config = VideoRetrievalConfig()
engine = OptimizedVideoSearchEngine(config)

# Build database
engine.build_database("videos/video_database")

# Search
results = engine.search_by_text("motorcycle at night", top_k=3)
for result in results:
    print(f"{result['video_name']}: {result['similarity_score']:.4f}")
```

## Validation

Run the test script to validate your installation:

```bash
python test_implementation.py
```

Expected output:
```
✓ Model loaded successfully
✓ Text embedding shape: (768,)
✓ Text embedding norm: 1.0000
✓ All core components working correctly!
```
