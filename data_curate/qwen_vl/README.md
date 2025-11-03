# Qwen3-VL Video Judgment Pipeline

Distributed video judgment pipeline for autonomous driving using Qwen3-VL-2B-Instruct. This module provides both standalone CLI inference and a Ray-based distributed pipeline for large-scale video judgment.

## Features

- ✅ **Distributed Processing**: Ray-based pipeline for large-scale video judgment
- ✅ **Binary Judgments**: Yes/No answers for queries like "Is there a pedestrian crossing?"
- ✅ **Standalone CLI**: Local inference for testing and development
- ✅ **Fast**: Optional Flash Attention 2 support with GPU acceleration
- ✅ **LakeFS Integration**: Reads from and writes to LakeFS parquet datasets
- ✅ **Multi-Camera**: Processes multiple camera views simultaneously

## Dependencies

Core dependencies:
```bash
pip install -r requirements.txt
# or manually:
pip install transformers torch torchvision pillow pyyaml
```

Optional for faster inference:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Command Line

```bash
# Identify objects in a driving scene
python infer.py \
    -m Qwen/Qwen2.5-VL-7B-Instruct \
    -i dashcam_image.jpg \
    -p "Identify all objects in this image."

# Analyze temporal sequence (multiple frames)
python infer.py \
    -m Qwen/Qwen2.5-VL-7B-Instruct \
    -i frame1.jpg -i frame2.jpg -i frame3.jpg \
    -p "Analyze lane changes and cut-in behaviors"

# With custom system prompt
python infer.py \
    -m Qwen/Qwen2.5-VL-7B-Instruct \
    -i image.jpg \
    -p "Identify all objects" \
    --system-prompt "You are an expert autonomous driving systems analyst."

# With Flash Attention (faster)
python infer.py \
    -m Qwen/Qwen2.5-VL-7B-Instruct \
    --flash-attn \
    -i dashcam.jpg \
    -p "Identify all objects in this image."
```

### Python API

```python
from infer import load_model, build_chat_messages, inference, load_prompts

# Load model
model, processor = load_model("Qwen/Qwen2.5-VL-7B-Instruct")

# Load prompts from prompts.yaml
prompts = load_prompts()
system_prompt = prompts['system_prompts']['role']
user_prompt = prompts['user_prompts']['judgments']

# Build chat messages
messages = build_chat_messages(
    media_paths="dashcam_image.jpg",
    prompt=user_prompt,
    media_type="image",
    system_prompt=system_prompt
)

# Run inference
output = inference(model, processor, messages)
print(output)
```

## File Structure

```
qwen_vl/
├── __init__.py             # Package initialization
├── config.py               # Configuration for distributed pipeline
├── infer.py                # Inference engine + CLI tool
├── generate_judgements.py  # Main distributed pipeline orchestrator
├── asset.py                # Dagster asset definition
├── stage.py                # Stage definition and output schema
├── prompts.yaml            # Prompt configuration
├── examples.ipynb          # Example notebook
├── README.md               # This file
└── requirements.txt        # Dependencies
```

## Architecture Overview

This module implements a two-stage video analysis pipeline:

**Stage 1: Cosmos Embedding** (separate module)
- Generates 768-dim embeddings for all video segments
- Enables fast semantic search and top-k retrieval

**Stage 2: QwenVL Judgment** (this module)
- Takes top-k video candidates from Stage 1
- Applies Qwen3-VL-2B-Instruct to judge each video against queries
- Returns binary Yes/No judgments with structured metadata

## Distributed Pipeline Usage

### Configuration

The pipeline configuration automatically loads judgment queries from `prompts.yaml`:

```python
class QwenVLJudgeConfig(BaseStageConfigV2):
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    judgements: List[str] = None  # Auto-loaded from prompts.yaml
    batch_size: int = 8
    concurrency: int = 8
    num_gpus_per_judge_actor: float = 0.5  # QwenVL requires GPU
    segment_desired_fps: float = 1.0
    process_camera_names_csv: str = "CAMERA_FRONT_WIDE,CAMERA_FRONT_NARROW,..."
```

Edit `prompts.yaml` to change the judgment query:

```yaml
system_prompts:
  role: "You are an expert autonomous driving systems analyst."

user_prompts:
  judgments: "Is there a pedestrian crossing the street?"
```

**Note**: Unlike embedding models, QwenVL requires GPU and will not run on CPU. The model automatically uses FP16 precision for efficient GPU inference.

### Running the Pipeline

```bash
# Initialize Ray and run the pipeline
python generate_judgements.py
```

The pipeline will:
1. Load video log slices from LakeFS
2. Spawn distributed Ray workers with GPU allocation
3. Process each video segment with each query
4. Save judgments to LakeFS as parquet files

### Output Schema

Parquet files contain:
- `row_id` (str): Unique identifier for video segment
- `identifiers` (struct): Metadata linking to source data
- `sensor_name` (str): Camera identifier (e.g., CAMERA_FRONT_WIDE)
- `query` (str): The judgment query text
- `judgment` (bool): True = Yes, False = No
- `logapps_metadata` (struct): Additional logging metadata
- `timestamp` (timestamp): Processing timestamp

### Adding Custom Queries

Queries are managed in `prompts.yaml` under `user_prompts.judgments`:

```yaml
# prompts.yaml
system_prompts:
  role: "You are an expert autonomous driving systems analyst."

user_prompts:
  judgments: "Is there a pedestrian crossing the street?"
```

The config automatically loads from YAML:

```python
# config.py - automatically loads from prompts.yaml
class QwenVLJudgeConfig(BaseStageConfigV2):
    judgements: List[str] = None  # Auto-loaded from YAML
    
# Or override directly in config if needed:
config = QwenVLJudgeConfig(
    judgements=["Is there a pedestrian crossing the street?"]
)
```

## Standalone CLI Usage

## Prompt Management

All prompts are managed in `prompts.yaml`. Currently configured for autonomous driving:

**System Prompt:**
- `default`: "You are an expert autonomous driving systems analyst."

**User Prompts:**
- `identify_objects`: "Identify all objects in this image."
- `temporal_analysis`: Multi-frame analysis for lane changes and cut-in detection

To modify prompts, simply edit `prompts.yaml` - no code changes needed!

## Python API for QwenVLJudge Class

For programmatic use in custom pipelines:

```python
from infer import QwenVLJudge
import numpy as np

# Initialize the judge
judge = QwenVLJudge(
    model_path="Qwen/Qwen3-VL-2B-Instruct",
    load_model_from_lakefs=False,
    use_flash_attn=True,
    max_new_tokens=128
)

# Prepare video frames (list of HWC uint8 numpy arrays)
frames = [...]  # Your video frames

# Judge against a query
query = "Is there a pedestrian crossing the street?"
result = judge.judge_video(frames, query)
print(f"Judgment: {'Yes' if result else 'No'}")

# Or use the __call__ method
result = judge(frames, query)
```

## Model Caching Strategy

The module uses intelligent caching to avoid redundant downloads:

**HuggingFace Mode** (default):
- Cache location: `~/.cache/huggingface/hub/` (persistent)
- Automatic caching by HuggingFace Transformers
- Shared across all processes on the same machine

**LakeFS Mode** (for internal deployments):
- Cache location: `/tmp/qwenvl_models_cache/` (configurable)
- Process-safe locks prevent duplicate downloads
- Completion markers track successful downloads

**Ray Distributed Mode**:
- Sets `HF_HOME=/tmp/qwenvl_hf_cache` for worker consistency
- Each worker checks cache before downloading
- First worker downloads, others wait and reuse

## Performance Considerations

**GPU Requirement** (Important):
- QwenVL **requires GPU** and will not run on CPU
- Model automatically uses FP16 precision for efficient inference
- Qwen3-VL-2B-Instruct requires ~4-6GB VRAM with FP16

**GPU Allocation**:
- `num_gpus_per_judge_actor: 0.5` allows 2 workers per GPU (recommended)
- `num_gpus_per_judge_actor: 1.0` dedicates one GPU per worker
- Adjust based on available GPU memory

**Concurrency Tuning**:
- Default `concurrency: 8` for balanced throughput
- Increase for more parallelism (monitor GPU memory)
- Decrease if experiencing OOM errors

**Frame Sampling**:
- `segment_desired_fps: 1.0` samples 1 frame per second
- Higher FPS provides more temporal detail but increases processing time
- Balance based on your use case requirements

## Integration with Cosmos Pipeline

The QwenVL judgment pipeline is designed as Stage 2 after Cosmos embedding (Stage 1):

```
[Video Logs] 
    ↓
[Cosmos Embeddings] → Generate 768-dim embeddings for all videos
    ↓
[Semantic Search] → Retrieve top-k candidates for each query
    ↓
[QwenVL Judgment] → Binary Yes/No judgment on candidates
    ↓
[Filtered Results] → Only videos that passed judgment
```

This two-stage approach combines:
- **Speed**: Cosmos embeddings enable fast semantic search
- **Accuracy**: QwenVL provides detailed visual understanding for final judgment

## Notes

- Function naming: `prepare_messages()` → `build_chat_messages()`, `run_inference()` → `inference()`
- Frame sequences are processed as multiple images with `media_type="image"`
- See `examples.ipynb` for complete autonomous driving examples
- The QwenVLJudge class is optimized for distributed Ray processing
- Model uses FP16 by default for 2x speedup on modern GPUs

