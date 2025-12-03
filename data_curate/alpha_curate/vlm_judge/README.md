# VLM Judge for ALFA Curate

## Overview

The VLM Judge module provides vision-language model (VLM) based verification for ALFA Curate search results. It acts as a **second-stage filter** after embedding-based similarity search to reduce false positives.

## Requirements

This module requires specific versions of transformers and PyTorch:

**Minimum Versions:**
- **transformers >= 4.47.0** (for Qwen3-VL support)
- **PyTorch >= 2.3.0** (required for transformers 4.47+)

```bash
# Install both together
pip install --upgrade 'transformers>=4.47.0' 'torch>=2.3.0'
```

**Important Notes:**
- The `AutoModelForImageTextToText` class used for Qwen2-VL was added in transformers 4.45.0
- **Versions 4.45.0 and 4.45.1 have known bugs** (shape mismatch errors with Idefics2 and potentially other models)
- **PyTorch 2.3+ is required** for transformers 4.47+ (provides `register_pytree_node` API)
- Using older PyTorch (< 2.3) will cause: `AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`

See `requirements.txt` for full dependencies.

## Architecture

```
ALFA Curate Pipeline:
1. Text-to-Video Embedding Search  →  Top-K candidates
2. VLM Judge Verification          →  Filtered candidates  
3. Materialization                 →  Final selections
```

## How It Works

1. **Embedding Search**: Cosmos embeddings find candidates matching text prompts
2. **VLM Verification**: Qwen-VL model judges each candidate by analyzing video frames
3. **Confidence Filtering**: Only candidates with VLM confidence above threshold pass

## Configuration

VLM judge is configured via `VLMJudgeConfig` in `AlfaCurateConfig`:

```python
from autonomy.perception.datasets.active_learning.alfa_curate.config import AlfaCurateConfig
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.config import VLMJudgeConfig

config = AlfaCurateConfig(
    vlm_judge=VLMJudgeConfig(
        enable_vlm_judge=True,                  # Enable/disable VLM filtering
        max_candidates_for_vlm=100,             # Max candidates to judge (top K)
        vlm_model_path="Qwen/Qwen3-VL-2B-Instruct",
        vlm_confidence_threshold=0.7,           # Min confidence to pass (0.0-1.0)
        segment_desired_fps=1.0,                # Frame sampling rate
        max_frames_per_segment=8,               # Max frames per video segment
        load_model_from_lakefs=False,           # Load from HF or LakeFS
        use_flash_attn=False,                   # Use Flash Attention 2
        gpu_memory_gb=6,                        # GPU memory needed (6GB for 2B model)
        num_gpus_per_worker=0.5,                # 0.5 = share GPU with 2 workers
        gpu_type="A100",                        # GPU type to request
    )
)
```

### Example: For 32B Model

```python
config = AlfaCurateConfig(
    vlm_judge=VLMJudgeConfig(
        vlm_model_path="Qwen/Qwen3-VL-32B-Instruct",
        gpu_memory_gb=64,                       # 64GB needed for 32B model
        num_gpus_per_worker=1.0,                # Full GPU per worker
        gpu_type="A100-80GB",                   # Must use 80GB variant
    )
)
```

### Key Parameters

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|----------------|
| `enable_vlm_judge` | Enable VLM filtering | `True` | Set to `False` to disable |
| `max_candidates_for_vlm` | Top K candidates to judge | `100` | Higher = slower but more thorough |
| `vlm_confidence_threshold` | Min confidence to pass | `0.7` | 0.7-0.8 for balanced precision/recall |
| `segment_desired_fps` | Frame sampling rate | `1.0` | 1.0 FPS for good coverage |
| `max_frames_per_segment` | Max frames to send to VLM | `8` | 8-16 frames typical |
| `gpu_memory_gb` | GPU memory needed (GB) | `6` | 2B: 6GB, 7B: 14GB, 32B: 64GB |
| `num_gpus_per_worker` | GPUs per worker | `0.5` | 0.5 for small models, 1.0 for 32B+ |
| `gpu_type` | GPU type to request | `"A100"` | "A100", "A100-80GB", "V100" |

## Usage

### In ALFA Curate Pipeline

VLM judge is automatically called in `generate_alfa_curate.py`:

```python
# Stage 1: Embedding search
results = select_slices_for_scenario(config, scenario)

# Stage 2: VLM judge (inside select_slices_for_scenario)
filtered_results = apply_vlm_judge(results, scenario, config)
```

### Standalone Usage

You can also use the VLM judge independently:

```python
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer import VLMJudge
import numpy as np

# Initialize judge
judge = VLMJudge(
    model_path="Qwen/Qwen3-VL-2B-Instruct",
    load_model_from_lakefs=False,
    max_new_tokens=256,
)

# Judge video frames
frames = [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(8)]
query = "Is there a pedestrian crossing the street?"

result = judge.judge_frames(frames, query, return_confidence=True)
print(f"Query: {result.query}")
print(f"Match: {result.match}")
print(f"Confidence: {result.confidence}")
print(f"Observation: {result.observation}")
print(f"Reason: {result.reason}")
```

### Example Output

```json
{
  "query": "Is there a pedestrian crossing the street?",
  "match": true,
  "confidence": 0.92,
  "observation": "In the video frames, I can see a person wearing dark clothing walking across a marked crosswalk from left to right. The pedestrian is approximately in the middle of the street crossing.",
  "reason": "The video clearly shows a pedestrian actively crossing the street in a crosswalk, which directly matches the query criteria. The pedestrian's movement and position confirm they are crossing, not just standing near the street."
}
```

## Components

### 1. `config.py` - Configuration
- `VLMJudgeConfig`: Dataclass for VLM judge settings

### 2. `infer.py` - Inference
- `VLMJudge`: Main class for VLM inference
- `VLMJudgeResult`: Result dataclass with structured output:
  - `query`: The query that was evaluated
  - `match`: True/False match result
  - `confidence`: 0.0-1.0 confidence score
  - `observation`: What the VLM observed in the frames
  - `reason`: Why the VLM made this judgment
  - `raw_response`: Raw JSON response from VLM

### 3. `utils.py` - Utilities
- `load_video_frames()`: Load frames from video file
- `get_video_path_for_slice()`: Resolve video path from manifest
- `load_frames_for_search_result()`: Load frames for a SearchResult

### 4. `prompts.yaml` - Prompt Configuration
Central configuration file for all VLM prompts:
- `system_prompts.role`: System role for the VLM
- `user_prompts.judgment`: Complete judgment prompt template with output format

## Customizing Prompts

All prompts are defined in `prompts.yaml` for easy customization:

```yaml
# System Prompts
system_prompts:
  role: "You are an expert autonomous driving systems analyst."
  instruction: "Analyze video frames carefully and provide accurate judgments."

# User Prompts
user_prompts:
  judgment: |
    {query}

    Analyze the video frames and respond ONLY with valid JSON in this exact format:
    
    {{
      "query": "<the question>",
      "match": true,
      "confidence": 0.95,
      "observation": "...",
      "reason": "..."
    }}

    Where:
    - query: the question being evaluated
    - match: true if the scenario matches, false otherwise
    - confidence: your confidence level from 0.0 to 1.0
    - observation: what you see in the video frames (be specific)
    - reason: why you gave this judgment
```

### Customizing the System Prompt

Edit `system_prompts.role` to change how the VLM perceives its role:

```yaml
system_prompts:
  role: "You are a safety-focused driving scenario expert. Prioritize accuracy over speed."
```

### Customizing the User Prompts

Edit `user_prompts.judgment` to change the complete prompt including output format:

```yaml
user_prompts:
  judgment: |
    {query}

    Your custom instructions here...
    Format: {{"match": true, "confidence": 0.9, "observation": "...", "reason": "..."}}
```

**Note**: Use double curly braces `{{` and `}}` for literal braces in the YAML (to avoid conflicts with the `{query}` template variable).

### Template Variables

The `user_prompts.judgment` template supports this variable:
- `{query}`: The judgment query (e.g., "Is there a pedestrian crossing?")

## Performance Considerations

### GPU Requirements

VLM models require GPU for inference. Memory requirements vary by model size:

| Model | GPU Memory (`gpu_memory_gb`) | `num_gpus_per_worker` | Recommended GPU |
|-------|----------------------------|---------------------|-----------------|
| Qwen3-VL-2B | 6 GB | 0.5 (share) | A100 40GB |
| Qwen3-VL-7B | 14 GB | 0.5 or 1.0 | A100 40GB |
| Qwen3-VL-32B | 64 GB | 1.0 (full) | A100 80GB |
| Qwen3-VL-72B | 144 GB | 2.0 (multi-GPU) | 2x A100 80GB |

**Key Points:**
- `gpu_memory_gb`: Estimated VRAM needed per worker (for documentation)
- `num_gpus_per_worker`: How many GPUs to allocate (0.5 = share with 2 workers)
- `gpu_type`: GPU hardware to request (e.g., "A100", "A100-80GB")
- Set `gpu_type=None` to use any available GPU

### Speed vs Quality Trade-offs

| Setting | Speed | Quality | Use Case |
|---------|-------|---------|----------|
| `max_candidates=50, fps=0.5, max_frames=4` | Fast | Lower | Quick filtering |
| `max_candidates=100, fps=1.0, max_frames=8` | Medium | Balanced | **Recommended** |
| `max_candidates=200, fps=2.0, max_frames=16` | Slow | Higher | High precision needed |

### Optimization Tips

1. **Limit Candidates**: Use `max_candidates_for_vlm` to judge only top results
2. **Adjust FPS**: Lower FPS (0.5-1.0) for faster inference
3. **Batch Processing**: VLM judge processes candidates sequentially (can be parallelized)
4. **Cache Model**: Set `load_model_from_lakefs=True` to cache model locally

## Example: Pedestrian Detection

```yaml
# prompts.yaml
scenarios:
  - name: "peds_crossing_road"
    prompts:
      - prompt: "People running across the road"
        camera_names: ["camera_front_wide", "camera_front_narrow"]
        similarity_threshold: 0.3
```

With VLM judge enabled:
1. **Embedding search** finds ~500 candidates with similarity ≥ 0.3
2. **VLM judge** evaluates top 100 candidates with detailed analysis:
   ```
   PASS [23/100]: slice_id=abc123, vlm_confidence=0.88, embedding_sim=0.65
     Observation: Two pedestrians wearing bright clothing are running across the crosswalk 
                  from the right side of the frame to the left, moving quickly.
     Reason: The video clearly shows multiple people actively running across the road in a 
             crosswalk, which matches the query for "people running across the road."
   ```
3. Only candidates where VLM confirms scenario (confidence ≥ 0.7) pass
4. Typical pass rate: 30-50% (filtering out 50-70% false positives)
