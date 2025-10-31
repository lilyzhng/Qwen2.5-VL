# Qwen3-VL Local Inference

Standalone local inference module for Qwen3-VL models using HuggingFace Transformers.

## Features

- ✅ **Standalone**: No dependency on `qwen-vl-utils`
- ✅ **Simple**: Clean, production-ready code
- ✅ **Flexible**: Supports images and videos
- ✅ **Fast**: Optional Flash Attention 2 support

## Dependencies

Only requires standard HuggingFace libraries:
```bash
pip install transformers torch torchvision pillow
```

Optional for faster inference:
```bash
pip install flash-attn --no-build-isolation
```

## Usage

### Command Line

```bash
# Single image
python local_inference.py \
    -m Qwen/Qwen3-VL-2B-Instruct \
    -i image.jpg \
    -p "Describe this image"

# Multiple images
python local_inference.py \
    -m Qwen/Qwen3-VL-2B-Instruct \
    -i img1.jpg -i img2.jpg \
    -p "Compare these images"

# Video
python local_inference.py \
    -m Qwen/Qwen3-VL-8B-Instruct \
    -v video.mp4 \
    -p "What happens in this video?" \
    --fps 2.0

# With Flash Attention (faster)
python local_inference.py \
    -m Qwen/Qwen3-VL-8B-Instruct \
    --flash-attn \
    -i image.jpg \
    -p "Analyze this image"
```

### Python API

```python
from qwen_vl import load_model, prepare_messages, run_inference

# Load model
model, processor = load_model("Qwen/Qwen3-VL-2B-Instruct")

# Prepare messages
messages = prepare_messages(
    media_paths="image.jpg",
    prompt="Describe this image",
    media_type="image"
)

# Run inference
output = run_inference(model, processor, messages)
print(output)
```

## File Structure

```
qwen_vl/
├── __init__.py          # Package initialization
├── local_inference.py   # Main inference script
├── README.md           # This file
└── requirements.txt    # Dependencies
```

## Notes

- Videos are processed as frame sequences internally
- The `media_type` parameter ("image" or "video") affects how the model samples/decodes input
- No `qwen-vl-utils` dependency required - all processing handled by transformers

