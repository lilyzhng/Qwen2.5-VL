#!/usr/bin/env python3
"""Simple test for Qwen3-VL compatibility"""

import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

# Test imports
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

print("\n✅ All imports successful!")
print("Environment is ready for Qwen3-VL")

