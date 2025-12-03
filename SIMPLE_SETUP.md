# Simple Qwen3-VL Setup for Linux

## Quick Setup

```bash
# 1. Create virtual environment
python3 -m venv qwen3_env
source qwen3_env/bin/activate

# 2. Install PyTorch 2.5.1 with CUDA 12.4
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# If 2.5.1 is not available, use 2.6.0:
# pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 3. Install Qwen3-VL
pip install "transformers>=4.57.0" accelerate qwen-vl-utils

# 4. Test it
python3 simple_test.py
```

## Expected Output

If everything works, you should see:
```
PyTorch version: 2.5.1+cu124
Transformers version: 4.57.x
CUDA available: True
CUDA version: 12.4
GPU: [your GPU name]

✅ All imports successful!
Environment is ready for Qwen3-VL
```

## That's it!

If the test passes, you're ready to use Qwen3-VL.

