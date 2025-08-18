#!/usr/bin/env python3
"""
Test script to verify performance monitoring infrastructure works.
This can be run without loading the actual Cosmos model.
"""

import time
import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_monitoring():
    """Test the performance monitoring system."""
    print("🔍 Testing Performance Monitoring Infrastructure...")
    
    # Test basic imports
    try:
        import psutil
        print("✅ psutil imported successfully")
        
        # Test basic CPU/RAM monitoring
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        print(f"   CPU Usage: {cpu_percent}%")
        print(f"   RAM Usage: {memory_info.percent}% ({memory_info.used / (1024**3):.1f}GB / {memory_info.total / (1024**3):.1f}GB)")
        
    except ImportError as e:
        print(f"❌ psutil import failed: {e}")
        return False
    
    # Test GPU monitoring
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"✅ GPUtil working - GPU: {gpu.name}")
            print(f"   GPU Load: {gpu.load * 100:.1f}%")
            print(f"   GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        else:
            print("⚠️  GPUtil working but no GPUs detected")
    except ImportError:
        print("⚠️  GPUtil not available (optional)")
    except Exception as e:
        print(f"⚠️  GPUtil error: {e}")
    
    # Test NVIDIA-ML-PY
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"✅ NVIDIA-ML-PY working - GPU: {name}")
            print(f"   GPU Utilization: {util.gpu}%")
            print(f"   Memory Utilization: {util.memory}%")
        else:
            print("⚠️  NVIDIA-ML-PY working but no GPUs detected")
    except ImportError:
        print("⚠️  NVIDIA-ML-PY not available (optional)")
    except Exception as e:
        print(f"⚠️  NVIDIA-ML-PY error: {e}")
    
    # Test PyTorch GPU monitoring
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated() / (1024**2):.1f}MB")
        print(f"   Memory Cached: {torch.cuda.memory_reserved() / (1024**2):.1f}MB")
        
        # Simulate GPU load
        print("\n🧪 Testing GPU load simulation...")
        with torch.cuda.device(0):
            # Create some tensors to use GPU memory
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            start_time = time.time()
            for i in range(100):
                z = torch.matmul(x, y)
                torch.cuda.synchronize()  # Ensure operations complete
            end_time = time.time()
            
            print(f"   Completed 100 matrix multiplications in {end_time - start_time:.2f}s")
            print(f"   Final GPU Memory: {torch.cuda.memory_allocated() / (1024**2):.1f}MB")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
    else:
        print("⚠️  PyTorch CUDA not available")
    
    print("\n✅ Monitoring infrastructure test completed!")
    return True

def simulate_workload():
    """Simulate a workload similar to video embedding extraction."""
    print("\n🎬 Simulating Video Embedding Workload...")
    
    # Simulate video processing workload
    batch_size = 4
    sequence_length = 8
    hidden_size = 768
    
    print(f"   Simulating batch_size={batch_size}, frames={sequence_length}, embedding_dim={hidden_size}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    # Simulate embedder network
    class MockEmbedder(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = MockEmbedder(hidden_size).to(device)
    
    # Simulate processing multiple videos
    total_inference_time = 0
    num_videos = 5
    
    for video_idx in range(num_videos):
        print(f"   Processing mock video {video_idx + 1}/{num_videos}...")
        
        # Simulate video frames as random tensors
        video_frames = torch.randn(batch_size, sequence_length, hidden_size, device=device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Process frames
            embeddings = model(video_frames.view(-1, hidden_size))
            embeddings = embeddings.view(batch_size, sequence_length, hidden_size)
            
            # Simulate temporal pooling
            video_embedding = embeddings.mean(dim=1)
            
            if device == 'cuda':
                torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        print(f"     Video {video_idx + 1} processed in {inference_time:.3f}s")
        print(f"     Output shape: {video_embedding.shape}")
        
        if torch.cuda.is_available():
            print(f"     GPU Memory: {torch.cuda.memory_allocated() / (1024**2):.1f}MB")
    
    avg_time = total_inference_time / num_videos
    print(f"\n📊 Simulation Results:")
    print(f"   Average inference time: {avg_time:.3f}s per video")
    print(f"   Throughput: {60 / avg_time:.1f} videos per minute")
    print(f"   Total time: {total_inference_time:.3f}s")
    
    # Cleanup
    del model, video_frames, embeddings, video_embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("🚀 Performance Monitoring Test Suite")
    print("=" * 50)
    
    # Test monitoring infrastructure
    monitoring_ok = test_monitoring()
    
    if monitoring_ok:
        # Run workload simulation
        simulate_workload()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\nNext steps:")
        print("1. Install optional monitoring packages:")
        print("   pip install GPUtil nvidia-ml-py")
        print("2. Run the full inference benchmark:")
        print("   python utils/inference_benchmark.py --help")
    else:
        print("\n❌ Some monitoring components failed.")
        print("Install missing dependencies with:")
        print("pip install psutil GPUtil nvidia-ml-py")
