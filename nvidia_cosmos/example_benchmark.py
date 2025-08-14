#!/usr/bin/env python3
"""
Example script showing how to use the NVIDIA Cosmos inference benchmark.
"""

from pathlib import Path
from utils.inference_benchmark import InferenceBenchmark
from config import VideoRetrievalConfig

def main():
    """Run a simple benchmark example."""
    print("🚀 NVIDIA Cosmos Embed Model Benchmark Example")
    print("=" * 50)
    
    # Load configuration
    config = VideoRetrievalConfig()
    
    # Set video directory
    video_dir = Path("videos/video_database")
    
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        print("Please ensure video files are available in the videos/video_database directory")
        return
    
    print(f"📁 Using video directory: {video_dir}")
    print(f"🤖 Model: {config.model_name}")
    print(f"🖥️  Device: {config.device}")
    
    try:
        # Create benchmark instance
        benchmark = InferenceBenchmark(config)
        
        # Run benchmark on a few videos
        print("\n🏃 Running benchmark...")
        results = benchmark.run_benchmark(video_dir, max_videos=3)
        
        # Display results
        benchmark.print_summary(results)
        
        # Save detailed results
        output_file = "cosmos_benchmark_example.json"
        benchmark.save_results(results, output_file)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure NVIDIA Cosmos model is available")
        print("2. Check CUDA/GPU availability if using GPU")
        print("3. Install missing dependencies: pip install -r requirements.txt")
        print("4. Test monitoring infrastructure: python utils/test_monitoring.py")

if __name__ == "__main__":
    main()
