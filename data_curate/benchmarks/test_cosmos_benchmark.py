#!/usr/bin/env python3
"""
Test script for the Cosmos CPU benchmark.
Validates the benchmark functionality without running full inference.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch
import tempfile
import json

from benchmarks.cosmos_cpu_benchmark import CosmosCPUBenchmark, PerformanceMonitor, CosmosCPUEmbedder
from core.config import VideoRetrievalConfig


class TestCosmosCPUBenchmark(unittest.TestCase):
    """Test cases for the Cosmos CPU benchmark."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.benchmark = CosmosCPUBenchmark()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        self.assertIsNotNone(self.benchmark.config)
        self.assertEqual(self.benchmark.config.device, "cpu")
        self.assertEqual(self.benchmark.config.batch_size, 1)
        
    def test_system_info(self):
        """Test system information gathering."""
        info = self.benchmark.get_system_info()
        
        # Check required fields
        required_fields = [
            'platform', 'processor', 'machine', 'cpu_count_physical',
            'cpu_count_logical', 'total_memory_gb', 'python_version',
            'torch_version', 'torch_num_threads', 'cuda_available'
        ]
        
        for field in required_fields:
            self.assertIn(field, info)
            
        # Check data types
        self.assertIsInstance(info['cpu_count_physical'], int)
        self.assertIsInstance(info['cpu_count_logical'], int)
        self.assertIsInstance(info['total_memory_gb'], float)
        self.assertIsInstance(info['cuda_available'], bool)
        
    def test_performance_monitor(self):
        """Test the CPU performance monitor."""
        monitor = PerformanceMonitor(monitor_interval=0.01)
        
        # Test start/stop without errors
        monitor.start_monitoring()
        import time
        time.sleep(0.05)  # Brief monitoring period
        stats = monitor.stop_monitoring()
        
        # Check that we got some metrics
        self.assertIsInstance(stats, dict)
        
        # Check for expected metric keys (some may be 0 if not available)
        expected_metrics = [
            'cpu_percent_avg', 'memory_mb_avg', 'cpu_temperature_avg', 'power_usage_watts_avg'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, stats)
            
    def test_config_cpu_optimization(self):
        """Test that configuration is optimized for CPU."""
        config = VideoRetrievalConfig()
        config.device = "cuda"  # Start with CUDA
        
        # Create CPU embedder (should force CPU)
        with patch('benchmarks.cosmos_cpu_benchmark.CosmosVideoEmbedder.__init__') as mock_init:
            mock_init.return_value = None
            
            embedder = CosmosCPUEmbedder(config)
            
            # Verify CPU was forced
            mock_init.assert_called_once()
            passed_config = mock_init.call_args[0][0]
            self.assertEqual(passed_config.device, "cpu")
            
    def test_save_load_results(self):
        """Test saving and loading benchmark results."""
        # Create sample results
        sample_results = {
            'benchmark_info': {
                'timestamp': '2024-01-01T12:00:00',
                'model_name': 'test_model',
                'device': 'cpu'
            },
            'system_info': {
                'cpu_count': 8,
                'total_memory_gb': 16.0
            },
            'summary': {
                'total_videos': 5,
                'avg_inference_time_seconds': 2.5
            },
            'detailed_results': []
        }
        
        # Test saving
        output_file = self.temp_dir / "test_results.json"
        self.benchmark.save_results(sample_results, str(output_file))
        
        # Verify file was created and contains correct data
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            loaded_results = json.load(f)
            
        self.assertEqual(loaded_results['benchmark_info']['model_name'], 'test_model')
        self.assertEqual(loaded_results['system_info']['cpu_count'], 8)
        
    def test_video_file_discovery(self):
        """Test video file discovery functionality."""
        # Create test video files
        test_videos = ['test1.mp4', 'test2.avi', 'test3.mov', 'not_video.txt']
        
        for video in test_videos:
            (self.temp_dir / video).touch()
            
        # Mock the run_benchmark method to test file discovery
        with patch.object(self.benchmark, 'setup_model', return_value=1.0), \
             patch.object(self.benchmark, 'benchmark_single_video') as mock_benchmark:
            
            mock_benchmark.return_value = {
                'video_path': 'test',
                'inference_time_seconds': 1.0,
                'embedding_shape': (512,),
                'embedding_size_mb': 0.002
            }
            
            results = self.benchmark.run_benchmark(self.temp_dir, max_videos=10)
            
            # Should find 3 video files (excluding .txt)
            self.assertEqual(results['summary']['total_videos'], 3)
            
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent directory
        with self.assertRaises(ValueError):
            self.benchmark.run_benchmark(Path("/non/existent/path"))
            
        # Test with directory containing no videos
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(ValueError):
            self.benchmark.run_benchmark(empty_dir)


def run_quick_validation():
    """Run a quick validation of the benchmark without full inference."""
    print("🧪 Running Cosmos CPU Benchmark Validation...")
    
    try:
        # Test system info gathering
        benchmark = CosmosCPUBenchmark()
        system_info = benchmark.get_system_info()
        
        print(f"✅ System info gathered successfully")
        print(f"   CPU: {system_info.get('cpu_model', 'Unknown')} ({system_info['cpu_count_logical']} cores)")
        print(f"   RAM: {system_info['total_memory_gb']:.1f} GB")
        print(f"   OS: {system_info.get('os_version', 'Unknown')}")
        
        # Test performance monitor
        monitor = PerformanceMonitor(monitor_interval=0.01)
        monitor.start_monitoring()
        import time
        time.sleep(0.1)
        stats = monitor.stop_monitoring()
        
        print(f"✅ Performance monitoring working")
        print(f"   CPU usage captured: {'cpu_percent_avg' in stats}")
        print(f"   Memory usage captured: {'memory_mb_avg' in stats}")
        
        # Test configuration
        config = benchmark.config
        print(f"✅ Configuration validated")
        print(f"   Device: {config.device}")
        print(f"   Batch size: {config.batch_size}")
        
        print("\n🎉 Validation completed successfully!")
        print("The benchmark is ready to run with real videos.")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run quick validation
    success = run_quick_validation()
    
    if success:
        print("\n" + "="*60)
        print("To run the full benchmark, use:")
        print("python benchmarks/cosmos_cpu_benchmark.py --video-dir /path/to/videos")
        print("="*60)
    
    # Run unit tests if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--run-tests':
        print("\n🧪 Running unit tests...")
        unittest.main(argv=[''], exit=False, verbosity=2)
