#!/usr/bin/env python3
"""
Performance Benchmark: Monitors inference time, GPU/CPU utilization, and memory usage.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime
import threading
import subprocess

# Try to import optional dependencies
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from core.config import VideoRetrievalConfig
from core.embedder import CosmosVideoEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance during inference."""
    
    def __init__(self, monitor_interval: float = 0.1):
        """
        Initialize performance monitor.
        
        Args:
            monitor_interval: How often to sample metrics (seconds)
        """
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_vram_mb': [],
            'gpu_utilization_percent': [],
            'timestamps': []
        }
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start monitoring system metrics."""
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self._calculate_stats()
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            gpu_vram_mb = 0
            gpu_utilization_percent = 0
            
            if torch.cuda.is_available():
                try:
                    # VRAM usage via PyTorch (more accurate for allocated memory)
                    gpu_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    
                    # GPU utilization percentage via nvidia-ml-py (preferred)
                    if HAS_PYNVML:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_utilization_percent = util.gpu
                            
                            # Also get VRAM from NVML for comparison/backup
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            nvml_vram_mb = mem_info.used / (1024 * 1024)
                            # Use the higher value between PyTorch and NVML
                            gpu_vram_mb = max(gpu_vram_mb, nvml_vram_mb)
                        except Exception as e:
                            logger.debug(f"NVML error: {e}")
                    
                    # Fallback to GPUtil
                    elif HAS_GPUTIL:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu_utilization_percent = gpus[0].load * 100
                                # Use GPUtil VRAM if PyTorch shows 0
                                if gpu_vram_mb == 0:
                                    gpu_vram_mb = gpus[0].memoryUsed
                        except Exception as e:
                            logger.debug(f"GPUtil error: {e}")
                            
                except Exception as e:
                    logger.debug(f"GPU monitoring error: {e}")
            
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_mb'].append(memory_mb)
            self.metrics['gpu_vram_mb'].append(gpu_vram_mb)
            self.metrics['gpu_utilization_percent'].append(gpu_utilization_percent)
            self.metrics['timestamps'].append(timestamp)
            
            time.sleep(self.monitor_interval)
            
    def _calculate_stats(self) -> Dict:
        """Calculate statistics from collected metrics."""
        if not self.metrics['timestamps']:
            return {}
            
        stats = {}
        for metric_name, values in self.metrics.items():
            if metric_name == 'timestamps':
                continue
            if values:
                stats[f'{metric_name}_avg'] = np.mean(values)
                stats[f'{metric_name}_max'] = np.max(values)
                stats[f'{metric_name}_min'] = np.min(values)
                stats[f'{metric_name}_std'] = np.std(values)
        
        return stats


class InferenceBenchmark:
    """Benchmark NVIDIA Cosmos Embed model inference performance."""
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize benchmark.
        
        Args:
            config: Configuration for the ALFA 0.1 retrieval
        """
        self.config = config or VideoRetrievalConfig()
        self.embedder = None
        self.results = []
        
    def setup_model(self):
        """Initialize the embedding model."""
        logger.info("Loading NVIDIA Cosmos Embed model...")
        start_time = time.time()
        
        try:
            self.embedder = CosmosVideoEmbedder(
                self.config
            )
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            return load_time
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def get_system_info(self) -> Dict:
        """Get system information."""
        info = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_count': torch.cuda.device_count(),
            })
            
        return info
        
    def benchmark_single_video(self, video_path: Path, monitor: PerformanceMonitor) -> Dict:
        """
        Benchmark inference on a single video.
        
        Args:
            video_path: Path to video file
            monitor: Performance monitor instance
            
        Returns:
            Dictionary with benchmark results
        """
        if not self.embedder:
            raise RuntimeError("Model not initialized. Call setup_model() first.")
            
        logger.info(f"Benchmarking video: {video_path.name}")
        
        # Ensure clean GPU state by clearing cache and synchronizing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        monitor.start_monitoring()
        
        start_time = time.time()
        try:
            embeddings = self.embedder.extract_video_embedding(Path(video_path))
            
            # Ensure GPU operations are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            inference_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Inference failed for {video_path}: {e}")
            monitor.stop_monitoring()
            return {'error': str(e), 'video_path': str(video_path)}
        
        performance_stats = monitor.stop_monitoring()
        
        result = {
            'video_path': str(video_path),
            'video_name': video_path.name,
            'inference_time_seconds': inference_time,
            'embedding_shape': embeddings.shape if embeddings is not None else None,
            'embedding_size_mb': embeddings.nbytes / (1024**2) if embeddings is not None else 0,
            **performance_stats
        }
        
        logger.info(f"Inference completed in {inference_time:.3f}s")
        return result
        
    def run_benchmark(self, video_dir: Path, max_videos: int = 10) -> Dict:
        """
        Run benchmark on multiple videos.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to test
            
        Returns:
            Complete benchmark results
        """
        logger.info("Starting NVIDIA Cosmos Embed inference benchmark...")
        
        system_info = self.get_system_info()
        logger.info(f"System: {system_info['cpu_count']} CPU cores, "
                   f"{system_info['total_memory_gb']:.1f}GB RAM")
        
        if system_info['cuda_available']:
            logger.info(f"GPU: {system_info['gpu_name']}, "
                       f"{system_info['gpu_memory_gb']:.1f}GB VRAM")
        else:
            logger.warning("CUDA not available - running on CPU")
            
        model_load_time = self.setup_model()
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = [
            f for f in video_dir.rglob('*') 
            if f.is_file() and f.suffix.lower() in video_extensions
        ][:max_videos]
        
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")
            
        logger.info(f"Found {len(video_files)} videos to benchmark")
        
        benchmark_results = []
        monitor = PerformanceMonitor()
        
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"Processing video {i}/{len(video_files)}: {video_path.name}")
            
            result = self.benchmark_single_video(video_path, monitor)
            benchmark_results.append(result)
            
            # Small delay between videos
            time.sleep(0.5)
            
        successful_results = [r for r in benchmark_results if 'error' not in r]
        
        if successful_results:
            inference_times = [r['inference_time_seconds'] for r in successful_results]
            gpu_vram_peaks = [r.get('gpu_vram_mb_max', 0) for r in successful_results]
            gpu_util_peaks = [r.get('gpu_utilization_percent_max', 0) for r in successful_results]
            cpu_peaks = [r.get('cpu_percent_max', 0) for r in successful_results]
            ram_peaks = [r.get('memory_mb_max', 0) for r in successful_results]
            
            summary = {
                'total_videos': len(video_files),
                'successful_videos': len(successful_results),
                'failed_videos': len(video_files) - len(successful_results),
                'avg_inference_time_seconds': np.mean(inference_times),
                'min_inference_time_seconds': np.min(inference_times),
                'max_inference_time_seconds': np.max(inference_times),
                'std_inference_time_seconds': np.std(inference_times),
                'avg_gpu_vram_mb': np.mean(gpu_vram_peaks),
                'max_gpu_vram_mb': np.max(gpu_vram_peaks),
                'avg_gpu_utilization_percent': np.mean(gpu_util_peaks),
                'max_gpu_utilization_percent': np.max(gpu_util_peaks),
                'avg_cpu_utilization_percent': np.mean(cpu_peaks),
                'max_cpu_utilization_percent': np.max(cpu_peaks),
                'avg_ram_usage_mb': np.mean(ram_peaks),
                'max_ram_usage_mb': np.max(ram_peaks),
            }
        else:
            summary = {'error': 'No successful inferences'}
            
        final_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config.model_name,
                'device': self.config.device,
                'model_load_time_seconds': model_load_time,
            },
            'system_info': system_info,
            'summary': summary,
            'detailed_results': benchmark_results,
        }
        
        return final_results
        
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
        
    def print_summary(self, results: Dict):
        """Print a formatted summary of benchmark results."""
        print("\n" + "="*80)
        print("NVIDIA COSMOS EMBED INFERENCE BENCHMARK RESULTS")
        print("="*80)
        
        # System info
        sys_info = results['system_info']
        print(f"\n🖥️  SYSTEM CONFIGURATION:")
        print(f"   CPU: {sys_info['cpu_count']} cores @ {sys_info.get('cpu_freq_mhz', 'N/A')} MHz")
        print(f"   RAM: {sys_info['total_memory_gb']:.1f} GB")
        
        if sys_info['cuda_available']:
            print(f"   GPU: {sys_info['gpu_name']}")
            print(f"   VRAM: {sys_info['gpu_memory_gb']:.1f} GB")
            print(f"   CUDA: {sys_info['cuda_version']}")
        else:
            print("   GPU: Not available (CPU inference)")
            
        # Model info
        bench_info = results['benchmark_info']
        print(f"\n🤖 MODEL CONFIGURATION:")
        print(f"   Model: {bench_info['model_name']}")
        print(f"   Device: {bench_info['device']}")
        print(f"   Load Time: {bench_info['model_load_time_seconds']:.2f}s")
        
        # Performance summary
        summary = results['summary']
        if 'error' not in summary:
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Total Videos: {summary['total_videos']}")
            print(f"   Successful: {summary['successful_videos']}")
            print(f"   Failed: {summary['failed_videos']}")
            
            print(f"\n⏱️  INFERENCE TIME:")
            print(f"   Average: {summary['avg_inference_time_seconds']:.3f}s per video")
            print(f"   Range: {summary['min_inference_time_seconds']:.3f}s - {summary['max_inference_time_seconds']:.3f}s")
            print(f"   Std Dev: {summary['std_inference_time_seconds']:.3f}s")
            
            # GPU info with context
            sys_info = results['system_info']
            if sys_info['cuda_available']:
                total_vram_gb = sys_info['gpu_memory_gb']
                peak_used_gb = summary['max_gpu_vram_mb'] / 1024
                vram_utilization = (peak_used_gb / total_vram_gb) * 100
                
                print(f"\nGPU VRAM USAGE:")
                print(f"   Peak VRAM: {summary['max_gpu_vram_mb']:.1f} MB ({peak_used_gb:.2f} GB)")
                print(f"   Total VRAM: {total_vram_gb:.1f} GB")
                print(f"   VRAM Utilization: {vram_utilization:.1f}% of total")
                print(f"   Available: {total_vram_gb - peak_used_gb:.1f} GB")
            
            print(f"\n⚡ GPU UTILIZATION:")
            print(f"   Peak Usage: {summary['max_gpu_utilization_percent']:.1f}%")
            print(f"   Avg Usage: {summary['avg_gpu_utilization_percent']:.1f}%")
            
            print(f"\nCPU UTILIZATION:")
            print(f"   Peak Usage: {summary['max_cpu_utilization_percent']:.1f}%")
            print(f"   Avg Usage: {summary['avg_cpu_utilization_percent']:.1f}%")
            
            print(f"\nRAM USAGE:")
            print(f"   Peak Usage: {summary['max_ram_usage_mb']:.1f} MB ({summary['max_ram_usage_mb']/1024:.1f} GB)")
            print(f"   Avg Usage: {summary['avg_ram_usage_mb']:.1f} MB ({summary['avg_ram_usage_mb']/1024:.1f} GB)")
            
            print(f"\nTHROUGHPUT:")
            print(f"   Videos per minute: {60 / summary['avg_inference_time_seconds']:.1f}")
            print(f"   Videos per hour: {3600 / summary['avg_inference_time_seconds']:.0f}")
        else:
            print(f"\n❌ BENCHMARK FAILED: {summary['error']}")


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA Cosmos Embed inference performance")
    parser.add_argument('--video-dir', '-d', type=str, 
                       default='/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/videos/video_database',
                       help='Directory containing video files')
    parser.add_argument('--max-videos', '-n', type=int, default=10,
                       help='Maximum number of videos to benchmark')
    parser.add_argument('--output', '-o', type=str,
                       default=f'cosmos_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='Output file for results')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Override device setting')
    
    args = parser.parse_args()
    
    if args.config:
        config = VideoRetrievalConfig.from_yaml(args.config)
    else:
        config = VideoRetrievalConfig()
        
    if args.device:
        config.device = args.device
        
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return 1
        
    try:
        benchmark = InferenceBenchmark(config)
        results = benchmark.run_benchmark(video_dir, args.max_videos)
        
        benchmark.save_results(results, args.output)
        benchmark.print_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
