#!/usr/bin/env python3
"""
NVIDIA Cosmos Embedding Model Ray-Accelerated CPU Benchmark
High-performance parallel processing using Ray for CPU-only inference.
Provides significant speedup over sequential processing on multi-core systems.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime
import platform
import os

# Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not available. Install with: pip install ray>=2.8.0")

# Import from existing infrastructure
from core.config import VideoRetrievalConfig
from core.embedder import CosmosVideoEmbedder
from core.exceptions import ModelLoadError, VideoLoadError, EmbeddingExtractionError

# Import the original benchmark for comparison
from benchmarks.cosmos_cpu_benchmark import CosmosCPUBenchmark, PerformanceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@ray.remote
class CosmosWorker:
    """Ray worker for distributed Cosmos embedding extraction."""
    
    def __init__(self, model_path: str, worker_id: int):
        """
        Initialize a Cosmos worker.
        
        Args:
            model_path: Path to the Cosmos model
            worker_id: Unique identifier for this worker
        """
        self.worker_id = worker_id
        self.model_path = model_path
        
        # Configure for CPU-only inference
        self.config = VideoRetrievalConfig()
        self.config.device = "cpu"
        self.config.model_name = model_path
        self.config.batch_size = 1  # Each worker processes one video at a time
        
        # Set CPU threads for this worker (divide total cores by number of workers)
        worker_threads = max(1, psutil.cpu_count() // 8)  # Assume max 8 workers
        torch.set_num_threads(worker_threads)
        
        logger.info(f"Worker {worker_id}: Initializing Cosmos model...")
        start_time = time.time()
        
        try:
            # Import the CPU embedder class
            from benchmarks.cosmos_cpu_benchmark import CosmosCPUEmbedder
            self.embedder = CosmosCPUEmbedder(self.config)
            load_time = time.time() - start_time
            logger.info(f"Worker {worker_id}: Model loaded in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Worker {worker_id}: Failed to load model: {e}")
            raise
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a single video and extract embeddings.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Extract embedding
            embedding = self.embedder.extract_video_embedding(Path(video_path))
            inference_time = time.time() - start_time
            
            return {
                'video_path': video_path,
                'video_name': Path(video_path).name,
                'embedding': embedding,
                'embedding_shape': embedding.shape,
                'embedding_size_mb': embedding.nbytes / (1024**2),
                'inference_time_seconds': inference_time,
                'fps': 1.0 / inference_time if inference_time > 0 else 0,
                'worker_id': self.worker_id,
                'success': True
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"Worker {self.worker_id}: Failed to process {video_path}: {e}")
            
            return {
                'video_path': video_path,
                'video_name': Path(video_path).name,
                'error': str(e),
                'inference_time_seconds': inference_time,
                'worker_id': self.worker_id,
                'success': False
            }
    
    def get_worker_info(self) -> Dict[str, Any]:
        """Get information about this worker."""
        return {
            'worker_id': self.worker_id,
            'model_path': self.model_path,
            'torch_threads': torch.get_num_threads(),
            'device': self.config.device
        }


class CosmosRayBenchmark:
    """Ray-accelerated benchmark for NVIDIA Cosmos on CPU."""
    
    def __init__(self, model_path: Optional[str] = None, num_workers: Optional[int] = None):
        """
        Initialize Ray benchmark.
        
        Args:
            model_path: Path to local Cosmos model
            num_workers: Number of Ray workers (auto-detected if None)
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for this benchmark. Install with: pip install ray>=2.8.0")
        
        self.model_path = model_path or VideoRetrievalConfig().model_name
        
        # Auto-detect optimal number of workers
        if num_workers is None:
            # Conservative approach: assume ~4GB per worker, leave some headroom
            available_memory_gb = psutil.virtual_memory().total / (1024**3)
            memory_per_worker_gb = 4.5  # Model + overhead
            max_workers_by_memory = int(available_memory_gb / memory_per_worker_gb)
            
            # Also consider CPU cores
            cpu_cores = psutil.cpu_count(logical=True)
            max_workers_by_cpu = min(8, cpu_cores)  # Cap at 8 workers
            
            self.num_workers = min(max_workers_by_memory, max_workers_by_cpu, 6)  # Conservative cap
            logger.info(f"Auto-detected {self.num_workers} workers (Memory limit: {max_workers_by_memory}, CPU limit: {max_workers_by_cpu})")
        else:
            self.num_workers = num_workers
        
        self.workers = []
        self.ray_initialized = False
        
    def initialize_ray(self):
        """Initialize Ray and create workers."""
        if self.ray_initialized:
            return
        
        logger.info("Initializing Ray...")
        
        # Initialize Ray with CPU-only resources
        ray.init(
            num_cpus=psutil.cpu_count(),
            ignore_reinit_error=True,
            logging_level=logging.ERROR  # Reduce Ray logging noise
        )
        
        logger.info(f"Creating {self.num_workers} Cosmos workers...")
        start_time = time.time()
        
        # Create workers
        self.workers = []
        worker_futures = []
        
        for i in range(self.num_workers):
            worker = CosmosWorker.remote(self.model_path, i)
            self.workers.append(worker)
            worker_futures.append(worker.get_worker_info.remote())
        
        # Wait for all workers to initialize
        worker_infos = ray.get(worker_futures)
        init_time = time.time() - start_time
        
        logger.info(f"All {self.num_workers} workers initialized in {init_time:.2f}s")
        for info in worker_infos:
            logger.info(f"  Worker {info['worker_id']}: {info['torch_threads']} threads")
        
        self.ray_initialized = True
    
    def shutdown_ray(self):
        """Shutdown Ray and cleanup."""
        if self.ray_initialized:
            logger.info("Shutting down Ray...")
            ray.shutdown()
            self.ray_initialized = False
    
    def get_system_info(self) -> Dict:
        """Get system information including Ray details."""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'ray_version': ray.__version__ if RAY_AVAILABLE else None,
            'num_ray_workers': self.num_workers,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        }
        
        # Platform-specific info
        try:
            system_name = platform.system()
            
            if system_name == 'Darwin':  # macOS
                mac_ver = platform.mac_ver()
                info['os_version'] = mac_ver[0]
                
                # Get CPU brand using system_profiler
                import subprocess
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Processor Name:' in line:
                            info['cpu_model'] = line.split(':')[-1].strip()
                        elif 'Total Number of Cores:' in line:
                            info['cpu_cores_total'] = int(line.split(':')[-1].strip())
                            
            elif system_name == 'Linux':
                # Get Linux distribution info
                try:
                    with open('/etc/os-release', 'r') as f:
                        os_release = f.read()
                        for line in os_release.split('\n'):
                            if line.startswith('PRETTY_NAME='):
                                info['os_version'] = line.split('=')[1].strip('"')
                                break
                except FileNotFoundError:
                    info['os_version'] = 'Linux (unknown distribution)'
                
                # Get CPU info from /proc/cpuinfo
                try:
                    with open('/proc/cpuinfo', 'r') as f:
                        cpuinfo = f.read()
                        for line in cpuinfo.split('\n'):
                            if line.startswith('model name'):
                                info['cpu_model'] = line.split(':')[1].strip()
                                break
                except FileNotFoundError:
                    pass
                    
        except Exception as e:
            logger.debug(f"Could not get detailed system info: {e}")
            
        return info
    
    def run_ray_benchmark(self, video_dir: Path, max_videos: int = 10) -> Dict:
        """
        Run Ray-accelerated benchmark.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to test
            
        Returns:
            Complete benchmark results
        """
        logger.info("Starting NVIDIA Cosmos Ray-accelerated CPU benchmark...")
        
        system_info = self.get_system_info()
        logger.info(f"System: {system_info.get('cpu_model', 'Unknown CPU')}")
        logger.info(f"CPU: {system_info['cpu_count_logical']} logical cores, "
                   f"{system_info['total_memory_gb']:.1f}GB RAM")
        logger.info(f"Ray Workers: {self.num_workers}")
        
        # Initialize Ray and workers
        init_start = time.time()
        self.initialize_ray()
        init_time = time.time() - init_start
        
        # Find video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = [
            f for f in video_dir.rglob('*') 
            if f.is_file() and f.suffix.lower() in video_extensions
        ][:max_videos]
        
        if not video_files:
            raise ValueError(f"No video files found in {video_dir}")
            
        logger.info(f"Found {len(video_files)} videos to benchmark")
        
        # Start system monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Distribute videos across workers
        logger.info("Distributing videos across Ray workers...")
        start_time = time.time()
        
        futures = []
        for i, video_path in enumerate(video_files):
            worker = self.workers[i % self.num_workers]
            future = worker.process_video.remote(str(video_path))
            futures.append(future)
        
        # Collect results with progress tracking
        logger.info("Processing videos in parallel...")
        results = []
        completed = 0
        
        while futures:
            # Wait for at least one task to complete
            ready, futures = ray.wait(futures, num_returns=1, timeout=1.0)
            
            for future in ready:
                result = ray.get(future)
                results.append(result)
                completed += 1
                
                if result['success']:
                    logger.info(f"Completed {completed}/{len(video_files)}: {result['video_name']} "
                              f"({result['inference_time_seconds']:.3f}s, Worker {result['worker_id']})")
                else:
                    logger.error(f"Failed {completed}/{len(video_files)}: {result['video_name']} "
                               f"(Worker {result['worker_id']})")
        
        total_time = time.time() - start_time
        performance_stats = monitor.stop_monitoring()
        
        # Calculate statistics
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        if successful_results:
            inference_times = [r['inference_time_seconds'] for r in successful_results]
            
            # Calculate per-worker statistics
            worker_stats = {}
            for worker_id in range(self.num_workers):
                worker_results = [r for r in successful_results if r['worker_id'] == worker_id]
                if worker_results:
                    worker_times = [r['inference_time_seconds'] for r in worker_results]
                    worker_stats[f'worker_{worker_id}'] = {
                        'videos_processed': len(worker_results),
                        'avg_time': np.mean(worker_times),
                        'total_time': sum(worker_times)
                    }
            
            summary = {
                'total_videos': len(video_files),
                'successful_videos': len(successful_results),
                'failed_videos': len(failed_results),
                'total_wall_time_seconds': total_time,
                'avg_inference_time_seconds': np.mean(inference_times),
                'min_inference_time_seconds': np.min(inference_times),
                'max_inference_time_seconds': np.max(inference_times),
                'std_inference_time_seconds': np.std(inference_times),
                'avg_fps': np.mean([r['fps'] for r in successful_results]),
                'total_cpu_time_seconds': sum(inference_times),
                'parallel_efficiency': sum(inference_times) / (total_time * self.num_workers),
                'speedup_vs_sequential': sum(inference_times) / total_time,
                'videos_per_minute': len(successful_results) * 60 / total_time,
                'videos_per_hour': len(successful_results) * 3600 / total_time,
                'worker_statistics': worker_stats,
                **performance_stats
            }
        else:
            summary = {'error': 'No successful inferences'}
        
        final_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_path,
                'device': 'cpu',
                'platform': platform.system().lower(),
                'benchmark_type': 'ray_parallel',
                'num_workers': self.num_workers,
                'ray_init_time_seconds': init_time,
                'ray_version': ray.__version__,
            },
            'system_info': system_info,
            'summary': summary,
            'detailed_results': results,
        }
        
        return final_results
    
    def run_comparison_benchmark(self, video_dir: Path, max_videos: int = 10) -> Dict:
        """
        Run both Ray and sequential benchmarks for comparison.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to test
            
        Returns:
            Comparison results
        """
        logger.info("Running Ray vs Sequential comparison benchmark...")
        
        # Run Ray benchmark
        ray_results = self.run_ray_benchmark(video_dir, max_videos)
        
        # Shutdown Ray to free memory
        self.shutdown_ray()
        
        # Run sequential benchmark for comparison
        logger.info("\nRunning sequential benchmark for comparison...")
        sequential_benchmark = CosmosCPUBenchmark(self.model_path)
        sequential_results = sequential_benchmark.run_benchmark(video_dir, max_videos)
        
        # Calculate comparison metrics
        ray_summary = ray_results['summary']
        seq_summary = sequential_results['summary']
        
        if 'error' not in ray_summary and 'error' not in seq_summary:
            comparison = {
                'ray_total_time': ray_summary['total_wall_time_seconds'],
                'sequential_total_time': seq_summary['avg_inference_time_seconds'] * seq_summary['successful_videos'],
                'speedup_factor': (seq_summary['avg_inference_time_seconds'] * seq_summary['successful_videos']) / ray_summary['total_wall_time_seconds'],
                'ray_throughput_per_hour': ray_summary['videos_per_hour'],
                'sequential_throughput_per_hour': seq_summary['videos_per_hour'],
                'throughput_improvement': ray_summary['videos_per_hour'] / seq_summary['videos_per_hour'],
                'ray_avg_per_video': ray_summary['avg_inference_time_seconds'],
                'sequential_avg_per_video': seq_summary['avg_inference_time_seconds'],
                'efficiency': ray_summary.get('parallel_efficiency', 0),
            }
        else:
            comparison = {'error': 'One or both benchmarks failed'}
        
        return {
            'comparison': comparison,
            'ray_results': ray_results,
            'sequential_results': sequential_results
        }
    
    def print_comparison_summary(self, results: Dict):
        """Print a formatted comparison summary."""
        print("\n" + "="*80)
        print("RAY vs SEQUENTIAL BENCHMARK COMPARISON")
        print("="*80)
        
        comparison = results['comparison']
        ray_results = results['ray_results']
        seq_results = results['sequential_results']
        
        # System info
        sys_info = ray_results['system_info']
        platform_name = ray_results['benchmark_info']['platform'].upper()
        print(f"\n🖥️  {platform_name} SYSTEM CONFIGURATION:")
        print(f"   Model: {sys_info.get('cpu_model', 'Unknown')}")
        print(f"   CPU: {sys_info['cpu_count_logical']} logical cores ({sys_info['cpu_count_physical']} physical)")
        print(f"   RAM: {sys_info['total_memory_gb']:.1f} GB")
        print(f"   Ray Workers: {sys_info['num_ray_workers']}")
        
        if 'error' not in comparison:
            ray_summary = ray_results['summary']
            seq_summary = seq_results['summary']
            
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   Total Videos: {ray_summary['total_videos']}")
            
            print(f"\n⏱️  TIMING RESULTS:")
            print(f"   Ray (Parallel):     {comparison['ray_total_time']:.2f}s total")
            print(f"   Sequential:         {comparison['sequential_total_time']:.2f}s total")
            print(f"   Speedup Factor:     {comparison['speedup_factor']:.2f}x")
            
            print(f"\n🚀 THROUGHPUT:")
            print(f"   Ray:                {comparison['ray_throughput_per_hour']:.0f} videos/hour")
            print(f"   Sequential:         {comparison['sequential_throughput_per_hour']:.0f} videos/hour")
            print(f"   Improvement:        {comparison['throughput_improvement']:.2f}x")
            
            print(f"\n⚡ EFFICIENCY:")
            print(f"   Parallel Efficiency: {comparison['efficiency']:.1%}")
            print(f"   Ray Avg per Video:   {comparison['ray_avg_per_video']:.3f}s")
            print(f"   Sequential per Video: {comparison['sequential_avg_per_video']:.3f}s")
            
            # Worker statistics
            if 'worker_statistics' in ray_summary:
                print(f"\n👥 WORKER UTILIZATION:")
                for worker_id, stats in ray_summary['worker_statistics'].items():
                    print(f"   {worker_id}: {stats['videos_processed']} videos, "
                          f"{stats['avg_time']:.3f}s avg")
        else:
            print(f"\n❌ COMPARISON FAILED: {comparison['error']}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return str(obj)
            
            json.dump(results, f, indent=2, default=convert_numpy)
        logger.info(f"Results saved to {output_path}")


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray-accelerated NVIDIA Cosmos CPU benchmark")
    parser.add_argument('--video-dir', '-d', type=str, 
                       default='data/videos/video_database',
                       help='Directory containing video files')
    parser.add_argument('--max-videos', '-n', type=int, default=8,
                       help='Maximum number of videos to benchmark (default: 8 for Ray)')
    parser.add_argument('--num-workers', '-w', type=int,
                       help='Number of Ray workers (auto-detected if not specified)')
    parser.add_argument('--output', '-o', type=str,
                       default=f'cosmos_ray_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='Output file for results')
    parser.add_argument('--model-path', '-m', type=str,
                       help='Path to local Cosmos model (uses default if not specified)')
    parser.add_argument('--compare', action='store_true',
                       help='Run comparison with sequential benchmark')
    parser.add_argument('--ray-only', action='store_true',
                       help='Run only Ray benchmark (no comparison)')
    
    args = parser.parse_args()
    
    if not RAY_AVAILABLE:
        logger.error("Ray is not installed. Please install with: pip install ray>=2.8.0")
        return 1
        
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return 1
        
    try:
        benchmark = CosmosRayBenchmark(args.model_path, args.num_workers)
        
        if args.ray_only:
            # Run only Ray benchmark
            results = benchmark.run_ray_benchmark(video_dir, args.max_videos)
            benchmark.save_results(results, args.output)
            
            # Print Ray results summary
            ray_summary = results['summary']
            if 'error' not in ray_summary:
                print(f"\n🚀 RAY BENCHMARK RESULTS:")
                print(f"   Total Time: {ray_summary['total_wall_time_seconds']:.2f}s")
                print(f"   Speedup: {ray_summary['speedup_vs_sequential']:.2f}x")
                print(f"   Throughput: {ray_summary['videos_per_hour']:.0f} videos/hour")
                print(f"   Efficiency: {ray_summary.get('parallel_efficiency', 0):.1%}")
            
        else:
            # Run comparison benchmark (default)
            results = benchmark.run_comparison_benchmark(video_dir, args.max_videos)
            benchmark.save_results(results, args.output)
            benchmark.print_comparison_summary(results)
        
        benchmark.shutdown_ray()
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Ensure Ray is shutdown
        try:
            ray.shutdown()
        except:
            pass


if __name__ == "__main__":
    exit(main())
