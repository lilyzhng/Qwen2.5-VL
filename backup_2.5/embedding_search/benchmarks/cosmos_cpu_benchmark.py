#!/usr/bin/env python3
"""
NVIDIA Cosmos Embedding Model CPU Benchmark (Cross-Platform)
Specialized benchmark for testing Cosmos embedding model performance using CPU only.
Supports Linux, macOS, and other Unix-like systems.
Reuses infrastructure from the existing embedding_search project.
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
import platform
import os

# Import from existing infrastructure
from core.config import VideoRetrievalConfig
from core.embedder import CosmosVideoEmbedder
from core.exceptions import ModelLoadError, VideoLoadError, EmbeddingExtractionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """CPU performance monitor for embedding benchmarks."""
    
    def __init__(self, monitor_interval: float = 0.1):
        """
        Initialize performance monitor for Unix-like systems.
        
        Args:
            monitor_interval: How often to sample metrics (seconds)
        """
        self.monitor_interval = monitor_interval
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'cpu_temperature': [],  # Platform-specific (if available)
            'power_usage_watts': [],  # Platform-specific (if available)
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
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (cross-platform, if available)."""
        # Try Linux sensors first
        try:
            result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['cpu', 'core', 'temp']) and '°C' in line:
                        # Extract temperature value
                        temp_part = line.split('°C')[0].split()[-1]
                        return float(temp_part.replace('+', ''))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
        
        # Try macOS powermetrics (requires admin privileges)
        if platform.system() == 'Darwin':
            try:
                result = subprocess.run(['sudo', 'powermetrics', '-n', '1', '-s', 'cpu_power'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'CPU die temperature' in line:
                            temp_str = line.split(':')[-1].strip().replace('C', '')
                            return float(temp_str)
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass
        
        # Try /sys/class/thermal on Linux
        try:
            thermal_zones = Path('/sys/class/thermal').glob('thermal_zone*')
            for zone in thermal_zones:
                temp_file = zone / 'temp'
                if temp_file.exists():
                    temp_millic = int(temp_file.read_text().strip())
                    return temp_millic / 1000.0  # Convert from millicelsius
        except (FileNotFoundError, ValueError, PermissionError):
            pass
            
        return None
        
    def _get_power_usage(self) -> Optional[float]:
        """Get power usage (cross-platform, if available)."""
        # Try macOS powermetrics (requires admin privileges)
        if platform.system() == 'Darwin':
            try:
                result = subprocess.run(['sudo', 'powermetrics', '-n', '1', '-s', 'cpu_power'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'CPU Power' in line and 'mW' in line:
                            power_str = line.split(':')[-1].strip().replace('mW', '')
                            return float(power_str) / 1000.0  # Convert to watts
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
                pass
        
        # Try Linux RAPL (Running Average Power Limit) interface
        try:
            rapl_path = Path('/sys/class/powercap/intel-rapl')
            if rapl_path.exists():
                for rapl_dir in rapl_path.glob('intel-rapl:*'):
                    energy_file = rapl_dir / 'energy_uj'
                    if energy_file.exists():
                        # This would need time-based sampling for power calculation
                        # For now, just return None as it's complex to implement properly
                        pass
        except (FileNotFoundError, PermissionError):
            pass
            
        return None
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            timestamp = time.time()
            
            # Standard metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            # Mac-specific metrics
            cpu_temp = self._get_cpu_temperature()
            power_usage = self._get_power_usage()
            
            self.metrics['cpu_percent'].append(cpu_percent)
            self.metrics['memory_mb'].append(memory_mb)
            self.metrics['cpu_temperature'].append(cpu_temp if cpu_temp is not None else 0)
            self.metrics['power_usage_watts'].append(power_usage if power_usage is not None else 0)
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
                non_zero_values = [v for v in values if v > 0]  # Filter out unavailable metrics
                if non_zero_values:
                    stats[f'{metric_name}_avg'] = np.mean(non_zero_values)
                    stats[f'{metric_name}_max'] = np.max(non_zero_values)
                    stats[f'{metric_name}_min'] = np.min(non_zero_values)
                    stats[f'{metric_name}_std'] = np.std(non_zero_values)
                else:
                    stats[f'{metric_name}_avg'] = 0
                    stats[f'{metric_name}_max'] = 0
                    stats[f'{metric_name}_min'] = 0
                    stats[f'{metric_name}_std'] = 0
        
        return stats


class CosmosCPUEmbedder(CosmosVideoEmbedder):
    """CPU-optimized version of CosmosVideoEmbedder."""
    
    def __init__(self, config: Optional[VideoRetrievalConfig] = None):
        """
        Initialize CPU-only Cosmos embedder.
        
        Args:
            config: Configuration object with device forced to CPU
        """
        # Force CPU usage regardless of config
        if config is None:
            config = VideoRetrievalConfig()
        
        # Override device to CPU for Mac benchmark
        config.device = "cpu"
        
        # Optimize for CPU inference
        config.batch_size = 4  # Smaller batches for CPU
        
        logger.info("Initializing Cosmos embedder for CPU-only inference")
        
        # Initialize parent with CPU-forced config
        super().__init__(config)
        
        # Additional CPU optimizations
        torch.set_num_threads(psutil.cpu_count())  # Use all CPU cores
        
        if hasattr(torch, 'set_flush_denormal'):
            torch.set_flush_denormal(True)  # Performance optimization
            
        logger.info(f"CPU threads set to: {torch.get_num_threads()}")


class CosmosCPUBenchmark:
    """Specialized benchmark for NVIDIA Cosmos on CPU."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize benchmark.
        
        Args:
            model_path: Path to local Cosmos model (uses default if None)
        """
        # Create CPU-optimized config
        self.config = VideoRetrievalConfig()
        
        if model_path:
            self.config.model_name = model_path
        
        # Force CPU device
        self.config.device = "cpu"
        self.config.batch_size = 4  # Optimize for CPU
        
        self.embedder = None
        self.results = []
        
    def setup_model(self):
        """Initialize the CPU-optimized embedding model."""
        logger.info("Loading NVIDIA Cosmos model for CPU inference...")
        start_time = time.time()
        
        try:
            self.embedder = CosmosCPUEmbedder(self.config)
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            return load_time
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def get_system_info(self) -> Dict:
        """Get cross-platform system information."""
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
            'torch_num_threads': torch.get_num_threads(),
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
                result = subprocess.run(['system_profiler', 'SPHardwareDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Processor Name:' in line:
                            info['cpu_model'] = line.split(':')[-1].strip()
                        elif 'Total Number of Cores:' in line:
                            info['cpu_cores_total'] = int(line.split(':')[-1].strip())
                        elif 'Memory:' in line and 'GB' in line:
                            info['memory_description'] = line.split(':')[-1].strip()
                            
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
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            logger.debug("Could not get detailed system info")
            
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
        
        # Clear any cached data
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        monitor.start_monitoring()
        
        start_time = time.time()
        try:
            embeddings = self.embedder.extract_video_embedding(video_path)
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
            'fps': 1.0 / inference_time if inference_time > 0 else 0,
            **performance_stats
        }
        
        logger.info(f"Inference completed in {inference_time:.3f}s")
        return result
        
    def run_benchmark(self, video_dir: Path, max_videos: int = 10) -> Dict:
        """
        Run comprehensive benchmark on multiple videos.
        
        Args:
            video_dir: Directory containing video files
            max_videos: Maximum number of videos to test
            
        Returns:
            Complete benchmark results
        """
        logger.info("Starting NVIDIA Cosmos CPU benchmark...")
        
        system_info = self.get_system_info()
        logger.info(f"System: {system_info.get('cpu_model', 'Unknown CPU')}")
        logger.info(f"CPU: {system_info['cpu_count_logical']} logical cores, "
                   f"{system_info['total_memory_gb']:.1f}GB RAM")
        logger.info(f"PyTorch threads: {system_info['torch_num_threads']}")
        
        if system_info.get('mps_available'):
            logger.info("Note: MPS (Metal Performance Shaders) available but using CPU for benchmark")
            
        model_load_time = self.setup_model()
        
        # Find video files
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
        
        # Process videos in batches
        batch_size = self.config.batch_size
        for i in range(0, len(video_files), batch_size):
            batch_videos = video_files[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {[v.name for v in batch_videos]}")
            
            # Time the entire batch
            monitor.start_monitoring()
            start_time = time.time()
            
            batch_results = self.embedder.extract_video_embeddings_batch(batch_videos, batch_size)
            
            batch_time = time.time() - start_time
            performance_stats = monitor.stop_monitoring()
            
            # Calculate per-video metrics from batch
            per_video_time = batch_time / len(batch_videos)
            for j, video_result in enumerate(batch_results):
                # Create individual results for each video in the batch
                result = {
                    'video_path': video_result['video_path'],
                    'video_name': batch_videos[j].name,
                    'inference_time_seconds': per_video_time,
                    'embedding_shape': video_result['embedding'].shape,
                    'embedding_size_mb': video_result['embedding'].nbytes / (1024**2),
                    'fps': 1.0 / per_video_time,
                    **performance_stats
                }
                benchmark_results.append(result)
            
            # Brief pause between videos
            time.sleep(0.5)
            
        # Calculate summary statistics
        successful_results = [r for r in benchmark_results if 'error' not in r]
        
        if successful_results:
            inference_times = [r['inference_time_seconds'] for r in successful_results]
            cpu_peaks = [r.get('cpu_percent_max', 0) for r in successful_results]
            ram_peaks = [r.get('memory_mb_max', 0) for r in successful_results]
            temps = [r.get('cpu_temperature_max', 0) for r in successful_results if r.get('cpu_temperature_max', 0) > 0]
            power = [r.get('power_usage_watts_max', 0) for r in successful_results if r.get('power_usage_watts_max', 0) > 0]
            
            summary = {
                'total_videos': len(video_files),
                'successful_videos': len(successful_results),
                'failed_videos': len(video_files) - len(successful_results),
                'avg_inference_time_seconds': np.mean(inference_times),
                'min_inference_time_seconds': np.min(inference_times),
                'max_inference_time_seconds': np.max(inference_times),
                'std_inference_time_seconds': np.std(inference_times),
                'avg_cpu_utilization_percent': np.mean(cpu_peaks),
                'max_cpu_utilization_percent': np.max(cpu_peaks),
                'avg_ram_usage_mb': np.mean(ram_peaks),
                'max_ram_usage_mb': np.max(ram_peaks),
                'avg_fps': np.mean([r['fps'] for r in successful_results]),
                'videos_per_minute': 60 / np.mean(inference_times),
                'videos_per_hour': 3600 / np.mean(inference_times),
            }
            
            # Add temperature and power if available
            if temps:
                summary.update({
                    'avg_cpu_temperature_c': np.mean(temps),
                    'max_cpu_temperature_c': np.max(temps),
                })
                
            if power:
                summary.update({
                    'avg_power_usage_watts': np.mean(power),
                    'max_power_usage_watts': np.max(power),
                })
        else:
            summary = {'error': 'No successful inferences'}
            
        final_results = {
            'benchmark_info': {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.config.model_name,
                'device': 'cpu',
                'platform': platform.system().lower(),
                'model_load_time_seconds': model_load_time,
                'torch_threads': torch.get_num_threads(),
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
        print("NVIDIA COSMOS CPU BENCHMARK RESULTS")
        print("="*80)
        
        # System info
        sys_info = results['system_info']
        platform_name = results['benchmark_info']['platform'].upper()
        print(f"\n🖥️  {platform_name} SYSTEM CONFIGURATION:")
        print(f"   Model: {sys_info.get('cpu_model', 'Unknown')}")
        print(f"   CPU: {sys_info['cpu_count_logical']} logical cores ({sys_info['cpu_count_physical']} physical)")
        print(f"   RAM: {sys_info['total_memory_gb']:.1f} GB")
        print(f"   OS: {sys_info.get('os_version', 'Unknown')}")
        print(f"   PyTorch Threads: {sys_info['torch_num_threads']}")
        
        # Model info
        bench_info = results['benchmark_info']
        print(f"\n🤖 MODEL CONFIGURATION:")
        print(f"   Model: {bench_info['model_name']}")
        print(f"   Device: CPU (forced)")
        print(f"   Load Time: {bench_info['model_load_time_seconds']:.2f}s")
        
        # Performance summary
        summary = results['summary']
        if 'error' not in summary:
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print(f"   Total Videos: {summary['total_videos']}")
            print(f"   Successful: {summary['successful_videos']}")
            print(f"   Failed: {summary['failed_videos']}")
            
            print(f"\n⏱️  INFERENCE PERFORMANCE:")
            print(f"   Average Time: {summary['avg_inference_time_seconds']:.3f}s per video")
            print(f"   Range: {summary['min_inference_time_seconds']:.3f}s - {summary['max_inference_time_seconds']:.3f}s")
            print(f"   Std Dev: {summary['std_inference_time_seconds']:.3f}s")
            print(f"   Average FPS: {summary['avg_fps']:.2f}")
            
            print(f"\n🚀 THROUGHPUT:")
            print(f"   Videos per minute: {summary['videos_per_minute']:.1f}")
            print(f"   Videos per hour: {summary['videos_per_hour']:.0f}")
            
            print(f"\n💻 CPU UTILIZATION:")
            print(f"   Peak Usage: {summary['max_cpu_utilization_percent']:.1f}%")
            print(f"   Average Usage: {summary['avg_cpu_utilization_percent']:.1f}%")
            
            print(f"\n🧠 MEMORY USAGE:")
            print(f"   Peak RAM: {summary['max_ram_usage_mb']:.1f} MB ({summary['max_ram_usage_mb']/1024:.1f} GB)")
            print(f"   Average RAM: {summary['avg_ram_usage_mb']:.1f} MB ({summary['avg_ram_usage_mb']/1024:.1f} GB)")
            
            # Temperature and power if available
            if 'max_cpu_temperature_c' in summary and summary['max_cpu_temperature_c'] > 0:
                print(f"\n🌡️  TEMPERATURE:")
                print(f"   Peak CPU Temp: {summary['max_cpu_temperature_c']:.1f}°C")
                print(f"   Average CPU Temp: {summary['avg_cpu_temperature_c']:.1f}°C")
                
            if 'max_power_usage_watts' in summary and summary['max_power_usage_watts'] > 0:
                print(f"\n⚡ POWER CONSUMPTION:")
                print(f"   Peak Power: {summary['max_power_usage_watts']:.2f}W")
                print(f"   Average Power: {summary['avg_power_usage_watts']:.2f}W")
        else:
            print(f"\n❌ BENCHMARK FAILED: {summary['error']}")
            
        print("\n" + "="*80)


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA Cosmos on CPU (Cross-Platform)")
    parser.add_argument('--video-dir', '-d', type=str, 
                       default='data/videos/video_database',
                       help='Directory containing video files')
    parser.add_argument('--max-videos', '-n', type=int, default=5,
                       help='Maximum number of videos to benchmark (default: 5 for CPU)')
    parser.add_argument('--output', '-o', type=str,
                       default=f'cosmos_cpu_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                       help='Output file for results')
    parser.add_argument('--model-path', '-m', type=str,
                       help='Path to local Cosmos model (uses default if not specified)')
    
    args = parser.parse_args()
        
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return 1
        
    try:
        benchmark = CosmosCPUBenchmark(args.model_path)
        results = benchmark.run_benchmark(video_dir, args.max_videos)
        
        benchmark.save_results(results, args.output)
        benchmark.print_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
