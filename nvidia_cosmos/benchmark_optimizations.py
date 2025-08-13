#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized implementation.
Shows performance improvements from techniques in the official NVIDIA implementation.
"""

import time
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import matplotlib.pyplot as plt

from video_search_v2 import VideoSearchEngine
from video_search_optimized import OptimizedVideoSearchEngine
from config import VideoRetrievalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark different implementations."""
    
    def __init__(self, video_dir: str, num_queries: int = 10):
        self.video_dir = Path(video_dir)
        self.num_queries = num_queries
        self.results = {}
        
    def benchmark_database_build(self):
        """Benchmark database building."""
        logger.info("Benchmarking database building...")
        
        # Original implementation
        config1 = VideoRetrievalConfig(database_path="benchmark_db_v2")
        engine1 = VideoSearchEngine(config=config1)
        
        start = time.time()
        engine1.build_database(self.video_dir, force_rebuild=True)
        time_v2 = time.time() - start
        
        # Optimized implementation
        config2 = VideoRetrievalConfig(database_path="benchmark_db_opt")
        engine2 = OptimizedVideoSearchEngine(config=config2)
        
        start = time.time()
        engine2.build_database(self.video_dir, force_rebuild=True)
        time_opt = time.time() - start
        
        self.results['database_build'] = {
            'original': time_v2,
            'optimized': time_opt,
            'speedup': time_v2 / time_opt if time_opt > 0 else 0
        }
        
        logger.info(f"Database build - Original: {time_v2:.2f}s, Optimized: {time_opt:.2f}s")
        logger.info(f"Speedup: {self.results['database_build']['speedup']:.2f}x")
        
        return engine1, engine2
    
    def benchmark_search(self, engine1: VideoSearchEngine, 
                        engine2: OptimizedVideoSearchEngine,
                        query_videos: List[Path]):
        """Benchmark search performance."""
        logger.info("Benchmarking search performance...")
        
        times_v2 = []
        times_opt = []
        times_opt_cached = []
        
        for query_video in query_videos[:self.num_queries]:
            # Original search
            start = time.time()
            results1 = engine1.search_by_video(query_video, top_k=10)
            times_v2.append(time.time() - start)
            
            # Optimized search (first run)
            start = time.time()
            results2 = engine2.search_by_video(query_video, top_k=10)
            times_opt.append(time.time() - start)
            
            # Optimized search (cached)
            start = time.time()
            results3 = engine2.search_by_video(query_video, top_k=10, use_cache=True)
            times_opt_cached.append(time.time() - start)
        
        self.results['search'] = {
            'original_mean': np.mean(times_v2),
            'optimized_mean': np.mean(times_opt),
            'optimized_cached_mean': np.mean(times_opt_cached),
            'speedup': np.mean(times_v2) / np.mean(times_opt),
            'speedup_cached': np.mean(times_v2) / np.mean(times_opt_cached)
        }
        
        logger.info(f"Search performance (mean of {self.num_queries} queries):")
        logger.info(f"  Original: {self.results['search']['original_mean']:.3f}s")
        logger.info(f"  Optimized: {self.results['search']['optimized_mean']:.3f}s")
        logger.info(f"  Optimized (cached): {self.results['search']['optimized_cached_mean']:.3f}s")
        logger.info(f"  Speedup: {self.results['search']['speedup']:.2f}x")
        logger.info(f"  Speedup (cached): {self.results['search']['speedup_cached']:.2f}x")
    
    def benchmark_text_search(self, engine1: VideoSearchEngine,
                            engine2: OptimizedVideoSearchEngine):
        """Benchmark text search performance."""
        logger.info("Benchmarking text search...")
        
        queries = [
            "car approaching cyclist",
            "pedestrian crossing street",
            "vehicle turning left",
            "car overtaking",
            "traffic at intersection"
        ]
        
        times_v2 = []
        times_opt = []
        
        for query in queries:
            # Original
            start = time.time()
            try:
                results1 = engine1.search_by_text(query, top_k=5)
            except:
                pass
            times_v2.append(time.time() - start)
            
            # Optimized
            start = time.time()
            try:
                results2 = engine2.search_by_text(query, top_k=5)
            except:
                pass
            times_opt.append(time.time() - start)
        
        self.results['text_search'] = {
            'original_mean': np.mean(times_v2),
            'optimized_mean': np.mean(times_opt),
            'speedup': np.mean(times_v2) / np.mean(times_opt) if np.mean(times_opt) > 0 else 0
        }
        
        logger.info(f"Text search - Original: {self.results['text_search']['original_mean']:.3f}s")
        logger.info(f"Text search - Optimized: {self.results['text_search']['optimized_mean']:.3f}s")
        logger.info(f"Speedup: {self.results['text_search']['speedup']:.2f}x")
    
    def plot_results(self):
        """Plot benchmark results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of execution times
        categories = ['DB Build', 'Search', 'Search\n(cached)', 'Text\nSearch']
        original_times = [
            self.results['database_build']['original'],
            self.results['search']['original_mean'],
            self.results['search']['original_mean'],  # No cache in original
            self.results['text_search']['original_mean']
        ]
        optimized_times = [
            self.results['database_build']['optimized'],
            self.results['search']['optimized_mean'],
            self.results['search']['optimized_cached_mean'],
            self.results['text_search']['optimized_mean']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, original_times, width, label='Original', color='blue', alpha=0.7)
        ax1.bar(x + width/2, optimized_times, width, label='Optimized', color='green', alpha=0.7)
        
        ax1.set_xlabel('Operation')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Speedup plot
        speedups = [
            self.results['database_build']['speedup'],
            self.results['search']['speedup'],
            self.results['search']['speedup_cached'],
            self.results['text_search']['speedup']
        ]
        
        ax2.bar(categories, speedups, color='orange', alpha=0.7)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Operation')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup (Original / Optimized)')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add speedup values on bars
        for i, v in enumerate(speedups):
            ax2.text(i, v + 0.1, f'{v:.1f}x', ha='center')
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', dpi=150)
        logger.info("Benchmark plot saved to benchmark_results.png")
        
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        logger.info("=" * 60)
        logger.info("NVIDIA Cosmos Video Retrieval - Performance Benchmark")
        logger.info("Comparing original vs optimized implementation")
        logger.info("=" * 60)
        
        # Get video files
        video_files = list(self.video_dir.glob("*.mp4"))
        if not video_files:
            logger.error(f"No video files found in {self.video_dir}")
            return
        
        logger.info(f"Found {len(video_files)} videos for benchmarking")
        
        # Run benchmarks
        engine1, engine2 = self.benchmark_database_build()
        self.benchmark_search(engine1, engine2, video_files)
        self.benchmark_text_search(engine1, engine2)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        logger.info("Optimizations applied (from official NVIDIA implementation):")
        logger.info("✓ FAISS for efficient similarity search")
        logger.info("✓ Batch normalization using faiss.normalize_L2")
        logger.info("✓ Embedding caching for repeated queries")
        logger.info("✓ Parquet format for efficient storage")
        logger.info("")
        logger.info("Average speedups:")
        logger.info(f"  Database building: {self.results['database_build']['speedup']:.1f}x faster")
        logger.info(f"  Video search: {self.results['search']['speedup']:.1f}x faster")
        logger.info(f"  Video search (cached): {self.results['search']['speedup_cached']:.1f}x faster")
        logger.info(f"  Text search: {self.results['text_search']['speedup']:.1f}x faster")
        
        # Plot results
        self.plot_results()


def main():
    """Run benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark video retrieval optimizations")
    parser.add_argument(
        '--video-dir',
        type=str,
        default='/Users/lilyzhang/Desktop/Qwen2.5-VL/nvidia_cosmos/videos/video_database',
        help='Directory containing videos'
    )
    parser.add_argument(
        '--num-queries',
        type=int,
        default=5,
        help='Number of queries to benchmark'
    )
    
    args = parser.parse_args()
    
    # Check if FAISS is available
    try:
        import faiss
        logger.info(f"FAISS version: {faiss.__version__}")
    except ImportError:
        logger.error("FAISS not installed! Install with: pip install faiss-cpu")
        return
    
    # Run benchmark
    benchmark = PerformanceBenchmark(args.video_dir, args.num_queries)
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
