#!/usr/bin/env python3
"""Analyze if temporal adaptive subsampling works well with sparse embeddings."""

import numpy as np
import pyarrow as pa
import sys
from unittest.mock import MagicMock

# Setup mocks
EMBEDDING = "embedding"
FRAMES_KEY = "frames"
IDENTIFIERS = "identifiers"
SLICE_ID = "slice_id"
START_NS = "start_ns"
TIMESTAMP_NS = "timestamp_ns"

constants_mock = MagicMock()
constants_mock.EMBEDDING = EMBEDDING
constants_mock.FRAMES_KEY = FRAMES_KEY
constants_mock.IDENTIFIERS = IDENTIFIERS
constants_mock.SLICE_ID = SLICE_ID
constants_mock.START_NS = START_NS
constants_mock.TIMESTAMP_NS = TIMESTAMP_NS

def mock_get_chunks(array, chunk_size):
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]

index_writer_mock = MagicMock()
index_writer_mock.get_chunks = mock_get_chunks

def mock_map_remote_to_args(func, args_list, disable_ray=False, dynamically_adjust_memory=False):
    for args in args_list:
        try:
            yield func(args)
        except Exception as e:
            yield e

ray_map_mock = MagicMock()
ray_map_mock.map_remote_to_args = mock_map_remote_to_args

sys.modules['autonomy'] = MagicMock()
sys.modules['autonomy.perception'] = MagicMock()
sys.modules['autonomy.perception.datasets'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels.pico'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels.pico.config'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.alfa_curate'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.alfa_curate.utils'] = MagicMock()
sys.modules['kits'] = MagicMock()
sys.modules['kits.scalex'] = MagicMock()
sys.modules['kits.scalex.dataset'] = MagicMock()
sys.modules['kits.scalex.dataset.constants'] = constants_mock
sys.modules['kits.scalex.dataset.index'] = MagicMock()
sys.modules['kits.scalex.dataset.index.index_writer'] = index_writer_mock
sys.modules['kits.scalex.dataset.stage_str'] = MagicMock()
sys.modules['kits.scalex.pipeline'] = MagicMock()
sys.modules['kits.scalex.pipeline.ray'] = MagicMock()
sys.modules['kits.scalex.pipeline.ray.map'] = ray_map_mock
sys.modules['platforms'] = MagicMock()
sys.modules['platforms.lakefs'] = MagicMock()
sys.modules['platforms.lakefs.client'] = MagicMock()

from pico import compute_embedding_change_rates

def analyze_sparse_embeddings():
    """Analyze whether temporal subsampling can distinguish pico rows with sparse embeddings."""
    
    print("=" * 90)
    print("ANALYSIS: Temporal Adaptive Subsampling with Sparse Embeddings (50 frames)")
    print("=" * 90)
    print()
    
    # Realistic scenario: 20 FPS, embeddings every 50 frames (2.5s), pico rows every 0.5s
    duration_sec = 10
    pico_interval_sec = 0.5
    embedding_interval_sec = 2.5  # Every 50 frames at 20 FPS
    
    num_pico_rows = int(duration_sec / pico_interval_sec)  # 20 rows
    num_embeddings = int(duration_sec / embedding_interval_sec) + 1  # 5 embeddings
    
    print(f"Setup:")
    print(f"  Duration: {duration_sec}s")
    print(f"  Pico rows: {num_pico_rows} (every {pico_interval_sec}s)")
    print(f"  Embeddings: {num_embeddings} (every {embedding_interval_sec}s = 50 frames)")
    print(f"  Ratio: {int(embedding_interval_sec / pico_interval_sec)} pico rows per embedding")
    print()
    
    # Create pico rows
    pico_rows = []
    for i in range(num_pico_rows):
        time_ns = int(i * pico_interval_sec * 1e9)
        pico_rows.append({
            SLICE_ID: "slice_1",
            FRAMES_KEY: [{TIMESTAMP_NS: time_ns}]
        })
    
    pico_table = pa.Table.from_pylist(pico_rows)
    
    # Create embeddings with progressively changing values
    embeddings_list = [[float(i), 0, 0] for i in range(num_embeddings)]
    embedding_timestamps = [int(i * embedding_interval_sec * 1e9) for i in range(num_embeddings)]
    
    embedding_table = pa.Table.from_pydict(
        {
            EMBEDDING: embeddings_list,
            START_NS: embedding_timestamps,
            IDENTIFIERS: [{SLICE_ID: "slice_1"} for _ in range(num_embeddings)],
        },
        schema=pa.schema([
            pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3])),
            pa.field(START_NS, pa.int64()),
            pa.field(IDENTIFIERS, pa.struct([pa.field(SLICE_ID, pa.string())])),
        ]),
    )
    
    # Compute change rates
    result_table = compute_embedding_change_rates(
        pico_table, embedding_table, temporal_window_size=5.0
    )
    
    velocities = result_table.column("embedding_velocity").to_pylist()
    accelerations = result_table.column("embedding_acceleration").to_pylist()
    
    print("Results:")
    print()
    print(f"{'Row':<5} {'Time(s)':<8} {'Velocity':<12} {'Acceleration':<12} {'Analysis'}")
    print("-" * 90)
    
    # Track unique values
    unique_velocities = set()
    unique_accelerations = set()
    velocity_groups = {}
    
    for i in range(len(velocities)):
        time_s = i * pico_interval_sec
        vel_str = f"{velocities[i]:.6f}" if velocities[i] is not None else "None"
        acc_str = f"{accelerations[i]:.6f}" if accelerations[i] is not None else "None"
        
        # Determine which embedding this row uses (approximate)
        emb_idx = int(time_s / embedding_interval_sec + 0.5)
        emb_idx = min(emb_idx, num_embeddings - 1)
        
        analysis = f"Uses emb {emb_idx}"
        
        # Track velocity grouping
        vel_key = vel_str if velocities[i] is None else f"{velocities[i]:.6f}"
        if vel_key not in velocity_groups:
            velocity_groups[vel_key] = []
        velocity_groups[vel_key].append(i)
        
        if velocities[i] is not None:
            unique_velocities.add(round(velocities[i], 6))
        if accelerations[i] is not None:
            unique_accelerations.add(round(accelerations[i], 6))
        
        print(f"{i:<5} {time_s:<8.1f} {vel_str:<12} {acc_str:<12} {analysis}")
    
    print()
    print("=" * 90)
    print("CONCERN VALIDATION:")
    print("=" * 90)
    print()
    
    print(f"Total pico rows: {num_pico_rows}")
    print(f"Total embeddings: {num_embeddings}")
    print(f"Expected pico rows per embedding: ~{int(embedding_interval_sec / pico_interval_sec)}")
    print()
    
    print(f"Unique velocity values: {len(unique_velocities)}")
    print(f"Unique acceleration values: {len(unique_accelerations)}")
    print()
    
    print("Velocity grouping (rows with same velocity):")
    for vel_key, rows in sorted(velocity_groups.items()):
        if len(rows) > 1:
            print(f"  Velocity {vel_key}: rows {rows} ({len(rows)} rows)")
    print()
    
    # The concern
    print("🔍 IS THE CONCERN VALID?")
    print()
    
    if len(unique_velocities) < num_pico_rows / 2:
        print("⚠️  YES - CONCERN IS VALID!")
        print()
        print("Issues identified:")
        print(f"  1. Only {len(unique_velocities)} unique velocities for {num_pico_rows} pico rows")
        print(f"  2. Multiple consecutive rows share the same velocity")
        print(f"  3. Temporal subsampling can't distinguish between these rows")
        print()
        print("Why this happens:")
        print(f"  • Embeddings every {embedding_interval_sec}s means ~5 pico rows share each embedding")
        print("  • Rows sharing the same current & reference embeddings → same velocity")
        print("  • Limited granularity for adaptive subsampling decisions")
        print()
        print("Impact:")
        print("  • Subsampling by velocity/acceleration threshold will drop GROUPS of rows")
        print("  • NOT fine-grained selection of individual interesting frames")
        print("  • Temporal dynamics at pico row level (0.5s) are NOT captured")
        print()
    else:
        print("✅ Concern appears to be less severe than expected")
        print(f"   We have {len(unique_velocities)} unique velocities for {num_pico_rows} rows")
    
    print("=" * 90)
    print("RECOMMENDATION:")
    print("=" * 90)
    print()
    print("For truly effective temporal adaptive subsampling with sparse embeddings:")
    print()
    print("Option 1: INCREASE EMBEDDING FREQUENCY (RECOMMENDED)")
    print("  • Generate embeddings every 10-20 frames instead of 50")
    print("  • This provides better temporal granularity")
    print("  • Allows subsampling to distinguish individual pico rows")
    print()
    print("Option 2: USE CURRENT IMPLEMENTATION WITH CAVEATS")
    print("  • Accept that subsampling works at embedding granularity (~2.5s)")
    print("  • NOT at pico row granularity (~0.5s)")
    print("  • Still useful for removing redundant ~2.5s segments")
    print()
    print("Option 3: USE DIFFERENT SUBSAMPLING STRATEGY")
    print("  • Clustering-based selection")
    print("  • Random sampling")
    print("  • Or wait until denser embeddings are available")
    print()

if __name__ == "__main__":
    analyze_sparse_embeddings()

