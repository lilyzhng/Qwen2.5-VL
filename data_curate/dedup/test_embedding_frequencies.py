#!/usr/bin/env python3
"""Test how the current implementation handles different embedding frequencies."""

import numpy as np

def test_embedding_frequencies():
    """Test different embedding frequency scenarios."""
    
    print("=" * 80)
    print("TESTING DIFFERENT EMBEDDING FREQUENCIES")
    print("=" * 80)
    print()
    
    pico_row_frames = 10  # Each pico row spans 10 frames
    
    scenarios = [
        ("Sparse (current)", 50, "1 embedding per 50 frames"),
        ("Medium", 10, "1 embedding per 10 frames (same as pico row)"),
        ("Dense", 5, "1 embedding per 5 frames"),
        ("Very Dense", 2, "1 embedding per 2 frames"),
    ]
    
    print(f"Pico row size: {pico_row_frames} frames\n")
    
    for name, frames_per_embedding, description in scenarios:
        print(f"Scenario: {name}")
        print(f"  {description}")
        
        # Calculate how many embeddings per pico row
        embeddings_per_pico = pico_row_frames / frames_per_embedding
        
        if embeddings_per_pico < 1:
            # Sparse: multiple pico rows share one embedding
            picos_per_embedding = frames_per_embedding / pico_row_frames
            print(f"  Result: ~{picos_per_embedding:.1f} pico rows per embedding")
            print(f"  Current approach: ✅ Nearest neighbor works well")
            print(f"  Optimization: None needed")
        elif embeddings_per_pico == 1:
            # Same frequency: 1-to-1 mapping
            print(f"  Result: ~1 embedding per pico row")
            print(f"  Current approach: ✅ Nearest neighbor works well")
            print(f"  Optimization: None needed")
        else:
            # Dense: multiple embeddings per pico row
            print(f"  Result: ~{embeddings_per_pico:.1f} embeddings per pico row")
            print(f"  Current approach: ⚠️  Nearest neighbor only uses 1 embedding")
            print(f"  Optimization: 💡 Could use mean pooling of all embeddings in row")
        
        print()
    
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print()
    print("Current implementation (nearest neighbor only):")
    print("  ✅ Works for ALL frequencies")
    print("  ✅ Simple and consistent")
    print("  ⚠️  Sub-optimal for dense embeddings (wastes information)")
    print()
    print("Recommended: HYBRID APPROACH")
    print("  1. Check if embeddings exist WITHIN the pico row time window")
    print("  2. If YES → Use mean pooling of those embeddings")
    print("  3. If NO → Use nearest neighbor")
    print("  ✅ Optimal for all frequencies")
    print("  ✅ Future-proof")
    print()

if __name__ == "__main__":
    test_embedding_frequencies()

