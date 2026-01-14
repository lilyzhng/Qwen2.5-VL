# VRU Preservation for Pico Deduplication

## Quick Start

```python
from autonomy.perception.datasets.human_labels.pico.pico_config import HumanLabelsPicoConfig

config = HumanLabelsPicoConfig(
    enable_vru_preservation=True,  # Preserve pedestrians, cyclists, etc.
    embedding_dedupe_threshold=0.5,
)
```

## Overview

Ensures frames with pedestrians, cyclists, and motorcyclists are never removed during deduplication by using existing human annotations instead of unreliable embeddings.

## Configuration

Add these settings to your `HumanLabelsPicoConfig`:

```python
from autonomy.perception.datasets.human_labels.pico.pico_config import HumanLabelsPicoConfig

config = HumanLabelsPicoConfig(
    # Enable VRU preservation
    enable_vru_preservation=True,
    
    # Specify which object classes to preserve (optional - defaults shown below)
    vru_classes=[
        "pedestrian",
        "cyclist",
        "motorcyclist",
        "bicycle",
        "motorcycle",
        "person",
    ],
    
    # Optional: specify a different annotations dataset
    # If not set, uses human_labels_gold_reference
    vru_annotations_reference="",
    
    # ... other config parameters
)
```

## How It Works

### 1. Build VRU Frame Index (One-time)

When deduplication starts, the system:

```python
# Scans the gold dataset annotations
vru_frame_index = build_vru_frame_index(
    gold_dataset=gold_dataset,
    vru_classes=config.vru_classes,
)

# Returns: set of (slice_id, timestamp_ns) tuples
# Example: {("slice_abc123", 1234567890), ("slice_def456", 9876543210), ...}
```

### 2. Deduplication with VRU Checking

During deduplication, each embedding is checked:

```python
for embedding_idx in candidate_for_exclusion:
    if check_embedding_has_vru(embedding_idx, embeddings_table, vru_frame_index):
        # Preserve this frame - move to included set
        included_indices.add(embedding_idx)
    else:
        # Safe to exclude
        excluded_indices.add(embedding_idx)
```

### 3. Result

- **VRU frames**: Always preserved, even if visually similar to other frames
- **Non-VRU frames**: Deduplicated normally based on embedding similarity

## Example Usage

### Basic Usage

```python
from autonomy.perception.datasets.human_labels.pico.pico_config import HumanLabelsPicoConfig
from autonomy.perception.datasets.human_labels.pico.ingredients import (
    dedupe_and_get_include_and_exclude_maps
)
from platforms.lakefs.client import LakeFS

# Configure with VRU preservation
config = HumanLabelsPicoConfig(
    enable_vru_preservation=True,
    embedding_dedupe_threshold=0.5,  # L2 distance threshold
    num_kmeans_clusters=1024,
)

# Run deduplication
lakefs = LakeFS()
include_map, exclude_map = dedupe_and_get_include_and_exclude_maps(config, lakefs)

# include_map contains all VRU frames + diverse non-VRU frames
# exclude_map contains only non-VRU duplicate frames
```

### Custom VRU Classes

For specific use cases, customize which classes to preserve:

```python
config = HumanLabelsPicoConfig(
    enable_vru_preservation=True,
    vru_classes=[
        "pedestrian",
        "child",
        "stroller",
        "wheelchair",
        "cyclist",
        "scooter_rider",
    ],
)
```

### Logging

The system provides detailed logs:

```
INFO: VRU preservation enabled. Building VRU frame index...
INFO: Building VRU frame index for classes: ['pedestrian', 'cyclist', 'motorcyclist', ...]
INFO: Built VRU frame index: 45,231 frames with VRU objects
INFO: Assigned clusters. Deduplicating embeddings.
DEBUG: VRU preservation: kept 127 frames with VRU objects in this cluster
INFO: <<<Deduplicated embeddings. Included 250,000, excluded 150,000.>>>
```

## Performance Characteristics

### Time Complexity

- **VRU index building**: O(N × M) where N = number of frames, M = avg annotations per frame
  - Typically runs once at the start
  - ~1-2 minutes for 1M frames (depends on annotation density)

- **VRU checking during dedup**: O(1) per frame
  - Fast hash table lookup: `(slice_id, timestamp) in vru_frame_index`

### Memory

- **VRU index**: ~32 bytes per VRU frame
  - 100K VRU frames ≈ 3.2 MB
  - Negligible compared to embeddings

### Scalability

The approach scales well because:
1. **One-time index building**: Only scan annotations once
2. **Efficient lookups**: Hash-based set membership testing
3. **Ray parallelization**: Deduplication tasks share the same VRU index

## Annotation Structure Compatibility

The VRU index builder tries multiple common annotation field names:

```python
# Supported annotation fields
annotation_fields = ['annotations', 'objects', 'labels', 'detections']

# Supported class name fields within annotations
class_fields = ['class_name', 'class', 'label', 'type', 'object_class']
```

This makes the system compatible with various annotation formats without modification.

## Testing

Run the VRU preservation test:

```bash
pytest data_curate/dedup/pico_test.py::test_vru_preservation -v
```

Expected output:
```
test_vru_preservation PASSED
Without VRU: included={0, 2}, excluded={1, 3}
With VRU: included={0, 1, 2}, excluded={3}
```

## Troubleshooting

### Issue: "VRU frame index built: 0 frames with VRU objects"

**Cause**: Annotation field names don't match expected formats

**Solutions**:
1. Check your annotation structure in the gold dataset
2. Verify `vru_classes` match the actual class names in annotations
3. Add custom field names to `build_vru_frame_index()` if needed

### Issue: "Failed to build VRU frame index"

**Cause**: Cannot access gold dataset or annotations

**Solutions**:
1. Verify `human_labels_gold_reference` is correct
2. Check LakeFS permissions
3. Ensure gold dataset has annotation data

### Issue: VRU frames still being excluded

**Cause**: Timestamp or slice_id mismatch between annotations and embeddings

**Solutions**:
1. Verify embeddings and annotations are from the same dataset version
2. Check that `START_NS` in embeddings matches `timestamp_ns` in annotations
3. Enable debug logging to see VRU checking details

## Design Rationale

### Why Not Use Embeddings for VRU Detection?

We considered using NVIDIA Cosmo or other embeddings to classify VRU presence:

❌ **Embedding-based approach**:
- Pro: No need to access annotations
- Con: Embeddings don't reliably capture object-level information
- Con: Requires additional inference (computational cost)
- Con: Black box - can't guarantee VRU detection
- Con: May miss rare VRU cases

✅ **Annotation-based approach**:
- Pro: Ground truth from human labels
- Pro: 100% accurate for labeled data
- Pro: No additional inference needed
- Pro: Frame-level precision
- Pro: Efficient with simple lookups

### Why Frame-Level Instead of Slice-Level?

**Slice-level** (10-second chunks):
- ❌ Coarse granularity - might preserve 9 seconds of non-VRU frames
- ❌ Might miss VRUs that appear briefly

**Frame-level**:
- ✅ Precise - only preserve frames with actual VRU objects
- ✅ Efficient - still O(1) lookup per frame
- ✅ Maximum data quality

## Future Enhancements

Potential improvements:

1. **Multi-level preservation**: Different thresholds for different object classes
   ```python
   preservation_classes = {
       "pedestrian": {"threshold": 0.0, "min_distance_m": None},  # Preserve all
       "cyclist": {"threshold": 0.3, "min_distance_m": 50},  # Preserve if close
       "vehicle": {"threshold": 0.8, "min_distance_m": None},  # Only very unique
   }
   ```

2. **Spatial filtering**: Only preserve VRUs within certain distance from ego vehicle

3. **Temporal context**: Preserve N frames before/after VRU appearance

4. **Rare pose preservation**: Keep unusual VRU poses even if scene is similar

## Summary

The VRU preservation feature provides:

- ✅ **Guaranteed safety**: Never lose critical VRU examples
- ✅ **High accuracy**: Uses human-labeled ground truth
- ✅ **Efficient**: Fast lookups with minimal memory overhead
- ✅ **Flexible**: Easy to configure which classes to preserve
- ✅ **Scalable**: Works with large datasets via Ray parallelization

Enable it with a single config flag to ensure your autonomous driving dataset maintains comprehensive VRU coverage while benefiting from deduplication.
