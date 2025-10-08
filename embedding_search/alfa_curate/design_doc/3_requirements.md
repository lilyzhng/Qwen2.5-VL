# 3. Requirements

### 3.1 Success Criteria

**Language-controlled Curation**
- Selection denoising reduces retrieval noise with minimal manual tuning. Precision@K improves by at least 20% over baseline score calibration.
- VLM Reranker achieves >80% precision on top-100 candidates for user-defined scenarios.

**Data Deduplication**
- Temporal deduplication reduces redundant segments by \~⅔ within boring slices while preserving key moments.
- Semantic deduplication removes \~30% of redundant data across slices.
- Training time and storage costs are reduced proportionally to deduplication rate.

**Data Partition**
- Slices are partitioned into slices\_clean and slices\_obstruction with >85% accuracy.
- Slice-level metadata (camera obstruction, fog, ISP issues) is provided for filtered data.

### 3.2 Out of Scope
- Automatic labeling or ground truth generation (VLM only re-ranks).
- Online curation (pipeline runs offline on pre-embedded data).
- Perceptual hash-based deduplication (focus is on embedding-based methods).


