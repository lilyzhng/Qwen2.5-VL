# 4. Development Plan

### 4.1 Dependency

### 4.2 Timeline 

| Phase | Work Item | Description | Priority | ETA |
|-------|-----------|-------------|----------|-----|
| **Phase 1: Selection Denoising** | | | | **2 Weeks** |
| | Multi-prompt fusion | Generate prompt variations, score slices against multiple prompts | P0 | Week 1-2 |
| | Multi-view fusion | Fuse scores across FNC/FWC cameras | P1 | Week 2 |
| | Hybrid search (BM25) | Blend embedding search with metadata (scene tags, ego-speed, time-of-day, weather, camera) | P1 | Week 2-3 |
| | Multi-encoder fusion (RRF) | Add lightweight second vector (CLIP/SigLIP), fuse rankings via Reciprocal Rank Fusion | P2 | Week 3 |
| **Phase 2: Temporal Deduplication** | | | | **2 Weeks** |
| | Within-slice deduplication | Remove ~⅔ redundant segments per slice using Cosmos-Embed1 similarity | P0 | Week 4-5 |
| | Evaluation & tuning | Validate key moments preserved, tune similarity thresholds | P0 | Week 5 |
| | Cross-slice clustering | Cluster slices using Cosmos-Embed1 embeddings | P1 | Week 9 |
| | Cluster-based sampling | Sample 50% per cluster or use learned importance weights | P1 | Week 9-10 |
| | Evaluation | Validate ~30% data reduction with minimal loss | P1 | Week 10 |
| **Phase 3: VLM Reranker** | | | | **3 Weeks** |
| | SML infrastructure setup | Work with SML team on Qwen VL inference pipeline with scalex | P0 | Week 6 |
| | VLM-as-Judge implementation | Top-K relevance reranking with auto-prompt from metadata | P0 | Week 6-7 |
| | Evaluation & iteration | Target >80% precision on top-100 candidates | P0 | Week 7-8 |
| **Phase 5: Data Partition** | | | | **Week 11-12** |
| | Clean vs obstruction classifier | Split corpus into slices_clean and slices_obstruction | P1 | Week 11 |
| | Metadata generation | Add slice-level metadata (camera obstruction, fog, ISP issues) | P1 | Week 11-12 |
| | Validation | Target >85% partition accuracy | P1 | Week 12 |
| **Phase 6: Integration & Evaluation** | | | | **Week 13-14** |
| | End-to-end pipeline | Integrate all components into unified curation pipeline | P0 | Week 13 |
| | Full evaluation | Measure Precision@K improvement (target: +20% over baseline) | P0 | Week 13-14 |
| | Documentation & handoff | User guide, API docs, deployment guide | P0 | Week 14 |

**Total estimated timeline: 14 weeks (~3.5 months)**

**Notes:**
- Priorities: P0 = Critical path, P1 = High value, P2 = Nice-to-have
- Selection denoising starts immediately as foundation
- Temporal dedup (P0) prioritized before semantic dedup (P1)
- VLM reranker blocks on SML team support (Week 6)
- Phases 4-5 can partially overlap with Phase 3
- Buffer time included for iteration and bug fixes

