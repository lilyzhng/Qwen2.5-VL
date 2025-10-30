# 2. Curation Algorithms

### **2.1 Language-controlled Curation**

[TODO] Insert system figure

The curation pipeline adds a layer on top of ALFA Search. It uses existing Cosmos video-level embeddings and LanceDB index, adding capabilities that target the pain points in Section 1.

#### **A. Selection Denoising**

**Existing baseline: Score calibration + thresholding.** Cosine scores are normalized per query. Items below a percentile floor are dropped to trim weak matches.

**Multi-prompt fusion.** Prompt variations are created, then every slice is scored against multiple prompts to improve retrieval accuracy.

**Multi-view fusion.** Scores are fused across multiple cameras:
- FNC has denser pixels and smaller FoV, similar to cosmos training set cameras. Good for small VRU retrieval.
- FWC has wide FoV and handles occluded scenes better.

**Hybrid search.** Embedding search is blended with BM25 (metadata such as scene tags, ego-speed, time-of-day, weather, camera). If accurate metadata exists, we can use it to boost performance.

**Multi-encoder fusion (RRF).** A lightweight second vector (e.g., CLIP/SigLIP frame-pool on sampled frames) can be added. Rankings are fused via Reciprocal Rank Fusion. At query time, both are retrieved and ranks fused for large precision@K gains.

#### **B. VLM Reranker**

**VLM-as-Judge (top-K only, relevance rerank).** A VLM reviews and re-orders top-k. This tightens precision without creating a compute bottleneck.

**Ask:** Request support from SML on setting up Qwen VL inference pipeline with scalex.
- Input: Top-K Manifest (slice\_id, query, metadata, camera, rank, emb\_score)
  - Along with a list of frames (8–16 frames) plus auto-prompt built from object, scene, spatial relation, and behavior.
- Prompt: "Does this video show a person running across the street in front of the vehicle? Return JSON {label:0|1, score:0..1, reason:'\<10 words\>'}."
- Output: slice\_id, vlm\_label (int: 0/1), vlm\_score (float: 0..1), vlm\_reason (str, ≤20 words)

### **2.2 Data Deduplication**

**[P0] Temporal deduplication (within a slice).** Identify key segments within boring slices.
- Use similarity in Cosmos-Embed1 embedding space. Get rid of \~⅔ segments per slice by calculating embedding similarity across N segments within the slice.

**[P1] Semantic deduplication (across slices).** Use similarity in Cosmos-Embed1 embedding space. Sample only 50% from each semantic cluster, or use learned importance per cluster to decide sampling ratio.

**Impact:** Filter out irrelevant or insufficiently challenging data. Much of the current data isn't hard enough or relevant for training.

### **2.3 Data Partition**

[TODO] Insert embedding figure
Figure: Based on vector distance in embedding space, slices\_clean and slices\_obstruction can be separated. Note that the above illustration demonstrates an idea; it is not generated from Cosmos embeddings.

**Split the corpus into slices\_clean and slices\_obstruction.** Provide slice-level metadata such as "Camera obstruction" and "Fog on lens."

**Filter imperfect/bad data.** For example, ISP issues.


