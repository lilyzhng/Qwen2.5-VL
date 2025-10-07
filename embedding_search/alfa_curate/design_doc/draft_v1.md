# PT-1756 ALFA Curate: All You Need is Automatic Data Curation

Authors: Lily Zhang  
Date: Oct 6, 2025

Summary

The design doc focuses on solving data problems that can empower ML engineers to both train faster and train better; and recommendations on what we can/should build between Offboard Models and Scalable ML teams. 

# 1\. Problem Statement

### **1.1 Challenges**

To ensure the product we build addresses real user pain points, I spent time interviewing several sensing ML practitioners and reviewing how they currently manage perception data. Through these discussions, three recurring pain points across these conversations: redundancy overhead, imbalance distribution, curation explainability.

**1\. Redundancy overhead**  
It is frequently noted that large portions of the corpus consist of visually or behaviorally similar slices that add little new information. Long, uneventful sequences inflate storage and training time while slowing iteration, and simple subsampling misses which segments are actually distinct and worth keeping.

* For example: BPA slices with slow ego speeds; select keyframes within each slice to avoid indexing near-identical content.  
* Motivation: Users should be able to cut compute without losing rare events by down-sampling redundancies with a user-controlled de-dup threshold.

**2\. Imbalance Distribution**  
Data is concentrated in common conditions, while rare but safety-critical scenarios remain sparse. This skews learning toward easy contexts and limits robustness in edge cases, especially when metadata is incomplete or noisy.

* For example: Sunny daytime highway dominates while night, adverse weather, and VRU interactions are scarce, impacting modules like BLIS, RCTA, and exit warning.  
* Motivation: Enrich and correct metadata, then apply scenario-aware weighting (e.g., “pedestrian emerging from behind the ego vehicle,” trailers, large vehicles) to raise coverage where it matters.

**3\. Curation explainability**  
Teams lack a consistent way to assess which slices contribute most to learning and why. Without interpretable signals, prioritization relies on intuition, slowing hard-example discovery and targeted improvements.

* Example: Hard-example mining needs an explanation of why slice A is harder than slice B.  
* Motivation: Attach quality and difficulty signals to metadata and use semantic cues (object type, scene, spatial relationship, actor behavior) to rank and justify selection.

### 1.2 Existing Strategies

ALFA Search today   
The ALFA Search app supports both text-to-video and video-to-video search. Today, we have end-to-end search working: embeddings are generated, indexed in LanceDB, and queried from the ALFA Search UI. In addition, the ALFA Curate has been integrated into Active Learning Framework for prompt based data selection.

While ALFA Search reliably returns neighbors, results can still be noisy and redundant without automatic reranking or dedup control. SemDeDup showed that you can remove \~50% on LAION‑like web data with little loss. Upcoming sections address these problems. 

# 2\. Design

### 2.1 Architecture Overview

The pipeline is another layer on top of ALFA search. It builds upon the existing cosmos video-level embeddings and lancedb index, but adds new capabilities directly targeting the pain

**A. Data Separation**

* Split the corpus into slices\_clean, slices\_obstruction  
* score every slice against multiple prompts. “Camera obstruction”, “Fog on lens”  
* \[P2\] Ask from [Shivam Gautam](mailto:sgautam@lat.ai): How can we filter imperfect/bad data?   
  * ISP issues  
  * Labeling errors   
  * Sensor obstruction


**B. Retrieval Denoising**

* Score calibration \+ thresholding (existing baseline). Normalize cosine scores per query and drop items below a percentile floor to trim weak matches.

* Hybrid search: Blend Cosmo cosine with a BM25 field and key metadata (scene tags, ego-speed bands, time-of-day, weather, camera). Tune a single α on a 1–2k judged set.  
  * This will keep Cosmo as primary; α≈0.7 cosine / 0.3 sparse is a strong starting point, which suppresses off-topic data points with near-zero infra.


* Geometry fix (one-time transform). Cosmo’s space is anisotropic (“clubbed”). Whitening \+ re-norm spreads neighborhoods so cosine actually reflects the nuances you care about (occlusion, actor interactions, motion patterns). That’s often enough to reduce noisy neighbors before you spend cycles on heavier rerankers.

* Multi-encoder fusion (RRF). Add a lightweight second vector (e.g., CLIP/SigLIP frame-pool on a few sampled frames). Fuse rankings via Reciprocal Rank Fusion.  
  * This stores a second vector column; at query time, retrieve with both, fuse ranks. large precision@K gains for minimal cost.

* VLM-as-Judge (top-K only, relevance rerank): Use a VLM only to re-order a small shortlist after the cheaper steps above. This tightens semantic precision without turning the VLM into a compute bottleneck.  
  * When: After hybrid \+ geometry \+ RRF, take top-K \= 150 candidates.  
  * What the VLM sees: A compact visual (8–16 frames or a 2–3 s GIF) plus a short query (or auto-prompt built from object, scene, spatial relation, behavior).

    

**C. Data Deduplication**  
Goal: Dataset restructuring to increase the number of rows, reduce the number of frames, and randomize the dataset

* \[P0\] Temporal deduplication of segments (within a slice)  
  * Motivating example from [Jack Arendt](mailto:jarendt@lat.ai): identify key segments within boring parking slices  
  * Option A: use ego speed based heuristics to decide stride  
  * Option B: Via similarity in Cosmos-Embed1 embeddings space  
    * Get rid of \~⅔ segments for a slice  
    * Calculate embedding for every frame and figure out relevance to closest   
* \[P1\] Perceptual deduplication (across slices)  
  * Similarity in Cosmos-Embed1 embeddings space  
  * Sample only 50% from each semantic cluster OR use learned importance per cluster to decide on sampling ratio   
* Semantic deduplication of segments   
  * Maybe when we have second stage VLM  
* Impact: Filter out irrelevant or insufficiently challenging data, because much of the current data isn't hard enough or relevant for training

**D. Data Rebalance** 

* Coverage balance (breadth). Can maximize diversity in embedding space to ensure diverse samples that properly represent the output distribution.   
  * Ensure the curated set is not dominated by easy, common conditions (e.g., sunny-day highway) and includes enough rare but important conditions (night, adverse weather, obstruction, VRU interactions). This prevents blind spots and improves generalization.  
* Hard Example Mining (depth).  Create challenging datasets based on related PCA FP scenarios, this is to curate datasets that are difficult for model training  
  * Intentionally up-weight hard examples that match PCA-related prompts (your known false-positive patterns). This accelerates learning on failure modes—i.e., hard-example mining—without overwhelming the model with only hard cases.  
* How:   
  * Compute one signal from the embedding space: density\_score\_i  
    * density\_bin ∈ {low, mid, high} where high \= rare, low \= redundant.  
  * Compute one task signal you already have: task relevance (e.g. PCA relevance)  
    * softly prioritize hard examples tied to your PCA false-positive prompts.  
    * pca\_relevance\_i  
  * Make a single final sampling weight  
    * blend diversity (density) and difficulty (PCA relevance) into the loader/sampler.  
    * wi​=(1+λdiv​ ⋅ density\_scorei​) × (1+λpca ​⋅ pca\_relevancei​)


### 2.2 Model Choice

# 3\. Requirements

### 3.1 Success Criteria

### 3.2 Out of Scope

# 4\. Development Plan

### 4.1 Dependency

### Applications / Active learning strategies

* Replace current strategy for obstruction scenarios?  
  * Currently using CLIP, which tagged all night scene as obstruction scenes

### Questions

* What is the priority? What matters the most for you?  
* Whether the data curation runs as an automated process or is based on manual queries  
* Whether to look purely in embedding space or use more statistical approaches  
* How to best structure the overall data pipeline architecture

**Automate Data Curation**

* Automatic queries generation, can pull metrics from bigquery, rank slices based on ped detection in the front frustum   
* Started to log metrics, can log more including training loss, use VLM agent to check loss changes over the time.

### 4.1 Productionization 

# Appendix

# Reference

[\[1\]](https://lancedb.com/docs/search/hybrid-search) Hybrid Search in LanceDB [https://lancedb.com/docs/search/hybrid-search](https://lancedb.com/docs/search/hybrid-search)  
[\[2\]](https://scholar.google.com/scholar_lookup?arxiv_id=2405.08209#d=gs_cit&t=1759805469248&u=%2Fscholar%3Fq%3Dinfo%3A8CHtZYpXH44J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) Hong, Rachel, et al. "Who's in and who's out? A case study of multimodal CLIP-filtering in DataComp." Proceedings of the 4th ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization. 2024\.  
[\[3\]](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LamRA_Large_Multimodal_Model_as_Your_Advanced_Retrieval_Assistant_CVPR_2025_paper.html) Liu, Yikun, et al. "Lamra: Large multimodal model as your advanced retrieval assistant." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025\.  
[\[4\]](https://arxiv.org/abs/2501.07972) Xu, Yifang, et al. "Zero-shot video moment retrieval via off-the-shelf multimodal large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39\. No. 9\. 2025\.  
[\[5\]](https://huggingface.co/spaces/HuggingFaceM4/FineVision) FineVision: Open Data Is All You Need [https://huggingface.co/spaces/HuggingFaceM4/FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision)  
[\[6\]](https://github.com/huggingface/large-scale-image-deduplication) Hugging Face Image Deduplication Toolkit [https://github.com/huggingface/large-scale-image-deduplication](https://github.com/huggingface/large-scale-image-deduplication)  
\[\] [datologyai](https://www.datologyai.com/blog/multimodal-plus-blogpost) CLIP Gets a Data Upgrade: Outperforming SoTA with Improved Data Curation Only

### 

