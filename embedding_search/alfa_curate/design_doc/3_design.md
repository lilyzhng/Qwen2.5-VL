# PT-1756 ALFA Curate: Design

Authors: Lily Zhang  
Date: Oct 6, 2025

# 2. Design

### 2.1 Architecture Overview

The pipeline is another layer on top of ALFA search. It builds upon the existing cosmos video-level embeddings and lancedb index, but adds new capabilities directly targeting the pain

**A. Data Separation**

* Split the corpus into slices\_clean, slices\_obstruction  
* score every slice against multiple prompts. "Camera obstruction", "Fog on lens"  
* \[P2\] Ask from [Shivam Gautam](mailto:sgautam@lat.ai): How can we filter imperfect/bad data?   
  * ISP issues  
  * Labeling errors   
  * Sensor obstruction


**B. Retrieval Denoising**

* Score calibration \+ thresholding (existing baseline). Normalize cosine scores per query and drop items below a percentile floor to trim weak matches.

* Hybrid search: Blend Cosmo cosine with a BM25 field and key metadata (scene tags, ego-speed bands, time-of-day, weather, camera). Tune a single α on a 1–2k judged set.  
  * This will keep Cosmo as primary; α≈0.7 cosine / 0.3 sparse is a strong starting point, which suppresses off-topic data points with near-zero infra.


* Geometry fix (one-time transform). Cosmo's space is anisotropic ("clubbed"). Whitening \+ re-norm spreads neighborhoods so cosine actually reflects the nuances you care about (occlusion, actor interactions, motion patterns). That's often enough to reduce noisy neighbors before you spend cycles on heavier rerankers.

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
