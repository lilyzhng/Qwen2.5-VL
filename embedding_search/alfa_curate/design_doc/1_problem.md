# PT-1756 ALFA Curate: Problem Statement

Authors: Lily Zhang  
Date: Oct 6, 2025

# 1. Problem Statement

### **1.1 Challenges**

To ensure the product we build addresses real user pain points, I spent time interviewing several sensing ML practitioners and reviewing how they currently manage perception data. Through these discussions, three recurring pain points across these conversations: redundancy overhead, imbalance distribution, curation explainability.

**1. Redundancy overhead**  
It is frequently noted that large portions of the corpus consist of visually or behaviorally similar slices that add little new information. Long, uneventful sequences inflate storage and training time while slowing iteration, and simple subsampling misses which segments are actually distinct and worth keeping.

* For example: BPA slices with slow ego speeds; select keyframes within each slice to avoid indexing near-identical content.  
* Motivation: Users should be able to cut compute without losing rare events by down-sampling redundancies with a user-controlled de-dup threshold.

**2. Imbalance Distribution**  
Data is concentrated in common conditions, while rare but safety-critical scenarios remain sparse. This skews learning toward easy contexts and limits robustness in edge cases, especially when metadata is incomplete or noisy.

* For example: Sunny daytime highway dominates while night, adverse weather, and VRU interactions are scarce, impacting modules like BLIS, RCTA, and exit warning.  
* Motivation: Enrich and correct metadata, then apply scenario-aware weighting (e.g., "pedestrian emerging from behind the ego vehicle," trailers, large vehicles) to raise coverage where it matters.

**3. Curation explainability**  
Teams lack a consistent way to assess which slices contribute most to learning and why. Without interpretable signals, prioritization relies on intuition, slowing hard-example discovery and targeted improvements.

* Example: Hard-example mining needs an explanation of why slice A is harder than slice B.  
* Motivation: Attach quality and difficulty signals to metadata and use semantic cues (object type, scene, spatial relationship, actor behavior) to rank and justify selection.

### 1.2 Existing Strategies

ALFA Search today   
The ALFA Search app supports both text-to-video and video-to-video search. Today, we have end-to-end search working: embeddings are generated, indexed in LanceDB, and queried from the ALFA Search UI. In addition, the ALFA Curate has been integrated into Active Learning Framework for prompt based data selection.

While ALFA Search reliably returns neighbors, results can still be noisy and redundant without automatic reranking or dedup control. SemDeDup showed that you can remove ~50% on LAION‑like web data with little loss. Upcoming sections address these problems.
