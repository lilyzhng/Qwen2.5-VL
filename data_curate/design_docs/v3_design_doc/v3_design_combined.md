# PT-1938 ALFA 2.0: Reason and Scale Your Dataset

Authors: [Lily Zhang](mailto:xzhang@lat.ai)  
Creation Date: Oct 27, 2025  
Jira Ticket: [PT-1938](https://latitudeai.atlassian.net/browse/PT-1938)  
Comment Close Date:   
Version: 0.1  
Status: **DRAFT** / ABORTED / READY FOR APPROVAL /APPROVED  
Have you submitted an invention disclosure? **No**/Yes

Stakeholders / Sign-offs

| Stakeholder Name (Team) | Status (Pending / Approved \<date\> / Changes Requested |
| :---- | :---- |
| [Nikita Jaipuria](mailto:njaipuria@lat.ai) (Offboard Models) |  |
| [Richard Kwant](mailto:rkwant@lat.ai) (Scalable ML) |  |
| Jack / Shivam  |  |
| Mrigesh |  |
| Michael / Nicolas  |  |
| Tom Lewis |  |
| Xiufeng / Yijung,  |  |

Summary

This design doc focuses on empowering scalable data curation with multimodal reasoning for accurate and efficient dataset selection.

Glossary

| Test-time scaling |  |
| :---- | :---- |
| Reranker | a second-stage model that reorders a first-stage retriever’s top-K results using richer signals (e.g., cross-encoder/VLM scores) to improve final relevance at low extra latency. |

# 1\. Introduction

## 1.1 Current State

ALFA Search is actively used by various teams (UM, SP, PAD, SE) for text-to-video and video-to-video search ([https://to/ALFA](https://to/ALFA)). LaTS ([https://to/lats](https://to/lats)) has adopted the text-to-video functionality and is used across Latitude.

ALFA Curate has been integrated into the Active Learning Framework for language-guided data selection. Users define scenarios (e.g., "People running across the road") with top\_k and similarity\_threshold to curate unlabeled data of interest for prioritized labeling. ALFA Curate allows users to filter retrieved results not only by similarity threshold, but also by applying hybrid SQL-based filters over metadata stored in BigQuery. This enables more precise data selection based on criteria such as geographic region, time-of-day, weather conditions, tags, annotations, and boolean logic.

## 1.1 Why Reason About Your Data ?

ALFA Curate returns slices with a user-defined threshold, but requires manual tuning. Multimodal embedding-based selection has information loss and does not guarantee fine-grained details. Due to fundamental limitations of embedding-based retrieval [\[7\]](https://scholar.google.com/scholar_lookup?arxiv_id=2508.21038), which map whole clips to a joint space for scene-level semantics, vector A (cow+grass+sky) can be very close to vector B (grass+sky). These coarse embeddings suffer from background domination, making it challenging to ensure whether a retrieved slice actually contains the queried objects, actions, or spatial relationships. Additionally, SQL filters rely on pre-existing annotations, which may be incomplete.

Most importantly, neither mechanism can judge false signals on temporal dynamics, spatial relationships, or multi-attribute complex properties. Today's system lacks a way to reason about the data selection quality. What's desired? We need an intelligent way to review, judge, and refine retrieved results.

# 2\. System Design

## 2.1 Two-Stage Architecture

\[TODO\] Update system figure

This design introduces a **two-stage retrieval-refinement architecture**. Stage 1 uses NVIDIA Cosmos embedding search to retrieve candidate slices based on semantic similarity threshold. Stage 2 applies Qwen 3.0 VL multimodal reasoning to re-score and rerank candidates based on semantic relevance. As VLMs are good at binary decisions, we ask the model to perform binary judgements, determining whether each video truly contains the user desired data targets, paired with a confidence threshold to filter uncertain predictions. 

**Impact of the work**

* Explainability: The VLM judge provides interpretable selection reasoning through structured observation and justification fields. Users can understand why each video was selected or rejected, enabling faster debugging and prompt refinement.  
* Shortened Feedback Loop:  By adding an intelligent reasoning layer between embedding retrieval and human review, the system enables rapid iteration on data selection without waiting for full human evaluation cycles. It can autonomously filter low-quality candidates, reducing the time for dataset delivery.  
* Assist iMerit Human Expert: The VLM judge serves as a first-pass filter to assist iMerit reviewers by removing false positives before human review. This reduces reviewer workload and allows experts to focus on edge cases requiring human judgment.  
* Foundation for Future Work: While labeling QA is out of scope for this design, the VLM reasoning infrastructure establishes a foundation for future applications such as label validation, annotation quality assessment, and quality control in the labeling pipeline

**Pseudocode**

| //=====================Stage 1: Embedding Search================================ |
| :---- |
| 1\. Initialize empty result set R 2\. Load embedding index from vector database 3\. For each text query q in scenario:      a. Compute embedding vector E(q)      b. Retrieve top-N video slices where cosine\_similarity(E(q), E(slice)) \> similarity\_threshold      c. Add retrieved slices to R 4\. Deduplicate R by slice\_id 5\. Sort R by embedding similarity (descending) |
| //======================Stage 2: VLM Judge=====================================  |
| 6\. If VLM judge disabled:      Return R 7\. Select top-K candidates from R (default K=100) 8\. Initialize VLM model M 9\. Initialize empty verified set V 10\. For each candidate c in top-K:      a. Extract N frames from video slice (e.g., 8 frames at 1 FPS)      b. Construct prompt: "Does this video show \[query\]? Return JSON with match, confidence, observation, reason"      c. Inference: (match, confidence, observation, reason) ← M(frames, prompt)      d. If match \== true AND confidence ≥ confidence\_threshold:           Add c to V 11\. Return V |


**Model Choice.** Qwen 3.0 VL ([HELP-67607](https://latitudeai.atlassian.net/browse/1/HELP-67607)) has been approved internally. The system can leverage the following Qwen 3.0 VL capabilities:

| Capability | Tasks for VLM Judge |
| :---- | :---- |
| **Spatial Perception** | Validate spatial relationships (e.g., "car approaching cyclist from left") |
| **Temporal Modeling** | Verify temporal sequences and motion patterns (e.g., "vehicle entering intersection then turning") |
| **Multi-attribute Reasoning** | Evaluate combination of scene attributes (e.g., "rainy night with occluded pedestrian") |
| **2D/3D Grounding** | Validate object positions and inter-object relationships |
| **Long Context Processing** | Process full video clips (up to 256K tokens), handle hours-long video with full recall and second-level indexing. |

**Top-K Filtering.** The VLM judge operates on the top-K results from embedding search (default K=100). VLM inference is compute-intensive, as each candidate requires processing temporal frames through a large vision-language model. The VLM-as-Judge reranker operates on top-K results to balance precision and computational cost.

**Frame Sampling**. Unlike the embedding model, which uses fixed frame counts, the VLM supports dynamic FPS sampling, enabling the model to comprehend videos at various sampling rates. In the first stage, the default configuration (1.0 FPS, max 8 frames) is bound by the requirements for NVIDIA cosmos embed model, but  in VLM judge second stage, users can increase segment\_desired\_fps to capture finer temporal dynamics. This flexibility allows the VLM to extract temporal details that the embedding representation may miss.

**Adaptive Fetch:** How to handle VLM judge filters bad candidates but creates fewer than k_desired selections? The system adaptively fetches additional candidates from embedding search. This approach uses real-time VLM pass rate to estimate how many additional candidates are needed, avoiding both computational waste.

```py
Input: k_desired, initial_batch_size=100, max_iterations=3, min_pass_rate=0.05
Output: vlm_verified_candidates

1. Initialize: verified = [], offset = 0, iteration = 0
2. While len(verified) < k_desired AND iteration < max_iterations:
   a. Determine batch size:
      If iteration == 0: batch_size = initial_batch_size
      Else:
         pass_rate = len(verified) / offset
         If pass_rate < min_pass_rate: break
         batch_size = int((k_desired - len(verified)) / pass_rate * 1.2)
   b. Fetch candidates from embedding search (offset, batch_size)
      If no more candidates: break
   c. Apply VLM judge, add verified candidates where match == true AND confidence >= threshold
   d. Update: offset += batch_size, iteration += 1
3. Return verified
Example:
- Easy query: Fetch 100 → 82 pass → Done (82% pass rate)
- Hard query: Fetch 100 → 18 pass → Fetch 546 more → 98 pass → Done (18% pass rate)
```

**Stride strategy:** Starting from the lowest-ranked candidate (e.g., \#100 in top-100), the VLM evaluates with a configurable stride (default stride=20). For scenarios where no candidates in the bottom 40-60 pass filtering, this approach reduces inference cost by 40-60% compared to sequential top-to-bottom processing.

```py
Input: candidates (sorted by embedding similarity, descending), stride (default = 20)
Output: vlm_verified_candidates

// ============= a. Stride Search =============
1. Start from lowest-ranked candidate (e.g., #100)
2. Evaluate candidates at stride intervals: #100, #80, #60, #40, #20...
3. For each candidate:
    If fails: skip all candidates between current and previous check
    If passes: record as first_success, proceed to b.
4. If no candidate passes: return empty list

// ============= b. Binary Search =============
5. Perform binary search between last_failure and first_success
6. Example: If #80 failed and #60 passed, test #70
    If #70 passes: test #75
    If #70 fails: test #65
7. Continue until precise cutoff boundary is located

// ============= c. Collect Results =============
8. Evaluate all candidates from rank 1 to cutoff_rank
9. Return candidates where match == true AND confidence >= confidence_threshold
```

**VLM-based Re-ranking:** The Stride strategy assumes embedding similarity is reliable (top-ranked = most relevant). However, embedding-based retrieval suffers from background domination and information loss, where high-similarity candidates may not contain the queried objects or actions. When embedding scores are unreliable, the system re-ranks verified candidates by VLM confidence scores instead of embedding similarity, ensuring the final selections are ordered by multimodal reasoning relevance rather than coarse semantic similarity.

```py
Input: verified_candidates (with embedding_score and vlm_confidence)
Output: reranked_candidates

1. For each candidate in verified_candidates:
   score = alpha * vlm_confidence + (1 - alpha) * embedding_score
   Where alpha controls ranking strategy:
   - alpha = 1.0: Pure VLM ranking (unreliable embeddings)
   - alpha = 0.5: Hybrid ranking (balance both signals)
   - alpha = 0.0: Pure embedding ranking (reliable embeddings, default)

2. Sort candidates by score (descending)

3. Return top k_desired candidates
```


## 2.2 Prompt Design

The VLM judge uses a two-part prompt structure: a system prompt that establishes the model's role, and a user prompt template that formats the judgment query with explicit output specifications.

**System Prompt:** The domain-specific framing improves relevance judgments for driving scenarios (e.g., understanding safety-critical events like pedestrians crossing, vehicle lane changes).

```py
You are an expert autonomous driving systems analyst.
```

**User Prompt Template:**

```py
{query}
Analyze the video frames and respond ONLY with valid JSON in this exact format:
{{
  "query": "<the question being evaluated>",
  "match": true,
  "confidence": 0.95,
  "observation": "Describe what you observe in the video frames",
  "reason": "Explain why you made this judgment based on your observations"
}}
Where:
- query: the question being evaluated
- match: true if the scenario matches, false otherwise
- confidence: your confidence level from 0.0 (not confident) to 1.0 (very confident)
- observation: what you see in the video frames (be specific)
- reason: why you gave this judgment based on what you observed
```

**Design rationale:**

1. **Query-conditioned Judgement:** The scenario query ({query}) appears before the output format specification. The {query} placeholder is populated at runtime with scenario-specific text (e.g., "Is there a pedestrian crossing the street?"). This ensures the VLM makes judgement oriented around the target query.   
2. **Structured JSON schema:** The prompt includes a complete JSON example with field names, types, and descriptions. This eliminates ambiguity and ensures feasible output.   
3. **Instructions specification:** The schema is shown both as a JSON example and as a bulleted list with field explanations. This redundancy reinforces the required structure and clarifies semantic intent for each field.  
4. **Observation-reason separation:** The prompt requires both observation (what is seen) and reason (why it satisfies/misses the query). This two-step structure improves judgment quality by forcing the model to ground its decision in visual evidence before providing justification.  
5. **Confidence threshold:** The prompt specifies confidence as a float from 0.0 to 1.0 rather than categorical labels (low/medium/high). This enables fine-grained threshold tuning for precision-recall trade-offs. 

**Structured output:** The VLM returns JSON with five fields: query, match (boolean), confidence (0-1), observation (visual description), and reason (justification). This structured format serves multiple purposes:

* **Interpretability:** Observation and reason fields provide debugging signals for false positives/negatives, enabling prompt refinement.  
* **Confidence calibration:** Confidence scores allow threshold tuning. Lower thresholds increase recall; higher thresholds increase precision.  
* **Iterations:** Storing VLM reasoning alongside selected slices creates an auditable record for data selection quality validation.

**Filtering logic:** Candidates pass the filter if match \== true and confidence \>= threshold (default 0.6). 

* This design captures two modes: (1) semantic mismatch (match=False), where the video content does not satisfy the query; (2) low confidence (confidence \< threshold), where the VLM is uncertain. Separating these conditions allows handling e.g., logging low-confidence rejections separately for prompt debugging.  
* Filtered logic can be materialized to store along with embedding scores, VLM confidence, and reasoning traces for downstream user visibility.

## 2.3 Inference Cost Optimizations

**Adaptive Processing Order.** The VLM processes candidates in ascending order of embedding similarity (lowest to highest rank). Since embedding similarity correlates with VLM match likelihood, low-ranked candidates typically fail verification, enabling early termination and cost savings.

**Resource Allocation.** GPU requirements scale with model size. The 2B parameter Qwen-VL model fits in \~6GB VRAM, allowing multiple workers per GPU (num\_gpus\_per\_worker=0.5). The 32B model requires 64GB, requiring full GPU allocation. We will benchmark with different model sizes to find the best tradeoff between inference cost and model capabilities. This configurability of different numbers of workers per GPU enables cost-performance optimization: smaller models for high-throughput scenarios, larger models where accuracy is critical.

## 2.4 Data Models

**Judge Configuration**

```py
@dataclass
class VLMJudgeConfig:
    """Configuration for VLM judge filtering."""
    
    #: Enable VLM judge filtering. If False, skip VLM filtering entirely.
    enable_vlm_judge: bool = True
    
    #: Maximum number of candidates to send to VLM judge (top K from embedding search).
    max_candidates_for_vlm: int = 100
    
    #: The VLM model path (HuggingFace ID or local path).
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    
    #: Confidence threshold for VLM judge (0.0-1.0).
    vlm_confidence_threshold: float = 0.6
    
    #: Desired frame rate for sampling video segments (frames per second).
    segment_desired_fps: float = 1.0
    
    #: Maximum number of frames to send to VLM per video segment.
    max_frames_per_segment: int = 8
    
    #: Maximum number of tokens to generate for each judgment.
    max_new_tokens: int = 256

    #: Number of GPUs to allocate per VLM judge worker.
    num_gpus_per_worker: float = 1.0

    #: GPU type to request (e.g., "A100", "H100").
    gpu_type: str = "A100"
```

**Judge Output**

```py
@dataclass
class VLMJudgeResult:
    """Result from VLM judge evaluation."""
    #: The query that was evaluated.
    query: str

    #: Whether the video matches the query (True = match, False = no match).
    match: bool

    #: Confidence score from the VLM (0.0-1.0). Higher means more confident.
    confidence: float

    #: What the VLM observed in the video frames.
    observation: str

    #: Reasoning for why the VLM made this judgment.
    reason: str

    #: Raw response text from the VLM.
    raw_response: str
```

# 3\. Requirements

## 3.1 Success Criteria

* **Rejection Rate Reduction**: \>30% reduction in iMerit human rejection rate averaged across test scenarios  
* **Precision@K:** \>80% precision on top K candidates for user-defined scenarios  
* **Cost Savings:** Reduce human review time by 30% through VLM pre-filtering.  
* **Explainability:** VLM explanations (observation \+ reason fields) help reviewers understand decisions

## 3.2 Dependencies

**AV packages:** Qwen VL has a dependency on **torch==2.4.1+cu121** and **transformers==4.57.0,** because **bluemind-common**v1.4.0 hard-pins **torch==2.1.1+cu121 ,** we cannot land the code in AV. We need bluemind’s support to bump up the torch version.

**Tooling Support:** Without the dashboard from Tom Lewis/labeling team to track rejection rates, we cannot measure retrieval quality or evaluate VLM Judge effectiveness. See section 4 for requirements for the rejection dashboard.

**Fallback**: If the rejection dashboard is delayed, we can use the sheet of human evaluation

# 4\. Evaluation

## 4.1 Objectives

This evaluation measures the success criteria defined in Section 3.1. The goal is to track how ALFA releases improve human rejection rates over time by filtering out low-quality candidates. The key question is: does each new ALFA version effectively reduce false positives that would be rejected by human reviewers?

Release Comparison

| Release | Pipeline | Expected Outcome |
| :---- | :---- | :---- |
| **ALFA 1.0 (Baseline)** | Embedding Search → Top-K Candidates → Human Review | All 100 candidates sent to humans |
| **ALFA 2.0 (With VLM Judge)** | Embedding Search → VLM Judge Filtering → Top-K Candidates → Human Review | VLM filters bad candidates → only X candidates sent to humans (X \< 100\) |

## 4.2 Mock Dashboard Design

Historical Comparison: Compare human rejection rates and review times between ALFA 1.0 (baseline without VLM) and ALFA 2.0 (with VLM Judge). It leverages evolving production data from each release to measure VLM Judge impact on human reviewer efficiency.

**Metrics**

* Human rejection rate (% of candidates humans reject)  
* Candidates sent to humans (number of candidates requiring human review)  
* Human review time per candidate (average minutes per candidate)  
* Total human review time per batch (primary metric: \# candidates × time per candidate)  
* Precision (% of candidates humans accept)


Note: Human review time per candidate may increase in ALFA 2.0 (VLM filters easy cases, leaving harder ones), but total review time per batch should decrease significantly due to volume reduction. Example: if per-candidate time ↑20% but volume ↓60%, then total time \= 120% × 40% \= 48% of original → 52% saved.

**Mock Design**  
**\[Figures\]**

**Dashboard Requirements**. The evaluation depends on a dashboard that tracks: 

1. Historical human rejection rate comparison  
   * Showing human rejection rate across ALFA releases  
   * Release comparisons: ALFA version, human rejection rate, candidates sent to humans, time per candidate, total time per batch, precision@100  
2. Candidate volume reduction  
   * Shows how scene candidates volume change across different releases  
3. Human review time per candidate  
   * Track whether per-candidate review time increases in ALFA 2.0, this helps understand if remaining candidates are harder edge cases  
4. Total human review time savings per batch  
   * Primary metric: total human review time per batch (e.g., per 100 candidates requested), which shows time saved (target: \>30% reduction in total time)  
5. Human rejection reasons  
   * iMerit human experts will check different boxes for quick rejection reason logging.   
   * Different rejection types: wrong object type, wrong behavior, wrong scene, wrong spatial relationship, background only

## 4.3 Evaluation Steps

Below scenarios cover diverse difficulty levels (easy, medium, hard), and are actively used by teams (UM, SP, PAD, SE). We measure human rejection rates historically comparing ALFA 1.0 (baseline, no VLM) vs ALFA 2.0 (with VLM Judge).

| ID | Scenario | Difficulty | ALFA 1.0 | ALFA 2.0 |
| :---- | :---- | :---- | :---- | :---- |
| S1 | "Pedestrian crossing the street" | Easy | TBD | \<10% |
| S2 | "Vehicle turning right at intersection" | Easy | TBD | \<12% |
| S3 | "Cyclist approaching from left" | Medium | TBD | \<15% |
| S4 | "Pedestrian partially occluded by parked vehicle" | Medium | TBD | \<20% |
| S5 | "Vehicle lane change behind slow-moving truck" | Hard | TBD | \<25% |
| S6 | "Rainy night intersection with pedestrian crossing" | Hard | TBD | \<30% |

**Steps**  
1\. Select 6 test scenarios (S1-S6 covering easy, medium, hard difficulty). For each scenario and ALFA version:

* Request top-100 candidates from embedding search  
* Extract human review data from iMerit dashboard  
* Record: rejection reason, review time per candidate, timestamp, number of candidates sent to humans

2\. Calculate metrics per scenario:

* Human rejection rate: (\# rejected by humans) / (\# sent to humans)  
* Average review time per candidate  
* Total human review time per batch: (\# candidates sent) × (avg time per candidate)  
* Precision: (\# accepted) / (\# sent to humans)

3: ALFA 1.0 Baseline (No VLM Judge)

* All 100 candidates sent to human reviewers  
* Output: ALFA 1.0 baseline metrics

4: ALFA 2.0 with VLM Judge

* VLM Judge filters candidates first → only X candidates sent to humans (where X ≤ 100\)  
* Calculate same metrics as Task 1  
* Compare to ALFA 1.0:  
  * Human rejection rate reduction: (ALFA 1.0 Rate \- ALFA 2.0 Rate) / ALFA 1.0 Rate  
  * Candidate volume reduction: (100 \- X) / 100  
  * Total human time saved: (ALFA 1.0 Total Time \- ALFA 2.0 Total Time) / ALFA 1.0 Total Time  
* Output: ALFA 2.0 metrics showing volume reduction and time savings

5: Analyze Impact and Identify optimization opportunities

* Compare rejection reasons between ALFA 1.0 and 2.0 (which types decreased most?)  
* Analyze per-candidate time changes (is ALFA 2.0 time higher? indicates harder edge cases)  
* Calculate time savings breakdown: volume reduction benefit vs. harder candidate penalty  
* Identify where VLM Judge works best and where it struggles, find optimization opportunities

6\. Desired Outcomes

| Metric | Formula | Target |
| :---- | :---- | :---- |
| **Human Rejection Rate** | (\# Rejected by Humans) / (\# Sent to Humans) × 100% | Decreasing (ALFA 2.0 \< ALFA 1.0) |
| **Human Rejection Rate Reduction** | (ALFA 1.0 Rate \- ALFA 2.0 Rate) / ALFA 1.0 Rate × 100% | \>30% |
| **Candidates Sent to Humans** | Number of candidates requiring human review after VLM filter | Decreasing (ALFA 2.0 \< ALFA 1.0) |
| **Candidate Volume Reduction** | (100 \- X) / 100 × 100% where X \= candidates sent in ALFA 2.0 | \>40% |
| **Total Human Review Time per Batch** | (\# Candidates Sent) × (Avg Time per Candidate) | Decreasing |
| **Total Human Time Saved** | (ALFA 1.0 Total \- ALFA 2.0 Total) / ALFA 1.0 Total × 100% | \>30% |
| **Human Review Time per Candidate** | Average seconds per candidate | Diagnostic (may increase) |
| **Precision@K** | (\# Accepted by Humans) / (\# Sent to Humans) | \>80% |

# 5\. Reference

[\[1\]](https://lancedb.com/docs/search/hybrid-search) Hybrid Search in LanceDB [https://lancedb.com/docs/search/hybrid-search](https://lancedb.com/docs/search/hybrid-search)  
[\[2\]](https://scholar.google.com/scholar_lookup?arxiv_id=2405.08209#d=gs_cit&t=1759805469248&u=%2Fscholar%3Fq%3Dinfo%3A8CHtZYpXH44J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den) Hong, Rachel, et al. "Who's in and who's out? A case study of multimodal CLIP-filtering in DataComp." Proceedings of the 4th ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization. 2024\.  
[\[3\]](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_LamRA_Large_Multimodal_Model_as_Your_Advanced_Retrieval_Assistant_CVPR_2025_paper.html) Liu, Yikun, et al. "Lamra: Large multimodal model as your advanced retrieval assistant." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025\.  
[\[4\]](https://arxiv.org/abs/2501.07972) Xu, Yifang, et al. "Zero-shot video moment retrieval via off-the-shelf multimodal large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39\. No. 9\. 2025\.  
[\[5\]](https://huggingface.co/spaces/HuggingFaceM4/FineVision) FineVision: Open Data Is All You Need [https://huggingface.co/spaces/HuggingFaceM4/FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision)  
[\[6\]](https://github.com/huggingface/large-scale-image-deduplication) Hugging Face Image Deduplication Toolkit [https://github.com/huggingface/large-scale-image-deduplication](https://github.com/huggingface/large-scale-image-deduplication)  
[\[7\]](https://scholar.google.com/scholar_lookup?arxiv_id=2508.21038) Weller, Orion, et al. "On the theoretical limitations of embedding-based retrieval." arXiv preprint arXiv:2508.21038 (2025).  
[\[8\]](https://scholar.google.com/scholar_lookup?arxiv_id=2303.09540) Abbas, Amro, et al. "Semdedup: Data-efficient learning at web-scale through semantic deduplication." arXiv preprint arXiv:2303.09540 (2023).  
[\[9\]](https://www.researchgate.net/publication/390718175_CLIP-CID_Efficient_CLIP_Distillation_via_Cluster-Instance_Discrimination)  Efficient CLIP Distillation via Cluster-Instance Discrimination 

# 

### 

