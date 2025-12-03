# 2. System Architecture

## **2.1 Two-Stage Architecture**

[TODO] Insert system figure
This design introduces a **two-stage retrieval-refinement architecture**. Stage 1 uses NVIDIA Cosmos embedding search to retrieve candidate slices based on semantic similarity threshold. Stage 2 applies Qwen 3.0 VL multimodal reasoning to re-score and rerank candidates based on semantic relevance. As VLMs are good at binary decisions, we ask the model to perform binary judgements, determining whether each video truly contains the user desired data targets, paired with a confidence threshold to filter uncertain predictions. The VLM-as-Judge reranker operates on top-K results to balance precision and computational cost.

## **2.2 System Architecture**

### **Algorithm Overview**

```
Algorithm: Two-Stage Video Slice Selection

Input: 
  - scenario (contains text queries)
  - config (contains similarity_threshold, confidence_threshold, top_K)
Output: 
  - verified_slices (list of video slices passing both stages)

// ============= Stage 1: Embedding Search =============
1. Initialize empty result set R
2. Load embedding index from vector database
3. For each text query q in scenario:
     a. Compute embedding vector E(q)
     b. Retrieve top-N video slices where cosine_similarity(E(q), E(slice)) > similarity_threshold
     c. Add retrieved slices to R
4. Deduplicate R by slice_id
5. Sort R by embedding similarity (descending)

// ============= Stage 2: VLM Verification =============
6. If VLM judge disabled:
     Return R
7. Select top-K candidates from R (default K=100)
8. Initialize VLM model M
9. Initialize empty verified set V
10. For each candidate c in top-K:
     a. Extract N frames from video slice (e.g., 8 frames at 1 FPS)
     b. Construct prompt: "Does this video show [query]? Return JSON with match, confidence, observation, reason"
     c. Inference: (match, confidence, observation, reason) ← M(frames, prompt)
     d. If match == true AND confidence ≥ confidence_threshold:
          Add c to V
11. Return V
```

### **Top-K Filtering Rationale**

The VLM judge operates on the top-K results from embedding search (default K=100). VLM inference is compute-intensive, as each candidate requires processing 8 frames through a large vision-language model. For scenarios generating 500 embedding matches, top-100 filtering delivers 80% cost reduction (5× speedup) while preserving recall, since relevant videos typically rank within the top 100. Lower K risks missing true positives; higher K doubles inference cost without proportional gains.

### **VLM Capabilities**

The system leverages the following Qwen 3.0 VL capabilities:

| Capability | Application in VLM Judge |
|------------|--------------------------|
| **Spatial Perception** | Validate spatial relationships (e.g., "car approaching cyclist from left") |
| **Temporal Modeling** | Verify temporal sequences and motion patterns (e.g., "vehicle entering intersection then turning") |
| **Multi-attribute Reasoning** | Evaluate conjunctions of scene attributes (e.g., "rainy night with occluded pedestrian") |
| **Visual Recognition** | Confirm object types, states, and semantic labels |
| **2D/3D Grounding** | Validate object positions and inter-object relationships |
| **Long Context Processing** | Process full video clips (up to 256K tokens) |

## **2.3 VLM Judge Design**

### **Frame Sampling**

The VLM operates on the same video segments retrieved by the embedding model but can apply denser temporal sampling. Unlike the embedding model, which uses fixed frame counts, the VLM supports dynamic FPS sampling, enabling the model to comprehend videos at various sampling rates.

[TODO] check the best fps for Qwen VL

The default configuration (1.0 FPS, max 8 frames) provides a conservative baseline, but users can increase `segment_desired_fps` to capture finer temporal dynamics—e.g., 2.0 FPS for fast-moving scenarios or events requiring frame-level precision. This flexibility allows the VLM to extract temporal details that the embedding representation may miss.

### **Prompt Design**

The VLM judge uses a two-part prompt structure defined in `prompts.yaml`: a system prompt that establishes the model's role, and a user prompt template that formats the judgment query with explicit output specifications.

**System Prompt:**
```
You are an expert autonomous driving systems analyst.
```

This brief role assignment primes the VLM to interpret video content through the lens of autonomous driving applications. The domain-specific framing improves relevance judgments for driving scenarios (e.g., understanding safety-critical events like pedestrians crossing, vehicle lane changes).

**User Prompt Template:**
```
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

1. **Query-first ordering:** The scenario query (`{query}`) appears before the output format specification. This ensures the VLM processes the judgment criteria before considering response structure, reducing the risk of format-driven rather than content-driven responses.

2. **Explicit JSON schema:** The prompt includes a complete JSON example with field names, types, and descriptions. This eliminates ambiguity and ensures parseable output. The `ONLY` directive and "exact format" phrasing discourage free-form responses that would fail JSON parsing.

3. **Dual-format specification:** The schema is shown both as a JSON example and as a bulleted list with field explanations. This redundancy reinforces the required structure and clarifies semantic intent for each field.

4. **Observation-reason separation:** The prompt requires both `observation` (what is seen) and `reason` (why it satisfies/violates the query). This two-step reasoning structure improves judgment quality by forcing the model to ground its decision in visual evidence before providing justification.

5. **Continuous confidence:** The prompt specifies confidence as a float from 0.0 to 1.0 rather than categorical labels (low/medium/high). This granular output enables fine-grained threshold tuning for precision-recall trade-offs.

All prompts are centralized in `prompts.yaml` for versioning and A/B testing. The `{query}` placeholder is populated at runtime with scenario-specific text (e.g., "Is there a pedestrian crossing the street?").

### **Output Schema and Filtering**

**Structured output:** The VLM returns JSON with five fields: query, match (boolean), confidence (0-1), observation (visual description), and reason (justification). This structured format serves multiple purposes:
- **Interpretability:** Observation and reason fields provide debugging signal for false positives/negatives, enabling prompt refinement.
- **Confidence calibration:** Continuous confidence scores allow threshold tuning. Lower thresholds increase recall; higher thresholds increase precision.
- **Auditability:** Storing VLM reasoning alongside selected slices creates an auditable record for data quality validation.

The prompt enforces strict JSON formatting to ensure parseable responses. Parse failures are handled conservatively (match=False, confidence=0.0) to prevent false positives from malformed outputs.

**Filtering logic:** Candidates pass the filter if `match == true` and `confidence >= confidence_threshold` (default 0.6). The `match` field is the **primary criterion**—VLMs excel at binary classification of whether a video contains specific objects or behaviors. The confidence threshold serves as a **complementary filter** to exclude uncertain predictions. This dual-condition design reflects two failure modes: (1) semantic mismatch (match=False), where the video content does not satisfy the query; (2) low confidence (confidence < confidence_threshold), where the VLM detects a match but is uncertain about the classification. Separating these conditions allows differential handling—e.g., logging low-confidence rejections separately for prompt debugging.

Filtered candidates are materialized to storage with embedding scores, VLM confidence, and reasoning traces for downstream analysis.

## **2.4 Optimization Strategies**

### **Adaptive Processing Order**

The VLM processes candidates in ascending order of embedding similarity (lowest to highest rank). Since embedding similarity correlates with VLM match likelihood, low-ranked candidates typically fail verification, enabling early termination and cost savings.

**Adaptive stride strategy:**

```
Algorithm: Stride-based VLM Processing

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

For scenarios where no candidates in the bottom 40-60 pass filtering, this approach reduces inference cost by 40-60% compared to sequential top-to-bottom processing.

**Alternative strategies under consideration:**

1. **Exponential backoff:** Instead of fixed stride, use exponentially increasing gaps (check #100, #95, #85, #70, #40...). This accelerates search in sparse regions but requires more careful tuning.

2. **Confidence-based thresholding:** Analyze the embedding similarity score distribution to predict a cutoff threshold without VLM evaluation. For example, if scores drop sharply below a certain rank (e.g., similarity < 0.3 at #85), skip all candidates below that point. This requires offline calibration to establish score-to-quality mappings.

3. **Batch sampling with early stopping:** Process candidates in batches (e.g., batches of 10). If an entire batch fails, skip the next batch and move to a higher rank. This leverages batch inference efficiency while maintaining early termination benefits.

4. **Two-stage lightweight filtering:** Use a smaller, faster VLM (e.g., Qwen-VL-2B) for initial coarse filtering on all top-K candidates, then apply the full model (e.g., Qwen-VL-32B) only to candidates that pass the lightweight filter. This trades two inference passes for higher overall throughput.

The current adaptive stride approach balances simplicity, effectiveness, and minimal hyperparameter tuning. Alternative strategies may offer marginal gains but introduce additional complexity or require offline calibration.

### **Resource Allocation**

GPU requirements scale with model size. The 2B parameter Qwen-VL model fits in ~6GB VRAM, allowing multiple workers per GPU (`num_gpus_per_worker=0.5`). The 32B model requires 64GB, necessitating full GPU allocation. This configurability enables cost-performance optimization: smaller models for high-throughput scenarios, larger models where accuracy is critical.

## **2.5 Data Models**

### **Configuration**

```python
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

### **Output**

```python
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
