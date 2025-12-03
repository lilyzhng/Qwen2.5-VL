# 4. Evaluation

## **4.1 Evaluation Objectives**

**Primary Goal:** Reduce the human rejection rate from iMerit reviewers by filtering out low-quality candidates before human review.

**Key Question:** Does VLM Judge effectively reduce false positives that would be rejected by human reviewers?

**Dependency:**
- **Dashboard Ready:** Required for evaluation metrics
- **Without Dashboard:** Evaluation cannot proceed; success metrics cannot be measured
- **Fallback:** If dashboard delayed, manually track rejections in spreadsheet (not scalable)


## **4.2 Evaluation Pipeline**

```
Method A: Baseline (No VLM)
  └─> Embedding Search → Top-K Candidates → Human Review (iMerit) → Rejection Rate

Method B: With VLM Judge
  └─> Embedding Search → VLM Judge Filtering → Top-K Candidates → Human Review (iMerit) → Rejection Rate

Comparison: Rejection Rate Reduction = (Method A Rate - Method B Rate) / Method A Rate
```

## **4.3 Evaluation Methodology**

### **Task 1: Baseline Measurement (No VLM)**

**Objective:** Establish current rejection rate without VLM filtering.

**Steps:**
1. Select 6 test scenarios from Section 4.4
2. Run embedding search with current thresholds
3. Send top-100 candidates per scenario to iMerit for review
4. Record rejection data via Tom Lewis's dashboard
5. Calculate baseline metrics:
   - Rejection rate per scenario
   - Average rejection rate across scenarios

**Output:** Baseline rejection rate for comparison

### **Task 2: VLM Judge Evaluation**

**Objective:** Measure rejection rate reduction with VLM Judge filtering.

**Steps:**
1. Use same 6 scenarios from Task 1
2. Run embedding search → VLM Judge filtering → Select top-100
3. Send VLM-filtered candidates to iMerit for review
4. Record rejection data via Tom Lewis's dashboard
5. Calculate rejection rate reduction:
   - Rejection rate per scenario
   - Average rejection rate across scenarios
   - Reduction: (Method A Rate - Method B Rate) / Method A Rate
6. Analyze filtering effectiveness:
   - How many false positives did VLM filter before human review?
   - What error types does VLM miss that humans still reject?

**Output:** Post-VLM rejection rate and reduction percentage

### **Task 3: VLM Judgment Quality Analysis**

**Objective:** Understand VLM decision quality to improve the system.

**Steps:**
1. Sample 100 candidates per scenario (mix of VLM pass/fail)
2. Human expert reviews each candidate and VLM judgment
3. Calculate VLM metrics:
   - **Precision:** Of candidates VLM passed, how many are truly relevant?
   - **Recall:** Of truly relevant candidates, how many did VLM pass?
   - **Agreement with iMerit:** Do VLM judgments align with iMerit rejections?
4. Error analysis:
   - False Positives: VLM passed but should reject (what types?)
   - False Negatives: VLM rejected but should pass (what types?)
   - Review VLM explanations (observation + reason fields) for quality

**Output:** VLM confusion matrix, error categories, improvement opportunities

## **4.4 Test Scenarios**

### **Scenario Selection Criteria**

Select scenarios that:
1. Are actively used by teams (UM, SP, PAD, SE)
2. Cover diverse difficulty levels (easy, medium, hard)
3. Represent common failure modes of embedding search

### **Recommended Test Scenarios**

| ID | Scenario | Difficulty | Expected Baseline Rejection Rate | Target Post-VLM Rate |
|----|----------|------------|----------------------------------|---------------------|
| S1 | "Pedestrian crossing the street" | Easy | ~15-20% | <10% |
| S2 | "Vehicle turning right at intersection" | Easy | ~20-25% | <12% |
| S3 | "Cyclist approaching from left" | Medium | ~30-40% | <15% |
| S4 | "Pedestrian partially occluded by parked vehicle" | Medium | ~35-45% | <20% |
| S5 | "Vehicle lane change behind slow-moving truck" | Hard | ~40-50% | <25% |
| S6 | "Rainy night intersection with pedestrian crossing" | Hard | ~45-55% | <30% |

### **Sample Size Guidelines**

- **Per Scenario:** 100-200 candidates for statistically significant results
- **Total:** 600-1200 candidates across 6 scenarios
- **Timing:** 2-4 weeks for human review depending on iMerit capacity

## **4.5 Evaluation Metrics**

| Metric | Formula | Target |
|--------|---------|--------|
| **Rejection Rate** | (# Rejected) / (# Total Sent) × 100% | - |
| **Rejection Rate Reduction** | (Method A - Method B) / Method A × 100% | >50% |
| **Precision@K** | (# Relevant in Top-K) / K | >80% @100 |
| **VLM Agreement** | (# VLM-Human Agree) / (# Total) | - |

**VLM-Human Agreement Categories:**
- True Positive: VLM pass + Human accept
- True Negative: VLM reject + Human would reject  
- False Positive: VLM pass + Human reject
- False Negative: VLM reject + Human would accept

## **4.6 Dependencies**

### **Critical Dependency: Tom Lewis's Rejection Rate Dashboard**

**Blocker:** Without the dashboard from Tom Lewis's labeling team to track rejection rates, we cannot measure retrieval quality or evaluate VLM Judge effectiveness.

**Dashboard Requirements:**

The evaluation depends on a dashboard that tracks:
1. **Method Identification: Rejection rate per batch** (with timestamp)
   - Required to compare Method A (baseline) vs. Method B (with VLM)
   - Must support time-series visualization to track trends
2. **Rejection reasons** (categorized by type)
   - Categories: wrong object, wrong action, wrong scene, background only, sensor issues, etc.
   - Required to understand what types of errors VLM Judge catches or misses
3. **Per-scenario breakdown**
   - Rejection rate and count for each test scenario (S1-S6)
   - Required to identify which scenarios benefit most from VLM filtering
4. **VLM confidence distribution** (Method B only)
   - Distribution of VLM confidence scores for passed candidates
   - Required to analyze if low-confidence VLM passes correlate with human rejections
5. **Candidate-level data export**
   - Required for Task 3 analysis (VLM judgment quality assessment)
6. **Cost analysis metrics**
   - Human review time per candidate
   - Time saved by VLM pre-filtering (for Method B)
7. **Data Schema:** Each candidate record should include:
   ```
   {
     "slice_id": str,
     "scenario": str,
     "batch_id": str,  // "method_a" or "method_b"
     "rejected": bool,
     "rejection_reason": str,  // e.g., "wrong_object", "wrong_action", "background_only"
     "reviewer_id": str,
     "timestamp": datetime
   }
   ```
