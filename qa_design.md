# VLM-Powered Labeling QA System

## 1. Problem Statement

### Current State
- **100% of labeled frames** require human review in the current QA pipeline
- Deterministic validators auto-resolve only **65–75%** of cases
- Remaining **25–35%** of uncertain cases still require manual review
- Human review is expensive, slow, and does not scale

### Failure Modes That Deterministic Rules Cannot Solve

| Failure Mode | Example | Why Rules Fail |
|--------------|---------|----------------|
| **Semantic Misclassification** | Pedestrian ↔ Motorcycle, Cyclist → Vehicle | Geometry/LiDAR cannot encode semantic meaning; size and motion are ambiguous at range |
| **Policy/Context Errors** | Parked vehicle marked as active SRO | Map data noisy; rules over-reject edge cases (shoulder, construction) |
| **Grouped Actor Confusion** | Multiple cyclists labeled as single vehicle | LiDAR points don't encode grouping semantics; no cheap rule reliably fixes this |
| **Missed Detections** | Clearly visible object at 90m not detected | Deterministic detectors don't reason over *absence*; only flag presence |

---

## 2. Proposed Solution

### Architecture: VLM as Targeted Arbiter

Use VLM **only** when:
1. Classifier likelihood < threshold (low confidence)
2. Map rules + perception disagree
3. High downstream severity

**Do NOT use VLM** for (Low ROI):
- Cases where deterministic rules are confident
- Pure geometric validation
- Broad scanning (too expensive)
- Box tightness / geometry refinement
- Yaw smoothing
- Z-height correction
- Stagnant ghost tracks
- Cases where rules already agree


---

### Three Priority Experiments

#### Experiment 1: Semantic Class Disambiguation
**Trigger:** Low classifier likelihood OR class conflict with size/motion priors

**Dataset Setup:**
- Use nuScenes mini
- Take GT 3D boxes for: pedestrian, bicycle, motorcycle
- Create synthetic label errors (relabel cyclist → pedestrian, etc.)
- Project 3D box → camera → crop ROI

**VLM Prompt Template:**
```
You are validating a 3D-labeled traffic participant.
Look ONLY at the highlighted region.

Choose exactly ONE class from:
{PEDESTRIAN, CYCLIST, MOTORCYCLIST, REGULAR_VEHICLE, TRUCK, BUS}

If the object is unclear, return REVIEW.
Do NOT guess. Do NOT hallucinate.
List up to 3 visible cues you used.
Output JSON only.
```

**Expected Output:**
```json
{
  "decision": "ACCEPT | REVIEW",
  "class": "CYCLIST",
  "evidence": ["visible bicycle frame", "upright riding posture"]
}
```

---

#### Experiment 2: False Positive / Ghost Box Detection
**Trigger:** Deterministic system suspects false negative (good visibility, no occlusion)

**Dataset Setup:**
- Take GT boxes
- Create fake boxes (shift laterally into empty space, rotate into background)
- Project and crop

**VLM Prompt Template:**
```
You are checking whether an object exists in the highlighted region.
Question: Is there a real physical traffic participant present?

Answer ONLY one: {YES, NO, UNCERTAIN}

If YES, specify which type at a high level.
If NO or UNCERTAIN, do not explain further.
Do not hallucinate objects.
```

**Expected Output:**
```json
{
  "exists": "YES",
  "type": "REGULAR_VEHICLE"
}
```

---

#### Experiment 3: Policy / SRO Eligibility Check
**Trigger:** Map rules + perception disagree

**Dataset Setup:**
- Select vehicles near curb / shoulder / parking
- Keep GT box correct
- Pretend downstream system marks them as "active SRO candidate"

**VLM Prompt Template:**
```
You are validating policy eligibility.
Based on the image:

Is this vehicle plausibly part of active drivable traffic flow?

Choose ONE: {ON_DRIVABLE_FLOW, PARKED_OR_SHOULDER, OFF_DRIVABLE}

Return REVIEW if uncertain.
Briefly cite visual context (1 sentence max).
```

**Expected Output:**
```json
{
  "classification": "PARKED_OR_SHOULDER",
  "note": "Vehicle is stationary next to curb, not aligned with lane"
}
```

---

#### Bonus: Template D — ID Continuity Across Occlusion
**Trigger:** Track resumes after occlusion with appearance drift

**VLM Prompt Template:**
```
Compare BEFORE and AFTER images.
Are these the same physical object?

Answer one: {SAME_OBJECT, DIFFERENT_OBJECT, UNCERTAIN}

List 2 visual cues.
```

---

## 3. Measurable Impact

### Target Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Human review rate | 100% | 25–30% |
| **Review reduction** | — | **70–75%** |
| VLM call rate | — | **< 5% of total frames** |
| ACCEPT precision | — | High (VLM as arbiter, not primary) |

### VLM-Addressable Error Distribution

| Error Type | % of All Data | VLM Helps? |
|------------|---------------|------------|
| Geometry / drift / stagnant | ~10–15% | ❌ |
| Semantic misclassification | ~8–10% | ✅ |
| Policy / SRO edge cases | ~5% | ✅ |
| Missed detections | ~3–5% | ✅ |
| True hard cases | ~5–8% | ❌ |

**VLM-relevant slice: ~15–20% of total data**

From experience + literature:
- VLM resolves **~60–70%** of semantic cases confidently
- Remaining go to human

---

### Success Criteria per Experiment

#### Experiment 1: Semantic Misclassification
| Metric | Target |
|--------|--------|
| Accuracy vs GT | ≥70–80% on clean, visible cases |
| Hallucination rate | Very low |
| Effective range | Strong at <50–60m |
| UNCERTAIN rate | Acceptable (not a failure) |

**Deliverable:** 6–8 side-by-side examples (image crop, wrong label, VLM output + evidence)

---

#### Experiment 2: Ghost Box Detection
| Metric | Target |
|--------|--------|
| True negative rate | High (NO when box is fake) |
| False positive rate | Low (YES when empty) |
| UNCERTAIN behavior | Acceptable at long range |

**Deliverable:** 4–6 "obviously empty" crops with VLM correctly saying NO

---

#### Experiment 3: Policy / SRO Check
| Metric | Target |
|--------|--------|
| Agreement with human intuition | High |
| Boundary case handling | Honest UNCERTAIN near ambiguous shoulders |

**Deliverable:** 3–4 parked vehicle examples with VLM reasoning aligned to human judgment

---

### Executive Summary Slide

> **VLM reduces human review by 70–75% while calling VLM on <5% of data**
>
> - Targets only semantic/policy edge cases where rules fail
> - Acts as human-like arbiter, not primary detector
> - High precision through targeted triggering
