# VLM Judge Test Prompts - Quick Reference

This document contains 10 carefully designed test prompts for evaluating the VLM-as-Judge refinement system. Each prompt targets specific capabilities and failure modes.

---

## Prompt Catalog

### Category A: Single Attribute (Easy - Baseline)

#### **Test Prompt 1: Basic Object Detection**
```yaml
Query: "Pedestrian crossing the street"
Difficulty: Easy
Expected Precision: >0.90
Key Challenge: Distinguish crossing vs. standing
VLM Focus: Object presence + motion detection
```

#### **Test Prompt 2: Scene Type Classification**
```yaml
Query: "Highway driving at night"
Difficulty: Easy
Expected Precision: >0.85
Key Challenge: Highway vs. urban road at night
VLM Focus: Scene understanding + lighting conditions
```

---

### Category B: Spatial Relationships (Medium)

#### **Test Prompt 3: Relative Position**
```yaml
Query: "Cyclist approaching from the left side of ego vehicle"
Difficulty: Medium
Expected FP Reduction: >50%
Key Challenge: Directional approach (left vs. right vs. front)
VLM Focus: Spatial reasoning + relative positioning
Failure Mode: Embedding returns any cyclist scene
```

#### **Test Prompt 4: Occlusion Handling**
```yaml
Query: "Pedestrian partially occluded by parked vehicle"
Difficulty: Medium
Expected Accuracy: >80%
Key Challenge: Partial visibility + occlusion source identification
VLM Focus: Spatial perception + object relationships
Failure Mode: Embedding ignores occlusion state
```

---

### Category C: Temporal Events (Medium-Hard)

#### **Test Prompt 5: Action Sequence**
```yaml
Query: "Vehicle enters intersection then turns right"
Difficulty: Medium-Hard
Expected Precision: >0.75
Expected Temporal Accuracy: >0.85
Key Challenge: Temporal ordering (enter → turn)
VLM Focus: Sequential event understanding + temporal modeling
Failure Mode: Embeddings capture spatial similarity but not temporal order
```

#### **Test Prompt 6: Behavior Pattern**
```yaml
Query: "Ego vehicle lane change behind slow-moving truck"
Difficulty: Hard
Expected Semantic Match: >0.70
Key Challenge: Causal relationship (lane change due to slow truck)
VLM Focus: Causal analysis + behavior understanding
Failure Mode: Missing causality in embedding space
```

---

### Category D: Multi-Attribute (Hard)

#### **Test Prompt 7: Complex Environmental Conditions**
```yaml
Query: "Rainy night urban intersection with pedestrian crossing and ego vehicle waiting at red light"
Difficulty: Hard
Expected Precision: >0.80
Expected All-Attributes-Match: >0.75
Attributes to Validate:
  - Rain (visible droplets, wet roads, low visibility)
  - Night (dark environment, artificial lighting)
  - Urban intersection (traffic lights, crosswalk)
  - Pedestrian actively crossing
  - Ego vehicle stationary at red light
Key Challenge: Compositional understanding of 5+ attributes
VLM Focus: Multi-attribute reasoning
Failure Mode: Embedding returns partial matches (e.g., rain but no pedestrian)
```

#### **Test Prompt 8: Rare Edge Case**
```yaml
Query: "Emergency vehicle with flashing lights approaching ego vehicle from behind in left lane during highway driving"
Difficulty: Very Hard
Expected FP Rate: <0.15
Expected Recall Preservation: >0.80
Attributes to Validate:
  - Emergency vehicle type (ambulance, fire truck, police car)
  - Flashing lights active
  - Approach from behind (not oncoming)
  - Left lane position
  - Highway context
Key Challenge: Rare class + complex spatial configuration
VLM Focus: Few-shot recognition + spatial reasoning
Failure Mode: Embedding struggles with rare classes
```

---

### Category E: Negative Case Filtering (Critical)

#### **Test Prompt 9: Ambiguity Resolution**
```yaml
Query: "Motorcyclist wearing helmet stopped at intersection"
Difficulty: Hard
Expected FP Reduction: >60%
Expected Type-Confusion Rate: <10%
Must Reject:
  - Bicyclist with helmet (wrong vehicle type)
  - Motorcyclist without helmet (missing attribute)
  - Motorcyclist moving through intersection (wrong behavior)
Key Challenge: Fine-grained distinction between similar classes
VLM Focus: Visual recognition + logical validation
Failure Mode: Embedding conflates motorcycles and bicycles
```

#### **Test Prompt 10: Sensor Obstruction Detection**
```yaml
Query: "Camera partially obstructed by water droplets or dirt affecting visibility of road ahead"
Difficulty: Hard
Expected Obstruction Detection: >0.85
Expected Severity Classification: >0.70
Must Distinguish From:
  - Rain in scene (but clear camera)
  - Fog (environmental condition vs. sensor issue)
  - Windshield wipers visible (not camera obstruction)
Key Challenge: Meta-perception (sensing the sensor state)
VLM Focus: Multi-level visual analysis
Failure Mode: Embedding cannot detect sensor-level artifacts
Severity Levels: [Minimal <10% coverage, Moderate 10-40%, Severe >40%]
```

---

## Evaluation Protocol

### For Each Test Prompt:

1. **Stage 1 Baseline**: Run embedding-only search, collect top-20 results
2. **Stage 2 Refinement**: Apply VLM judge, collect top-5 refined results
3. **Metrics Collection**:
   - Precision@5 (Stage 1 vs Stage 2)
   - Recall@5 (Stage 1 vs Stage 2)
   - False Positive Count
   - Semantic Match Accuracy (Stage 2 only)
   - Explanation Coherence Score (Stage 2 only)

4. **Manual Review**: Sample 10-20 results for human validation

### Success Criteria Summary

| Prompt | Difficulty | Primary Metric | Target |
|--------|-----------|----------------|---------|
| 1-2 | Easy | Precision | >0.85 |
| 3-4 | Medium | FP Reduction | >50% |
| 5-6 | Medium-Hard | Semantic Match | >0.70 |
| 7-8 | Hard | All-Attributes | >0.75 |
| 9-10 | Critical | Type-Confusion | <10% |

### Overall System Goals

- **Average Precision Improvement**: +30-50% across all prompts
- **False Positive Reduction**: 60-80% on average
- **Semantic Match Accuracy**: >85% on validatable attributes
- **Zero Catastrophic Failures**: No prompt with <50% of Stage 1 recall

---

## Prompt Engineering Tips

### General Structure
```
[Role Definition] → [Context] → [Task] → [Evaluation Criteria] → [Output Format]
```

### Key Elements for Automotive Queries

1. **Temporal Markers**: "before", "after", "then", "during", "while"
2. **Spatial Prepositions**: "from", "to", "behind", "left of", "approaching"
3. **Conditional Phrases**: "if", "when", "while", "during"
4. **Attribute Modifiers**: "partially", "fully", "actively", "stationary"
5. **Environmental Context**: "at night", "in rain", "on highway", "at intersection"

### Anti-Patterns to Avoid

❌ **Vague**: "Car and person"
✅ **Specific**: "Pedestrian crossing street in front of approaching vehicle"

❌ **Ambiguous**: "Vehicle turning"
✅ **Precise**: "Vehicle enters intersection then executes right turn"

❌ **Single-word**: "Rain"
✅ **Contextual**: "Rainy weather with reduced visibility affecting camera view"

---

## Quick Test Commands

```python
# Run single prompt test
python tests/test_vlm_judge.py --prompt-id 1 --top-k 20 --final-n 5

# Run category test (all prompts in category)
python tests/test_vlm_judge.py --category "spatial" --visualize

# Run full evaluation suite
python tests/test_vlm_judge.py --run-all --output results.json

# Compare Stage 1 vs Stage 2
python tests/compare_stages.py --prompts 1,2,3,4,5 --metrics precision,recall,f1
```

---

## Expected Results Summary

Based on preliminary analysis of embedding search limitations:

| Prompt Category | Stage 1 Precision | Stage 2 Expected | Improvement |
|----------------|------------------|-----------------|-------------|
| **Single Attribute** | 0.75 | 0.90 | +20% |
| **Spatial Relations** | 0.60 | 0.85 | +42% |
| **Temporal Events** | 0.55 | 0.75 | +36% |
| **Multi-Attribute** | 0.45 | 0.80 | +78% |
| **Negative Filtering** | 0.50 | 0.85 | +70% |
| **Overall Average** | 0.57 | 0.83 | **+46%** |

---

## Next Steps

1. ✅ Define test prompts (COMPLETE)
2. ⏳ Collect ground truth annotations for each prompt
3. ⏳ Implement VLM judge pipeline
4. ⏳ Run evaluation on all 10 prompts
5. ⏳ Analyze failure modes and iterate
6. ⏳ Document best practices and production guidelines
