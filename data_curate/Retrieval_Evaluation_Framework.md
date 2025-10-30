# Retrieval Evaluation Framework

## Summary

This one-pager describes the evaluation framework for ALFA 0.1 embedding search, covering both text-to-video and video-to-video search capabilities. 

### Ground Truth Processing
- Load annotated video dataset from `unified_annotation.csv`
- Extract semantic keywords across 5 categories:
  - **Object Type**: vehicles, pedestrians, cyclists
  - **Actor Behavior**: entering path, stationary, crossing
  - **Spatial Relation**: corridor, adjacent, split positions
  - **Ego Behavior**: turning, lane change, straight
  - **Scene Type**: urban, highway, intersection, weather
- User can select to evaluate Object Type + Actor Behavior groups, e.g. "cyclist entering the ego lane " 


## Evaluation Metrics

### Primary Metric: Recall@K
```
Recall@K = |Relevant ∩ Retrieved_K| / K
```
- `Relevant`: Set of ground truth relevant videos
  - **Video-to-Video**: A video is relevant if it contains ALL keywords of the query video
  - **Text-to-Video**: A video is relevant if matching ALL the query keyword(s).
- `Retrieved_K`: Top K retrieved videos  
- `K ∈ {1, 3, 5}`: Standard evaluation cutoffs
- Per-query recall computation + Aggregation across query set

It answers: "Of the K results shown, how many are actually relevant?"

### Additional Metrics

### 1. Precision@K
```
Precision@K = |Relevant ∩ Retrieved_K| / |Relevant|
```
*Not implemented yet but valuable for understanding coverage of relevant set*

### 2. Mean Average Precision (MAP)
```
MAP = Σ(AP(q)) / |Q|
AP = Σ(Precision@k × rel(k)) / |Relevant|
```
*Useful for ranking quality assessment across multiple queries*

### 3. Normalized Discounted Cumulative Gain (NDCG)
```
DCG@K = Σ(relevance_i / log2(i + 1))
NDCG@K = DCG@K / IDCG@K
```
*Unlike Recall@k which penalizes systems for retrieving fewer than k documents, NDCG@k normalizes against the ideal ranking possible with available results.*

## Evaluation Algorithms

### Algorithm 1: Text-to-Video Evaluation with Semantic Keyword Matching

```
Algorithm 1 Text-to-video retrieval evaluation with semantic matching
────────────────────────────────────────────────────────────────────
1:   Input: text query Q, ground truth GT = {(slice_id, keywords)}, top-K value
2:   Fixed keywords vocabulary V = {v₁, v₂, ..., vₙ} from GT
3:   
4:   Initialize KeyBERT model with vocabulary V
5:   query_keywords = []
6:   
7:   for each n-gram (1 to 3 words) in preprocess(Q) do
8:       similarity_scores = KeyBERT.extract(n-gram, V)
9:       if max(similarity_scores) > threshold then
10:          mapped_keyword = argmax(similarity_scores)
11:          query_keywords.append(mapped_keyword)
12:      end if
13:  end for
14:  
15:  relevant_videos = V₁ ∩ V₂ ∩ ... ∩ Vₙ where Vᵢ = GT.videos_with_keyword(kᵢ)
16:  
17:  search_results = search_engine.search_by_text(Q, top_k=K)
18:  retrieved_ids = [r.slice_id for r in search_results]
19:  
20:  true_positives = 0
21:  for i in range(min(K, len(retrieved_ids))) do
22:      video_keywords = GT.get_keywords(retrieved_ids[i])
23:      // Check semantic match: all query keywords must be present
24:      if query_keywords ⊆ video_keywords then
25:          true_positives += 1
26:      end if
27:  end for
28:  
29:  Recall@K = true_positives / K
30:  
31:  return Recall@K, query_keywords, retrieved_ids
```

### Algorithm 2: Video-to-Video Retrieval Evaluation

```
Algorithm 2 Video-to-video retrieval evaluation
────────────────────────────────────────────────────────────────────
1:   Input: query video Q, ground truth GT = {(slice_id, keywords)}, top-K value
2:   Fixed keywords vocabulary V = {v₁, v₂, ..., vₙ} from GT
3:   Pre-computed embeddings E = {(slice_id, embedding)}
4:   
5:   Extract keywords from query video
6:   query_keywords = GT.get_keywords(Q) 
7:   
8:   video_query_embedding = E.get_embedding(Q)
9:   search_results = similarity_search(video_query_embedding, top_k=K)
10:  retrieved_ids = [r.slice_id for r in search_results if r.slice_id ≠ Q]
11:  
12:  true_positives = 0
13:  for i in range(min(K, len(retrieved_ids))) do
14:      retrieved_video_keywords = GT.get_keywords(retrieved_ids[i])
15:      // Check semantic match: all query keywords must be present
16:      if query_keywords ⊆ retrieved_video_keywords then
17:          true_positives += 1
18:      end if
19:  end for
20:  
21:  Recall@K = true_positives / K
22:  
23:  return Recall@K, query_keywords, retrieved_ids
```
**Workflow Example:**
```
1. Select semantic groups: [pv_object_type, pv_actor_behavior]
2. Choose query video: urban_cyclist_crossing_001.mp4
3. System identifies 5 relevant videos based on keyword overlap
4. Retrieves top-5 results, calculates:
   - Recall@1: 0.0 (top result not relevant)
   - Recall@3: 0.667 (2/3 results relevant)
   - Recall@5: 0.800 (4/5 results relevant)
5. Visual inspection reveals embedding bias toward visual similarity over semantic matching
```

## Example
**Text-to-Video Search:**
```
Query: "cyclist entering ego lane"
Extracted Keywords: [bicyclist, entering ego path]
Retrieved@5: [0.92, 0.88, 0.85, 0.53, 0.12] #embedding similarity
Threshold: >= 0.15
Rank #5 Result: car2pedestrian_001.mp4
  Objects: ❌ bicyclist
  Behavior: ✅ entering ego path
  → Partial match (1/2 keywords)
Recall@1: 1.0
Recall@3: 1.0
Recall@5: 0.8
```

**Video-to-Video Search:**
```
Query: highway_merge_002.mp4
Keywords: [highway, lane change, small vehicle]
Relevant Videos: 8
Top-5 Similarities: [0.95, 0.91, 0.32, 0.10, 0.09]
Recall@5: 0.6
```
