# PT-1938 ALFA Curate

Authors: [Lily Zhang](mailto:xzhang@lat.ai)  
Creation Date: Oct 6, 2025  
Jira Ticket: [PT-1938](https://latitudeai.atlassian.net/browse/PT-1938)  
Comment Close Date:   
Version: 0.1  
Status: **DRAFT** / ABORTED / READY FOR APPROVAL /APPROVED  
Have you submitted an invention disclosure? **No**/Yes


Summary

This design doc focuses on building a scalable data curation pipeline that empowers ML engineers to train faster and better.

Glossary

| Temporal deduplication  | Within the same slice, N segments (8 s) can be generated for 1 slice (30-40s). temporal\_dedup removes or down-weights near-identical segments.  |
| :---- | :---- |
| Perceptual deduplication  | Across slices, perceptual\_dedup removes or down-weights pixel-level or near-exact visual duplicates. |
| Semantic deduplication | Across slices, semantic\_dedup removes or down-weights content that is different in pixels but equivalent in meaning. |
| Hybrid Search  | A query method that combines multiple search techniques, such as semantic and keyword search. |
| Hard deduplication  | hard\_dedup removes items identified as duplicates based on exact, perceptual, or semantic similarity so only one representative remains.  |
| Soft deduplication | soft\_dedup down-weights items identified as duplicates based on exact, perceptual, or semantic similarity to reduce redundancy without deleting data. |

# 1. Problem
ALFA Search is actively used by various teams (UM, SP, PAD, SE) for text-to-video and video-to-video search ([https://to/alfa](https://to/alfa)). LaTS ([https://to/lats](https://to/lats)) has adopted the text-to-video functionality and is used across Latitude. 


ALFA Curate has been integrated into the Active Learning Framework for language-guided data selection. Users define scenarios (e.g., "People running across the road") with top\_k and similarity\_threshold to curate unlabeled data of interest for prioritized labeling.

### **1.1 Retrieval Noises**
ALFA Search returns slices with a user-defined threshold, but requires manual tuning. Due to fundamental limitations of embedding-based retrieval [\[7\]](https://scholar.google.com/scholar_lookup?arxiv_id=2508.21038), which map whole clips to a joint space for scene-level semantics, slice A (cow+grass+sky) can be very close to slice B (grass+sky). These coarse models suffer from background domination and information loss, resulting in noisy search results.

What's desired? We need an automatic way to review and refine retrieved results with minimal manual tuning. 

### **1.2 Redundancy Overhead**
[TODO] Insert a HPC/Bluemind usecase figure
The total amount of compute on HPC is fixed, but there are growing number features and datasets to be trained. Large portions of the corpus consist of visually or behaviorally similar slices that add little new information. Long, uneventful sequences (e.g., BPA slices with slow ego speeds) inflate storage and training time while slowing iteration.

**What's desired?** The ability to cut compute without losing rare events by down-sampling redundancies with a user-controlled deduplication threshold. SemDeDup demonstrated \~50% removal of LAION‑like web data with minimal loss. 

