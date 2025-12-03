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


# 1. Introduction

## 1.1 Current State
ALFA Search is actively used by various teams (UM, SP, PAD, SE) for text-to-video and video-to-video search ([https://to/ALFA](https://to/ALFA)). LaTS ([https://to/lats](https://to/lats)) has adopted the text-to-video functionality and is used across Latitude. 

ALFA Curate has been integrated into the Active Learning Framework for language-guided data selection. Users define scenarios (e.g., "People running across the road") with top\_k and similarity\_threshold to curate unlabeled data of interest for prioritized labeling. ALFA Curate allows users to filter retrieved results not only by similarity threshold, but also by applying hybrid SQL-based filters over metadata stored in BigQuery. This enables more precise data selection based on criteria such as geographic region, time-of-day, weather conditions, tags, annotations, and complex boolean logic.

## 1.2 Why reason about your data?
ALFA Curate returns slices with a user-defined threshold, but requires manual tuning. Multimodal embedding-based selection has information loss and does not guarantee fine-grained details. Due to fundamental limitations of embedding-based retrieval [\[7\]](https://scholar.google.com/scholar_lookup?arxiv_id=2508.21038), which map whole clips to a joint space for scene-level semantics, slice A (cow+grass+sky) can be very close to slice B (grass+sky). These coarse embeddings suffer from background domination, making it impossible to ensure whether a retrieved slice actually contains the queried objects, actions, or spatial relationships. Additionally, SQL filters rely on pre-existing annotations, which may be incomplete. 

Most importantly, neither mechanism can reason about temporal dynamics, spatial relationships, or multi-attribute complex properties. Today's system lacks a way to reason about the data quality. What's desired? We need an intelligent way to review, judge, and refine retrieved results.

