# 1\. Introduction

## 1.1 Current State

**Redundancy Problem.** As typical driving scenes videos, the data from each slice is highly correlated. Large portions of the corpus consist of visually or behaviorally similar slices that add little new information. For example, in some highway scenarios, the scenery and arrangement of actors around the vehicle may change slowly. In some extreme cases like parking lots, the scene may be entirely static. These long, uneventful sequences inflate storage and training time while slowing iteration. 

**Dataset Structure Problem.** ScaleX datasets used for training perception models are composed of parquet files, each with \~10 rows. Each row corresponds to a slice, which is a driving sequence with \~300 frames from about \~15 seconds of data. The total size of each parquet file is about 100 MB. During training, data is loaded from each file in groups of 8 frames. Because each file has only \~10 slices bundled and sampling is done in groups of 8, the same slices will be selected together repeatedly. This structure limits diversity during sampling.

## 1.2 Purpose

SemDeDup[\[1\]](https://arxiv.org/abs/2303.09540) showed in the left (a) figure that image data can be visualized in embedding space, and in the right (b) figure, you can remove \~50% of LAION-like web data with little loss. Users should be able to cut compute without losing rare events by down-sampling redundancies with a user-controlled de-dup threshold.

The purpose of this design is to:

1. Restructure the dataset to promote higher diversity during sampling.  
2. Remove duplicate data to reduce the dataset size and potentially speed up model convergence.

#### Diversity Goals

* **PerceptualDedup**: Perceptual deduplication removes or down-weights pixel-level or near-exact visual duplicates. With UM model trends using only one or two frames per slice, perceptual diversity becomes critical to ensure each training example provides unique visual information. Dinov2 / v3 can be used for perceptual similarity comparison.
  * **TemporalDedup** (a special case of perceptual similarity over time): Within a slice, we want to avoid keeping many consecutive frames that look almost identical (e.g., static parking lot scenes, slow highway driving). Temporal deduplication downsamples these redundant segments. 
* **SemanticDedup**: Across slices, semantic deduplication removes or down-weights content that is different in pixels but equivalent in meaning (e.g., multiple boring highway drives with different vehicles and lighting). This design uses NVIDIA Cosmos Embedding to identify semantically similar scenes and balance representation of rare but critical scenarios.

# 2\. System Design

Prior work on semantic deduplication (e.g., SemDeDup, FAIR filtering) uses CLIP-style vision–language embeddings because their primary goal is to remove semantic redundancy in web-scale image–text data. Perceptual near-duplicates are mostly handled by simpler filters. 

For driving data, temporal redundancy is a perceptual issue we haven’t solved yet. So we first need to improve perceptual diversity (over time and viewpoint) before targeting semantic diversity at the scenario level.

We’ll approach this problem in multiple phases.

## 2.1 \- Phase 1: Restructure the dataset to promote diversity

Very simply, we’ll transform the dataset from having N rows, each with M frames, to having (N \* M / 10\) rows, each with 10 frames.  As a result, each row of the dataset will correspond to 0.5 seconds of a slice.

This conversion will be done using a Ray data pipeline that does the following for batches of 3000 rows:

1. Load and transform each row into M / 10 rows, each with 10 frames.  
2. Globally shuffle the dataset  
3. Save the dataset into parquet files, each with 2000 rows and a row group size of 50

This change will mean that each file optimistically has \~2000 unique slices, and the row group size will allow the files to be iterated over in a memory efficient way.  During testing, a sample file could be iterated over in 1.63 seconds (the original 10 bundled files take \~600 ms).  To compensate for the change in time, we’ll increase the batch size during sampling to at least 24\.

### Changes to FrameSamplingViewV2

Currently FrameSamplingViewV2 iterates over parquet files via 

```py
for record_batch in parquet_file.iter_batches(batch_size=1, use_threads=False):
```

This approach works because the row group size in the bundled dataset is 1, but it’s not performant for pico files.  We’ll update the implementation to iterate over row groups and then iterate over slices of each group:

```py
for i in range(pqfile.num_row_groups):
    group = pqfile.read_row_group(i, use_threads=False)
    for j in range(group.num_rows):
        data = group.slice(j, 1)
```

This approach was shown to have a minimal change for existing files, but to be much faster for pico files.

## 2.2 \- Phase 2: Temporal Subsampling

To get an initial sense of dataset redundancy, we'll downsample the dataset and measure performance impact. If we can drop 50% or 90% of frames with minimal degradation, that provides strong evidence for dataset redundancy and justifies more aggressive semantic dedup. This is cheap to implement and run as an ablation. 

We'll train models with the original bundled dataset and decimated pico datasets to understand the impact of each approach.

* Approach 1: Periodic Subsampling, Downsample by removing frames at regular intervals:  
  * Remove 1 of every 2 frames  
  * Remove 9 of every 10 frames  
* Approach 2: Uniform Random Row Drop  
  * After generating pico rows (each \= 10-frame chunk), treat all pico rows as independent samples and select a random subset with probability 𝑝.  
* Trade-offs  
* Periodic subsampling shrinks the dataset but preserves temporal bias. Static parking lot scenes and fast lane change sequences are treated identically. Critical events (cut-ins, pedestrians, braking) might be randomly deleted while keeping redundant highway scenes.  
* Uniform random row drop shrinks the dataset while roughly preserving the original distribution and avoiding extra structured bias.

## 2.3 \- Phase 3: Semantic deduplication

split → embed → semantic dedup

Semantic deduplication of datasets has been shown to speed up convergence of models (SemDeDup[\[1\]](https://arxiv.org/abs/2303.09540) also being used by Nemo Curator [\[2\]](https://docs.nvidia.com/nemo-framework/user-guide/24.09/datacuration/semdedup.html)). We will implement a similar approach within the ScaleX dataset generation framework using a Pico dataset as the input.

**Original Proposal (Density-Based)**

1. Embedding Generation: We’ll use embeddings that are already generated as part of the existing features pipelines.  Options include DinoV2, CLIP, and Cosmos-Embed1.  We’ll start with DinoV2 because it’s fast to generate and can be easily computed on every 10th frame of the dataset (the features datasets will need to be updated to reduce the stride).  
2. Clustering Method: We’ll perform KMeans clustering on the dataset.  
3. Within each cluster, we’ll iteratively do the following:  
   1. Select the highest density point and move it to the output dataset  
   2. Discard points within a threshold distance of highest density point

**Alternative: Standard SemDeDup (Cosine-Similarity-Based)**

1. Embedding Generation: Use pre-computed embeddings (DinoV2, CLIP, or Cosmos-Embed1)  
2. Spherical K-means Clustering: L2-normalize embeddings and perform K-means (this makes K-means optimize for angular/cosine distance)  
3. Compute Cosine Distance to Centroid: For each point, compute cosine\_distance \= 1 \- cosine\_similarity to centroid  
4. Within-Cluster Deduplication:   
* Sort by cosine distance to centroid (descending \= most unique first)  
* For each point j, compute max cosine similarity to earlier points i \< j  
* Keep point j if max\_similarity ≤ (1 \- ε)

**Embedding Limitations.**  
It doesn’t matter if it’s DINO, Cosmo, or DINOv3. They all map a high-dimensional image → a low-dimensional vector. That vector mostly captures: global layout (“highway vs city vs parking lot”), big objects, coarse semantics. They will lose information: small or rare actors, fine-grained geometry,

* hybrid approach: overlay labels/metadata w embeddings   
  * Stratify first by labels/metadata, then dedup inside each bucket:  
    * scenario type (highway/city/parking),  
    * weather / time of day,  
    * driving behaviors (cut-in, pedestrian, braking, etc.),  
    * camera/view.

# 3\. Requirements

* The dataset is generated from an existing dataset and has the same schema, making it directly usable for training.  
* Deduplication of the dataset must improve overall model performance or convergence time.


# Reference

[\[1\]](https://arxiv.org/abs/2303.09540) Abbas, Amro, et al. "Semdedup: Data-efficient learning at web-scale through semantic deduplication." *arXiv preprint arXiv:2303.09540* (2023).  
[\[2\]](https://docs.nvidia.com/nemo-framework/user-guide/24.09/datacuration/semdedup.html) Nemo Curator [https://docs.nvidia.com/nemo-framework/user-guide/24.09/datacuration/semdedup.html](https://docs.nvidia.com/nemo-framework/user-guide/24.09/datacuration/semdedup.html)  
