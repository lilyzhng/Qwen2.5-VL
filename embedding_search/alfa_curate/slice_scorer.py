"""Prompt-based active learning strategy using Cosmos text search over LanceDB.

This implements a variant of the active learning SliceScorer where:
- process_slice returns the slice identifier (map stage)
- compute_slice_scores aggregates slice ids and runs text queries per prompt
  to score/filter only those slice ids that are in the candidate set (reduce stage)

Supports two scoring modes:
1. "independent" (default): Each prompt is scored independently, best score wins
2. "softmax": Uses softmax to find best matching prompt per slice (like multi_caption.py)
   - More accurate when prompts are mutually exclusive
   - Prevents over-selection from dominant prompts

1. Initialization Phase
Loads configuration (branch, model size, prompt YAML path)
Reads prompts from YAML file (e.g., "pedestrians crossing street")
Each prompt has a threshold and top_k setting

2. Reduce Phrase
a) Collect candidate slice IDs: Creates a set of IDs to score
b) For each prompt, perform text search: 
    Converts prompt text → embedding using Cosmos model
    Searches LanceDB for similar video slices
    Returns top_k most similar results
c) Noise prevention check
d) Deduplicate by base video
e) Filter and score candidates

Let's say we have: 1000 video slices to evaluate
3 prompts: "pedestrians crossing", "vehicle turning", "motorcyclist at night"

Map phase: 1000 calls to process_slice → returns 1000 slice IDs

Reduce phase:
For "pedestrians crossing": Search returns 200 results, 50 match our candidates, 20 pass threshold
For "vehicle turning": Search returns 200 results, 30 match our candidates, 15 pass threshold
For "motorcyclist at night": Search returns 200 results, 10 match our candidates, 5 pass threshold
Final output: ~40 unique slices with scores (some slices may match multiple prompts)

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (  # type: ignore
    DataModelReader,
    SimpleSliceScorerBase,
)

import numpy as np

from .config import StrategyConfig, PromptConfig
from .utils import (  # type: ignore
    run_text_query,
    deduplicate_by_base_slice_id,
    distance_to_similarity,
    text_to_embedding,
    load_table,
)

def _load_prompts_from_yaml(path: str | Path) -> list[PromptConfig]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    
    configs = []
    for item in data.get("scenarios", []):
        # Extract prompt (required field)
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Each scenario must have a non-empty 'prompt' field")
            
        config_args = {"prompt": prompt}
        
        if "threshold" in item:
            config_args["threshold"] = float(item["threshold"])
        
        if "top_k" in item:
            config_args["top_k"] = int(item["top_k"])
        
        configs.append(PromptConfig(**config_args))
    
    return configs


class SliceScorer(SimpleSliceScorerBase[StrategyConfig]):
    """Active learning scorer that selects slices matching text prompts.

    Map stage (process_slice): returns the slice id.
    Reduce stage (compute_slice_scores): runs text queries for each prompt and
    filters to the candidate slice ids, assigning scores based on similarity.
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        if config.load_prompts_from_yaml and config.prompt_yaml_path:
            self._prompts: list[PromptConfig] = _load_prompts_from_yaml(config.prompt_yaml_path)
        else:
            self._prompts: list[PromptConfig] = config.prompts

    def process_slice(self, data_model_reader: DataModelReader) -> Optional[str]:
        """Return slice id for aggregation in reduce stage."""
        return getattr(data_model_reader, "id", None)

    def compute_slice_scores(self, scores: list[tuple[str, Any]]) -> list[tuple[str, float]]:
        """Score candidate slices by running text queries for each prompt."""
        candidate_slice_ids: set[str] = {slice_id for slice_id, _ in scores if slice_id}
        if not candidate_slice_ids:
            return []

        if self.config.scoring_mode == "softmax":
            return self._compute_softmax_scores(candidate_slice_ids)
        else:  # independent mode
            return self._compute_independent_scores(candidate_slice_ids)

    def _compute_independent_scores(self, candidate_slice_ids: set[str]) -> list[tuple[str, float]]:
        """Original scoring mode: each prompt scored independently."""
        results: dict[str, float] = {}

        for prompt in self._prompts:
            raw_results = run_text_query(
                self.config.branch, prompt.prompt, prompt.top_k, self.config.model_size
            )
            
            # Noise prevention: skip prompt if best result is below threshold
            if raw_results and self.config.min_best_similarity > 0:
                best_similarity = distance_to_similarity(raw_results[0].get("_distance", 2.0))
                if best_similarity < self.config.min_best_similarity:
                    continue
            
            # Apply deduplication if configured
            if self.config.deduplicate_by_base_video:
                query_results = deduplicate_by_base_slice_id(raw_results)
            else:
                query_results = raw_results

            for row in query_results:
                slice_id = str(row.get("row_id", ""))
                if not slice_id or slice_id not in candidate_slice_ids:
                    continue
                    
                similarity = distance_to_similarity(float(row.get("_distance", 2.0)))
                if similarity < prompt.threshold:
                    continue

                # Keep best score across prompts
                results[slice_id] = max(results.get(slice_id, 0), similarity)

        # Lower scores = higher priority
        return [(slice_id, -score) for slice_id, score in results.items()]

    def _compute_softmax_scores(self, candidate_slice_ids: set[str]) -> list[tuple[str, float]]:
        """Softmax scoring mode: finds best matching prompt per slice using softmax."""
        if not self._prompts:
            return []
        
        # Get embeddings for all prompts
        prompt_embeddings = np.array([
            text_to_embedding(prompt.prompt, self.config.model_size)
            for prompt in self._prompts
        ])  # Shape: (num_prompts, embedding_dim)
        
        table = load_table(self.config.branch)
        results: dict[str, float] = {}
        
        # Process in batches to avoid memory issues
        candidate_list = list(candidate_slice_ids)
        batch_size = self.config.batch_size
        
        for i in range(0, len(candidate_list), batch_size):
            batch_ids = candidate_list[i:i + batch_size]
            
            # Fetch embeddings for this batch of candidates
            batch_data = table.search().where(f"row_id IN {batch_ids}").limit(len(batch_ids)).to_list()
            
            for row in batch_data:
                slice_id = row.get("row_id")
                if not slice_id:
                    continue
                
                # Get slice embedding
                slice_embedding = np.array(row.get("embedding", []))
                if len(slice_embedding) == 0:
                    continue
                
                # Compute similarities with all prompts
                similarities = np.dot(prompt_embeddings, slice_embedding)
                
                # Apply softmax to get probabilities
                # Scale by temperature
                scaled_similarities = similarities * self.config.softmax_temperature
                exp_sims = np.exp(scaled_similarities - np.max(scaled_similarities))  # Stability
                softmax_probs = exp_sims / np.sum(exp_sims)
                
                # Find best matching prompt
                best_prompt_idx = np.argmax(softmax_probs)
                best_prob = softmax_probs[best_prompt_idx]
                best_prompt = self._prompts[best_prompt_idx]
                
                # Apply threshold on probability instead of raw similarity
                if best_prob >= best_prompt.threshold:
                    results[slice_id] = best_prob
        
        # Lower scores = higher priority
        return [(slice_id, -score) for slice_id, score in results.items()]


