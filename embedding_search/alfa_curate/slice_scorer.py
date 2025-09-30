"""Implementation of ALFA based data selection strategy."""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from autonomy.perception.datasets.active_learning.alfa_curate.config import PromptConfig, StrategyConfig
from autonomy.perception.datasets.active_learning.alfa_curate.utils import (  # type: ignore
    distance_to_similarity,
    load_table,
    parse_row_id,
    run_text_query,
    text_to_embedding,
)
from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (  # type: ignore
    DataModelReader,
    SimpleSliceScorerBase,
)


def _load_prompts_from_yaml(path: str | Path) -> list[PromptConfig]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    configs = []
    for item in data.get("scenarios", []):
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Each scenario must have a non-empty 'prompt' field")

        config_args = {"prompt": prompt}

        if "threshold" in item:
            config_args["threshold"] = float(item["threshold"])

        if "top_k" in item:
            config_args["top_k"] = int(item["top_k"])

        if "scoring_mode" in item:
            config_args["scoring_mode"] = str(item["scoring_mode"])

        configs.append(PromptConfig(**config_args))

    return configs


class SliceScorer(SimpleSliceScorerBase[StrategyConfig]):
    """Active learning scorer that selects slices matching text prompts."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        if config.prompt_yaml_path:
            self._prompts: list[PromptConfig] = _load_prompts_from_yaml(config.prompt_yaml_path)

    def process_slice(self, data_model_reader: DataModelReader) -> Optional[str]:
        """Return slice id for aggregation in reduce stage."""
        return getattr(data_model_reader, "id", None)

    def compute_slice_scores(self, scores: list[tuple[str, Any]]) -> list[tuple[str, float]]:
        """Score candidate slices by running text queries for each prompt."""
        candidate_slice_ids: set[str] = {slice_id for slice_id, _ in scores if slice_id}
        if not candidate_slice_ids:
            return []

        # Separate prompts by mode
        independent_prompts = []
        softmax_prompts = []
        
        for prompt in self._prompts:
            mode = prompt.scoring_mode or self.config.scoring_mode
            if mode == "softmax":
                softmax_prompts.append(prompt)
            else:
                independent_prompts.append(prompt)
        
        # Collect all scores from both modes
        all_results: dict[str, float] = {}
        
        if independent_prompts:
            independent_scores = self._compute_independent_scores(candidate_slice_ids, independent_prompts)
            for slice_id, score in independent_scores:
                all_results[slice_id] = max(all_results.get(slice_id, 0), abs(score))  # abs because scores are negative
        
        if softmax_prompts:
            softmax_scores = self._compute_softmax_scores(candidate_slice_ids, softmax_prompts)
            for slice_id, score in softmax_scores:
                all_results[slice_id] = max(all_results.get(slice_id, 0), abs(score))
        
        # Apply final deduplication by base slice id if configured
        if self.config.deduplicate_by_base_video:
            all_results = self._deduplicate_by_base_slice_id(all_results)
        
        # Lower scores = higher priority
        return [(slice_id, -score) for slice_id, score in all_results.items()]

    def _compute_independent_scores(self, candidate_slice_ids: set[str], prompts: list[PromptConfig]) -> list[tuple[str, float]]:
        """Independent scoring mode: Each prompt scored independently."""
        results: dict[str, float] = {}

        for prompt in prompts:
            raw_results = run_text_query(self.config.branch, prompt.prompt, prompt.top_k, self.config.model_size)

            # Noise prevention: skip prompt if best result is below threshold
            if raw_results and self.config.min_best_similarity > 0:
                best_similarity = distance_to_similarity(raw_results[0].get("_distance", 2.0))
                if best_similarity < self.config.min_best_similarity:
                    continue

            # No dedup here - will be done at the end if configured
            for row in raw_results:
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

    def _compute_softmax_scores(self, candidate_slice_ids: set[str], prompts: list[PromptConfig]) -> list[tuple[str, float]]:
        """Softmax scoring mode: finds best matching prompt per slice using softmax."""
        if not prompts:
            return []

        # Get embeddings for all prompts
        prompt_embeddings = np.array(
            [text_to_embedding(prompt.prompt, self.config.model_size) for prompt in prompts]
        )  # Shape: (num_prompts, embedding_dim)

        table = load_table(self.config.branch)
        results: dict[str, float] = {}

        candidate_list = list(candidate_slice_ids)
        batch_size = self.config.batch_size

        for i in range(0, len(candidate_list), batch_size):
            batch_ids = candidate_list[i : i + batch_size]

            # Fetch embeddings for this batch of candidates
            batch_data = table.search().where(f"row_id IN {batch_ids}").limit(len(batch_ids)).to_list()

            for row in batch_data:
                slice_id = row.get("row_id")
                if not slice_id:
                    continue

                slice_embedding = np.array(row.get("embedding", []))
                if len(slice_embedding) == 0:
                    continue

                # Compute similarities with all prompts
                similarities = np.dot(prompt_embeddings, slice_embedding)

                # Apply softmax to get probabilities, scale by temperature
                scaled_similarities = similarities * self.config.softmax_temperature
                exp_sims = np.exp(scaled_similarities - np.max(scaled_similarities))
                softmax_probs = exp_sims / np.sum(exp_sims)

                best_prompt_idx = np.argmax(softmax_probs)
                best_prob = softmax_probs[best_prompt_idx]
                best_prompt = prompts[best_prompt_idx]

                if best_prob >= best_prompt.threshold:
                    results[slice_id] = best_prob

        # Lower scores = higher priority
        return [(slice_id, -score) for slice_id, score in results.items()]

    def _deduplicate_by_base_slice_id(self, results: dict[str, float]) -> dict[str, float]:
        """Keep only the best (highest score) segment per base slice id."""
        best_by_base: dict[str, tuple[str, float]] = {}  # base_id -> (slice_id, score)
        
        for slice_id, score in results.items():
            base_id, _, _, _ = parse_row_id(slice_id)
            
            if base_id not in best_by_base or score > best_by_base[base_id][1]:
                best_by_base[base_id] = (slice_id, score)
        
        return {slice_id: score for slice_id, score in best_by_base.values()}

