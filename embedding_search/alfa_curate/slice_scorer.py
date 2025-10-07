"""Implementation of ALFA based data selection strategy."""


from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from autonomy.perception.datasets.active_learning.alfa_curate.config import PromptConfig, StrategyConfig, TaskStrategy
from autonomy.perception.datasets.active_learning.alfa_curate.utils import (  # type: ignore
    distance_to_similarity,
    get_model_logit_scale,
    load_table,
    parse_row_id,
    run_text_query,
    text_to_embedding,
)
from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (  # type: ignore
    DataModelReader,
    SimpleSliceScorerBase,
)


def _load_config_from_yaml(path: str | Path) -> tuple[list[PromptConfig], dict[str, TaskStrategy]]:
    """Load both prompts and task strategies from YAML file.
    
    Returns:
        Tuple of (prompts, task_strategies)
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Load task strategies from "tasks:" section
    task_strategies: dict[str, TaskStrategy] = {}
    for task_name, task_config in data.get("tasks", {}).items():
        strategy_args = {"task_name": task_name}
        
        if "priority" in task_config:
            strategy_args["priority"] = int(task_config["priority"])
        
        if "scoring_mode" in task_config:
            strategy_args["scoring_mode"] = str(task_config["scoring_mode"])
        
        if "separate_dataset" in task_config:
            strategy_args["separate_dataset"] = bool(task_config["separate_dataset"])
        
        task_strategies[task_name] = TaskStrategy(**strategy_args)

    # Load prompts from "scenarios:" section
    prompts = []
    for item in data.get("scenarios", []):
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Each scenario must have a non-empty 'prompt' field")

        config_args = {"prompt": prompt}

        if "task" in item:
            config_args["task"] = str(item["task"])

        if "threshold" in item:
            config_args["threshold"] = float(item["threshold"])

        if "top_k" in item:
            config_args["top_k"] = int(item["top_k"])

        prompts.append(PromptConfig(**config_args))

    return prompts, task_strategies


class SliceScorer(SimpleSliceScorerBase[StrategyConfig]):
    """Active learning scorer that selects slices matching text prompts."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        if config.prompt_yaml_path:
            self._prompts, yaml_task_strategies = _load_config_from_yaml(config.prompt_yaml_path)

            for task_name, strategy in yaml_task_strategies.items():
                config.task_strategies[task_name] = strategy
        else:
            self._prompts: list[PromptConfig] = []

    def process_slice(self, data_model_reader: DataModelReader) -> Optional[str]:
        """Return slice id for aggregation in reduce stage."""
        return getattr(data_model_reader, "id", None)

    def compute_slice_scores(self, scores: list[tuple[str, Any]]) -> list[tuple[str, float]]:
        """Score candidate slices using task-based strategies."""
        candidate_slice_ids: set[str] = {slice_id for slice_id, _ in scores if slice_id}
        if not candidate_slice_ids:
            return []

        # Group prompts by task
        prompts_by_task: dict[str, list[PromptConfig]] = {}
        for prompt in self._prompts:
            task_name = prompt.task or "default"
            prompts_by_task.setdefault(task_name, []).append(prompt)

        # Get tasks sorted by priority
        task_order = sorted(
            prompts_by_task.keys(),
            key=lambda t: self._get_task_strategy(t).priority
        )

        # Process tasks in priority order
        all_task_results: dict[str, dict[str, float]] = {}  # task -> {slice_id: score}
        remaining_candidates = candidate_slice_ids.copy()

        for task_name in task_order:
            task_prompts = prompts_by_task[task_name]
            task_strategy = self._get_task_strategy(task_name)

            # Score using task-specific strategy
            task_results = self._score_task(remaining_candidates, task_prompts, task_strategy)

            if task_results:
                all_task_results[task_name] = task_results

                # If task separates dataset, remove from candidate pool for subsequent tasks
                if task_strategy.separate_dataset:
                    remaining_candidates -= set(task_results.keys())

        # Combine all task results
        combined_results = self._combine_task_results(all_task_results)

        # Apply final deduplication by base slice id if configured
        if self.config.deduplicate_by_base_video:
            combined_results = self._deduplicate_by_base_slice_id(combined_results)

        # Lower scores = higher priority
        return [(slice_id, -score) for slice_id, score in combined_results.items()]

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

        # Get the learned logit_scale from the model (cached, so only loaded once)
        logit_scale = get_model_logit_scale(self.config.model_size)

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

                # Apply softmax to get probabilities, scale by learned logit_scale
                scaled_similarities = similarities * logit_scale
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

    def _get_task_strategy(self, task_name: str) -> TaskStrategy:
        """Get strategy for a task, or create default if not configured."""
        if task_name in self.config.task_strategies:
            return self.config.task_strategies[task_name]
        
        # Create default strategy for unconfigured tasks
        return TaskStrategy(
            task_name=task_name,
            priority=100,  # Default priority
            scoring_mode=self.config.scoring_mode,  # Use global scoring mode
            separate_dataset=False,
        )

    def _score_task(
        self, 
        candidate_slice_ids: set[str], 
        prompts: list[PromptConfig], 
        strategy: TaskStrategy
    ) -> dict[str, float]:
        """Score a task's prompts using the task strategy."""
        if not candidate_slice_ids or not prompts:
            return {}

        # All prompts in a task use the task's scoring mode
        task_results: dict[str, float] = {}

        if strategy.scoring_mode == "softmax":
            softmax_scores = self._compute_softmax_scores(candidate_slice_ids, prompts)
            for slice_id, score in softmax_scores:
                task_results[slice_id] = max(task_results.get(slice_id, 0), abs(score))
        else:  # independent
            independent_scores = self._compute_independent_scores(candidate_slice_ids, prompts)
            for slice_id, score in independent_scores:
                task_results[slice_id] = max(task_results.get(slice_id, 0), abs(score))

        return task_results

    def _combine_task_results(self, all_task_results: dict[str, dict[str, float]]) -> dict[str, float]:
        """Combine results from all tasks."""
        combined: dict[str, float] = {}

        for task_name, task_results in all_task_results.items():
            for slice_id, score in task_results.items():
                # Use max score across all tasks
                combined[slice_id] = max(combined.get(slice_id, 0), score)

        return combined

