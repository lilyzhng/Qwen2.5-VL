"""Prompt-based active learning strategy using Cosmos text search over LanceDB.

This implements a variant of the active learning SliceScorer where:
- process_slice returns the slice identifier (map stage)
- compute_slice_scores aggregates slice ids and runs text queries per prompt
  to score/filter only those slice ids that are in the candidate set (reduce stage)

References:
- See `alfa_curate/readme.md` for requirements
- See `dinov2/dino_obstruction.py` for the general pattern of SliceScorer usage
- See `interface/streamlit_app_v2.py` for Cosmos text search implementation

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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (  # type: ignore
    DataModelReader,
    SimpleSliceScorerBase,
)

from .utils import (  # type: ignore
    run_text_query,
    deduplicate_by_base_slice_id,
    distance_to_similarity,
)


@dataclass
class PromptSpec:
    prompt: str
    threshold: float = 0.2
    top_k: int = 200


@dataclass
class PromptScorerConfig:
    branch: str = "main"  # LanceDB branch to read from
    model_size: str = "medium"  # Cosmos model size
    prompt_yaml_path: str = str(Path(__file__).with_name("prompts.yaml"))  # Path to prompts YAML


def _load_prompts_from_yaml(path: str | Path) -> list[PromptSpec]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [
        PromptSpec(
            prompt=str(item.get("prompt", "")).strip(),
            threshold=float(item.get("threshold", 0.2)),
            top_k=int(item.get("top_k", 200)),
        )
        for item in data.get("prompts", [])
    ]


class PromptQuerySliceScorer(SimpleSliceScorerBase[PromptScorerConfig]):
    """Active learning scorer that selects slices matching text prompts.

    Map stage (process_slice): returns the slice id.
    Reduce stage (compute_slice_scores): runs text queries for each prompt and
    filters to the candidate slice ids, assigning scores based on similarity.
    """

    def __init__(self, config: PromptScorerConfig) -> None:
        super().__init__(config)
        self._prompts: list[PromptSpec] = _load_prompts_from_yaml(self.config.prompt_yaml_path)

    def process_slice(self, data_model_reader: DataModelReader) -> Optional[str]:
        """Return slice id for aggregation in reduce stage."""
        return getattr(data_model_reader, "id", None)

    def compute_slice_scores(self, scores: list[tuple[str, Any]]) -> list[tuple[str, float]]:
        """Score candidate slices by running text queries for each prompt."""
        candidate_slice_ids: set[str] = {slice_id for slice_id, _ in scores if slice_id}
        if not candidate_slice_ids:
            return []

        results: dict[str, float] = {}

        for prompt in self._prompts:
            raw_results = run_text_query(
                self.config.branch, prompt.prompt, prompt.top_k, self.config.model_size
            )
            
            # Noise prevention: skip prompt if best result is below threshold
            if raw_results:
                best_similarity = distance_to_similarity(raw_results[0].get("_distance", 2.0))
                if best_similarity < prompt.threshold:
                    continue
            
            query_results = deduplicate_by_base_slice_id(raw_results)

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


