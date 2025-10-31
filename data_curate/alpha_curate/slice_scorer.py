"""Implementation of alpha based data selection strategy."""


import logging
from pathlib import Path
from typing import Any, Final


from ruamel.yaml import YAML


from autonomy.perception.datasets.active_learning.alfa_curate.config import StrategyConfig
from autonomy.perception.datasets.active_learning.alfa_curate.data_types import PromptConfig
from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import generate_prompt
from autonomy.perception.datasets.active_learning.alfa_curate.score_fusion import fuse_scores
from autonomy.perception.datasets.active_learning.alfa_curate.utils import (
   SearchResult,
   deduplicate_by_base_slice,
   load_table,
   parse_row_id,
   run_text_query,
)
from autonomy.perception.datasets.active_learning.framework.dataset.dataset_abstraction import DatasetAbstraction
from autonomy.perception.datasets.active_learning.framework.lakefs_support import (
   HUMAN_LABELS_NAME,
   LOG_SLICES_SILVER_NAME,
)
from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (
   DataModelReader,
   SimpleSliceScorerBase,
   compute_slice_ids_to_process,
)
from kits.ml.onnx.model_management.av_path import access_av_path


_LOGGER: Final = logging.getLogger(__name__)


MAX_LANCE_DISTANCE: Final[float] = 2.0




def load_scenarios_from_yaml(path: str | Path) -> list[PromptConfig]:
   """Load scenarios from a YAML file."""
   with open(path, "r") as f:
       yaml = YAML(typ="safe")
       data = yaml.load(f)


   return [PromptConfig.from_dict(item) for item in data.get("scenarios", [])]  # type: ignore [attr-defined]




class SliceScorer(SimpleSliceScorerBase[StrategyConfig]):
   """Active learning scorer that selects slices matching text prompts."""


   def __init__(self, config: StrategyConfig, use_prompt_expansion: bool | None = None) -> None:
       """Initialize the slice scorer.
       
       Args:
           config: Strategy configuration
           use_prompt_expansion: If True, use prompt expansion and fusion for better recall.
                                If None, uses config.use_prompt_expansion
       """
       super().__init__(config)
       if config.prompt_yaml_path:
           self.scenarios: list[PromptConfig] = load_scenarios_from_yaml(access_av_path(config.prompt_yaml_path))
       self.slices_to_process: set[str] = set()
       # Use parameter if provided, otherwise use config value
       self.use_prompt_expansion = use_prompt_expansion if use_prompt_expansion is not None else config.use_prompt_expansion


   def process_slice(self, data_model_reader: DataModelReader) -> None:
       """It works on the slice level, not neeed for alpha curate because slice level info is in the DB."""
       # TODO(SML-4678): refactor active learning with new base class


   def get_slices_to_process(self, input_datasets: dict[str, DatasetAbstraction]) -> set[str]:
       """Determines which keys should be processed."""


       log_slices_silver = input_datasets[LOG_SLICES_SILVER_NAME]
       human_labels_gold = input_datasets[HUMAN_LABELS_NAME]


       self.slices_to_process = compute_slice_ids_to_process(
           include=[log_slices_silver.get_ids()],
           exclude=[human_labels_gold.get_ids()],
       )
       _LOGGER.info("Number of unlabelled slices to process: %d", len(self.slices_to_process))
       return self.slices_to_process


   def compute_slice_scores(self, map_results: dict[str, float]) -> dict[str, float]:
       """Score candidate slices by running text queries for each prompt."""
       table = load_table(self.config.repo, self.config.lance_db_branch, self.config.table_name)
       total_results = {}

       for scenario in self.scenarios:
           if self.use_prompt_expansion:
               # Expand prompt and fuse scores from multiple variants
               _LOGGER.info("Expanding prompt: '%s'", scenario.prompt)
               expanded = generate_prompt(scenario.prompt, num_variants=self.config.num_variants)
               
               variant_results = {}
               for variant in expanded.positive_variants:
                   results = run_text_query(table, variant.text, scenario.top_k, self.config.model_size)
                   variant_results[variant.text] = results
               
               fused_scores = fuse_scores(expanded, variant_results)
               
               # Convert to system format (lower = better, so negate)
               for row_id, similarity in fused_scores.items():
                   base_slice_id = parse_row_id(row_id)[0]
                   if base_slice_id not in total_results or -similarity < total_results[base_slice_id]:
                       total_results[base_slice_id] = -similarity
           else:
               # Original: single query without expansion
               results = self._retrieve_top_ranks(table, scenario)
               if self.config.deduplicate:
                   results = deduplicate_by_base_slice(results)
               
               # Lower score = better
               total_results.update({parse_row_id(result.row_id)[0]: -result.similarity for result in results})

       # Filter to unlabeled slices
       filtered = {s: score for s, score in total_results.items() if s in self.slices_to_process}
       _LOGGER.info("Scored %d slices", len(filtered))
       return filtered


   def _filter_results(self, results: list[SearchResult], scenario: PromptConfig) -> list[SearchResult]:
       """Filtering based on similarity."""


       filtered = [r for r in results if r.similarity >= scenario.threshold]


       return sorted(filtered, key=lambda x: x.similarity, reverse=True)


   def _retrieve_top_ranks(self, table: Any, scenario: PromptConfig) -> list[SearchResult]:
       """Each prompt scored independently.


       Strategy:
       1. Retrieve more than K results initially (multiplier-based)
       2. Filter by similarity threshold
       3. Return top K from filtered results
       """
       initial_k = scenario.top_k * self.config.retrieval_multiplier


       retrieved_results = run_text_query(
           table,
           scenario.prompt,
           initial_k,
           self.config.model_size,
       )


       # filter by similarity threshold
       filtered_results = self._filter_results(retrieved_results, scenario)


       # return top K (or fewer if not enough results pass threshold)
       return filtered_results[: scenario.top_k]




