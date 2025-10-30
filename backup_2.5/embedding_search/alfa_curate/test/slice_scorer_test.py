"""Unit tests for ALFA curate data selection strategy."""


from unittest.mock import Mock, patch


import numpy as np
import pytest


from autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer import SliceScorer
from autonomy.perception.datasets.active_learning.alfa_curate.utils import SearchResult
from kits.ml.datasets.identifiers import Identifiers




def create_test_search_result(
   row_id: str = "test_slice",
   distance: float = 0.1,
   identifiers: Identifiers | None = None,
) -> SearchResult:
   """Helper to create SearchResult for tests."""
   if identifiers is None:
       identifiers = Identifiers(
           log_id="test_log",
           slice_id="test_clip",
       )


   return SearchResult(
       identifiers=identifiers,
       row_id=row_id,
       logapps_metadata=None,
       sensor_name="test_sensor",
       start_ns=np.int64(1000000000),
       end_ns=np.int64(1000000010),
       image_paths=["test.jpg"],
       embedding=np.zeros(512, dtype=np.float32),
       distance=distance,
   )




@pytest.fixture(name="config")
def get_config() -> Mock:
   """Create a mock StrategyConfig for testing."""
   config = Mock()
   config.prompt_yaml_path = (
       "autonomy/perception/datasets/active_learning/alfa_curate/tests/resources/test_config.yaml"
   )
   config.branch = "test_branch"
   config.repo = "test_repo"
   config.table_name = "test_table"
   config.model_size = "small"
   config.deduplicate = False
   config.batch_size = 100
   config.retrieval_multiplier = 2
   return config




def test_slice_scorer_init_with_yaml(config: Mock) -> None:
   """Test SliceScorer initialization with YAML file."""
   scorer = SliceScorer(config)
   assert len(scorer.scenarios) == 2
   assert scorer.scenarios[0].prompt == "People running across the road"
   assert scorer.scenarios[0].threshold == 0.5
   assert scorer.scenarios[1].prompt == "Car making left turn"
   assert scorer.scenarios[1].threshold == 0.5




def test_compute_slice_scores(config: Mock) -> None:
   """Test independent scoring mode with both above and below threshold scenarios."""
   with (
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table") as mock_load_table,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query") as mock_run_query,
   ):
       mock_load_table.return_value = Mock()
       scorer = SliceScorer(config)
       scorer.slices_to_process = {"slice_1", "slice_2"}


       # First test scenario, results below threshold (0.5)
       # distance=1.3 -> similarity = 1 - 1.3/2 = 0.35
       # distance=1.4 -> similarity = 1 - 1.4/2 = 0.30
       # Both below threshold (0.5), so should be filtered out
       mock_run_query.return_value = [
           create_test_search_result(row_id="slice_1", distance=1.3),
           create_test_search_result(row_id="slice_2", distance=1.4),
       ]


       result = scorer.compute_slice_scores({})


       # Revert negative sign to get actual similarity values
       result = {slice_id: -score for slice_id, score in result.items()}
       assert len(result) == 0
       assert mock_run_query.call_count == 2


       mock_run_query.reset_mock()


       # Second scenario, result above threshold (0.5)
       # distance=0.3 -> similarity = 1 - 0.3/2 = 0.85
       mock_run_query.return_value = [
           create_test_search_result(row_id="slice_1", distance=0.3),
       ]


       result = scorer.compute_slice_scores({})
       # Revert negative sign to get actual similarity values
       result = {slice_id: -score for slice_id, score in result.items()}


       # If threshold is 0.5, similarity=0.85 should pass
       assert len(result) == 1
       assert mock_run_query.call_count == 2




def test_deduplication_enabled(config: Mock) -> None:
   """Test that deduplication is called when enabled."""
   with (
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table") as mock_load_table,
       patch(
           "autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.deduplicate_by_base_slice"
       ) as mock_dedupe,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query") as mock_run_query,
   ):
       mock_load_table.return_value = Mock()
       config.deduplicate = True
       scorer = SliceScorer(config)


       test_result = create_test_search_result(row_id="slice_1", distance=0.3)
       mock_run_query.return_value = [test_result]


       mock_dedupe.return_value = [test_result]


       scorer.compute_slice_scores({})


       assert mock_dedupe.call_count == len(scorer.scenarios)


def test_prompt_expansion_disabled(config: Mock) -> None:
   """Test that when expansion is disabled, only original query is used."""
   with (
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table") as mock_load_table,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query") as mock_run_query,
   ):
       mock_load_table.return_value = Mock()
       config.use_prompt_expansion = False
       scorer = SliceScorer(config)
       scorer.slices_to_process = {"slice_1"}
       
       test_result = create_test_search_result(row_id="slice_1", distance=0.3)
       mock_run_query.return_value = [test_result]
       
       scorer.compute_slice_scores({})
       
       # Should call run_text_query once per scenario (2 scenarios, no expansion)
       assert mock_run_query.call_count == 2


def test_prompt_expansion_enabled(config: Mock) -> None:
   """Test that prompt expansion generates multiple variants and fuses scores."""
   with (
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table") as mock_load_table,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query") as mock_run_query,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.generate_prompt") as mock_expand,
       patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.fuse_scores") as mock_fuse,
   ):
       mock_load_table.return_value = Mock()
       config.use_prompt_expansion = True
       config.num_variants = 4
       
       # Mock VariedPrompts
       from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import PromptVariant, VariedPrompts
       
       mock_expanded = VariedPrompts(
           original="people running across the road",
           positive_variants=[
               PromptVariant("people running across the road"),
               PromptVariant("video of people running across the road"),
               PromptVariant("footage of people running across the road"),
               PromptVariant("scene showing people running across the road"),
               PromptVariant("clip of people running across the road"),
           ],
           negative_variants=[],
       )
       mock_expand.return_value = mock_expanded
       
       test_result = create_test_search_result(row_id="slice_1", distance=0.3)
       mock_run_query.return_value = [test_result]
       mock_fuse.return_value = {"slice_1": 0.85}
       
       scorer = SliceScorer(config)
       scorer.slices_to_process = {"slice_1"}
       
       result = scorer.compute_slice_scores({})
       
       # Should expand prompt for each scenario
       assert mock_expand.call_count == 2
       
       # Should call run_text_query for each variant (5 variants × 2 scenarios)
       assert mock_run_query.call_count == 10
       
       # Should fuse scores for each scenario
       assert mock_fuse.call_count == 2
       
       # Result should contain the fused score
       assert len(result) == 1


def test_prompt_expansion_people_running() -> None:
   """Test prompt expansion for 'people running across the road'."""
   from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import generate_prompt
   
   query = "people running across the road"
   expanded = generate_prompt(query, num_variants=4)
   
   # Should have original + 4 variants
   assert len(expanded.positive_variants) == 5
   
   # Original should be first
   assert expanded.positive_variants[0].text == query
   
   # Should have template variants
   variant_texts = [v.text for v in expanded.positive_variants]
   assert "video of people running across the road" in variant_texts
   assert "footage of people running across the road" in variant_texts
   assert "scene showing people running across the road" in variant_texts
   assert "clip of people running across the road" in variant_texts


def test_prompt_expansion_camera_obstruction() -> None:
   """Test prompt expansion for 'camera obstruction at daytime'."""
   from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import generate_prompt
   
   query = "camera obstruction at daytime"
   expanded = generate_prompt(query, num_variants=4)
   
   # Should have original + 4 variants (no synonyms for these words)
   assert len(expanded.positive_variants) == 5
   
   # Original should be first
   assert expanded.positive_variants[0].text == query
   
   # Should only have template variants (no synonyms in dictionary)
   variant_texts = [v.text for v in expanded.positive_variants]
   assert "video of camera obstruction at daytime" in variant_texts
   assert "footage of camera obstruction at daytime" in variant_texts
   assert "scene showing camera obstruction at daytime" in variant_texts
   assert "clip of camera obstruction at daytime" in variant_texts


def test_prompt_expansion_num_variants_control() -> None:
   """Test that num_variants parameter controls the number of variants generated."""
   from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import generate_prompt
   
   query = "person running across the road"
   
   # Test with different num_variants
   for num in [2, 4, 8]:
       expanded = generate_prompt(query, num_variants=num)
       # Original + num_variants (capped by available variants)
       assert len(expanded.positive_variants) <= num + 1
