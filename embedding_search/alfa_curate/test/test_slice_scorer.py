import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from autonomy.perception.datasets.active_learning.alfa_curate.config import StrategyConfig
from autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer import SliceScorer

# Test data constants
TEST_PROMPTS = [
    {"prompt": "cars driving", "threshold": 0.7, "top_k": 10, "scoring_mode": "independent"},
    {"prompt": "pedestrians crossing", "threshold": 0.6, "top_k": 15, "scoring_mode": "independent"},
]

TEST_SLICES = [("slice_1", None), ("slice_2", None)]


class MockPromptConfig:
    """Simple prompt configuration for testing."""
    def __init__(self, prompt: str, threshold: float = 0.7, top_k: int = 10, scoring_mode: str = None):
        self.prompt = prompt
        self.threshold = threshold
        self.top_k = top_k
        self.scoring_mode = scoring_mode


@pytest.fixture(name="config")
def get_config():
    """Create a mock StrategyConfig with default values."""
    config = Mock(spec=StrategyConfig)
    config.prompt_yaml_path = None
    config.scoring_mode = "independent"
    config.branch = "test_branch"
    config.model_size = "small"
    config.min_best_similarity = 0.0
    config.deduplicate_by_base_video = False
    config.batch_size = 100
    config.softmax_temperature = 1.0
    return config


@pytest.fixture
def sample_prompts():
    """Create sample PromptConfig objects from default data."""
    return [MockPromptConfig(**prompt_data) for prompt_data in TEST_PROMPTS]


@pytest.fixture
def mock_query_results():
    """Common query result patterns for tests."""
    return {
        "high_similarity": [
            {"row_id": "slice_1", "_distance": 0.3}, 
            {"row_id": "slice_2", "_distance": 0.5}],
        "low_similarity": [{"row_id": "slice_1", "_distance": 0.8}],
        "embeddings": [
            {"row_id": "slice_1", "embedding": [0.9, 0.1, 0.0]},
            {"row_id": "slice_2", "embedding": [0.1, 0.9, 0.0]},
        ],
    }


def create_mock_table():
    """Create a mock table with search chain for testing."""
    mock_table = Mock()
    mock_search = Mock()
    mock_table.search.return_value = mock_search
    mock_search.where.return_value = mock_search
    mock_search.limit.return_value = mock_search
    return mock_table, mock_search


def test_slice_scorer_init_with_yaml(config: StrategyConfig) -> None:
    """Test SliceScorer initialization with YAML file."""
    yaml_content = {"scenarios": TEST_PROMPTS}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        temp_path = f.name

    try:
        config.prompt_yaml_path = temp_path
        scorer = SliceScorer(config)
        assert len(scorer._prompts) == len(TEST_PROMPTS)
        for i, expected in enumerate(TEST_PROMPTS):
            assert scorer._prompts[i].prompt == expected["prompt"]
    finally:
        Path(temp_path).unlink()


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_compute_independent_scores(mock_dist_to_sim, mock_run_query, config, sample_prompts, mock_query_results):
    """Test independent scoring mode with both above and below threshold scenarios."""
    scorer = SliceScorer(config)
    scorer._prompts = sample_prompts

    # Test high similarity scenario
    mock_run_query.return_value = mock_query_results["high_similarity"]
    mock_dist_to_sim.side_effect = [0.8, 0.7, 0.8, 0.7]

    result = scorer.compute_slice_scores(TEST_SLICES)

    assert len(result) == 2, "Expected 2 results for high similarity scenario"
    assert all(score < 0 for _, score in result), "All scores should be negative"
    assert mock_run_query.call_count == len(sample_prompts), f"Expected {len(sample_prompts)} query calls"

    # Test low similarity scenario
    mock_run_query.reset_mock()
    mock_dist_to_sim.reset_mock()
    mock_run_query.return_value = mock_query_results["low_similarity"]
    mock_dist_to_sim.return_value = 0.5

    result = scorer.compute_slice_scores(list(TEST_SLICES[0]))

    assert len(result) == 0, "Expected no results for low similarity scenario"
    assert mock_run_query.call_count == len(sample_prompts), f"Expected {len(sample_prompts)} query calls"


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.text_to_embedding")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table")
def test_compute_softmax_scores(mock_load_table, mock_text_to_embedding, config, sample_prompts, mock_query_results):
    """Test softmax scoring mode that handles multiple prompts."""
    config.scoring_mode = "softmax"
    scorer = SliceScorer(config)
    scorer._prompts = sample_prompts

    mock_text_to_embedding.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]

    mock_table, mock_search = create_mock_table()
    mock_search.to_list.return_value = mock_query_results["embeddings"]
    mock_load_table.return_value = mock_table

    result = scorer.compute_slice_scores(TEST_SLICES)

    assert all(score <= 0 for _, score in result), "All softmax scores should be non-positive"


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.parse_row_id")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_deduplication_enabled(mock_dist_to_sim, mock_run_query, mock_parse, config, sample_prompts, mock_query_results):
    """Test that deduplication is called when enabled."""
    config.deduplicate_by_base_video = True
    scorer = SliceScorer(config)
    scorer._prompts = sample_prompts

    # Mock query results with segments from same base video
    mock_run_query.return_value = [
        {"row_id": "video1_segment_0_100_camera1", "_distance": 0.3},
        {"row_id": "video1_segment_100_200_camera1", "_distance": 0.2},
    ]
    mock_parse.side_effect = [
        ("video1", "0", "100", "camera1"),
        ("video1", "100", "200", "camera1"),
        ("video1", "0", "100", "camera1"),
        ("video1", "100", "200", "camera1"),
    ]
    mock_dist_to_sim.side_effect = [0.85, 0.90, 0.85, 0.90]

    result = scorer.compute_slice_scores([("video1_segment_0_100_camera1", None), ("video1_segment_100_200_camera1", None)])

    # With dedup enabled, should only keep the best segment per base video
    assert len(result) == 1, "Expected 1 result after deduplication by base video"
    assert result[0][0] == "video1_segment_100_200_camera1", "Should keep the segment with highest score (0.90)"


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_min_best_similarity_filtering(mock_dist_to_sim, mock_run_query, config, sample_prompts, mock_query_results):
    """Test filtering based on min_best_similarity."""
    config.min_best_similarity = 0.9
    scorer = SliceScorer(config)
    scorer._prompts = sample_prompts

    mock_run_query.return_value = mock_query_results["low_similarity"]
    mock_dist_to_sim.return_value = 0.8

    result = scorer.compute_slice_scores(list(TEST_SLICES[0]))

    assert len(result) == 0, "No results expected when similarity (0.8) < threshold (0.9)"


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.text_to_embedding")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_mixed_mode_fusion(mock_dist_to_sim, mock_run_query, mock_load_table, mock_text_to_embedding, config, mock_query_results):
    """Test fusion of independent and softmax scoring modes."""
    config.scoring_mode = "independent"  # Global default
    scorer = SliceScorer(config)
    
    # Create mixed mode prompts: first uses independent, second uses softmax
    scorer._prompts = [
        MockPromptConfig("cars driving", threshold=0.7, top_k=10, scoring_mode="independent"),
        MockPromptConfig("pedestrians", threshold=0.6, top_k=15, scoring_mode="softmax"),
    ]

    # Mock independent mode query
    mock_run_query.return_value = mock_query_results["high_similarity"]
    mock_dist_to_sim.side_effect = [0.8, 0.75]

    # Mock softmax mode embeddings
    mock_text_to_embedding.return_value = np.array([1.0, 0.0, 0.0])
    mock_table, mock_search = create_mock_table()
    mock_search.to_list.return_value = mock_query_results["embeddings"]
    mock_load_table.return_value = mock_table

    result = scorer.compute_slice_scores(TEST_SLICES)

    # Should have results from both modes
    assert len(result) > 0, "Expected results from fusion of both modes"
    assert mock_run_query.call_count == 1, "Independent mode should be called once"
    assert mock_text_to_embedding.call_count == 1, "Softmax mode should process one prompt"


def test_per_prompt_scoring_mode_override(config):
    """Test that per-prompt scoring_mode overrides global config."""
    yaml_content = {
        "scenarios": [
            {"prompt": "test1", "threshold": 0.5, "scoring_mode": "softmax"},
            {"prompt": "test2", "threshold": 0.5},  # Uses global default
            {"prompt": "test3", "threshold": 0.5, "scoring_mode": "independent"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        temp_path = f.name

    try:
        config.prompt_yaml_path = temp_path
        config.scoring_mode = "independent"
        scorer = SliceScorer(config)
        
        assert len(scorer._prompts) == 3
        assert scorer._prompts[0].scoring_mode == "softmax", "First prompt should override to softmax"
        assert scorer._prompts[1].scoring_mode is None, "Second prompt should use None (will use global default)"
        assert scorer._prompts[2].scoring_mode == "independent", "Third prompt should override to independent"
    finally:
        Path(temp_path).unlink()


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.parse_row_id")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_final_deduplication_keeps_best_score(mock_dist_to_sim, mock_run_query, mock_parse, config):
    """Test that final deduplication keeps the segment with the best score."""
    config.deduplicate_by_base_video = True
    scorer = SliceScorer(config)
    scorer._prompts = [
        MockPromptConfig("test query", threshold=0.5, top_k=10, scoring_mode="independent"),
    ]

    # Return multiple segments from same base video with different scores
    mock_run_query.return_value = [
        {"row_id": "base_video_segment_0_100_cam1", "_distance": 0.4},   # similarity = 0.8
        {"row_id": "base_video_segment_100_200_cam1", "_distance": 0.2}, # similarity = 0.9 (best)
        {"row_id": "base_video_segment_200_300_cam1", "_distance": 0.6}, # similarity = 0.7
    ]
    
    # All segments have same base_id
    mock_parse.side_effect = [
        ("base_video", "0", "100", "cam1"),
        ("base_video", "100", "200", "cam1"),
        ("base_video", "200", "300", "cam1"),
    ]
    
    mock_dist_to_sim.side_effect = [0.8, 0.9, 0.7]

    result = scorer.compute_slice_scores([
        ("base_video_segment_0_100_cam1", None),
        ("base_video_segment_100_200_cam1", None),
        ("base_video_segment_200_300_cam1", None),
    ])

    # Should only keep one segment (the one with highest score = 0.9)
    assert len(result) == 1, "Deduplication should keep only one segment per base video"
    assert result[0][0] == "base_video_segment_100_200_cam1", "Should keep segment with best score (0.9)"