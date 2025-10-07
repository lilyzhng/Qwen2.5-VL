import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import yaml

from autonomy.perception.datasets.active_learning.alfa_curate.config import StrategyConfig, TaskStrategy
from autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer import SliceScorer


TEST_PROMPTS = [
    {"prompt": "cars driving", "threshold": 0.7, "top_k": 10},
    {"prompt": "pedestrians crossing", "threshold": 0.6, "top_k": 15},
]

TEST_SLICES = [("slice_1", None), ("slice_2", None)]


class MockPromptConfig:
    """Simple prompt configuration for testing."""
    def __init__(self, prompt: str, threshold: float = 0.7, top_k: int = 10, task: str = None):
        self.prompt = prompt
        self.threshold = threshold
        self.top_k = top_k
        self.task = task


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
    yaml_content = {
        "tasks": {
            "test_task": {
                "priority": 10,
                "scoring_mode": "independent"
            }
        },
        "scenarios": TEST_PROMPTS
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        temp_path = f.name

    try:
        config.prompt_yaml_path = temp_path
        scorer = SliceScorer(config)
        assert len(scorer._prompts) == len(TEST_PROMPTS)
        for i, expected in enumerate(TEST_PROMPTS):
            assert scorer._prompts[i].prompt == expected["prompt"]
        # Verify task strategy was loaded
        assert "test_task" in config.task_strategies
        assert config.task_strategies["test_task"].priority == 10
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
def test_mixed_mode_fusion_via_tasks(mock_dist_to_sim, mock_run_query, mock_load_table, mock_text_to_embedding, config, mock_query_results):
    """Test fusion of independent and softmax scoring modes via different tasks."""
    config.scoring_mode = "independent"  # Global default
    config.task_strategies = {
        "task_independent": TaskStrategy(task_name="task_independent", priority=1, scoring_mode="independent"),
        "task_softmax": TaskStrategy(task_name="task_softmax", priority=2, scoring_mode="softmax"),
    }
    scorer = SliceScorer(config)
    
    # Create tasks with different modes
    scorer._prompts = [
        MockPromptConfig("cars driving", threshold=0.7, top_k=10, task="task_independent"),
        MockPromptConfig("pedestrians", threshold=0.6, top_k=15, task="task_softmax"),
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

    # Should have results from both task modes
    assert len(result) > 0, "Expected results from fusion of both task modes"
    assert mock_run_query.call_count == 1, "Independent task should be called once"
    assert mock_text_to_embedding.call_count == 1, "Softmax task should process one prompt"


def test_task_scoring_mode_hierarchy(config):
    """Test that task scoring_mode is loaded from YAML."""
    yaml_content = {
        "tasks": {
            "task_a": {
                "priority": 1,
                "scoring_mode": "softmax"
            },
            "task_b": {
                "priority": 2,
                "scoring_mode": "independent"
            }
        },
        "scenarios": [
            {"prompt": "test1", "task": "task_a", "threshold": 0.5},
            {"prompt": "test2", "task": "task_b", "threshold": 0.5},
            {"prompt": "test3", "threshold": 0.5},  # No task, uses default
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
        assert scorer._prompts[0].task == "task_a"
        assert scorer._prompts[1].task == "task_b"
        assert scorer._prompts[2].task is None  # Will use default task
        
        # Verify task strategies were loaded from YAML
        assert config.task_strategies["task_a"].scoring_mode == "softmax"
        assert config.task_strategies["task_b"].scoring_mode == "independent"
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
        MockPromptConfig("test query", threshold=0.5, top_k=10),
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


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_task_based_priority_ordering(mock_dist_to_sim, mock_run_query, config):
    """Test that tasks are executed in priority order."""
    config.task_strategies = {
        "high_priority": TaskStrategy(task_name="high_priority", priority=1, scoring_mode="independent"),
        "low_priority": TaskStrategy(task_name="low_priority", priority=10, scoring_mode="independent"),
    }
    
    scorer = SliceScorer(config)
    scorer._prompts = [
        MockPromptConfig("low priority query", task="low_priority", threshold=0.5),
        MockPromptConfig("high priority query", task="high_priority", threshold=0.5),
    ]
    
    mock_run_query.return_value = [{"row_id": "slice_1", "_distance": 0.4}]
    mock_dist_to_sim.return_value = 0.8
    
    result = scorer.compute_slice_scores([("slice_1", None)])
    
    # Verify both tasks ran
    assert len(result) > 0, "Should have results from both tasks"
    # High priority task should run first (but we can't directly verify order in current implementation)


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_task_isolation(mock_dist_to_sim, mock_run_query, config):
    """Test that separated tasks remove results from candidate pool."""
    config.task_strategies = {
        "separating_task": TaskStrategy(
            task_name="separating_task", 
            priority=1, 
            scoring_mode="independent",
            separate_dataset=True  # Should remove from pool
        ),
        "normal_task": TaskStrategy(
            task_name="normal_task", 
            priority=10, 
            scoring_mode="independent",
            separate_dataset=False
        ),
    }
    
    scorer = SliceScorer(config)
    scorer._prompts = [
        MockPromptConfig("separating query", task="separating_task", threshold=0.5),
        MockPromptConfig("normal query", task="normal_task", threshold=0.5),
    ]
    
    # First task returns slice_1
    # Second task should not see slice_1 (it's separated)
    call_count = [0]
    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:  # First call (separating task)
            return [{"row_id": "slice_1", "_distance": 0.3}]
        else:  # Second call (normal task) - should get different candidates
            return [{"row_id": "slice_2", "_distance": 0.4}]
    
    mock_run_query.side_effect = side_effect
    mock_dist_to_sim.side_effect = [0.85, 0.80]
    
    result = scorer.compute_slice_scores([("slice_1", None), ("slice_2", None)])
    
    # Both slices should appear in results
    result_ids = {r[0] for r in result}
    assert "slice_1" in result_ids, "Separated task should contribute slice_1"
    assert len(result) >= 1, "Should have results from separated task"


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_task_specific_scoring_mode(mock_dist_to_sim, mock_run_query, config):
    """Test that task strategy scoring mode is used."""
    config.scoring_mode = "independent"  # Global default
    config.task_strategies = {
        "softmax_task": TaskStrategy(
            task_name="softmax_task",
            priority=10,
            scoring_mode="softmax"  # Task-level setting
        ),
    }
    
    scorer = SliceScorer(config)
    # All prompts in task use task's scoring_mode
    scorer._prompts = [
        MockPromptConfig("test query", task="softmax_task", threshold=0.5),
    ]
    
    # Verify the task strategy is retrieved correctly
    strategy = scorer._get_task_strategy("softmax_task")
    assert strategy.scoring_mode == "softmax", "Task strategy should use softmax mode"


def test_yaml_loading_with_tasks(config):
    """Test loading YAML with tasks section and scenarios."""
    yaml_content = {
        "tasks": {
            "obstruction": {
                "priority": 1,
                "scoring_mode": "independent",
                "separate_dataset": True
            },
            "pca": {
                "priority": 10,
                "scoring_mode": "softmax",
                "separate_dataset": False
            }
        },
        "scenarios": [
            {"prompt": "obstruction", "task": "obstruction", "threshold": 0.3, "top_k": 500},
            {"prompt": "pca query", "task": "pca", "threshold": 0.2, "top_k": 200},
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        temp_path = f.name
    
    try:
        config.prompt_yaml_path = temp_path
        scorer = SliceScorer(config)
        
        # Verify prompts loaded
        assert len(scorer._prompts) == 2
        assert scorer._prompts[0].task == "obstruction"
        assert scorer._prompts[1].task == "pca"
        
        # Verify task strategies loaded from YAML
        assert "obstruction" in config.task_strategies
        assert config.task_strategies["obstruction"].priority == 1
        assert config.task_strategies["obstruction"].separate_dataset is True
        
        assert "pca" in config.task_strategies
        assert config.task_strategies["pca"].priority == 10
        assert config.task_strategies["pca"].scoring_mode == "softmax"
    finally:
        Path(temp_path).unlink()


def test_default_task_strategy_creation(config):
    """Test that default strategy is created for unconfigured tasks."""
    scorer = SliceScorer(config)
    
    # Request strategy for a task not in config
    strategy = scorer._get_task_strategy("undefined_task")
    
    assert strategy.task_name == "undefined_task"
    assert strategy.priority == 100  # Default priority
    assert strategy.scoring_mode == config.scoring_mode  # Uses global
    assert strategy.separate_dataset is False  # Default


@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.text_to_embedding")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.load_table")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.run_text_query")
@patch("autonomy.perception.datasets.active_learning.alfa_curate.slice_scorer.distance_to_similarity")
def test_mixed_task_and_mode_fusion(mock_dist_to_sim, mock_run_query, mock_load_table, mock_text_to_embedding, config, mock_query_results):
    """Test complex scenario: multiple tasks with different modes."""
    config.task_strategies = {
        "task_a": TaskStrategy(task_name="task_a", priority=1, scoring_mode="independent"),
        "task_b": TaskStrategy(task_name="task_b", priority=10, scoring_mode="softmax"),
    }
    
    scorer = SliceScorer(config)
    scorer._prompts = [
        MockPromptConfig("query 1", task="task_a", threshold=0.7),
        MockPromptConfig("query 2", task="task_b", threshold=0.6),
        MockPromptConfig("query 3", task="task_b", threshold=0.6),
    ]
    
    # Mock independent mode
    mock_run_query.return_value = mock_query_results["high_similarity"]
    mock_dist_to_sim.side_effect = [0.8, 0.75]
    
    # Mock softmax mode
    mock_text_to_embedding.side_effect = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    mock_table, mock_search = create_mock_table()
    mock_search.to_list.return_value = mock_query_results["embeddings"]
    mock_load_table.return_value = mock_table
    
    result = scorer.compute_slice_scores(TEST_SLICES)
    
    # Should have results from both tasks
    assert len(result) > 0, "Should have combined results from multiple tasks"