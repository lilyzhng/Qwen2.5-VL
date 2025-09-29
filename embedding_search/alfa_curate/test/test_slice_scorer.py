"""Simplified test for SliceScorer demonstrating key functionality with mock data."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import numpy.typing as npt

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Mock external dependencies before importing
sys.modules['lancedb'] = MagicMock()
sys.modules['autonomy'] = MagicMock()
sys.modules['autonomy.perception'] = MagicMock()
sys.modules['autonomy.perception.datasets'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.framework'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.framework.slice_scorer_base'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.base_config'] = MagicMock()
sys.modules['autonomy.perception.datasets.features'] = MagicMock()
sys.modules['autonomy.perception.datasets.features.cosmos'] = MagicMock()
sys.modules['autonomy.perception.datasets.features.cosmos.infer'] = MagicMock()
sys.modules['kits'] = MagicMock()
sys.modules['kits.scalex'] = MagicMock()
sys.modules['kits.scalex.dataset'] = MagicMock()
sys.modules['kits.scalex.dataset.instances'] = MagicMock()
sys.modules['kits.scalex.dataset.instances.lance_dataset'] = MagicMock()
sys.modules['platforms'] = MagicMock()
sys.modules['platforms.lakefs'] = MagicMock()
sys.modules['platforms.lakefs.client'] = MagicMock()

# Create mock base classes
class MockBaseStrategyConfig:
    pass

class SimpleSliceScorerBase:
    def __class_getitem__(cls, item):
        class GenericBase(cls):
            pass
        return GenericBase
    
    def __init__(self, config):
        self.config = config

# Set up mocked modules
sys.modules['autonomy.perception.datasets.active_learning.base_config'].BaseStrategyConfig = MockBaseStrategyConfig
sys.modules['autonomy.perception.datasets.active_learning.framework.slice_scorer_base'].SimpleSliceScorerBase = SimpleSliceScorerBase
sys.modules['autonomy.perception.datasets.active_learning.framework.slice_scorer_base'].DataModelReader = MagicMock

# Mock Cosmos
VIDEO_EMBED_DIM = 768
class MockCosmos:
    def __init__(self, model_size="medium", load_model_from_lakefs=False):
        self.model_size = model_size
    
    def text_embedding(self, text):
        return np.random.randn(VIDEO_EMBED_DIM).astype(np.float32)

sys.modules['autonomy.perception.datasets.features.cosmos.infer'].Cosmos = MockCosmos

# Now import our modules
from alfa_curate.slice_scorer import SliceScorer
from alfa_curate.config import StrategyConfig, PromptConfig
from alfa_curate.utils import distance_to_similarity


@dataclass
class Identifiers:
    log_id: str
    scene_id: Optional[str] = None

@dataclass  
class LogappsMetadata:
    tags: List[str]
    description: Optional[str] = None

@dataclass
class EmbeddedVideoFixed:
    """Mock video data matching the LanceDB schema."""
    identifiers: Identifiers
    row_id: str
    logapps_metadata: Optional[LogappsMetadata]
    sensor_name: str
    start_ns: np.int64
    end_ns: np.int64
    image_paths: List[str]
    embedding: npt.NDArray[np.float32]


class TestSliceScorerSimple(unittest.TestCase):
    """Simplified tests focusing on core SliceScorer functionality."""
    
    def test_distance_to_similarity(self):
        """Test the distance to similarity conversion."""
        # Perfect match
        self.assertEqual(distance_to_similarity(0.0), 1.0)
        # Maximum distance
        self.assertEqual(distance_to_similarity(2.0), 0.0)
        # Intermediate
        self.assertAlmostEqual(distance_to_similarity(1.0), 0.5)
    
    @patch('alfa_curate.slice_scorer.run_text_query')
    def test_slice_scorer_basic_flow(self, mock_run_text_query):
        """Test basic SliceScorer workflow with mocked data."""
        # Configure mock to return some results
        mock_run_text_query.return_value = [
            {"row_id": "video_001_segment_1000_2000_front", "_distance": 0.5},
            {"row_id": "video_002_segment_3000_4000_front", "_distance": 0.8},
            {"row_id": "video_003_segment_5000_6000_front", "_distance": 1.2}
        ]
        
        # Create config
        config = StrategyConfig()
        config.branch = "test"
        config.model_size = "medium"
        config.scoring_mode = "independent"
        config.prompts = [
            PromptConfig(prompt="pedestrian crossing", threshold=0.4, top_k=10)
        ]
        config.load_prompts_from_yaml = False
        config.deduplicate_by_base_video = False
        config.min_best_similarity = 0.0
        config.batch_size = 10
        
        # Create scorer
        scorer = SliceScorer(config)
        
        # Test process_slice
        mock_reader = MagicMock()
        mock_reader.id = "test_id"
        self.assertEqual(scorer.process_slice(mock_reader), "test_id")
        
        # Test compute_slice_scores
        candidate_scores = [
            ("video_001_segment_1000_2000_front", None),
            ("video_002_segment_3000_4000_front", None),
            ("video_999_segment_9000_9999_front", None)  # Not in results
        ]
        
        results = scorer.compute_slice_scores(candidate_scores)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # Only 2 candidates match
        
        # Results should be sorted by score (negative for priority)
        result_dict = dict(results)
        self.assertIn("video_001_segment_1000_2000_front", result_dict)
        self.assertIn("video_002_segment_3000_4000_front", result_dict)
        self.assertNotIn("video_999_segment_9000_9999_front", result_dict)
        
        # Check scores are negative (lower = higher priority)
        for _, score in results:
            self.assertLess(score, 0)
    
    @patch('alfa_curate.slice_scorer.load_table')
    @patch('alfa_curate.slice_scorer.text_to_embedding')
    def test_softmax_scoring_mode(self, mock_text_to_embedding, mock_load_table):
        """Test softmax scoring mode."""
        # Create mock embeddings
        embeddings = {
            "pedestrian": np.array([1.0, 0.0, 0.0] + [0.0] * (VIDEO_EMBED_DIM - 3), dtype=np.float32),
            "vehicle": np.array([0.0, 1.0, 0.0] + [0.0] * (VIDEO_EMBED_DIM - 3), dtype=np.float32),
            "video1": np.array([0.9, 0.1, 0.0] + [0.0] * (VIDEO_EMBED_DIM - 3), dtype=np.float32),
            "video2": np.array([0.1, 0.9, 0.0] + [0.0] * (VIDEO_EMBED_DIM - 3), dtype=np.float32)
        }
        
        # Normalize embeddings
        for key in embeddings:
            embeddings[key] = embeddings[key] / np.linalg.norm(embeddings[key])
        
        # Mock text embeddings
        mock_text_to_embedding.side_effect = lambda text, size: (
            embeddings["pedestrian"] if "pedestrian" in text else embeddings["vehicle"]
        )
        
        # Mock table with search results
        mock_table = MagicMock()
        mock_table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
            {"row_id": "video_001", "embedding": embeddings["video1"].tolist()},
            {"row_id": "video_002", "embedding": embeddings["video2"].tolist()}
        ]
        mock_load_table.return_value = mock_table
        
        # Create config for softmax mode
        config = StrategyConfig()
        config.branch = "test"
        config.model_size = "medium"
        config.scoring_mode = "softmax"
        config.prompts = [
            PromptConfig(prompt="pedestrian crossing", threshold=0.3, top_k=10),
            PromptConfig(prompt="vehicle turning", threshold=0.3, top_k=10)
        ]
        config.load_prompts_from_yaml = False
        config.deduplicate_by_base_video = False
        config.min_best_similarity = 0.0
        config.batch_size = 10
        config.softmax_temperature = 5.0
        
        scorer = SliceScorer(config)
        
        # Test with candidates
        candidate_scores = [("video_001", None), ("video_002", None)]
        results = scorer.compute_slice_scores(candidate_scores)
        
        # Both videos should be scored
        self.assertEqual(len(results), 2)
        result_dict = dict(results)
        
        # video_001 should have better score (more similar to pedestrian)
        # video_002 should have better score (more similar to vehicle)
        # But both should be included since they pass threshold
        self.assertIn("video_001", result_dict)
        self.assertIn("video_002", result_dict)


if __name__ == '__main__':
    unittest.main()
