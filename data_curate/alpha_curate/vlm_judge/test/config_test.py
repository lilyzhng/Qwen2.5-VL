"""Unit tests for VLM Judge configuration."""

from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.config import (
    VLMJudgeConfig,
)


def test_config_custom_values() -> None:
    """Test VLMJudgeConfig with custom values."""
    config = VLMJudgeConfig(
        enable_vlm_judge=False,
        max_candidates_for_vlm=50,
        vlm_model_path="custom/model",
        vlm_confidence_threshold=0.8,
    )
    
    assert config.enable_vlm_judge is False
    assert config.max_candidates_for_vlm == 50
    assert config.vlm_model_path == "custom/model"
    assert config.vlm_confidence_threshold == 0.8

