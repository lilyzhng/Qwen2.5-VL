"""Unit tests for VLM Judge inference."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch

from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer import (
    VLMJudge,
    VLMJudgeResult,
    _get_cache_paths,
    _load_model,
    _load_prompts_from_yaml,
    load_qwenvl_model,
)


def test_vlm_judge_result() -> None:
    """Test VLMJudgeResult creation."""
    result = VLMJudgeResult(
        query="Is there a red car?",
        match=True,
        confidence=0.95,
        observation="I see a red sedan",
        reason="The vehicle matches",
        raw_response='{"match": true}',
    )
    
    assert result.query == "Is there a red car?"
    assert result.match is True
    assert result.confidence == 0.95


def test_get_cache_paths() -> None:
    """Test cache path generation."""
    model_path = "Qwen/Qwen3-VL-2B-Instruct"
    cache_dir, temp_dir, completion_marker = _get_cache_paths(model_path)
    
    # Check special characters are replaced
    assert "/" not in cache_dir.name
    assert ":" not in cache_dir.name
    assert temp_dir.name.endswith(".downloading")
    assert completion_marker.name == ".download_complete"


def test_load_model() -> None:
    """Test model loading with and without flash attention."""
    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer.AutoModelForImageTextToText") as mock_model,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer.AutoProcessor") as mock_processor,
    ):
        mock_model_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # Test without flash attention
        model, processor = _load_model("test/model", use_flash_attn=False)
        assert model == mock_model_instance
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert "attn_implementation" not in call_kwargs
        
        # Test with flash attention
        model, processor = _load_model("test/model", use_flash_attn=True)
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs["attn_implementation"] == "flash_attention_2"


def test_load_qwenvl_model_from_lakefs_cached(tmp_path: Path) -> None:
    """Test loading cached model from LakeFS."""
    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer._get_cache_paths") as mock_get_paths,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer._load_model") as mock_load,
    ):
        # Setup cached model
        mock_cache_dir = tmp_path / "cached_model"
        mock_cache_dir.mkdir()
        mock_completion = mock_cache_dir / ".download_complete"
        mock_completion.touch()
        mock_get_paths.return_value = (mock_cache_dir, tmp_path / "temp", mock_completion)
        
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_load.return_value = (mock_model, mock_processor)
        
        model, processor = load_qwenvl_model("test/model", load_model_from_lakefs=True)
        
        # Should load from cache
        mock_load.assert_called_once_with(str(mock_cache_dir), False)


def test_load_prompts_from_yaml() -> None:
    """Test loading prompts from YAML."""
    mock_yaml = """
system_prompts:
  role: "You are an expert."
user_prompts:
  judgment: "{query}"
"""
    with patch("builtins.open", mock_open(read_data=mock_yaml)):
        prompts = _load_prompts_from_yaml()
        assert "system_prompts" in prompts
        assert "user_prompts" in prompts


def test_vlm_judge_frames_valid_response() -> None:
    """Test judge_frames with valid JSON response."""
    import numpy as np
    
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
    query = "Is there a pedestrian?"
    
    mock_response = {
        "query": query,
        "match": True,
        "confidence": 0.9,
        "observation": "I see a person",
        "reason": "Person is visible",
    }
    
    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer.load_qwenvl_model") as mock_load,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer._load_prompts_from_yaml") as mock_prompts,
    ):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_load.return_value = (mock_model, mock_processor)
        mock_prompts.return_value = {"system_prompts": {}, "user_prompts": {"judgment": "{query}"}}
        
        # Mock model generation
        mock_model.device = "cuda"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_processor.apply_chat_template.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
        mock_processor.batch_decode.return_value = [json.dumps(mock_response)]
        
        judge = VLMJudge(model_path="test/model")
        result = judge.judge_frames(frames, query)
        
        assert result.match is True
        assert result.confidence == 0.9
        assert result.observation == "I see a person"
