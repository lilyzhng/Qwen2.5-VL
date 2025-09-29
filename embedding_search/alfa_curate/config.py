"""Configuration for prompt-based active learning strategy using Cosmos text search.

This module defines the configuration structure for the ALFA curate active learning
strategy that uses text prompts to select relevant video slices.
"""

from typing import List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Try to import from autonomy package if available
try:
    from autonomy.perception.datasets.active_learning.base_config import (
        BaseStrategyConfig,  # pragma: ignore-dep
    )
    base_class = BaseStrategyConfig
except ImportError:
    # Fallback if running outside autonomy environment
    base_class = object


@dataclass
class PromptConfig:
    """Configuration for a single prompt used in active learning selection."""
    
    #: The text prompt to search for
    prompt: str
    
    #: Minimum similarity score (0.0-1.0) to keep a slice
    threshold: float = 0.2
    
    #: Number of nearest neighbors to retrieve per prompt
    top_k: int = 200


class StrategyConfig(base_class if base_class is not object else object):
    """Configuration for the prompt-based active learning selection strategy."""
    
    #: LanceDB branch to read from
    branch: str = "main"
    
    #: Cosmos model size for text embeddings ("small", "medium", or "large")
    model_size: str = "medium"
    
    #: Path to YAML file containing prompt definitions
    prompt_yaml_path: str = str(Path(__file__).with_name("prompts.yaml"))
    
    #: Scoring mode for multiple prompts
    #: - "softmax": Each slice assigned to best matching prompt (recommended)
    #: - "independent": Slices can match multiple prompts
    scoring_mode: str = "softmax"
    
    #: Temperature parameter for softmax scoring (higher = more selective)
    softmax_temperature: float = 10.0
    
    #: Maximum number of slices to process in a single batch (memory optimization)
    batch_size: int = 100
    
    #: Whether to deduplicate results by base video ID
    deduplicate_by_base_video: bool = True
    
    #: Minimum similarity of best result to consider prompt valid (noise prevention)
    min_best_similarity: float = 0.0
    
    #: List of prompts (can be loaded from YAML or defined directly)
    prompts: List[PromptConfig] = field(default_factory=list)
    
    #: Whether to load prompts from YAML file on initialization
    load_prompts_from_yaml: bool = True
    
    #: Number of concurrent workers for processing (if using Ray)
    num_workers: Optional[int] = None
    
    #: Whether to cache text embeddings for prompts
    cache_prompt_embeddings: bool = True
    
    #: Path to cache directory for embeddings
    cache_dir: Optional[str] = None
    
    #: Verbosity level for logging (0=quiet, 1=normal, 2=verbose)
    verbosity: int = 1


@dataclass
class EvaluationConfig:
    """Configuration for evaluating the active learning selection results."""
    
    #: Minimum number of slices required per prompt for evaluation
    min_slices_per_prompt: int = 5
    
    #: Whether to generate visualization of selected slices
    generate_visualizations: bool = True
    
    #: Output directory for evaluation results
    output_dir: str = "./alfa_curate_results"
    
    #: Whether to export selected slice IDs to file
    export_slice_ids: bool = True
    
    #: Format for exported slice IDs ("json", "csv", or "parquet")
    export_format: str = "json"
