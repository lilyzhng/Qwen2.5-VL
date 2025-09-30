from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for a single prompt used in selection."""

    #: The text prompt to search for
    prompt: str

    #: Minimum similarity score (0.0-1.0) to keep a slice
    threshold: float = 0.2

    #: Number of nearest neighbors to retrieve per prompt
    top_k: int = 200

    #: Scoring mode override for this specific prompt
    #: - None: Use the global scoring_mode from StrategyConfig
    #: - "independent": This prompt retrieves slices independently
    #: - "softmax": This prompt participates in softmax scoring with other softmax prompts
    scoring_mode: Optional[str] = None

    class Config:
        allow_mutation = False


class StrategyConfig(BaseModel):
    """Configuration for the alfa-based active learning selection strategy."""

    #: LanceDB branch to read from
    branch: str = "main"

    #: Cosmos model size for text embeddings
    model_size: str = "Cosmos-Embed1-448p"

    #: Path to YAML file containing prompts
    prompt_yaml_path: Optional[str] = Field(
        default=str(Path(__file__).with_name("prompts.yaml"))
    )

    #: Global scoring mode for multiple prompts (can be overridden per-prompt)
    #: - "softmax": Computes similarities for all prompts using softmax, then selects best matching prompt per slice
    #: - "independent": Each prompt retrieves top_k slices independently, scores are merged using max
    #: Mixed mode fusion: If different prompts use different modes, their results are combined
    scoring_mode: str = "softmax"

    #: Temperature parameter for softmax scoring (higher = more selective)
    softmax_temperature: float = 10.0

    #: Maximum number of slices to process in a single batch (memory optimization)
    batch_size: int = 1000

    #: Whether to deduplicate results by base slice ID after scoring
    #: When enabled, only the best scoring segment per base video is kept
    deduplicate_by_base_video: bool = True

    #: Minimum similarity of best result to consider prompt valid
    min_best_similarity: float = 0.0

    #: List of prompts
    prompts: List[PromptConfig] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

