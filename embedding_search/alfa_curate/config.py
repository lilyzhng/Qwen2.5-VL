from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStrategy(BaseModel):
    """Strategy configuration for a specific task."""

    #: Task identifier
    task_name: str

    #: Execution order (lower = earlier, useful for filtering tasks)
    priority: int = 100

    #: Scoring mode for this task's prompts ("independent" or "softmax")
    scoring_mode: str = "softmax"

    #: Whether to separate this task's results into a distinct dataset (removed from candidate pool for subsequent tasks)
    separate_dataset: bool = False

    class Config:
        allow_mutation = False


class PromptConfig(BaseModel):
    """Configuration for a single prompt used in selection."""

    #: The text prompt to search for
    prompt: str

    #: Task this prompt belongs to (used for grouping and applying task-specific strategies)
    #: If None, uses "default" task with global scoring_mode
    task: Optional[str] = None

    #: Minimum similarity score (0.0-1.0) to keep a slice
    threshold: float = 0.2

    #: Number of nearest neighbors to retrieve per prompt
    top_k: int = 200

    class Config:
        allow_mutation = False


class StrategyConfig(BaseModel):
    """Configuration for the alfa-based active learning selection strategy."""

    #: LanceDB branch to read from
    branch: str = "main"

    #: Cosmos model size for text embeddings
    model_size: str = "Cosmos-Embed1-448p"

    #: Path to YAML file containing prompts
    prompt_yaml_path: Optional[str] = "prompts.yaml"

    #: Global default scoring mode (used for tasks without explicit strategy)
    #: - "softmax": Computes similarities for all prompts using learned logit_scale from model, then selects best matching prompt per slice
    #: - "independent": Each prompt retrieves top_k slices independently, scores are merged using max
    #: Note: Task-level strategies override this setting
    scoring_mode: str = "softmax"

    #: Maximum number of slices to process in a single batch (memory optimization)
    batch_size: int = 1000

    #: Whether to deduplicate results by base slice ID after scoring
    #: When enabled, only the best scoring segment per base video is kept
    deduplicate_by_base_video: bool = True

    #: Minimum similarity of best result to consider prompt valid
    min_best_similarity: float = 0.0

    #: List of prompts
    prompts: List[PromptConfig] = Field(default_factory=list)

    #: Task-specific strategies (task_name -> TaskStrategy)
    #: Loaded from YAML file's "tasks:" section
    #: If a prompt's task is not defined here, default strategy is created using global scoring_mode
    task_strategies: Dict[str, TaskStrategy] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

