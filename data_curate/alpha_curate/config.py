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
    """Configuration for the alpha-based active learning selection strategy."""

    # Repo
    repo: str = "sensing--features--cosmos-index"

    #: LanceDB branch to read from
    lance_db_branch: str = "main"

    #: Cosmos model size for text embeddings
    model_size: str = "Cosmos-Embed1-448p"

    #: Path to YAML file containing prompts
    prompt_yaml_path: Optional[str] = "prompts.yaml"

    #: Maximum number of slices to process in a single batch (memory optimization)
    batch_size: int = 1000

    #: Whether to deduplicate results by base slice ID after scoring
    #: When enabled, only the best scoring segment per base video is kept
    deduplicate: bool = True

    #: Minimum similarity of best result to consider prompt valid
    retrieval_multiplier: int = 3

    #: Prompt expansion and fusion settings
    #: Enable prompt expansion for better recall with CLIP models
    use_prompt_expansion: bool = True
    
    #: Number of variants to generate per prompt (default: 6)
    #: More variants = better recall but slower (linear cost)
    num_variants: int = 6

