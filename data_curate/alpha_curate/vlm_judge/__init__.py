"""VLM Judge module for filtering ALFA Curate results."""

from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.config import VLMJudgeConfig
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer import VLMJudge

__all__ = ["VLMJudgeConfig", "VLMJudge"]

