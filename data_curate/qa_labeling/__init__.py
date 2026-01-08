"""VLM-powered Labeling QA System.

This module implements Experiment 1: Semantic Class Disambiguation
using VLM as a targeted arbiter for resolving semantic misclassifications
(pedestrian/cyclist/motorcycle confusion).
"""

from .config import (
    SEMANTIC_CLASSES,
    VRU_CLASSES,
    NUSCENES_TO_QA_CLASS,
    SEMANTIC_PROMPT,
    GENERATION_CONFIG,
)
from .vlm_judge import SemanticQAJudge, SelfConsistencyResult
from .data_prep import NuScenesDataLoader, SyntheticErrorInjector
from .evaluate import SemanticQAEvaluator

__all__ = [
    # Config
    "SEMANTIC_CLASSES",
    "VRU_CLASSES",
    "NUSCENES_TO_QA_CLASS",
    "SEMANTIC_PROMPT",
    "GENERATION_CONFIG",
    # VLM Judge
    "SemanticQAJudge",
    "SelfConsistencyResult",
    # Data
    "NuScenesDataLoader",
    "SyntheticErrorInjector",
    # Evaluation
    "SemanticQAEvaluator",
]

