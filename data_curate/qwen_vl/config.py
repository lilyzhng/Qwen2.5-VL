"""Configuration class for the QwenVL judgment dataset."""

from pathlib import Path
from typing import Final, List

import yaml

from kits.scalex.dataset.config import BaseStageConfigV2
from kits.scalex.platforms.sensor_names import (
    CAMERA_BACK_ULTRAWIDE,
    CAMERA_FRONT_NARROW,
    CAMERA_FRONT_ULTRAWIDE,
    CAMERA_FRONT_WIDE,
    CAMERA_LEFT_ULTRAWIDE,
    CAMERA_RIGHT_ULTRAWIDE,
)

# Model configuration
QWENVL_MODEL_PATH: Final = "Qwen/Qwen3-VL-2B-Instruct"


def _load_judgements_from_yaml() -> List[str]:
    """Load judgment queries from prompts.yaml file.
    
    Returns:
        List of judgment query strings loaded from user_prompts.judgements.
        Returns a default list if file not found or key missing.
    """
    try:
        prompts_file = Path(__file__).parent / "prompts.yaml"
        if prompts_file.exists():
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f) or {}
                user_prompts = prompts.get("user_prompts", {})

                judgments = user_prompts.get("judgments")
                if judgments is None:
                    judgments = user_prompts.get("judgements")

                if judgments:
                    if isinstance(judgments, str):
                        return [judgments]
                    if isinstance(judgments, list):
                        return [str(item) for item in judgments if item]
    except Exception:
        pass
    
    # Default fallback
    return ["Is there a pedestrian crossing the street?"]


class QwenVLJudgeConfig(BaseStageConfigV2):
    """Config for the QwenVL judgment stage."""

    #: The number of CPUs to request for dataset processing.
    num_cluster_cpus: int = 48

    #: The output branch.
    branch: str = "main"

    #: A string reference to the log slices silver stage.
    log_slices_silver_reference: str = "sensing--log-slices--silver/main"

    #: The model path (HuggingFace ID or local path).
    model_path: str = QWENVL_MODEL_PATH

    #: When true, loads model from lakefs. Otherwise, loads from HuggingFace.
    load_model_from_lakefs: bool = False

    #: List of judgment queries to evaluate against each video segment.
    judgements: List[str] = None

    #: The number of references to process at the same time and save in one file.
    batch_size: int = 8

    #: The concurrency for generating judgments.
    concurrency: int = 8

    #: The number of CPUs to request per judge actor.
    num_cpus_per_judge_actor: float = 2.0

    #: The number of GPUs to request per judge actor.
    num_gpus_per_judge_actor: float = 0.5

    #: How many seconds consecutive video segments overlap.
    segment_overlapping_secs: float = 5.0

    #: Desired frame rate for sampling within each video segment
    segment_desired_fps: float = 1.0

    #: Maximum number of tokens to generate for each judgment
    max_new_tokens: int = 128

    #: If True, remove all existing files in the repo at the begining of the job.
    delete_bulk_files: bool = False

    #: Camera names to process for generating judgments.
    process_camera_names_csv: str = ",".join(
        [
            # Front-facing cameras
            CAMERA_FRONT_WIDE,
            CAMERA_FRONT_NARROW,
            # SVC ultra-wide cameras
            CAMERA_FRONT_ULTRAWIDE,
            CAMERA_BACK_ULTRAWIDE,
            CAMERA_LEFT_ULTRAWIDE,
            CAMERA_RIGHT_ULTRAWIDE,
        ]
    )

    def __post_init__(self):
        """Load judgment queries from prompts.yaml if not explicitly set."""
        super().__post_init__()
        if self.judgements is None:
            self.judgements = _load_judgements_from_yaml()
