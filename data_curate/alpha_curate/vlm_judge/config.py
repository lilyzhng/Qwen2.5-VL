"""Configuration for VLM Judge in ALFA Curate."""

from dataclasses import dataclass
from typing import Final

# Model configuration
DEFAULT_VLM_MODEL_PATH: Final = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass
class VLMJudgeConfig:
    """Configuration for VLM judge filtering.
    
    This config is used within ALFA Curate to control how the VLM judge
    filters and re-ranks search results.
    """

    #: Enable VLM judge filtering. If False, skip VLM filtering entirely.
    enable_vlm_judge: bool = True

    #: Maximum number of candidates to send to VLM judge (top K from embedding search).
    max_candidates_for_vlm: int = 100

    #: The VLM model path (HuggingFace ID or local path).
    vlm_model_path: str = DEFAULT_VLM_MODEL_PATH

    #: When true, loads model from lakefs. Otherwise, loads from HuggingFace.
    load_model_from_lakefs: bool = False

    #: Whether to use Flash Attention 2 for faster inference.
    use_flash_attn: bool = False

    #: Maximum number of tokens to generate for each judgment (increased for detailed reasoning).
    max_new_tokens: int = 256

    #: Desired frame rate for sampling video segments (frames per second).
    segment_desired_fps: float = 1.0

    #: Maximum number of frames to send to VLM per video segment.
    max_frames_per_segment: int = 8

    #: Confidence threshold for VLM judge (0.0-1.0). Videos with confidence below this are filtered out.
    vlm_confidence_threshold: float = 0.7

    #: Number of GPUs to allocate per VLM judge worker.
    num_gpus_per_worker: float = 0.5

    #: GPU type to request (e.g., "A100", "H100"). None means any GPU type.
    gpu_type: str = "A100"

    #: Batch size for VLM inference.
    batch_size: int = 4

