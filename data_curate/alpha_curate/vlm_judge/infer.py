"""VLM Judge inference for ALFA Curate candidate verification."""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Final

import numpy as np
import numpy.typing as npt
import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

from kits.scalex.artifacts.lock import ProcessSafeFileLock
from kits.scalex.hpc.tiered_file_system import tiered_filesystem
from platforms.lakefs.client import LakeFS

# Cache directories for model storage
HF_HOME: Final = "/tmp/qwenvl_hf_cache"
HF_CACHE_DIR: Final = HF_HOME + os.sep + "modules"
LAKEFS_MODEL_CACHE_DIR: Final = "/tmp/qwenvl_models_cache"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_MODULES_CACHE"] = HF_CACHE_DIR
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(LAKEFS_MODEL_CACHE_DIR, exist_ok=True)

_LOGGER: Final = logging.getLogger(__name__)


def _load_model(
    model_path: str, use_flash_attn: bool = False
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load QwenVL model and processor.

    Args:
        model_path: Path to the model directory or HuggingFace model identifier.
        use_flash_attn: Whether to use Flash Attention.

    Returns:
        Tuple of (model, processor).
    """
    model_kwargs = {
        "torch_dtype": torch.float16, 
        "device_map": "auto", 
        "trust_remote_code": True,
    }

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def _get_cache_paths(model_path: str) -> tuple[Path, Path, Path]:
    """Get cache directory paths for a model.
    
    Returns:
        (cache_dir, temp_dir, completion_marker)
    """
    safe_name = model_path.replace("/", "_").replace(":", "_")
    cache_dir = Path(LAKEFS_MODEL_CACHE_DIR) / safe_name
    temp_dir = cache_dir.with_suffix(".downloading")
    completion_marker = cache_dir / ".download_complete"
    return cache_dir, temp_dir, completion_marker


def _download_from_lakefs(model_path: str, cache_dir: Path, temp_dir: Path) -> None:
    """Download model from LakeFS to cache directory."""
    lakefs_path = f"models/{model_path.replace('/', '_').replace(':', '_')}"
    
    # Get file list from LakeFS
    lakefs = LakeFS()
    files = list(lakefs.list_objects("sensing-models", "main", lakefs_path))
    if not files:
        raise FileNotFoundError(f"Model not found in lakefs://sensing-models/main/{lakefs_path}")
    
    # Clean up and create temp directory
    for dir_path in [temp_dir, cache_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    temp_dir.mkdir(parents=True)
    
    # Download all files
    try:
        tiered_fs = tiered_filesystem()
        for file_info in files:
            filename = file_info.path.split("/")[-1]
            tiered_fs.get_file(file_info.physical_address, str(temp_dir / filename))
            _LOGGER.info("Downloaded %s", filename)
        temp_dir.rename(cache_dir)  # Atomic move
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def load_qwenvl_model(
    model_path: str, load_model_from_lakefs: bool, use_flash_attn: bool = False
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load QwenVL model from HuggingFace or LakeFS.

    Args:
        model_path: HuggingFace model ID or LakeFS path.
        load_model_from_lakefs: If True, load from LakeFS; otherwise from HuggingFace.
        use_flash_attn: Whether to use Flash Attention 2.

    Returns:
        Tuple of (model, processor).
    """
    # Load from HuggingFace
    if not load_model_from_lakefs:
        _LOGGER.info("Loading model from HuggingFace: %s", model_path)
        return _load_model(model_path, use_flash_attn)
    
    # Load from LakeFS with caching
    cache_dir, temp_dir, completion_marker = _get_cache_paths(model_path)
    
    if completion_marker.exists():
        _LOGGER.info("Loading cached model from %s", cache_dir)
        return _load_model(str(cache_dir), use_flash_attn)
    
    safe_name = model_path.replace("/", "_").replace(":", "_")
    lock = ProcessSafeFileLock(f"/tmp/qwenvl_download_{safe_name}.lock")
    
    with lock:
        if completion_marker.exists():
            _LOGGER.info("Model was cached by another worker")
            return _load_model(str(cache_dir), use_flash_attn)
        
        _LOGGER.info("Downloading model from LakeFS to %s", cache_dir)
        _download_from_lakefs(model_path, cache_dir, temp_dir)
        completion_marker.touch()
        _LOGGER.info("Successfully cached model: %s", model_path)
    
    return _load_model(str(cache_dir), use_flash_attn)


def _load_prompts_from_yaml() -> Dict:
    """Load prompts from prompts.yaml file.

    Returns:
        Dictionary containing prompts configuration.

    Raises:
        RuntimeError: If prompts.yaml cannot be loaded.
    """
    prompts_file = Path(__file__).parent / "prompts.yaml"
    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f) or {}
            return prompts
    except Exception as e:
        _LOGGER.error("Failed to load prompts.yaml: %s", e)
        raise RuntimeError(f"Failed to load prompts.yaml: {e}")


@dataclass
class VLMJudgeResult:
    """Result from VLM judge evaluation."""

    #: The query that was evaluated.
    query: str

    #: Whether the video matches the query (True = match, False = no match).
    match: bool

    #: Confidence score from the VLM (0.0-1.0). Higher means more confident.
    confidence: float

    #: What the VLM observed in the video frames.
    observation: str

    #: Reasoning for why the VLM made this judgment.
    reason: str

    #: Raw response text from the VLM.
    raw_response: str


@dataclass
class VLMJudge:
    """VLM Judge for verifying ALFA Curate candidates.
    
    This class loads a vision-language model (Qwen-VL 3.0) and provides methods
    to judge whether video frames match specific queries. It's used to filter
    out false positives from embedding-based similarity search.
    """

    #: The model path (HuggingFace ID or LakeFS path).
    model_path: str

    #: When true, loads model from lakefs. Otherwise, loads from HuggingFace.
    load_model_from_lakefs: bool = False

    #: Whether to use Flash Attention 2 for faster inference.
    use_flash_attn: bool = False

    #: Maximum number of tokens to generate.
    max_new_tokens: int = 128

    #: The model instance.
    model: AutoModelForImageTextToText = field(init=False, repr=False)

    #: The processor instance.
    processor: AutoProcessor = field(init=False, repr=False)

    #: Prompts configuration loaded from YAML.
    prompts_config: Dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the VLM model and processor."""
        _LOGGER.info("Loading VLM model: %s", self.model_path)
        self.model, self.processor = load_qwenvl_model(
            self.model_path, self.load_model_from_lakefs, self.use_flash_attn
        )
        _LOGGER.info("VLM model loaded successfully")
        
        self.prompts_config = _load_prompts_from_yaml()
        _LOGGER.debug("Loaded prompts configuration from YAML")

    def judge_frames(
        self, 
        frames: List[npt.NDArray[np.uint8]], 
        query: str,
    ) -> VLMJudgeResult:
        """Judge if video frames match the query with detailed reasoning.

        Args:
            frames: List of video frames.
            query: The judgment query.

        Returns:
            VLMJudgeResult containing match, confidence, observation, reason, and raw response.
        """
        content = []

        # Add all frames as images
        for frame in frames:
            content.append({"type": "image", "image": frame})

        # Build prompt from template (output format is included in user_prompts.judgment)
        prompt_template = self.prompts_config.get("user_prompts", {}).get(
            "judgment",
            "{query}\n\nRespond with JSON containing query, match, confidence, observation, and reason."
        )
        
        judgment_prompt = prompt_template.format(query=query)

        content.append({"type": "text", "text": judgment_prompt})

        # Build messages with optional system prompt
        messages = []
        system_role = self.prompts_config.get("system_prompts", {}).get("role")
        if system_role:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_role}]
            })
        
        messages.append({"role": "user", "content": content})

        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        response = output_text[0].strip()

        # Parse JSON response
        try:
            parsed = json.loads(response)
            match = parsed.get("match", False)
            confidence = parsed.get("confidence", 1.0 if match else 0.0)
            observation = parsed.get("observation", "")
            reason = parsed.get("reason", "")
            query_returned = parsed.get("query", query)
            
            return VLMJudgeResult(
                query=query_returned,
                match=match,
                confidence=float(confidence),
                observation=observation,
                reason=reason,
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _LOGGER.error("Failed to parse judgment from response: %s. Error: %s", response, e)
            # Return conservative result on parse failure
            return VLMJudgeResult(
                query=query,
                match=False,
                confidence=0.0,
                observation="Failed to parse VLM response",
                reason=f"Error parsing response: {str(e)}",
                raw_response=response,
            )

    def __call__(
        self, 
        frames: List[npt.NDArray[np.uint8]], 
        query: str,
        return_confidence: bool = True,
    ) -> VLMJudgeResult:
        """Convenience method to call judge_frames."""
        return self.judge_frames(frames, query, return_confidence)

