#!/usr/bin/env python3
"""
Qwen3-VL Local Inference Script
Supports image and video inference using locally downloaded model weights with HuggingFace Transformers.
"""

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Final

import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from kits.scalex.artifacts.lock import ProcessSafeFileLock
from kits.scalex.hpc.tiered_file_system import tiered_filesystem
from platforms.lakefs.client import LakeFS

HF_HOME: Final = "/tmp/qwenvl_hf_cache"
HF_CACHE_DIR: Final = HF_HOME + os.sep + "modules"
LAKEFS_MODEL_CACHE_DIR: Final = "/tmp/qwenvl_models_cache"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_MODULES_CACHE"] = HF_CACHE_DIR
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(LAKEFS_MODEL_CACHE_DIR, exist_ok=True)

_LOGGER: Final = logging.getLogger(__name__)


def load_model(
    model_path: str, use_flash_attn: bool = False
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load QwenVL model and processor.
    
    Note: QwenVL requires GPU. Model will be loaded to CUDA device automatically.

    Args:
        model_path: Path to the model directory or HuggingFace model identifier.
        use_flash_attn: Whether to use Flash Attention 2.

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


def build_chat_messages(
    media_paths: Union[str, List[str]], 
    prompt: str,
    media_type: str = "image",
    system_prompt: Optional[str] = None,
    fps: Optional[float] = None,
    max_pixels: Optional[int] = None
) -> List[Dict]:
    """
    Build chat message structure for model inference with images or videos.
    
    Creates a properly formatted conversation structure with system and user messages,
    including media files and text prompts. This follows the chat API format expected
    by Qwen3-VL models.
    
    Note: Videos are ultimately processed as frames, so the distinction between
    "image" and "video" mainly affects how the model samples/decodes the input.
    
    Args:
        media_paths: Single file path (str) or list of file paths
        prompt: Text prompt/question for the user message
        media_type: Type of media - "image" or "video"
        system_prompt: Optional system prompt (e.g., for task-specific behavior)
        fps: FPS for video frame sampling (video only)
        max_pixels: Maximum pixels for frames (video only)
    
    Returns:
        List of formatted message dictionaries for the model's chat template
    
    Example:
        >>> messages = build_chat_messages(
        ...     media_paths="image.jpg",
        ...     prompt="What objects are in this image?",
        ...     system_prompt="You are a helpful assistant"
        ... )
    """
    messages = []
    
    # Add system prompt if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    
    # Handle both single file and multiple files
    if isinstance(media_paths, str):
        media_paths = [media_paths]
    
    # Build content with media files
    content = []
    for media_path in media_paths:
        media_content = {
            "type": media_type,
            media_type: os.path.abspath(media_path)
        }
        
        # Add optional parameters (typically for video)
        if fps is not None:
            media_content["fps"] = fps
        if max_pixels is not None:
            media_content["max_pixels"] = max_pixels
        
        content.append(media_content)
    
    # Add text prompt
    content.append({"type": "text", "text": prompt})
    
    messages.append({
        "role": "user",
        "content": content
    })
    
    return messages


def inference(model, processor, messages: List[Dict], max_new_tokens: int = 1024) -> str:
    """
    Run inference using HuggingFace Transformers.
    
    Args:
        model: Loaded model
        processor: Loaded processor
        messages: Prepared messages
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated text
    """
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


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
    safe_name = model_path.replace("/", "_").replace(":", "_")
    lakefs_path = f"models/{safe_name}"
    
    lakefs = LakeFS()
    files = list(lakefs.list_objects("sensing-models", "main", lakefs_path))
    if not files:
        raise FileNotFoundError(
            f"Model not found in lakefs://sensing-models/main/{lakefs_path}"
        )
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        tiered_fs = tiered_filesystem()
        for file_info in files:
            filename = file_info.path.rpartition("/")[-1]
            target_path = temp_dir / filename
            tiered_fs.get_file(file_info.physical_address, str(target_path))
            _LOGGER.info("Downloaded %s", filename)
        
        # Atomic move to final location
        temp_dir.rename(cache_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def load_qwenvl_model_and_processor(
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
        return load_model(model_path, use_flash_attn)
    
    # Load from LakeFS with caching
    cache_dir, temp_dir, completion_marker = _get_cache_paths(model_path)
    
    if completion_marker.exists():
        _LOGGER.info("Loading cached model from %s", cache_dir)
        return load_model(str(cache_dir), use_flash_attn)
    
    safe_name = model_path.replace("/", "_").replace(":", "_")
    lock = ProcessSafeFileLock(f"/tmp/qwenvl_download_{safe_name}.lock")
    
    with lock:
        if completion_marker.exists():
            _LOGGER.info("Model was cached by another worker")
            return load_model(str(cache_dir), use_flash_attn)
        
        _LOGGER.info("Downloading model from LakeFS to %s", cache_dir)
        _download_from_lakefs(model_path, cache_dir, temp_dir)
        completion_marker.touch()
        _LOGGER.info("Successfully cached model: %s", model_path)
    
    return load_model(str(cache_dir), use_flash_attn)


@dataclass
class QwenVLJudge:
    """Class to run video judgment inference with Qwen-VL.

    This class loads a Qwen-VL model and provides methods to judge whether
    video frames match specific queries (e.g., "Is there a pedestrian crossing?").
    """

    #: The model path
    model_path: str

    #: When true, loads model from lakefs. Otherwise, loads from HuggingFace.
    load_model_from_lakefs: bool = False

    #: Whether to use Flash Attention 2 for faster inference.
    use_flash_attn: bool = False

    #: Maximum number of tokens to generate.
    max_new_tokens: int = 128

    #: The model instance.
    model: AutoModelForImageTextToText = field(init=False)

    #: The processor instance.
    processor: AutoProcessor = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the QwenVL model and processor."""
        sys.path.append(os.getenv("HF_MODULES_CACHE") or "")

        # Load the model weights and instantiate the models.
        self.model, self.processor = load_qwenvl_model_and_processor(
            self.model_path, self.load_model_from_lakefs, self.use_flash_attn
        )

    def judge_video(self, frames: List[npt.NDArray[np.uint8]], query: str) -> bool:
        """Judge if video frames match the query. Returns Yes/No as boolean.

        Args:
            frames: List of video frames as HWC uint8 numpy arrays.
            query: The judgment query (e.g., "Is there a pedestrian crossing?").

        Returns:
            Boolean indicating whether the video matches the query (True = Yes, False = No).
        """
        content = []

        for frame in frames:
            content.append({"type": "image", "image": frame})

        # Format query to request structured JSON output
        structured_prompt = (
            f"{query}\n\n"
            f"Respond ONLY with valid JSON in this exact format:\n"
            f'{{"judgment": true}}  or  {{"judgment": false}}'
        )
        content.append({"type": "text", "text": structured_prompt})

        messages = [{"role": "user", "content": content}]

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
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        response = output_text[0].strip()

        # Parse JSON response
        try:
            parsed = json.loads(response)
            return parsed["judgment"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            _LOGGER.error(f"Failed to parse judgment from response: {response}. Error: {e}")
            raise ValueError(f"Invalid judgment response format: {response}") from e

    def __call__(self, frames: List[npt.NDArray[np.uint8]], query: str) -> bool:
        """Convenience method to call judge_video."""
        return self.judge_video(frames, query)

