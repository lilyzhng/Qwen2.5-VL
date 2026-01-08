"""Standalone VLM Judge for semantic class disambiguation.

This module implements self-consistency voting for VLM-based labeling QA.
No internal framework dependencies (no kits.scalex, no LakeFS).
"""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Final, Any

import numpy as np
import torch

from .config import (
    SEMANTIC_PROMPT,
    SEMANTIC_CLASSES,
    GENERATION_CONFIG,
    VISUAL_ANCHOR_PREFIX,
    TWO_VIEW_PREFIX,
    GHOST_BOX_PROMPT,
    get_decision,
)

_LOGGER: Final = logging.getLogger(__name__)

# Default model - Qwen3-VL-8B for best quality
DEFAULT_MODEL_ID: Final[str] = "Qwen/Qwen3-VL-8B-Instruct"


def load_model(
    model_path: str = DEFAULT_MODEL_ID,
    use_flash_attn: bool = False,
) -> tuple:
    """Load Qwen-VL model and processor from HuggingFace.
    
    Args:
        model_path: HuggingFace model ID or local path
        use_flash_attn: Whether to use Flash Attention 2
        
    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    _LOGGER.info("Loading VLM model: %s", model_path)
    
    # Determine device - prefer MPS on Mac, CUDA on Linux/Windows
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        _LOGGER.info("Using Apple MPS (Metal) backend")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        _LOGGER.info("Using CUDA backend")
    else:
        device = "cpu"
        dtype = torch.float32
        _LOGGER.info("Using CPU backend (slow)")
    
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    
    # device_map="auto" doesn't work well with MPS, load to CPU then move
    if device == "mps":
        model_kwargs["device_map"] = None  # Load to CPU first
    else:
        model_kwargs["device_map"] = "auto"
    
    if use_flash_attn and device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    
    # Move to MPS if needed
    if device == "mps":
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    _LOGGER.info("Model loaded successfully on %s", device)
    return model, processor


def resize_image_for_inference(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Resize image to fit within max_size while maintaining aspect ratio.
    
    This helps avoid MPS memory issues on Mac.
    
    Args:
        image: Input image (H, W, C)
        max_size: Maximum dimension size
        
    Returns:
        Resized image
    """
    from PIL import Image as PILImage
    
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    # Calculate new size
    if h > w:
        new_h = max_size
        new_w = int(w * max_size / h)
    else:
        new_w = max_size
        new_h = int(h * max_size / w)
    
    pil_img = PILImage.fromarray(image)
    pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
    return np.array(pil_img)


def build_semantic_qa_messages(
    image: np.ndarray,
    prompt: str = SEMANTIC_PROMPT,
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """Build chat messages for semantic QA task.
    
    Args:
        image: Input image as numpy array (H, W, C)
        prompt: The prompt template
        system_prompt: Optional system prompt
        
    Returns:
        List of message dicts for the model
    """
    # Resize image to avoid MPS memory issues
    image = resize_image_for_inference(image, max_size=512)
    
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })
    
    # Build user message with image and prompt
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]
    
    messages.append({
        "role": "user",
        "content": user_content,
    })
    
    return messages


def build_two_view_messages(
    target_image: np.ndarray,
    context_image: np.ndarray,
    prompt: str = SEMANTIC_PROMPT,
) -> List[Dict]:
    """Build chat messages for two-view classification.
    
    Provides both a tight TARGET view and a wider CONTEXT view.
    Prepends TWO_VIEW_PREFIX to the base prompt.
    
    Args:
        target_image: Tight crop of target object (H, W, C)
        context_image: Wider context view (H, W, C)
        prompt: The base classification prompt
        
    Returns:
        List of message dicts for the model
    """
    # Resize both images
    target_image = resize_image_for_inference(target_image, max_size=512)
    context_image = resize_image_for_inference(context_image, max_size=512)
    
    # Prepend two-view prefix to prompt
    full_prompt = TWO_VIEW_PREFIX + prompt
    
    # Build user message with both images and prompt
    user_content = [
        {"type": "image", "image": target_image},
        {"type": "image", "image": context_image},
        {"type": "text", "text": full_prompt},
    ]
    
    messages = [{
        "role": "user",
        "content": user_content,
    }]
    
    return messages


def parse_vlm_response(response: str) -> Dict[str, Any]:
    """Parse VLM JSON response, handling common formatting issues.
    
    Args:
        response: Raw text response from VLM
        
    Returns:
        Parsed dict with 'class' and 'evidence' fields
    """
    # Try to extract JSON from response
    response = response.strip()
    
    # Handle markdown code blocks
    if "```json" in response:
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)
    elif "```" in response:
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)
    
    try:
        parsed = json.loads(response)
        return {
            "class": parsed.get("class", "REVIEW"),
            "evidence": parsed.get("evidence", []),
            "raw_response": response,
        }
    except json.JSONDecodeError:
        # Try to extract class from text
        for cls in SEMANTIC_CLASSES:
            if cls in response.upper():
                return {
                    "class": cls,
                    "evidence": [],
                    "raw_response": response,
                }
        
        return {
            "class": "REVIEW",
            "evidence": [],
            "raw_response": response,
            "parse_error": True,
        }


@dataclass
class SelfConsistencyResult:
    """Result from self-consistency voting."""
    
    # Individual sample predictions
    samples: List[str]
    
    # Majority vote result
    predicted_class: str
    
    # How many samples agreed (1, 2, or 3)
    agreement: int
    
    # Decision based on agreement level
    decision: str  # ACCEPT or REVIEW
    
    # Evidence from the majority prediction
    evidence: List[str]
    
    # All raw responses for debugging
    raw_responses: List[str] = field(default_factory=list)


@dataclass
class GhostBoxResult:
    """Result from ghost box detection."""
    
    # Individual sample predictions
    samples: List[str]  # List of "YES", "NO", "UNCERTAIN"
    
    # Majority vote result
    exists: str  # "YES", "NO", or "UNCERTAIN"
    
    # Agreement count
    agreement: int
    
    # Decision based on agreement
    decision: str  # "ACCEPT" (confident) or "REVIEW" (uncertain)
    
    # Object type if exists=YES
    object_type: Optional[str] = None
    
    # Evidence from the VLM responses
    evidence: List[str] = field(default_factory=list)
    
    # Raw responses for debugging
    raw_responses: List[str] = field(default_factory=list)
    
    # All raw responses for debugging
    raw_responses: List[str] = field(default_factory=list)


@dataclass
class SemanticQAJudge:
    """VLM Judge for semantic class disambiguation using self-consistency.
    
    This class loads a vision-language model and uses self-consistency voting
    to determine the semantic class of objects in ROI images.
    """
    
    # Model configuration
    model_path: str = DEFAULT_MODEL_ID
    use_flash_attn: bool = False
    
    # Generation config
    num_samples: int = 3
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.95
    max_new_tokens: int = 256
    
    # Model instances
    model: Any = field(init=False, repr=False, default=None)
    processor: Any = field(init=False, repr=False, default=None)
    
    def __post_init__(self) -> None:
        """Load model on initialization."""
        self.model, self.processor = load_model(
            self.model_path, self.use_flash_attn
        )
        
        # Override with config values
        config = GENERATION_CONFIG
        self.num_samples = config.get("num_samples", self.num_samples)
        self.temperature = config.get("temperature", self.temperature)
        self.do_sample = config.get("do_sample", self.do_sample)
        self.top_p = config.get("top_p", self.top_p)
        self.max_new_tokens = config.get("max_new_tokens", self.max_new_tokens)
    
    def _run_single_inference(
        self,
        image: np.ndarray,
        prompt: str = SEMANTIC_PROMPT,
    ) -> Dict[str, Any]:
        """Run single inference pass.
        
        Args:
            image: Input image (H, W, C)
            prompt: The prompt to use
            
        Returns:
            Parsed response dict
        """
        messages = build_semantic_qa_messages(image, prompt)
        
        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        
        response = output_text[0].strip()
        return parse_vlm_response(response)
    
    def judge_image(
        self,
        image: np.ndarray,
        current_label: str = "UNKNOWN",
        prompt: str = SEMANTIC_PROMPT,
        use_visual_anchor: bool = False,
    ) -> SelfConsistencyResult:
        """Judge an image using self-consistency voting.
        
        Runs N inference passes with stochastic sampling and uses
        majority vote to determine the final prediction.
        
        Args:
            image: Input image (H, W, C)
            current_label: The current label (used for comparison in code, not shown to VLM)
            prompt: The prompt template for forced-choice classification
            use_visual_anchor: If True, prepend visual anchor prefix to prompt
            
        Returns:
            SelfConsistencyResult with prediction and agreement info
        """
        # Use prompt directly - forced choice classification doesn't show current_label
        formatted_prompt = prompt
        
        # Prepend visual anchor prefix if needed
        if use_visual_anchor:
            formatted_prompt = VISUAL_ANCHOR_PREFIX + formatted_prompt
        
        samples = []
        evidence_lists = []
        raw_responses = []
        
        for i in range(self.num_samples):
            result = self._run_single_inference(image, formatted_prompt)
            samples.append(result["class"])
            evidence_lists.append(result.get("evidence", []))
            raw_responses.append(result.get("raw_response", ""))
            
            _LOGGER.debug("Sample %d: %s", i + 1, result["class"])
        
        # Majority vote
        counter = Counter(samples)
        predicted_class, agreement = counter.most_common(1)[0]
        
        # Get evidence from majority prediction
        evidence = []
        for i, cls in enumerate(samples):
            if cls == predicted_class and evidence_lists[i]:
                evidence = evidence_lists[i]
                break
        
        # Determine decision based on agreement
        decision = get_decision(agreement, predicted_class)
        
        result = SelfConsistencyResult(
            samples=samples,
            predicted_class=predicted_class,
            agreement=agreement,
            decision=decision,
            evidence=evidence,
            raw_responses=raw_responses,
        )
        
        _LOGGER.info(
            "Self-consistency result: %s (agreement: %d/3, decision: %s)",
            predicted_class, agreement, decision,
        )
        
        return result
    
    def judge_two_view(
        self,
        target_image: np.ndarray,
        context_image: np.ndarray,
        current_label: str = "UNKNOWN",
        prompt: str = SEMANTIC_PROMPT,
    ) -> SelfConsistencyResult:
        """Judge using two views: tight TARGET and wider CONTEXT.
        
        Uses self-consistency voting with both images provided.
        
        Args:
            target_image: Tight crop of target object (H, W, C)
            context_image: Wider context view (H, W, C)
            current_label: The current label (for code comparison, not shown to VLM)
            prompt: The base prompt template
            
        Returns:
            SelfConsistencyResult with prediction and agreement info
        """
        samples = []
        evidence_lists = []
        raw_responses = []
        
        for i in range(self.num_samples):
            # Build two-view messages
            messages = build_two_view_messages(target_image, context_image, prompt)
            
            # Run inference
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            
            response = output_text[0].strip()
            result = parse_vlm_response(response)
            
            samples.append(result["class"])
            evidence_lists.append(result.get("evidence", []))
            raw_responses.append(result.get("raw_response", ""))
            
            _LOGGER.debug("Two-view sample %d: %s", i + 1, result["class"])
        
        # Majority vote
        counter = Counter(samples)
        predicted_class, agreement = counter.most_common(1)[0]
        
        # Get evidence from majority prediction
        evidence = []
        for i, cls in enumerate(samples):
            if cls == predicted_class and evidence_lists[i]:
                evidence = evidence_lists[i]
                break
        
        # Determine decision based on agreement
        decision = get_decision(agreement, predicted_class)
        
        result = SelfConsistencyResult(
            samples=samples,
            predicted_class=predicted_class,
            agreement=agreement,
            decision=decision,
            evidence=evidence,
            raw_responses=raw_responses,
        )
        
        _LOGGER.info(
            "Two-view result: %s (agreement: %d/3, decision: %s)",
            predicted_class, agreement, decision,
        )
        
        return result
    
    def judge_batch(
        self,
        images: List[np.ndarray],
        current_labels: List[str],
        prompt: str = SEMANTIC_PROMPT,
    ) -> List[SelfConsistencyResult]:
        """Judge multiple images.
        
        Args:
            images: List of input images
            current_labels: List of current labels for each image
            prompt: The prompt template to use
            
        Returns:
            List of SelfConsistencyResult
        """
        results = []
        for i, (image, label) in enumerate(zip(images, current_labels)):
            _LOGGER.info("Processing image %d/%d (label: %s)", i + 1, len(images), label)
            result = self.judge_image(image, current_label=label, prompt=prompt)
            results.append(result)
        
        return results
    
    def judge_ghost_box(
        self,
        image: np.ndarray,
    ) -> GhostBoxResult:
        """Check if a region contains a real object (ghost box detection).
        
        Uses self-consistency voting to determine if the highlighted region
        contains a real physical traffic participant or is empty/background.
        
        Args:
            image: Input image (H, W, C) - should be a crop of the region to check
            
        Returns:
            GhostBoxResult with exists decision
        """
        samples = []
        object_types = []
        evidence_list = []
        raw_responses = []
        
        _LOGGER.info("Running ghost box detection (num_samples=%d)", self.num_samples)
        
        # Build messages
        messages = build_semantic_qa_messages(image, GHOST_BOX_PROMPT)
        
        for i in range(self.num_samples):
            # Run inference
            with torch.no_grad():
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            
            response = output_text[0].strip()
            parsed = self._parse_ghost_box_response(response)
            
            samples.append(parsed["exists"])
            object_types.append(parsed.get("type"))
            evidence_list.append(parsed.get("evidence", []))
            raw_responses.append(response)
            
            _LOGGER.debug("Ghost box sample %d: %s", i + 1, parsed["exists"])
        
        # Majority vote
        counter = Counter(samples)
        exists, agreement = counter.most_common(1)[0]
        
        # Get object type if exists=YES (take from majority)
        object_type = None
        if exists == "YES":
            type_counter = Counter([t for t in object_types if t])
            if type_counter:
                object_type = type_counter.most_common(1)[0][0]
        
        # Get evidence from the majority prediction
        evidence = []
        for i, sample in enumerate(samples):
            if sample == exists and evidence_list[i]:
                evidence = evidence_list[i]
                break
        
        # Decision: ACCEPT if 3/3 or 2/3, REVIEW if no consensus
        if agreement >= 2:
            decision = "ACCEPT"
        else:
            decision = "REVIEW"
        
        result = GhostBoxResult(
            samples=samples,
            exists=exists,
            agreement=agreement,
            decision=decision,
            object_type=object_type,
            evidence=evidence,
            raw_responses=raw_responses,
        )
        
        _LOGGER.info(
            "Ghost box result: %s (agreement: %d/3, decision: %s, type: %s)",
            exists, agreement, decision, object_type,
        )
        
        return result
    
    def _parse_ghost_box_response(self, response: str) -> Dict[str, Any]:
        """Parse ghost box VLM response.
        
        Args:
            response: Raw text response from VLM
            
        Returns:
            Dict with 'exists', 'type', 'evidence' fields
        """
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
        
        try:
            parsed = json.loads(response)
            return {
                "exists": parsed.get("exists", "UNCERTAIN"),
                "type": parsed.get("type"),
                "evidence": parsed.get("evidence", []),
            }
        except json.JSONDecodeError:
            # Try to extract exists from text
            response_upper = response.upper()
            if "EXISTS" in response_upper and "YES" in response_upper:
                return {"exists": "YES", "type": None, "evidence": [response]}
            elif "EXISTS" in response_upper and "NO" in response_upper:
                return {"exists": "NO", "type": None, "evidence": [response]}
            else:
                return {"exists": "UNCERTAIN", "type": None, "evidence": [response]}

