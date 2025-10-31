#!/usr/bin/env python3
"""
Qwen3-VL Local Inference Script
Supports image and video inference using locally downloaded model weights with HuggingFace Transformers.
"""

import argparse
import os
from typing import List, Dict, Optional, Union

from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path: str, use_flash_attn: bool = False, device_map: str = "auto"):
    """
    Load Qwen3-VL model using HuggingFace Transformers.
    
    Args:
        model_path: Path to the local model weights or HuggingFace model ID
        use_flash_attn: Whether to use Flash Attention 2 for faster inference
        device_map: Device placement strategy
    
    Returns:
        model, processor
    """
    print(f"Loading model from: {model_path}")
    
    # Build model loading kwargs
    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": device_map,
        "trust_remote_code": True
    }
    
    # Add flash attention if requested
    if use_flash_attn:
        print("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Model loaded successfully on device: {model.device}")
    return model, processor


def prepare_messages(
    media_paths: Union[str, List[str]], 
    prompt: str,
    media_type: str = "image",
    system_prompt: Optional[str] = None,
    fps: Optional[float] = None,
    max_pixels: Optional[int] = None
) -> List[Dict]:
    """
    Prepare messages for inference with images or videos.
    
    Note: Videos are ultimately processed as frames, so the distinction between
    "image" and "video" mainly affects how the model samples/decodes the input.
    
    Args:
        media_paths: Single file path (str) or list of file paths
        prompt: Text prompt/question
        media_type: Type of media - "image" or "video"
        system_prompt: Optional system prompt
        fps: FPS for video frame sampling (video only)
        max_pixels: Maximum pixels for frames (video only)
    
    Returns:
        Formatted messages for the model
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


def run_inference(model, processor, messages: List[Dict], max_new_tokens: int = 1024) -> str:
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
    
    # Generate
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Local Inference Script (HuggingFace)")
    
    # Model configuration
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to local model weights or HuggingFace model ID"
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use Flash Attention 2 for faster inference"
    )
    
    # Input configuration
    parser.add_argument(
        "--image", "-i",
        type=str,
        action="append",
        help="Path to input image(s). Can be specified multiple times for multiple images."
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Path to input video"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt/question"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt (optional)"
    )
    
    # Generation configuration
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for video sampling (optional)"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Maximum pixels for video frames (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.image and not args.video:
        parser.error("At least one of --image or --video must be provided")
    if args.image and args.video:
        parser.error("Cannot process both image and video in the same request")
    
    # Load model
    model, processor = load_model(
        args.model_path,
        use_flash_attn=args.flash_attn
    )
    
    # Prepare messages
    if args.video:
        print(f"Processing video: {args.video}")
        messages = prepare_messages(
            media_paths=args.video,
            prompt=args.prompt,
            media_type="video",
            system_prompt=args.system_prompt,
            fps=args.fps,
            max_pixels=args.max_pixels
        )
    else:
        num_images = len(args.image)
        print(f"Processing {num_images} image{'s' if num_images > 1 else ''}: {', '.join(args.image)}")
        messages = prepare_messages(
            media_paths=args.image,
            prompt=args.prompt,
            media_type="image",
            system_prompt=args.system_prompt
        )
    
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating response...")
    print("-" * 80)
    
    # Run inference
    output = run_inference(model, processor, messages, args.max_new_tokens)
    
    print(output)
    print("-" * 80)


if __name__ == "__main__":
    main()

