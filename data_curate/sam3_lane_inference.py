#!/usr/bin/env python3
"""
SAM3 Lane Detection Inference on nuScenes Mini Dataset

This script runs SAM3 text-prompted segmentation on 5 selected nuScenes scenes
to detect lane markings and road boundaries, with comprehensive visualization.

Supports two inference modes:
- frame: Per-frame inference (no temporal tracking, faster for testing)
- video: Session-based video inference with temporal tracking/propagation

Prerequisites:
1. Accept the SAM3 license at: https://huggingface.co/facebook/sam3
2. Login to HuggingFace: huggingface-cli login
3. Install dependencies:
   pip install git+https://github.com/huggingface/transformers torchvision
   # For video mode, install sam3 from source:
   # pip install -e /path/to/sam3_repo

Target Scenes:
- scene-0061: Easy daylight sanity check
- scene-0553: Urban clutter/occlusions stress
- scene-0103: Hard visibility/lighting stress  
- scene-0916: Topology complexity stress
- scene-1094: Adverse condition (night + rain)
"""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor


@dataclass
class SceneConfig:
    """Configuration for a nuScenes scene to process."""
    name: str
    token: str
    description: str
    test_purpose: str
    pass_criteria: str


# Target scenes based on the plan
TARGET_SCENES = [
    SceneConfig(
        name="scene-0061",
        token="cc8c0bf57f984915a77078b10eb33198",
        description="Parked truck, construction, intersection, turn left, following a van",
        test_purpose="Easy daylight driving sanity check",
        pass_criteria="Both lane boundaries visible in near/mid field; masks not fragmented",
    ),
    SceneConfig(
        name="scene-0553",
        token="6f83169d067343658251f72e1dd17dbc",
        description="Wait at intersection, bicycle, large truck, peds crossing crosswalk",
        test_purpose="Urban clutter/occlusions stress",
        pass_criteria="Doesn't jump to car edges; doesn't label crosswalk stripes as lanes",
    ),
    SceneConfig(
        name="scene-0103",
        token="fcbccedd61424f1b85dcbf8f897f9754",
        description="Many peds right, wait for turning car, long bike rack left, cyclist",
        test_purpose="Hard visibility/lighting stress",
        pass_criteria="Dominant lane structure retained in near-field despite lighting",
    ),
    SceneConfig(
        name="scene-0916",
        token="325cef682f064c55a255f2625c533b75",
        description="Parking lot, bicycle rack, parked bicycles, bus, many peds",
        test_purpose="Topology complexity stress",
        pass_criteria="Split lines not incorrectly merged; main lane stable",
    ),
    SceneConfig(
        name="scene-1094",
        token="de7d80a1f5fb4c3e82ce8a4f213b450a",
        description="Night, after rain, many peds, PMD, jaywalker, truck, scooter",
        test_purpose="Adverse condition (rain + night)",
        pass_criteria="Performance under wet reflections and night glare",
    ),
]

# Lane-related text prompts to use
LANE_PROMPTS = [
    "lane markings",
    "road boundary",
    "lane lines",
    "road edges",
]


class NuScenesLoader:
    """Simple loader for nuScenes mini dataset."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.meta_root = self.data_root / "v1.0-mini"
        self.samples_root = self.data_root / "samples"
        
        # Load metadata
        with open(self.meta_root / "scene.json") as f:
            self.scenes = {s["token"]: s for s in json.load(f)}
        
        with open(self.meta_root / "sample.json") as f:
            self.samples = {s["token"]: s for s in json.load(f)}
        
        with open(self.meta_root / "sample_data.json") as f:
            self.sample_data = {s["token"]: s for s in json.load(f)}
    
    def _get_channel_from_filename(self, filename: str) -> str:
        """Extract channel name from filename path.
        
        Examples:
            samples/CAM_FRONT/xxx.jpg -> CAM_FRONT
            sweeps/RADAR_FRONT/xxx.pcd -> RADAR_FRONT
        """
        parts = filename.split("/")
        if len(parts) >= 2:
            return parts[1]  # e.g., CAM_FRONT
        return ""
    
    def get_scene_samples(self, scene_token: str, camera: str = "CAM_FRONT") -> list[dict]:
        """Get all camera samples for a scene."""
        scene = self.scenes[scene_token]
        samples = []
        
        # Walk through the sample chain
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = self.samples[sample_token]
            
            # Find the camera sample_data
            for sd_token in self._get_sample_data_tokens(sample_token):
                sd = self.sample_data[sd_token]
                channel = self._get_channel_from_filename(sd["filename"])
                if channel == camera:
                    image_path = self.data_root / sd["filename"]
                    if image_path.exists():
                        samples.append({
                            "token": sd_token,
                            "timestamp": sd["timestamp"],
                            "image_path": str(image_path),
                            "filename": sd["filename"],
                        })
                    break
            
            sample_token = sample.get("next", "")
            if not sample_token:
                break
        
        return samples
    
    def _get_sample_data_tokens(self, sample_token: str) -> list[str]:
        """Get all sample_data tokens for a sample."""
        return [
            sd["token"] for sd in self.sample_data.values()
            if sd.get("sample_token") == sample_token
        ]


class Sam3LaneInference:
    """SAM3-based lane detection inference (per-frame mode, no temporal tracking).
    
    Uses HuggingFace Transformers implementation which works on CPU, MPS, and CUDA.
    On CUDA, can optionally use the native SAM3 repo implementation for better performance
    (requires sam3 repo installed with triton).
    """
    
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        use_native_sam3: bool = False,
    ):
        """Initialize SAM3 lane inference.
        
        Args:
            model_id: HuggingFace model ID for SAM3
            device: Device to use (cuda, mps, cpu). Auto-detected if None.
            confidence_threshold: Confidence threshold for detections
            use_native_sam3: If True and CUDA available, use native sam3 repo implementation.
                           If False or CUDA not available, use HuggingFace transformers.
        """
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Determine which implementation to use
        self.use_native = False
        if use_native_sam3 and self.device == "cuda":
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor
                self.use_native = True
                print("Using native SAM3 implementation (CUDA + triton)")
            except ImportError as e:
                print(f"Native SAM3 not available ({e}), falling back to HuggingFace transformers")
                self.use_native = False
        
        if self.use_native:
            self._init_native_sam3()
        else:
            self._init_huggingface_sam3()
    
    def _init_huggingface_sam3(self):
        """Initialize using HuggingFace Transformers implementation."""
        print("Loading SAM3 model via HuggingFace Transformers...")
        model_path = self._prepare_model_dir()
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def _init_native_sam3(self):
        """Initialize using native SAM3 repo implementation (CUDA only)."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as NativeSam3Processor
        
        print("Loading SAM3 model via native implementation...")
        self.model = build_sam3_image_model(device=self.device)
        self.native_processor = NativeSam3Processor(
            self.model, 
            confidence_threshold=self.confidence_threshold
        )
        print("Model loaded successfully!")

    def _prepare_model_dir(self) -> str:
        """Download the model and ensure processor config naming is compatible."""
        cache_dir = Path(__file__).resolve().parent / "data" / "sam3_model"
        cache_dir.mkdir(parents=True, exist_ok=True)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        local_dir = snapshot_download(
            self.model_id,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        processor_cfg = Path(local_dir) / "processor_config.json"
        preprocessor_cfg = Path(local_dir) / "preprocessor_config.json"
        if processor_cfg.exists() and not preprocessor_cfg.exists():
            shutil.copy(processor_cfg, preprocessor_cfg)

        return local_dir
    
    @torch.no_grad()
    def run_inference(
        self,
        image: Image.Image,
        prompts: list[str],
    ) -> dict:
        """Run SAM3 inference with text prompts.
        
        Args:
            image: PIL Image to process
            prompts: List of text prompts
            
        Returns:
            Dictionary with masks, boxes, and scores for each prompt
        """
        if self.use_native:
            return self._run_inference_native(image, prompts)
        else:
            return self._run_inference_huggingface(image, prompts)
    
    def _run_inference_native(self, image: Image.Image, prompts: list[str]) -> dict:
        """Run inference using native SAM3 implementation."""
        results = {}
        
        # Set the image once
        state = self.native_processor.set_image(image)
        
        for prompt in prompts:
            # Reset prompts for new query
            self.native_processor.reset_all_prompts(state)
            
            # Re-set image backbone (reset clears it)
            state = self.native_processor.set_image(image, state)
            
            # Run text prompt
            state = self.native_processor.set_text_prompt(prompt, state)
            
            # Extract masks and scores
            masks = []
            scores = []
            
            if "masks" in state and state["masks"] is not None:
                mask_tensor = state["masks"]
                score_tensor = state.get("scores", None)
                
                for i in range(mask_tensor.shape[0]):
                    mask = mask_tensor[i, 0].cpu().numpy().astype(np.uint8)
                    masks.append(mask)
                    if score_tensor is not None and i < len(score_tensor):
                        scores.append(float(score_tensor[i].cpu()))
                    else:
                        scores.append(1.0)
            
            results[prompt] = {
                "masks": masks,
                "scores": scores,
                "num_detections": len(masks),
            }
        
        return results
    
    def _run_inference_huggingface(self, image: Image.Image, prompts: list[str]) -> dict:
        """Run inference using HuggingFace Transformers implementation."""
        results = {}
        original_size = image.size  # (width, height)
        
        for prompt in prompts:
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Run model
            outputs = self.model(**inputs)
            
            # Post-process using instance segmentation
            # target_sizes should be (height, width)
            target_sizes = [(original_size[1], original_size[0])]
            
            processed = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=self.confidence_threshold,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )
            
            # Extract masks and scores from processed output
            masks = []
            scores = []
            
            if processed and len(processed) > 0:
                result = processed[0]
                
                # Result contains 'segmentation' (combined mask) and 'segments_info'
                if "segments_info" in result:
                    for segment in result["segments_info"]:
                        segment_id = segment.get("id", 0)
                        score = segment.get("score", 1.0)
                        
                        # Extract mask for this segment from segmentation
                        if "segmentation" in result:
                            seg_map = result["segmentation"]
                            if isinstance(seg_map, torch.Tensor):
                                seg_map = seg_map.cpu().numpy()
                            mask = (seg_map == segment_id).astype(np.uint8)
                            masks.append(mask)
                            scores.append(float(score))
                
                # Alternative: direct mask output
                elif "masks" in result:
                    for i, mask in enumerate(result["masks"]):
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        masks.append(mask)
                        score = result.get("scores", [1.0] * len(result["masks"]))[i]
                        scores.append(float(score))
            
            results[prompt] = {
                "masks": masks,
                "scores": scores,
                "num_detections": len(masks),
            }
        
        return results


class Sam3VideoInference:
    """SAM3-based video inference with temporal tracking and propagation.
    
    Uses the SAM3 video predictor API for session-based inference,
    which maintains temporal consistency across frames.
    
    NOTE: Requires CUDA and triton. Will raise an error if CUDA is not available.
    """
    
    def __init__(self, gpus_to_use: Optional[list[int]] = None):
        """Initialize the video predictor.
        
        Args:
            gpus_to_use: List of GPU indices to use. If None, uses all available GPUs.
        
        Raises:
            RuntimeError: If CUDA is not available (video mode requires CUDA + triton)
        """
        # Check CUDA availability first
        if not torch.cuda.is_available():
            raise RuntimeError(
                "SAM3 video mode requires CUDA. Please use --mode frame on non-CUDA devices, "
                "or run on a machine with NVIDIA GPU."
            )
        
        # Import sam3 video predictor (requires sam3 installed from source)
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as e:
            raise ImportError(
                "sam3 package not found or missing dependencies. Please install from source:\n"
                "  cd sam3_repo && pip install -e .\n"
                f"Original error: {e}"
            )
        
        if gpus_to_use is None:
            gpus_to_use = list(range(torch.cuda.device_count()))
        
        print(f"Initializing SAM3 video predictor with GPUs: {gpus_to_use}")
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
        self.session_id = None
        print("Video predictor initialized!")
    
    def start_session(self, video_path: str) -> str:
        """Start a new video session.
        
        Args:
            video_path: Path to video file (MP4) or directory of JPEG frames
            
        Returns:
            Session ID
        """
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        self.session_id = response["session_id"]
        return self.session_id
    
    def reset_session(self):
        """Reset the current session (clear all prompts)."""
        if self.session_id:
            self.predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=self.session_id,
                )
            )
    
    def add_text_prompt(self, frame_index: int, text: str) -> dict:
        """Add a text prompt on a specific frame.
        
        Args:
            frame_index: Frame index to add the prompt on
            text: Text prompt describing what to segment
            
        Returns:
            Output dict with masks for the prompted frame
        """
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self.session_id,
                frame_index=frame_index,
                text=text,
            )
        )
        return response["outputs"]
    
    def propagate_in_video(self) -> dict[int, dict]:
        """Propagate masks through the entire video.
        
        Returns:
            Dictionary mapping frame_index -> outputs
        """
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self.session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        return outputs_per_frame
    
    def close_session(self):
        """Close the current session and free resources."""
        if self.session_id:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=self.session_id,
                )
            )
            self.session_id = None
    
    def shutdown(self):
        """Shutdown the predictor and free all resources."""
        self.predictor.shutdown()
    
    @staticmethod
    def prepare_frames_directory(image_paths: list[str], temp_dir: str) -> str:
        """Prepare a directory of numbered JPEG frames for video inference.
        
        SAM3 video predictor expects frames named as <frame_index>.jpg
        
        Args:
            image_paths: List of image paths in order
            temp_dir: Temporary directory to store renamed frames
            
        Returns:
            Path to the frames directory
        """
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        for i, src_path in enumerate(image_paths):
            dst_path = frames_dir / f"{i:05d}.jpg"
            # Copy or symlink the frame
            shutil.copy(src_path, dst_path)
        
        return str(frames_dir)
    
    @staticmethod
    def extract_masks_from_output(output: dict, image_size: tuple[int, int]) -> dict:
        """Extract masks from video predictor output format.
        
        Args:
            output: Output from video predictor
            image_size: (height, width) of the image
            
        Returns:
            Dictionary with masks and object IDs
        """
        masks = []
        obj_ids = []
        
        if output is None:
            return {"masks": masks, "obj_ids": obj_ids}
        
        # Video predictor returns masks keyed by object ID
        for obj_id, mask_data in output.items():
            if isinstance(mask_data, dict) and "mask" in mask_data:
                mask = mask_data["mask"]
            elif isinstance(mask_data, (np.ndarray, torch.Tensor)):
                mask = mask_data
            else:
                continue
            
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Ensure mask is correct size
            if mask.shape != image_size:
                mask = cv2.resize(mask.astype(np.float32), (image_size[1], image_size[0]))
            
            masks.append((mask > 0.5).astype(np.uint8))
            obj_ids.append(obj_id)
        
        return {"masks": masks, "obj_ids": obj_ids}


class Visualizer:
    """Visualization utilities for lane detection results."""
    
    # Color palette for different prompts
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    @staticmethod
    def create_overlay(
        image: np.ndarray,
        results: dict,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Create mask overlay on image.
        
        Args:
            image: Original image as numpy array (H, W, 3)
            results: Dictionary of results from inference
            alpha: Transparency for overlay
            
        Returns:
            Image with mask overlays
        """
        overlay = image.copy()
        
        for i, (prompt, data) in enumerate(results.items()):
            color = Visualizer.COLORS[i % len(Visualizer.COLORS)]
            
            for mask in data["masks"]:
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Create colored mask
                mask_binary = mask > 0.5
                overlay[mask_binary] = (
                    alpha * np.array(color) + 
                    (1 - alpha) * overlay[mask_binary]
                ).astype(np.uint8)
        
        return overlay
    
    @staticmethod
    def create_side_by_side(
        original: np.ndarray,
        overlay: np.ndarray,
    ) -> np.ndarray:
        """Create side-by-side comparison.
        
        Args:
            original: Original image
            overlay: Overlay image
            
        Returns:
            Side-by-side image
        """
        return np.hstack([original, overlay])
    
    @staticmethod
    def create_legend(prompts: list[str], height: int = 100) -> np.ndarray:
        """Create a legend for the visualization."""
        width = 400
        legend = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        y_offset = 20
        for i, prompt in enumerate(prompts):
            color = Visualizer.COLORS[i % len(Visualizer.COLORS)]
            # Draw color box
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            # Draw text
            cv2.putText(
                legend, prompt, (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
            y_offset += 20
        
        return legend


def create_video_from_frames(
    frame_paths: list[str],
    output_path: str,
    fps: float = 12.0,
) -> None:
    """Create MP4 video from frames.
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Output video path
        fps: Frames per second
    """
    if not frame_paths:
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for path in frame_paths:
        frame = cv2.imread(path)
        writer.write(frame)
    
    writer.release()
    print(f"Video saved to: {output_path}")


def run_frame_mode(args, loader: NuScenesLoader, target_scenes: list[SceneConfig]):
    """Run per-frame inference mode (no temporal tracking)."""
    print("Running in FRAME mode (per-frame inference, no temporal tracking)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inferencer
    use_native = getattr(args, 'use_native_sam3', False)
    print("Initializing SAM3 frame inference...")
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        use_native_sam3=use_native,
    )
    
    visualizer = Visualizer()
    
    # Process each target scene
    for scene_config in target_scenes:
        print(f"\n{'='*60}")
        print(f"Processing {scene_config.name}")
        print(f"Purpose: {scene_config.test_purpose}")
        print(f"{'='*60}")
        
        # Check if scene exists
        if scene_config.token not in loader.scenes:
            print(f"Warning: Scene {scene_config.name} not found in dataset, skipping...")
            continue
        
        # Create output directories
        scene_output = output_dir / scene_config.name
        overlays_dir = scene_output / "overlays"
        comparisons_dir = scene_output / "comparisons"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        
        # Get samples for scene
        samples = loader.get_scene_samples(scene_config.token)
        
        # Apply frame selection if specified
        if args.frames:
            frame_indices = [int(f) for f in args.frames.split(",")]
            samples = [samples[i] for i in frame_indices if i < len(samples)]
        elif args.max_samples:
            samples = samples[:args.max_samples]
        
        print(f"Processing {len(samples)} samples")
        
        overlay_paths = []
        
        # Process each sample
        for idx, sample in enumerate(tqdm(samples, desc=f"Processing {scene_config.name}")):
            try:
                # Load image
                image = Image.open(sample["image_path"])
                image_np = np.array(image)
                
                # Run inference
                results = inferencer.run_inference(image, LANE_PROMPTS)
                
                # Create visualizations
                overlay = visualizer.create_overlay(image_np, results)
                comparison = visualizer.create_side_by_side(image_np, overlay)
                
                # Save visualizations
                frame_name = Path(sample["filename"]).stem
                
                overlay_path = overlays_dir / f"{frame_name}_overlay.jpg"
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                overlay_paths.append(str(overlay_path))
                
                comparison_path = comparisons_dir / f"{frame_name}_comparison.jpg"
                cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error processing {sample['filename']}: {e}")
                continue
        
        # Create video from overlays
        if overlay_paths and not args.frames:
            video_path = scene_output / f"{scene_config.name}_lanes.mp4"
            create_video_from_frames(overlay_paths, str(video_path))
        
        # Save scene summary
        summary = {
            "scene_name": scene_config.name,
            "description": scene_config.description,
            "test_purpose": scene_config.test_purpose,
            "pass_criteria": scene_config.pass_criteria,
            "num_samples_processed": len(samples),
            "prompts_used": LANE_PROMPTS,
            "mode": "frame",
        }
        
        with open(scene_output / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {scene_output}")


def run_video_mode(args, loader: NuScenesLoader, target_scenes: list[SceneConfig]):
    """Run session-based video inference with temporal tracking."""
    print("Running in VIDEO mode (session-based with temporal tracking)")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video inferencer
    print("Initializing SAM3 video inference...")
    inferencer = Sam3VideoInference()
    
    visualizer = Visualizer()
    
    try:
        # Process each target scene
        for scene_config in target_scenes:
            print(f"\n{'='*60}")
            print(f"Processing {scene_config.name}")
            print(f"Purpose: {scene_config.test_purpose}")
            print(f"{'='*60}")
            
            # Check if scene exists
            if scene_config.token not in loader.scenes:
                print(f"Warning: Scene {scene_config.name} not found in dataset, skipping...")
                continue
            
            # Create output directories
            scene_output = output_dir / scene_config.name / "video_mode"
            overlays_dir = scene_output / "overlays"
            comparisons_dir = scene_output / "comparisons"
            overlays_dir.mkdir(parents=True, exist_ok=True)
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all samples for scene
            all_samples = loader.get_scene_samples(scene_config.token)
            print(f"Found {len(all_samples)} total samples")
            
            # Determine which frames to output
            if args.frames:
                output_frame_indices = [int(f) for f in args.frames.split(",")]
            elif args.max_samples:
                output_frame_indices = list(range(min(args.max_samples, len(all_samples))))
            else:
                output_frame_indices = list(range(len(all_samples)))
            
            # Create temp directory with numbered frames for video predictor
            with tempfile.TemporaryDirectory() as temp_dir:
                print("Preparing frames for video inference...")
                image_paths = [s["image_path"] for s in all_samples]
                frames_dir = Sam3VideoInference.prepare_frames_directory(image_paths, temp_dir)
                
                # Start video session
                print("Starting video session...")
                inferencer.start_session(frames_dir)
                
                # Process each prompt separately (video mode handles one concept at a time)
                all_prompt_results = {}
                
                for prompt in LANE_PROMPTS:
                    print(f"Processing prompt: '{prompt}'")
                    
                    # Reset session for new prompt
                    inferencer.reset_session()
                    
                    # Add prompt on first frame
                    prompt_frame = args.prompt_frame if hasattr(args, 'prompt_frame') else 0
                    print(f"  Adding prompt on frame {prompt_frame}...")
                    inferencer.add_text_prompt(prompt_frame, prompt)
                    
                    # Propagate through video
                    print("  Propagating through video...")
                    outputs = inferencer.propagate_in_video()
                    all_prompt_results[prompt] = outputs
                
                # Generate visualizations for selected frames
                print(f"Generating visualizations for frames: {output_frame_indices}")
                overlay_paths = []
                
                for frame_idx in tqdm(output_frame_indices, desc="Creating visualizations"):
                    if frame_idx >= len(all_samples):
                        continue
                    
                    sample = all_samples[frame_idx]
                    image = Image.open(sample["image_path"])
                    image_np = np.array(image)
                    image_size = (image_np.shape[0], image_np.shape[1])
                    
                    # Combine results from all prompts
                    combined_results = {}
                    for prompt, outputs in all_prompt_results.items():
                        if frame_idx in outputs:
                            mask_data = Sam3VideoInference.extract_masks_from_output(
                                outputs[frame_idx], image_size
                            )
                            combined_results[prompt] = {
                                "masks": mask_data["masks"],
                                "scores": [1.0] * len(mask_data["masks"]),
                                "num_detections": len(mask_data["masks"]),
                            }
                        else:
                            combined_results[prompt] = {
                                "masks": [],
                                "scores": [],
                                "num_detections": 0,
                            }
                    
                    # Create visualizations
                    overlay = visualizer.create_overlay(image_np, combined_results)
                    comparison = visualizer.create_side_by_side(image_np, overlay)
                    
                    # Save visualizations
                    frame_name = Path(sample["filename"]).stem
                    
                    overlay_path = overlays_dir / f"{frame_idx:03d}_{frame_name}_overlay.jpg"
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    overlay_paths.append(str(overlay_path))
                    
                    comparison_path = comparisons_dir / f"{frame_idx:03d}_{frame_name}_comparison.jpg"
                    cv2.imwrite(str(comparison_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
                # Close the session
                inferencer.close_session()
            
            # Create video from overlays if we have more than a few frames
            if len(overlay_paths) > 3 and not args.frames:
                video_path = scene_output / f"{scene_config.name}_lanes_video.mp4"
                create_video_from_frames(overlay_paths, str(video_path))
            
            # Save scene summary
            summary = {
                "scene_name": scene_config.name,
                "description": scene_config.description,
                "test_purpose": scene_config.test_purpose,
                "pass_criteria": scene_config.pass_criteria,
                "num_samples_total": len(all_samples),
                "frames_output": output_frame_indices,
                "prompts_used": LANE_PROMPTS,
                "mode": "video",
            }
            
            with open(scene_output / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"Results saved to: {scene_output}")
    
    finally:
        # Cleanup
        inferencer.shutdown()


def main():
    """Main entry point."""
    import argparse
    
    # Get the script directory for default paths
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="SAM3 Lane Detection on nuScenes")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["frame", "video"],
        default="frame",
        help="Inference mode: 'frame' for per-frame (no tracking), 'video' for session-based with temporal tracking",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(script_dir / "data" / "v1.0-mini"),
        help="Path to nuScenes mini data root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "data" / "sam3_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per scene (for testing)",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame indices to process (e.g., '0,4,9' for frames 1,5,10)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Process only this scene (e.g., 'scene-1094')",
    )
    parser.add_argument(
        "--prompt-frame",
        type=int,
        default=0,
        help="Frame index to add text prompt on (for video mode)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--use-native-sam3",
        action="store_true",
        help="Use native SAM3 repo implementation instead of HuggingFace (requires CUDA + triton)",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    
    # Initialize loader
    print("Initializing nuScenes loader...")
    loader = NuScenesLoader(str(data_root))
    
    # Filter scenes if specified
    if args.scene:
        target_scenes = [s for s in TARGET_SCENES if s.name == args.scene]
        if not target_scenes:
            print(f"Error: Scene '{args.scene}' not found in TARGET_SCENES")
            return
    else:
        target_scenes = TARGET_SCENES
    
    # Run appropriate mode
    if args.mode == "frame":
        run_frame_mode(args, loader, target_scenes)
    elif args.mode == "video":
        # Check CUDA availability for video mode
        if not torch.cuda.is_available():
            print("\n" + "="*60)
            print("WARNING: CUDA not available. Video mode requires CUDA + triton.")
            print("Falling back to frame mode (no temporal tracking).")
            print("To use video mode with temporal tracking, run on a CUDA-enabled machine.")
            print("="*60 + "\n")
            run_frame_mode(args, loader, target_scenes)
        else:
            run_video_mode(args, loader, target_scenes)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

