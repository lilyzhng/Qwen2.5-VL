#!/usr/bin/env python3
"""
SAM3 Lane Detection Inference on nuScenes Mini Dataset

This script runs SAM3 text-prompted segmentation on 5 selected nuScenes scenes
to detect lane markings and road boundaries, with comprehensive visualization.

Prerequisites:
1. Accept the SAM3 license at: https://huggingface.co/facebook/sam3
2. Login to HuggingFace: huggingface-cli login
3. Install dependencies:
   pip install git+https://github.com/huggingface/transformers torchvision

Target Scenes:
- scene-0061: Easy daylight sanity check
- scene-0553: Urban clutter/occlusions stress
- scene-0103: Hard visibility/lighting stress  
- scene-0916: Topology complexity stress
- scene-1094: Adverse condition (night + rain)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
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
    """SAM3-based lane detection inference."""
    
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
    ):
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
        
        # Load model and processor
        print("Loading SAM3 model and processor...")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
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


def main():
    """Main entry point."""
    import argparse
    
    # Get the script directory for default paths
    script_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="SAM3 Lane Detection on nuScenes")
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
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detections",
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing nuScenes loader...")
    loader = NuScenesLoader(str(data_root))
    
    print("Initializing SAM3 inference...")
    inferencer = Sam3LaneInference(
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    
    visualizer = Visualizer()
    
    # Process each target scene
    for scene_config in TARGET_SCENES:
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
        if args.max_samples:
            samples = samples[:args.max_samples]
        
        print(f"Found {len(samples)} samples")
        
        overlay_paths = []
        
        # Process each sample
        for sample in tqdm(samples, desc=f"Processing {scene_config.name}"):
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
        if overlay_paths:
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
        }
        
        with open(scene_output / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {scene_output}")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

