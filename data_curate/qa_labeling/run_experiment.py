#!/usr/bin/env python3
"""Run semantic class disambiguation experiment.

This script runs Experiment 1 from the QA design doc:
- Load nuScenes mini VRU annotations
- Inject synthetic labeling errors
- Run VLM-based classification with self-consistency voting
- Evaluate and report results

Usage:
    python -m qa_labeling.run_experiment --help
    python -m qa_labeling.run_experiment --max-samples 50 --output-dir results/
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM semantic class disambiguation experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "v1.0-mini",
        help="Path to nuScenes data root",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "qa_results",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of samples to evaluate",
    )
    
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.5,
        help="Fraction of samples to inject synthetic errors",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID or local path",
    )
    
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use Flash Attention 2 (requires flash-attn package)",
    )
    
    parser.add_argument(
        "--min-visibility",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Minimum visibility level (1-4)",
    )
    
    parser.add_argument(
        "--max-distance",
        type=float,
        default=60.0,
        help="Maximum distance from ego vehicle (meters)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data without running VLM inference",
    )
    
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save ROI crop images to disk (works with --dry-run)",
    )
    
    parser.add_argument(
        "--tight-crop",
        action="store_true",
        help="Use tight 3D-projected bounding box instead of padded crop (reduces distractors)",
    )
    
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance samples across VRU classes (PEDESTRIAN, CYCLIST, MOTORCYCLIST)",
    )
    
    parser.add_argument(
        "--use-existing-crops",
        action="store_true",
        help="Use existing crops from metadata.json instead of regenerating",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Single image inference mode
    parser.add_argument(
        "--single-image",
        type=Path,
        default=None,
        help="Run VLM inference on a single image file",
    )
    
    parser.add_argument(
        "--current-label",
        type=str,
        default="UNKNOWN",
        help="Current label to verify (used with --single-image)",
    )
    
    parser.add_argument(
        "--use-visual-anchor",
        action="store_true",
        help="Use visual anchor prompt (for images with TARGET box drawn)",
    )
    
    return parser.parse_args()


def run_single_image_inference(args: argparse.Namespace) -> int:
    """Run VLM inference on a single image file.
    
    Args:
        args: Parsed command line arguments with --single-image
        
    Returns:
        Exit code (0 for success)
    """
    from PIL import Image as PILImage
    from .vlm_judge import SemanticQAJudge
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    image_path = args.single_image
    if not image_path.exists():
        _LOGGER.error("Image file does not exist: %s", image_path)
        return 1
    
    _LOGGER.info("=" * 60)
    _LOGGER.info("SINGLE IMAGE VLM INFERENCE")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Image: %s", image_path)
    _LOGGER.info("Current label: %s", args.current_label)
    _LOGGER.info("Visual anchor: %s", args.use_visual_anchor)
    _LOGGER.info("Model: %s", args.model_path)
    _LOGGER.info("")
    
    # Load image
    _LOGGER.info("Loading image...")
    image = np.array(PILImage.open(image_path).convert("RGB"))
    _LOGGER.info("Image shape: %s", image.shape)
    
    # Initialize VLM judge
    _LOGGER.info("")
    _LOGGER.info("Loading VLM model...")
    judge = SemanticQAJudge(
        model_path=args.model_path,
        num_samples=3,
    )
    
    # Run inference
    _LOGGER.info("")
    _LOGGER.info("Running VLM inference with self-consistency voting...")
    result = judge.judge_image(
        image, 
        current_label=args.current_label,
        use_visual_anchor=args.use_visual_anchor,
    )
    
    # Print results
    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("RESULT")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Predicted class: %s", result.predicted_class)
    _LOGGER.info("Agreement: %d/3", result.agreement)
    _LOGGER.info("Decision: %s", result.decision)
    _LOGGER.info("All samples: %s", result.samples)
    _LOGGER.info("Evidence: %s", result.evidence)
    
    # Print summary
    print("")
    print("=" * 60)
    print(f"VLM Triage: {result.predicted_class}")
    print(f"Agreement: {result.agreement}/3 → {result.decision}")
    print(f"Evidence: {', '.join(result.evidence)}")
    print("=" * 60)
    
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    """Run the semantic QA experiment.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    # Import here to avoid slow imports on --help
    from .data_prep import (
        NuScenesDataLoader,
        SyntheticErrorInjector,
        prepare_roi_samples,
        ROISample,
    )
    from .vlm_judge import SemanticQAJudge
    from .evaluate import (
        SemanticQAEvaluator,
        create_side_by_side_visualization,
        save_results_json,
    )
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate data path
    if not args.data_root.exists():
        _LOGGER.error("Data root does not exist: %s", args.data_root)
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    _LOGGER.info("=" * 60)
    _LOGGER.info("SEMANTIC CLASS DISAMBIGUATION EXPERIMENT")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Data root: %s", args.data_root)
    _LOGGER.info("Output dir: %s", args.output_dir)
    _LOGGER.info("Max samples: %d", args.max_samples)
    _LOGGER.info("Error rate: %.1f%%", args.error_rate * 100)
    _LOGGER.info("Model: %s", args.model_path)
    
    # Check if using existing crops
    if args.use_existing_crops:
        _LOGGER.info("")
        _LOGGER.info("Loading existing crops from metadata.json...")
        
        import json
        from PIL import Image as PILImage
        
        crops_dir = args.output_dir / "crops"
        metadata_path = crops_dir / "metadata.json"
        
        if not metadata_path.exists():
            _LOGGER.error("No metadata.json found at %s", metadata_path)
            return 1
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Reconstruct ROISample objects from metadata
        samples = []
        for item in metadata:
            crop_path = crops_dir / item["filename"]
            if not crop_path.exists():
                _LOGGER.warning("Crop not found: %s", crop_path)
                continue
            
            roi_image = np.array(PILImage.open(crop_path))
            
            sample = ROISample(
                annotation_token=item["annotation_token"],
                sample_token=item["sample_token"],
                camera_name="CAM_FRONT",  # Default
                image_path=crop_path,
                roi_image=roi_image,
                bbox_2d=tuple(item["bbox_2d"]) if item.get("bbox_2d") else None,
                bbox_2d_tight=tuple(item["bbox_2d_tight"]) if item.get("bbox_2d_tight") else None,
                gt_class=item["gt_class"],
                category_name=item.get("category_name", ""),
                injected_class=item.get("injected_class"),
                distance=item.get("distance", 0.0),
            )
            samples.append(sample)
        
        _LOGGER.info("Loaded %d existing samples", len(samples))
        
        if not samples:
            _LOGGER.error("No valid samples found in metadata!")
            return 1
    else:
        # Step 1: Load nuScenes data
        _LOGGER.info("")
        _LOGGER.info("Step 1: Loading nuScenes data...")
        loader = NuScenesDataLoader(data_root=args.data_root)
        
        # Step 2: Prepare ROI samples
        _LOGGER.info("")
        _LOGGER.info("Step 2: Preparing ROI samples...")
        samples = prepare_roi_samples(
            loader=loader,
            max_samples=args.max_samples,
            min_visibility=args.min_visibility,
            max_distance=args.max_distance,
            balance_classes=args.balance_classes,
        )
        
        if not samples:
            _LOGGER.error("No samples found! Check data path and filters.")
            return 1
        
        _LOGGER.info("Prepared %d samples", len(samples))
        
        # Step 3: Inject synthetic errors
        _LOGGER.info("")
        _LOGGER.info("Step 3: Injecting synthetic labeling errors...")
        injector = SyntheticErrorInjector(
            error_rate=args.error_rate,
            seed=args.seed,
        )
        samples = injector.inject_errors(samples)
    
    n_clean = sum(1 for s in samples if s.injected_class is None)
    n_corrupted = sum(1 for s in samples if s.injected_class is not None)
    _LOGGER.info("Clean samples: %d, Corrupted samples: %d", n_clean, n_corrupted)
    
    # Save crops if requested
    if args.save_crops:
        _LOGGER.info("")
        _LOGGER.info("Saving ROI crops to disk...")
        crops_dir = args.output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        from PIL import Image as PILImage
        import json
        
        metadata = []
        for i, sample in enumerate(samples):
            if sample.roi_image is None:
                continue
            
            # Save crop image
            crop_filename = f"crop_{i:04d}.png"
            crop_path = crops_dir / crop_filename
            PILImage.fromarray(sample.roi_image).save(crop_path)
            
            # Collect metadata
            metadata.append({
                "index": i,
                "filename": crop_filename,
                "annotation_token": sample.annotation_token,
                "sample_token": sample.sample_token,
                "gt_class": sample.gt_class,
                "category_name": sample.category_name,
                "injected_class": sample.injected_class,
                "distance": round(sample.distance, 2),
                "bbox_2d": sample.bbox_2d,
                "bbox_2d_tight": sample.bbox_2d_tight,
            })
        
        # Save metadata JSON
        metadata_path = crops_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        _LOGGER.info("Saved %d crops to %s", len(metadata), crops_dir)
        _LOGGER.info("Metadata saved to %s", metadata_path)
    
    if args.dry_run:
        _LOGGER.info("")
        _LOGGER.info("Dry run complete. Skipping VLM inference.")
        
        # Print sample distribution
        from collections import Counter
        gt_dist = Counter(s.gt_class for s in samples)
        _LOGGER.info("Ground truth distribution:")
        for cls, count in gt_dist.most_common():
            _LOGGER.info("  %s: %d", cls, count)
        
        return 0
    
    # Step 4: Load VLM model
    _LOGGER.info("")
    _LOGGER.info("Step 4: Loading VLM model...")
    judge = SemanticQAJudge(
        model_path=args.model_path,
        use_flash_attn=args.use_flash_attn,
    )
    
    # Step 5: Run inference with self-consistency
    _LOGGER.info("")
    _LOGGER.info("Step 5: Running VLM inference with self-consistency voting...")
    evaluator = SemanticQAEvaluator()
    
    for i, sample in enumerate(samples):
        _LOGGER.info("Processing sample %d/%d", i + 1, len(samples))
        
        if sample.roi_image is None:
            _LOGGER.warning("Sample %d has no ROI image, skipping", i + 1)
            continue
        
        # Determine current label to show to VLM:
        # - If corrupted: show the injected (wrong) label
        # - If clean: show the GT label
        current_label = sample.injected_class if sample.injected_class else sample.gt_class
        
        # Run VLM inference with current label for QA verification
        result = judge.judge_image(sample.roi_image, current_label=current_label)
        
        # Add to evaluator
        evaluator.add_result(sample, result)
    
    # Step 6: Compute and report metrics
    _LOGGER.info("")
    _LOGGER.info("Step 6: Computing evaluation metrics...")
    
    report = evaluator.format_report()
    print("\n" + report)
    
    # Save report to file
    report_path = args.output_dir / "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    _LOGGER.info("Saved report to %s", report_path)
    
    # Save detailed results as JSON
    metrics = evaluator.compute_metrics()
    json_path = args.output_dir / "results.json"
    save_results_json(metrics, evaluator.sample_results, json_path)
    
    # Step 7: Create visualizations
    _LOGGER.info("")
    _LOGGER.info("Step 7: Creating visualizations...")
    
    # Visualize error cases
    error_samples = evaluator.get_error_samples()
    if error_samples:
        viz_dir = args.output_dir / "error_cases"
        create_side_by_side_visualization(
            error_samples, viz_dir, max_examples=8
        )
    
    # Visualize review cases
    review_samples = evaluator.get_review_samples()
    if review_samples:
        viz_dir = args.output_dir / "review_cases"
        create_side_by_side_visualization(
            review_samples, viz_dir, max_examples=8
        )
    
    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("EXPERIMENT COMPLETE")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Overall Accuracy: %.1f%%", metrics.accuracy * 100)
    _LOGGER.info("Review Rate: %.1f%%", metrics.review_rate * 100)
    _LOGGER.info("Results saved to: %s", args.output_dir)
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Check for single image mode
    if args.single_image is not None:
        return run_single_image_inference(args)
    
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())

