#!/usr/bin/env python3
"""Run VLM-powered labeling QA experiments.

This script supports multiple experiments:
- Experiment 1: Semantic class disambiguation
- Experiment 2: Ghost box detection (false positives)

Usage:
    python -m qa_labeling.run_experiment --help
    python -m qa_labeling.run_experiment --experiment semantic --max-samples 50
    python -m qa_labeling.run_experiment --experiment ghost --max-samples 10
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


def create_ghost_box_visualization(
    ghost_samples,
    results,
    output_dir: Path,
) -> Path:
    """Create two-row visualization: top=ghost crops, bottom=VLM analysis.
    
    Args:
        ghost_samples: List of GhostBoxSample objects
        results: List of dicts with 'ghost_sample' and 'vlm_result'
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    from datetime import datetime
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = font_title
        font_small = font_title
    
    # Cell dimensions
    cell_width = 350
    cell_height = 350
    text_height = 150  # More space for VLM analysis text
    padding = 20
    title_height = 60
    
    # Colors
    color_yes = (220, 53, 69)      # Red for YES (incorrect - ghost should be empty)
    color_no = (40, 167, 69)        # Green for NO (correct - detected as empty)
    color_uncertain = (255, 193, 7) # Yellow for UNCERTAIN (needs review)
    
    num_samples = len(results)
    
    # Create top row cells (ghost box crops)
    top_cells = []
    for i, item in enumerate(results):
        ghost = item["ghost_sample"]
        vlm_result = item["vlm_result"]
        
        # Create cell
        cell = PILImage.new('RGB', (cell_width, cell_height + text_height), 'white')
        draw = ImageDraw.Draw(cell)
        
        # Resize and paste crop
        crop_img = PILImage.fromarray(ghost.roi_image)
        crop_img.thumbnail((cell_width - 20, cell_height - 20), PILImage.Resampling.LANCZOS)
        
        # Background
        draw.rectangle([0, 0, cell_width-1, cell_height-1], fill=(245, 245, 245))
        draw.rectangle([0, 0, cell_width-1, cell_height-1], outline=(150, 150, 150), width=3)
        
        # Center crop
        img_x = (cell_width - crop_img.width) // 2
        img_y = (cell_height - crop_img.height) // 2
        cell.paste(crop_img, (img_x, img_y))
        
        # Text below
        y = cell_height + 10
        draw.text((10, y), f"Ghost Box #{i+1}", fill=(60, 60, 60), font=font_label)
        y += 28
        draw.text((10, y), f"Shift: {ghost.shift_type}", fill=(100, 100, 100), font=font_small)
        y += 22
        draw.text((10, y), f"Original: {ghost.original_gt_class}", fill=(100, 100, 100), font=font_small)
        y += 22
        draw.text((10, y), "(Expected: NO or UNCERTAIN)", fill=(120, 120, 120), font=font_small)
        
        top_cells.append(cell)
    
    # Create bottom row cells (VLM analysis)
    bottom_cells = []
    for i, item in enumerate(results):
        ghost = item["ghost_sample"]
        vlm_result = item["vlm_result"]
        
        # Determine color based on VLM result
        if vlm_result.exists == "NO":
            border_color = color_no
            result_text = "✓ CORRECT"
            triage_label = "EMPTY"
        elif vlm_result.exists == "UNCERTAIN":
            border_color = color_uncertain
            result_text = "⚠ REVIEW"
            triage_label = "UNCERTAIN"
        else:  # YES
            border_color = color_yes
            result_text = "✗ INCORRECT"
            triage_label = f"CONTAINS {vlm_result.object_type or 'OBJECT'}"
        
        # Create cell
        cell = PILImage.new('RGB', (cell_width, cell_height + text_height), 'white')
        draw = ImageDraw.Draw(cell)
        
        # Background with result color
        if vlm_result.exists == "NO":
            bg_color = (240, 255, 240)  # Light green
        elif vlm_result.exists == "UNCERTAIN":
            bg_color = (255, 250, 230)  # Light yellow
        else:
            bg_color = (255, 240, 240)  # Light red
        
        draw.rectangle([0, 0, cell_width-1, cell_height-1], fill=bg_color)
        draw.rectangle([0, 0, cell_width-1, cell_height-1], outline=border_color, width=6)
        
        # VLM analysis text
        y = 30
        draw.text((cell_width//2, y), result_text, fill=border_color, font=font_title, anchor="mm")
        
        y = 90
        draw.text((cell_width//2, y), "VLM Triage:", fill=(60, 60, 60), font=font_label, anchor="mm")
        y += 35
        draw.text((cell_width//2, y), triage_label, fill=border_color, font=font_title, anchor="mm")
        
        y += 50
        draw.text((cell_width//2, y), f"Agreement: {vlm_result.agreement}/3", fill=(100, 100, 100), font=font_small, anchor="mm")
        
        y += 25
        draw.text((cell_width//2, y), f"Decision: {vlm_result.decision}", fill=(100, 100, 100), font=font_small, anchor="mm")
        
        # Add evidence if available
        if vlm_result.evidence:
            y += 30
            draw.text((10, y), "Evidence:", fill=(80, 80, 80), font=font_small)
            y += 18
            for evidence_item in vlm_result.evidence[:3]:  # Show up to 3 evidence items
                # Wrap long text - show more characters
                if len(evidence_item) > 50:
                    # Split into two lines if needed
                    words = evidence_item.split()
                    line1 = []
                    line2 = []
                    current_line = line1
                    char_count = 0
                    for word in words:
                        if char_count + len(word) < 45 and current_line == line1:
                            line1.append(word)
                            char_count += len(word) + 1
                        else:
                            line2.append(word)
                    
                    if line1:
                        draw.text((10, y), f"• {' '.join(line1)}", fill=(100, 100, 100), font=font_small)
                        y += 16
                    if line2:
                        text2 = ' '.join(line2)
                        if len(text2) > 48:
                            text2 = text2[:45] + "..."
                        draw.text((14, y), text2, fill=(100, 100, 100), font=font_small)
                        y += 16
                else:
                    draw.text((10, y), f"• {evidence_item}", fill=(100, 100, 100), font=font_small)
                    y += 18
        
        bottom_cells.append(cell)
    
    # Create final grid image
    grid_width = num_samples * cell_width + (num_samples + 1) * padding
    grid_height = title_height + (cell_height + text_height) + padding + title_height + (cell_height + text_height) + padding * 2
    
    grid = PILImage.new('RGB', (grid_width, grid_height), color=(250, 250, 250))
    draw_grid = ImageDraw.Draw(grid)
    
    # Title for top row
    title_top = 'Ghost Box Crops (Shifted/Misaligned Bounding Boxes)'
    text_bbox = draw_grid.textbbox((0, 0), title_top, font=font_title)
    text_x = (grid_width - (text_bbox[2] - text_bbox[0])) // 2
    draw_grid.text((text_x, padding + 10), title_top, fill='black', font=font_title)
    
    # Paste top row
    y_top = padding + title_height
    for i, cell in enumerate(top_cells):
        x = padding + i * (cell_width + padding)
        grid.paste(cell, (x, y_top))
    
    # Title for bottom row
    title_bottom = 'VLM Analysis Results (Ghost Box Detection)'
    y_title_bottom = y_top + cell_height + text_height + padding
    text_bbox = draw_grid.textbbox((0, 0), title_bottom, font=font_title)
    text_x = (grid_width - (text_bbox[2] - text_bbox[0])) // 2
    draw_grid.text((text_x, y_title_bottom + 10), title_bottom, fill='black', font=font_title)
    
    # Paste bottom row
    y_bottom = y_title_bottom + title_height
    for i, cell in enumerate(bottom_cells):
        x = padding + i * (cell_width + padding)
        grid.paste(cell, (x, y_bottom))
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"ghost_analysis_{timestamp}.png"
    grid.save(output_path, quality=95)
    
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM-powered labeling QA experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="semantic",
        choices=["semantic", "ghost"],
        help="Which experiment to run (semantic=Exp1, ghost=Exp2)",
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


def run_ghost_box_experiment(args: argparse.Namespace) -> int:
    """Run the ghost box detection experiment.
    
    Creates synthetic ghost boxes by shifting real annotations,
    then evaluates whether VLM correctly identifies empty regions.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    from .data_prep import (
        NuScenesDataLoader,
        prepare_ghost_box_samples,
        GhostBoxSample,
    )
    from .vlm_judge import SemanticQAJudge
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate data path
    if not args.data_root.exists():
        _LOGGER.error("Data root does not exist: %s", args.data_root)
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "ghost_crops"
    crops_dir.mkdir(exist_ok=True)
    
    _LOGGER.info("=" * 60)
    _LOGGER.info("GHOST BOX DETECTION EXPERIMENT")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Data root: %s", args.data_root)
    _LOGGER.info("Output dir: %s", args.output_dir)
    _LOGGER.info("Max samples: %d", args.max_samples)
    _LOGGER.info("Model: %s", args.model_path)
    
    import json
    from PIL import Image as PILImage
    from datetime import datetime
    import numpy as np
    
    # Check if we should use existing crops
    if args.use_existing_crops:
        _LOGGER.info("")
        _LOGGER.info("Step 1: Loading existing ghost box crops...")
        
        # Find the most recent metadata file
        metadata_files = list(crops_dir.glob('ghost_metadata_*.json'))
        if not metadata_files:
            _LOGGER.error("No ghost metadata found in %s", crops_dir)
            _LOGGER.error("Run without --use-existing-crops first to generate crops")
            return 1
        
        metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime)
        _LOGGER.info("Loading metadata from: %s", metadata_path)
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Load crops into GhostBoxSample objects
        ghost_samples = []
        for item in metadata:
            crop_path = crops_dir / item['filename']
            if not crop_path.exists():
                _LOGGER.error("Crop file not found: %s", crop_path)
                continue
            
            roi_image = np.array(PILImage.open(crop_path))
            
            ghost_samples.append(GhostBoxSample(
                original_annotation_token=item['original_annotation_token'],
                sample_token=item['sample_token'],
                camera_name=item['camera_name'],
                image_path=Path(item.get('image_path', '')),
                roi_image=roi_image,
                bbox_2d=tuple(item['bbox_2d']),
                bbox_2d_original=tuple(item['bbox_2d_original']),
                shift_type=item['shift_type'],
                shift_vector=tuple(item['shift_vector']),
                original_gt_class=item['original_gt_class'],
                distance=item.get('distance', 0.0),
            ))
        
        _LOGGER.info("Loaded %d existing ghost box crops", len(ghost_samples))
        
    else:
        # Generate new crops
        # Step 1: Load nuScenes data
        _LOGGER.info("")
        _LOGGER.info("Step 1: Loading nuScenes data...")
        loader = NuScenesDataLoader(data_root=args.data_root)
        
        # Step 2: Prepare ghost box samples
        _LOGGER.info("")
        _LOGGER.info("Step 2: Preparing ghost box samples (shifting real annotations)...")
        ghost_samples = prepare_ghost_box_samples(
            loader=loader,
            num_samples=args.max_samples,
            min_visibility=args.min_visibility,
            max_distance=args.max_distance,
        )
        
        if not ghost_samples:
            _LOGGER.error("No ghost box samples could be created!")
            return 1
        
        _LOGGER.info("Created %d ghost box samples", len(ghost_samples))
        
        # Step 3: Save crops (always save for ghost boxes)
        _LOGGER.info("")
        _LOGGER.info("Step 3: Saving ghost box crops...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = []
        
        for i, ghost in enumerate(ghost_samples):
            filename = f"ghost_crop_{i:04d}_{ghost.shift_type}.png"
            crop_path = crops_dir / filename
            
            # Save crop image
            PILImage.fromarray(ghost.roi_image).save(crop_path)
            
            # Save metadata
            metadata.append({
                "filename": filename,
                "original_annotation_token": ghost.original_annotation_token,
                "sample_token": ghost.sample_token,
                "camera_name": ghost.camera_name,
                "shift_type": ghost.shift_type,
                "shift_vector": ghost.shift_vector,
                "original_gt_class": ghost.original_gt_class,
                "bbox_2d": ghost.bbox_2d,
                "bbox_2d_original": ghost.bbox_2d_original,  # Original bbox before shift
            })
            
            _LOGGER.info(
                "Saved ghost crop %d/%d: %s (shift=%s, original=%s)",
                i + 1, len(ghost_samples), filename,
                ghost.shift_type, ghost.original_gt_class,
            )
        
        # Save metadata
        metadata_path = crops_dir / f"ghost_metadata_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        _LOGGER.info("Saved metadata to: %s", metadata_path)
    
    # If dry-run, stop here
    if args.dry_run:
        _LOGGER.info("")
        _LOGGER.info("=" * 60)
        _LOGGER.info("DRY RUN COMPLETE (VLM inference skipped)")
        _LOGGER.info("=" * 60)
        _LOGGER.info("Ghost crops saved to: %s", crops_dir)
        return 0
    
    # Step 4: Initialize VLM judge
    _LOGGER.info("")
    _LOGGER.info("Step 4: Loading VLM model...")
    judge = SemanticQAJudge(
        model_path=args.model_path,
        num_samples=3,  # Self-consistency with 3 samples
    )
    
    # Step 5: Run VLM inference on ghost boxes
    _LOGGER.info("")
    _LOGGER.info("Step 5: Running VLM ghost box detection...")
    
    results = []
    for i, ghost in enumerate(ghost_samples):
        _LOGGER.info("")
        _LOGGER.info("Processing ghost box %d/%d (shift=%s)...", 
                     i + 1, len(ghost_samples), ghost.shift_type)
        
        result = judge.judge_ghost_box(ghost.roi_image)
        
        results.append({
            "ghost_sample": ghost,
            "vlm_result": result,
        })
        
        _LOGGER.info(
            "Result: exists=%s (agreement: %d/3, decision: %s, type: %s)",
            result.exists, result.agreement, result.decision, result.object_type,
        )
    
    # Step 6: Evaluate results
    _LOGGER.info("")
    _LOGGER.info("Step 6: Evaluating results...")
    
    correct = 0
    total = len(results)
    review_count = 0
    
    for item in results:
        vlm_result = item["vlm_result"]
        
        # Ghost boxes should be detected as NO or UNCERTAIN
        if vlm_result.exists == "NO":
            correct += 1
        elif vlm_result.exists == "UNCERTAIN":
            review_count += 1
            # Count UNCERTAIN as partially correct
            correct += 0.5
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    review_rate = (review_count / total) * 100 if total > 0 else 0
    
    _LOGGER.info("")
    _LOGGER.info("=" * 60)
    _LOGGER.info("GHOST BOX DETECTION RESULTS")
    _LOGGER.info("=" * 60)
    _LOGGER.info("Total ghost boxes: %d", total)
    _LOGGER.info("Correctly detected as empty (NO): %d", int(correct))
    _LOGGER.info("Accuracy: %.1f%%", accuracy)
    _LOGGER.info("Review rate (UNCERTAIN): %.1f%%", review_rate)
    
    # Step 7: Save results
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = args.output_dir / f"ghost_results_{timestamp}.json"
    with open(results_path, "w") as f:
        results_data = []
        for item in results:
            ghost = item["ghost_sample"]
            vlm_result = item["vlm_result"]
            results_data.append({
                "shift_type": ghost.shift_type,
                "original_class": ghost.original_gt_class,
                "vlm_exists": vlm_result.exists,
                "vlm_agreement": vlm_result.agreement,
                "vlm_decision": vlm_result.decision,
                "vlm_type": vlm_result.object_type,
                "vlm_evidence": vlm_result.evidence,
            })
        json.dump({
            "accuracy": accuracy,
            "review_rate": review_rate,
            "total_samples": total,
            "results": results_data,
        }, f, indent=2)
    
    _LOGGER.info("Results saved to: %s", results_path)
    
    # Step 8: Create visualization
    _LOGGER.info("")
    _LOGGER.info("Step 8: Creating visualization...")
    
    viz_path = create_ghost_box_visualization(
        ghost_samples=ghost_samples,
        results=results,
        output_dir=args.output_dir,
    )
    _LOGGER.info("Visualization saved to: %s", viz_path)
    
    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Check for single image mode
    if args.single_image is not None:
        return run_single_image_inference(args)
    
    # Route to appropriate experiment
    if args.experiment == "ghost":
        return run_ghost_box_experiment(args)
    else:
        return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())

