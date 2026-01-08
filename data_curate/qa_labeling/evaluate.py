"""Evaluation metrics and visualization for semantic QA experiments.

Computes:
- Accuracy vs ground truth
- Confusion matrix
- Agreement distribution (3/3, 2/3, 1/3)
- REVIEW rate
- Distance-based analysis
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Final

from PIL import Image

from .config import SEMANTIC_CLASSES, VRU_CLASSES, DISTANCE_BUCKETS
from .data_prep import ROISample
from .vlm_judge import SelfConsistencyResult

_LOGGER: Final = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from evaluating VLM predictions against ground truth."""
    
    # Overall metrics
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # Agreement distribution
    agreement_3_3: int = 0  # All 3 samples agreed
    agreement_2_3: int = 0  # 2 of 3 agreed
    agreement_1_3: int = 0  # No majority
    
    # Decision distribution
    accept_count: int = 0
    review_count: int = 0
    review_rate: float = 0.0
    
    # Accuracy by decision type
    accuracy_when_accept: float = 0.0
    accuracy_when_review: float = 0.0
    
    # Confusion matrix: confusion_matrix[gt_class][pred_class] = count
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Per-class metrics
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    per_class_count: Dict[str, int] = field(default_factory=dict)
    
    # Distance-based metrics
    accuracy_by_distance: Dict[str, float] = field(default_factory=dict)
    count_by_distance: Dict[str, int] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result for a single sample."""
    
    sample: ROISample
    prediction: SelfConsistencyResult
    is_correct: bool
    gt_class: str
    pred_class: str
    

@dataclass
class SemanticQAEvaluator:
    """Evaluator for semantic QA experiments."""
    
    # Results storage
    sample_results: List[SampleResult] = field(default_factory=list)
    
    def add_result(
        self,
        sample: ROISample,
        prediction: SelfConsistencyResult,
    ) -> None:
        """Add a single evaluation result.
        
        Args:
            sample: The ROI sample with ground truth
            prediction: The VLM prediction result
        """
        # For samples with injected errors, we want to see if VLM recovers GT
        gt_class = sample.gt_class
        pred_class = prediction.predicted_class
        
        is_correct = (pred_class == gt_class)
        
        self.sample_results.append(SampleResult(
            sample=sample,
            prediction=prediction,
            is_correct=is_correct,
            gt_class=gt_class,
            pred_class=pred_class,
        ))
    
    def compute_metrics(self) -> EvaluationResult:
        """Compute all evaluation metrics.
        
        Returns:
            EvaluationResult with all computed metrics
        """
        result = EvaluationResult()
        result.total_samples = len(self.sample_results)
        
        if result.total_samples == 0:
            return result
        
        # Initialize confusion matrix
        for gt_cls in SEMANTIC_CLASSES:
            result.confusion_matrix[gt_cls] = defaultdict(int)
        
        # Counters for various metrics
        correct_when_accept = 0
        correct_when_review = 0
        accept_total = 0
        review_total = 0
        
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        distance_correct = defaultdict(int)
        distance_total = defaultdict(int)
        
        for sr in self.sample_results:
            # Overall accuracy
            if sr.is_correct:
                result.correct_predictions += 1
            
            # Agreement distribution
            agreement = sr.prediction.agreement
            if agreement == 3:
                result.agreement_3_3 += 1
            elif agreement == 2:
                result.agreement_2_3 += 1
            else:
                result.agreement_1_3 += 1
            
            # Decision distribution
            decision = sr.prediction.decision
            if decision == "ACCEPT":
                result.accept_count += 1
                accept_total += 1
                if sr.is_correct:
                    correct_when_accept += 1
            else:
                result.review_count += 1
                review_total += 1
                if sr.is_correct:
                    correct_when_review += 1
            
            # Confusion matrix
            result.confusion_matrix[sr.gt_class][sr.pred_class] += 1
            
            # Per-class metrics
            per_class_total[sr.gt_class] += 1
            if sr.is_correct:
                per_class_correct[sr.gt_class] += 1
            
            # Distance-based metrics
            distance = sr.sample.distance
            for min_d, max_d, label in DISTANCE_BUCKETS:
                if min_d <= distance < max_d:
                    distance_total[label] += 1
                    if sr.is_correct:
                        distance_correct[label] += 1
                    break
        
        # Compute final metrics
        result.accuracy = result.correct_predictions / result.total_samples
        result.review_rate = result.review_count / result.total_samples
        
        if accept_total > 0:
            result.accuracy_when_accept = correct_when_accept / accept_total
        if review_total > 0:
            result.accuracy_when_review = correct_when_review / review_total
        
        # Per-class accuracy
        for cls in per_class_total:
            result.per_class_count[cls] = per_class_total[cls]
            if per_class_total[cls] > 0:
                result.per_class_accuracy[cls] = (
                    per_class_correct[cls] / per_class_total[cls]
                )
        
        # Distance-based accuracy
        for label in distance_total:
            result.count_by_distance[label] = distance_total[label]
            if distance_total[label] > 0:
                result.accuracy_by_distance[label] = (
                    distance_correct[label] / distance_total[label]
                )
        
        return result
    
    def get_error_samples(self) -> List[SampleResult]:
        """Get all samples where prediction was incorrect."""
        return [sr for sr in self.sample_results if not sr.is_correct]
    
    def get_review_samples(self) -> List[SampleResult]:
        """Get all samples that were marked for review."""
        return [
            sr for sr in self.sample_results 
            if sr.prediction.decision == "REVIEW"
        ]
    
    def format_report(self) -> str:
        """Format a human-readable evaluation report.
        
        Returns:
            Formatted string report
        """
        metrics = self.compute_metrics()
        
        lines = [
            "=" * 60,
            "SEMANTIC QA EVALUATION REPORT",
            "=" * 60,
            "",
            "## Overall Metrics",
            f"Total Samples: {metrics.total_samples}",
            f"Accuracy: {metrics.accuracy:.1%}",
            f"Review Rate: {metrics.review_rate:.1%}",
            "",
            "## Agreement Distribution",
            f"  3/3 agree: {metrics.agreement_3_3} ({metrics.agreement_3_3/metrics.total_samples:.1%})",
            f"  2/3 agree: {metrics.agreement_2_3} ({metrics.agreement_2_3/metrics.total_samples:.1%})",
            f"  1/3 agree: {metrics.agreement_1_3} ({metrics.agreement_1_3/metrics.total_samples:.1%})",
            "",
            "## Decision Distribution",
            f"  ACCEPT: {metrics.accept_count} ({metrics.accept_count/metrics.total_samples:.1%})",
            f"  REVIEW: {metrics.review_count} ({metrics.review_count/metrics.total_samples:.1%})",
            "",
            "## Accuracy by Decision",
            f"  When ACCEPT: {metrics.accuracy_when_accept:.1%}",
            f"  When REVIEW: {metrics.accuracy_when_review:.1%}",
            "",
            "## Per-Class Performance",
        ]
        
        for cls in VRU_CLASSES:
            if cls in metrics.per_class_accuracy:
                acc = metrics.per_class_accuracy[cls]
                count = metrics.per_class_count[cls]
                lines.append(f"  {cls}: {acc:.1%} (n={count})")
        
        lines.extend([
            "",
            "## Distance-Based Performance",
        ])
        
        for _, _, label in DISTANCE_BUCKETS:
            if label in metrics.accuracy_by_distance:
                acc = metrics.accuracy_by_distance[label]
                count = metrics.count_by_distance[label]
                lines.append(f"  {label}: {acc:.1%} (n={count})")
        
        lines.extend([
            "",
            "## Confusion Matrix",
        ])
        
        # Format confusion matrix
        header = "GT \\ Pred".ljust(15) + " ".join(
            cls[:8].ljust(10) for cls in VRU_CLASSES
        )
        lines.append(header)
        lines.append("-" * len(header))
        
        for gt_cls in VRU_CLASSES:
            row = gt_cls.ljust(15)
            for pred_cls in VRU_CLASSES:
                count = metrics.confusion_matrix.get(gt_cls, {}).get(pred_cls, 0)
                row += str(count).ljust(10) + " "
            lines.append(row)
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def format_confusion_matrix_table(self) -> str:
        """Format confusion matrix as a markdown table.
        
        Returns:
            Markdown-formatted confusion matrix
        """
        metrics = self.compute_metrics()
        
        lines = ["| GT \\ Pred |"]
        lines[0] += " | ".join(cls for cls in VRU_CLASSES) + " |"
        lines.append("|" + "|".join(["---"] * (len(VRU_CLASSES) + 1)) + "|")
        
        for gt_cls in VRU_CLASSES:
            row = f"| {gt_cls} |"
            for pred_cls in VRU_CLASSES:
                count = metrics.confusion_matrix.get(gt_cls, {}).get(pred_cls, 0)
                row += f" {count} |"
            lines.append(row)
        
        return "\n".join(lines)


def create_side_by_side_visualization(
    sample_results: List[SampleResult],
    output_dir: Path,
    max_examples: int = 8,
) -> List[Path]:
    """Create side-by-side visualizations of samples.
    
    Args:
        sample_results: List of sample results to visualize
        output_dir: Directory to save visualizations
        max_examples: Maximum number of examples to create
        
    Returns:
        List of paths to created images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    for i, sr in enumerate(sample_results[:max_examples]):
        if sr.sample.roi_image is None:
            continue
        
        # Create visualization image
        roi = sr.sample.roi_image
        pil_img = Image.fromarray(roi)
        
        # Add text annotation
        from PIL import ImageDraw
        
        # Create a larger canvas with space for text
        text_height = 100
        canvas_width = max(roi.shape[1], 400)
        canvas_height = roi.shape[0] + text_height
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
        
        # Paste ROI image
        x_offset = (canvas_width - roi.shape[1]) // 2
        canvas.paste(pil_img, (x_offset, 0))
        
        # Add text
        draw = ImageDraw.Draw(canvas)
        
        # Prepare text
        gt_class = sr.gt_class
        injected = sr.sample.injected_class
        pred_class = sr.pred_class
        agreement = sr.prediction.agreement
        decision = sr.prediction.decision
        is_correct = "✓" if sr.is_correct else "✗"
        
        text_lines = [
            f"GT: {gt_class}" + (f" | Injected: {injected}" if injected else ""),
            f"Pred: {pred_class} ({agreement}/3 agree) → {decision}",
            f"Result: {is_correct} {'Correct' if sr.is_correct else 'Incorrect'}",
        ]
        
        y_pos = roi.shape[0] + 10
        for line in text_lines:
            draw.text((10, y_pos), line, fill="black")
            y_pos += 25
        
        # Save
        output_path = output_dir / f"sample_{i+1:02d}.png"
        canvas.save(output_path)
        created_files.append(output_path)
        
        _LOGGER.info("Created visualization: %s", output_path)
    
    return created_files


def save_results_json(
    metrics: EvaluationResult,
    sample_results: List[SampleResult],
    output_path: Path,
) -> None:
    """Save evaluation results to JSON.
    
    Args:
        metrics: Computed metrics
        sample_results: Individual sample results
        output_path: Path to save JSON file
    """
    import json
    
    data = {
        "metrics": {
            "total_samples": metrics.total_samples,
            "accuracy": metrics.accuracy,
            "review_rate": metrics.review_rate,
            "agreement_distribution": {
                "3_of_3": metrics.agreement_3_3,
                "2_of_3": metrics.agreement_2_3,
                "1_of_3": metrics.agreement_1_3,
            },
            "decision_distribution": {
                "accept": metrics.accept_count,
                "review": metrics.review_count,
            },
            "accuracy_by_decision": {
                "when_accept": metrics.accuracy_when_accept,
                "when_review": metrics.accuracy_when_review,
            },
            "per_class_accuracy": metrics.per_class_accuracy,
            "accuracy_by_distance": metrics.accuracy_by_distance,
        },
        "samples": [
            {
                "annotation_token": sr.sample.annotation_token,
                "gt_class": sr.gt_class,
                "injected_class": sr.sample.injected_class,
                "pred_class": sr.pred_class,
                "agreement": sr.prediction.agreement,
                "decision": sr.prediction.decision,
                "is_correct": sr.is_correct,
                "distance": sr.sample.distance,
                "evidence": sr.prediction.evidence,
            }
            for sr in sample_results
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    _LOGGER.info("Saved results to %s", output_path)

