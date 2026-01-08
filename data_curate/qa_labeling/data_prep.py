"""nuScenes data loader and synthetic error injection for QA experiments.

Uses the official nuscenes-devkit for proper coordinate transformations
and camera projections.

This module handles:
1. Loading nuScenes annotations via nuscenes-devkit
2. Projecting 3D boxes to 2D camera coordinates (using devkit utilities)
3. Cropping ROI images around annotations
4. Injecting synthetic labeling errors for evaluation
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Final

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, view_points

from .config import NUSCENES_TO_QA_CLASS, VRU_CLASSES, GHOST_BOX_SHIFTS

_LOGGER: Final = logging.getLogger(__name__)


@dataclass
class Annotation3D:
    """A 3D annotation from nuScenes."""
    
    token: str
    sample_token: str
    instance_token: str
    category_name: str
    qa_class: str  # Mapped from category_name
    box: Box  # nuScenes Box object with all transformations
    visibility: int
    num_lidar_pts: int
    distance: float = 0.0  # Distance from ego vehicle


@dataclass
class ROISample:
    """A cropped ROI sample for VLM evaluation."""
    
    annotation_token: str
    sample_token: str
    camera_name: str
    image_path: Path
    roi_image: Optional[np.ndarray] = None  # Cropped region
    bbox_2d: Optional[Tuple[int, int, int, int]] = None  # [x1, y1, x2, y2] padded crop coords
    bbox_2d_tight: Optional[Tuple[int, int, int, int]] = None  # [x1, y1, x2, y2] tight 3D projection
    gt_class: str = ""  # Ground truth class (mapped, e.g., PEDESTRIAN)
    category_name: str = ""  # Original nuScenes category (e.g., human.pedestrian.adult)
    injected_class: Optional[str] = None  # Synthetic error class (if any)
    distance: float = 0.0  # Distance from ego
    is_ghost: bool = False  # True if this is a synthetic ghost box (empty region)
    ghost_shift_type: Optional[str] = None  # Type of shift applied (e.g., "shift_right")


@dataclass
class GhostBoxSample:
    """A ghost box sample - a shifted box that should show empty/background."""
    
    original_annotation_token: str
    sample_token: str
    camera_name: str
    image_path: Path
    roi_image: Optional[np.ndarray] = None
    bbox_2d: Optional[Tuple[int, int, int, int]] = None  # Shifted bbox
    bbox_2d_original: Optional[Tuple[int, int, int, int]] = None  # Original bbox before shift
    shift_type: str = ""  # e.g., "shift_right", "shift_left"
    shift_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (dx, dy, dz) in pixels
    original_gt_class: str = ""  # What the real object was
    distance: float = 0.0


@dataclass 
class NuScenesDataLoader:
    """Loader for nuScenes dataset using official devkit."""
    
    data_root: str
    version: str = "v1.0-mini"
    verbose: bool = False
    
    # NuScenes instance
    nusc: NuScenes = field(init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize NuScenes devkit."""
        _LOGGER.info("Loading NuScenes %s from %s", self.version, self.data_root)
        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.data_root,
            verbose=self.verbose,
        )
        _LOGGER.info(
            "Loaded %d samples, %d annotations",
            len(self.nusc.sample),
            len(self.nusc.sample_annotation),
        )
    
    def get_box_in_sensor_frame(
        self,
        sample_data_token: str,
        annotation_token: str,
    ) -> Optional[Box]:
        """Get annotation box in sensor (camera) coordinate frame.
        
        Args:
            sample_data_token: Token for the camera sample_data
            annotation_token: Token for the annotation
            
        Returns:
            Box in sensor frame, or None if not valid
        """
        # Get the annotation box in global frame
        ann = self.nusc.get("sample_annotation", annotation_token)
        box = Box(
            ann["translation"],
            ann["size"],
            Quaternion(ann["rotation"]),
        )
        
        # Get sensor and ego pose
        sd = self.nusc.get("sample_data", sample_data_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        
        # Transform: global -> ego -> sensor
        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(Quaternion(cs["rotation"]).inverse)
        
        return box
    
    def get_annotations_for_sample(
        self, sample_token: str, camera_name: str = "CAM_FRONT"
    ) -> List[Annotation3D]:
        """Get 3D annotations visible in a specific camera.
        
        Args:
            sample_token: The sample token
            camera_name: Which camera to check visibility for
            
        Returns:
            List of Annotation3D objects
        """
        sample = self.nusc.get("sample", sample_token)
        
        # Get camera sample_data
        cam_token = sample["data"][camera_name]
        cam_data = self.nusc.get("sample_data", cam_token)
        cs = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        
        # Get camera intrinsics
        camera_intrinsic = np.array(cs["camera_intrinsic"])
        
        # Get image dimensions
        img_path = Path(self.nusc.dataroot) / cam_data["filename"]
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        annotations = []
        
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            
            # Get category
            category_name = ann["category_name"]
            qa_class = NUSCENES_TO_QA_CLASS.get(category_name)
            if not qa_class:
                continue  # Skip unmapped categories
            
            # Get visibility
            visibility = int(ann["visibility_token"])
            
            # Get box in sensor frame
            box = self.get_box_in_sensor_frame(cam_token, ann_token)
            if box is None:
                continue
            
            # Check if box is in front of camera
            if box.center[2] <= 0:
                continue  # Behind camera
            
            # Check if box is visible in image using devkit utility
            if not box_in_image(box, camera_intrinsic, (img_width, img_height), vis_level=1):
                continue
            
            # Distance from ego (in sensor frame, depth is z)
            distance = np.linalg.norm(box.center)
            
            annotations.append(Annotation3D(
                token=ann_token,
                sample_token=sample_token,
                instance_token=ann["instance_token"],
                category_name=category_name,
                qa_class=qa_class,
                box=box,
                visibility=visibility,
                num_lidar_pts=ann["num_lidar_pts"],
                distance=distance,
            ))
        
        return annotations
    
    def project_box_to_image(
        self,
        box: Box,
        camera_intrinsic: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """Project 3D box corners to 2D and get bounding box.
        
        Args:
            box: Box in sensor (camera) frame
            camera_intrinsic: 3x3 camera intrinsic matrix
            
        Returns:
            (x1, y1, x2, y2) bounding box in pixels
        """
        # Get 8 corners of the box
        corners_3d = box.corners()  # (3, 8) array
        
        # Project to image using devkit utility
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
        
        # Get bounding box
        x1 = int(np.min(corners_2d[0, :]))
        y1 = int(np.min(corners_2d[1, :]))
        x2 = int(np.max(corners_2d[0, :]))
        y2 = int(np.max(corners_2d[1, :]))
        
        return (x1, y1, x2, y2)
    
    def get_all_sample_tokens(self) -> List[str]:
        """Get all sample tokens."""
        return [s["token"] for s in self.nusc.sample]
    
    def get_camera_image_path(self, sample_token: str, camera_name: str) -> Path:
        """Get path to camera image for a sample."""
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][camera_name]
        cam_data = self.nusc.get("sample_data", cam_token)
        return Path(self.nusc.dataroot) / cam_data["filename"]
    
    def get_camera_intrinsic(self, sample_token: str, camera_name: str) -> np.ndarray:
        """Get camera intrinsic matrix for a sample."""
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][camera_name]
        cam_data = self.nusc.get("sample_data", cam_token)
        cs = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        return np.array(cs["camera_intrinsic"])


def crop_roi_from_image(
    image: np.ndarray,
    bbox_2d: Tuple[int, int, int, int],
    padding: float = 0.3,
    min_padding_px: int = 50,
    min_crop_size: int = 100,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop ROI from image with padding.
    
    Adds padding around the bounding box to give context. Uses both
    percentage-based and minimum pixel padding to ensure narrow objects
    get enough surrounding context.
    
    Args:
        image: Input image (H, W, C)
        bbox_2d: Bounding box [x1, y1, x2, y2]
        padding: Padding ratio around the box (default 30%)
        min_padding_px: Minimum padding in pixels on each side (default 50px)
        min_crop_size: Minimum crop dimension in pixels (default 100px)
        
    Returns:
        Cropped image and adjusted bbox
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_2d
    
    # Calculate padding - use max of percentage and minimum pixels
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_w = max(int(box_w * padding), min_padding_px)
    pad_h = max(int(box_h * padding), min_padding_px)
    
    # Apply padding
    x1_padded = max(0, x1 - pad_w)
    y1_padded = max(0, y1 - pad_h)
    x2_padded = min(w, x2 + pad_w)
    y2_padded = min(h, y2 + pad_h)
    
    # Ensure minimum crop size
    crop_w = x2_padded - x1_padded
    crop_h = y2_padded - y1_padded
    
    if crop_w < min_crop_size:
        extra = (min_crop_size - crop_w) // 2
        x1_padded = max(0, x1_padded - extra)
        x2_padded = min(w, x2_padded + extra)
    
    if crop_h < min_crop_size:
        extra = (min_crop_size - crop_h) // 2
        y1_padded = max(0, y1_padded - extra)
        y2_padded = min(h, y2_padded + extra)
    
    cropped = image[y1_padded:y2_padded, x1_padded:x2_padded]
    return cropped, (x1_padded, y1_padded, x2_padded, y2_padded)


def crop_tight_from_image(
    image: np.ndarray,
    bbox_2d: Tuple[int, int, int, int],
    margin: float = 0.15,
) -> np.ndarray:
    """Crop tightly around the 3D-projected bounding box with minimal margin.
    
    This is the preferred approach for VLM inference as it reduces
    distractor objects that might confuse the model.
    
    Args:
        image: Input image (H, W, C)
        bbox_2d: Tight bounding box from 3D projection [x1, y1, x2, y2]
        margin: Small margin ratio (default 15%)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_2d
    
    # Add small margin
    box_w = x2 - x1
    box_h = y2 - y1
    margin_w = int(box_w * margin)
    margin_h = int(box_h * margin)
    
    x1_m = max(0, x1 - margin_w)
    y1_m = max(0, y1 - margin_h)
    x2_m = min(w, x2 + margin_w)
    y2_m = min(h, y2 + margin_h)
    
    return image[y1_m:y2_m, x1_m:x2_m]


def needs_context_view(
    bbox_2d: Tuple[int, int, int, int],
    min_dimension: int = 40,
    max_aspect_ratio: float = 4.0,
) -> bool:
    """Check if a tight crop needs a context view for disambiguation.
    
    Returns True if the bbox is:
    - Too small (either dimension < min_dimension)
    - Too thin/tall (aspect ratio > max_aspect_ratio)
    
    Args:
        bbox_2d: Tight bounding box [x1, y1, x2, y2]
        min_dimension: Minimum acceptable dimension in pixels
        max_aspect_ratio: Maximum acceptable aspect ratio (width/height or height/width)
        
    Returns:
        True if context view is needed
    """
    x1, y1, x2, y2 = bbox_2d
    w = x2 - x1
    h = y2 - y1
    
    # Too small
    if w < min_dimension or h < min_dimension:
        return True
    
    # Extreme aspect ratio
    aspect = max(w / h, h / w) if min(w, h) > 0 else float('inf')
    if aspect > max_aspect_ratio:
        return True
    
    return False


def create_two_view_crops(
    image: np.ndarray,
    bbox_2d_tight: Tuple[int, int, int, int],
    target_margin: float = 0.15,
    context_scale: float = 2.5,
    max_context_size: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create two-view crops: tight TARGET view and larger CONTEXT view.
    
    View A (TARGET): Tight crop with small margin for decision-making
    View B (CONTEXT): Same center, larger area for disambiguation
    
    Args:
        image: Full camera image (H, W, C)
        bbox_2d_tight: Tight bounding box from 3D projection [x1, y1, x2, y2]
        target_margin: Margin ratio for target view (default 15%)
        context_scale: How much larger context view should be (default 2.5x area)
        max_context_size: Maximum dimension for context view (cap)
        
    Returns:
        (target_crop, context_crop)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_2d_tight
    
    # Box dimensions and center
    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # === View A: TARGET (tight + small margin) ===
    margin_w = int(box_w * target_margin)
    margin_h = int(box_h * target_margin)
    
    target_x1 = max(0, x1 - margin_w)
    target_y1 = max(0, y1 - margin_h)
    target_x2 = min(w, x2 + margin_w)
    target_y2 = min(h, y2 + margin_h)
    
    target_crop = image[target_y1:target_y2, target_x1:target_x2]
    
    # === View B: CONTEXT (same center, larger area) ===
    # Scale up dimensions by sqrt(context_scale) to get context_scale times the area
    scale_factor = np.sqrt(context_scale)
    context_half_w = int((box_w * scale_factor) / 2)
    context_half_h = int((box_h * scale_factor) / 2)
    
    # Cap to max_context_size
    context_half_w = min(context_half_w, max_context_size // 2)
    context_half_h = min(context_half_h, max_context_size // 2)
    
    context_x1 = max(0, cx - context_half_w)
    context_y1 = max(0, cy - context_half_h)
    context_x2 = min(w, cx + context_half_w)
    context_y2 = min(h, cy + context_half_h)
    
    context_crop = image[context_y1:context_y2, context_x1:context_x2]
    
    return target_crop, context_crop


def create_masked_crop(
    image: np.ndarray,
    bbox_2d: Tuple[int, int, int, int],
    padding: float = 0.3,
    min_padding_px: int = 50,
    mask_mode: str = "darken",
    darken_factor: float = 0.2,
    blur_radius: int = 25,
) -> np.ndarray:
    """Create a crop with everything outside the target box masked/blurred.
    
    This forces VLM attention on the target object by reducing visual
    saliency of distractors outside the bounding box.
    
    Args:
        image: Input image (H, W, C)
        bbox_2d: Target bounding box [x1, y1, x2, y2]
        padding: Padding ratio for the crop (default 30%)
        min_padding_px: Minimum padding in pixels
        mask_mode: How to mask outside - "darken", "blur", or "black"
        darken_factor: How much to darken (0=black, 1=original) for "darken" mode
        blur_radius: Blur kernel size for "blur" mode
        
    Returns:
        Masked/cropped image with target emphasized
    """
    from PIL import ImageFilter
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_2d
    
    # Calculate padded crop region
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = max(int(box_w * padding), min_padding_px)
    pad_h = max(int(box_h * padding), min_padding_px)
    
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(w, x2 + pad_w)
    crop_y2 = min(h, y2 + pad_h)
    
    # Crop the region first
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_h, crop_w = cropped.shape[:2]
    
    # Calculate target box position within the crop
    target_x1 = x1 - crop_x1
    target_y1 = y1 - crop_y1
    target_x2 = x2 - crop_x1
    target_y2 = y2 - crop_y1
    
    # Clamp to crop bounds
    target_x1 = max(0, target_x1)
    target_y1 = max(0, target_y1)
    target_x2 = min(crop_w, target_x2)
    target_y2 = min(crop_h, target_y2)
    
    # Create mask (True = keep original, False = mask)
    mask = np.zeros((crop_h, crop_w), dtype=bool)
    mask[target_y1:target_y2, target_x1:target_x2] = True
    
    if mask_mode == "black":
        # Black out everything outside
        result = np.zeros_like(cropped)
        result[mask] = cropped[mask]
        
    elif mask_mode == "darken":
        # Darken pixels outside the box
        result = cropped.copy()
        result[~mask] = (cropped[~mask] * darken_factor).astype(np.uint8)
        
    elif mask_mode == "blur":
        # Blur pixels outside the box
        pil_img = Image.fromarray(cropped)
        blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_arr = np.array(blurred)
        result = blurred_arr.copy()
        result[mask] = cropped[mask]
        
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    
    # Optionally draw a subtle border around the target
    # (helps VLM understand what "target" means)
    border_color = [0, 255, 0]  # Green
    border_width = 3
    
    # Top border
    result[target_y1:target_y1+border_width, target_x1:target_x2] = border_color
    # Bottom border
    result[target_y2-border_width:target_y2, target_x1:target_x2] = border_color
    # Left border
    result[target_y1:target_y2, target_x1:target_x1+border_width] = border_color
    # Right border
    result[target_y1:target_y2, target_x2-border_width:target_x2] = border_color
    
    return result


@dataclass
class SyntheticErrorInjector:
    """Inject synthetic labeling errors for evaluation."""
    
    # Error injection rates
    error_rate: float = 0.5  # What fraction of samples to corrupt
    seed: int = 42
    
    # Confusion pairs for VRU classes (realistic confusions)
    # - Motorcyclist ↔ Cyclist (both two-wheeled with rider)
    # - Cyclist ↔ Pedestrian (both human-powered, upright posture)
    # - Pedestrian ↔ Cyclist (similar silhouette at distance)
    confusion_pairs: Dict[str, List[str]] = field(default_factory=lambda: {
        "PEDESTRIAN": ["CYCLIST"],           # Pedestrian confused with cyclist
        "CYCLIST": ["PEDESTRIAN", "MOTORCYCLIST"],  # Cyclist confused with either
        "MOTORCYCLIST": ["CYCLIST"],         # Motorcyclist confused with cyclist
    })
    
    def __post_init__(self) -> None:
        """Initialize random state."""
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def inject_errors(
        self, samples: List[ROISample]
    ) -> List[ROISample]:
        """Inject synthetic errors into samples.
        
        Args:
            samples: List of ROI samples with GT labels
            
        Returns:
            Samples with some labels corrupted (injected_class set)
        """
        n_errors = int(len(samples) * self.error_rate)
        error_indices = random.sample(range(len(samples)), min(n_errors, len(samples)))
        
        for idx in error_indices:
            sample = samples[idx]
            gt_class = sample.gt_class
            
            if gt_class in self.confusion_pairs:
                # Pick a random confusion class
                wrong_class = random.choice(self.confusion_pairs[gt_class])
                sample.injected_class = wrong_class
        
        n_injected = sum(1 for s in samples if s.injected_class is not None)
        _LOGGER.info(
            "Injected %d synthetic errors into %d samples",
            n_injected,
            len(samples),
        )
        
        return samples


def prepare_roi_samples(
    loader: NuScenesDataLoader,
    max_samples: int = 100,
    min_visibility: int = 2,
    min_lidar_pts: int = 5,
    max_distance: float = 60.0,
    camera_name: str = "CAM_FRONT",
    balance_classes: bool = False,
) -> List[ROISample]:
    """Prepare ROI samples from nuScenes for VLM evaluation.
    
    Args:
        loader: NuScenes data loader
        max_samples: Maximum number of samples to prepare
        min_visibility: Minimum visibility level (1-4)
        min_lidar_pts: Minimum LiDAR points for valid annotation
        max_distance: Maximum distance from ego vehicle
        camera_name: Which camera to use
        balance_classes: If True, try to get equal samples per VRU class
        
    Returns:
        List of ROI samples ready for VLM evaluation
    """
    # If balancing, collect all samples first then subsample
    if balance_classes:
        return _prepare_balanced_samples(
            loader, max_samples, min_visibility, min_lidar_pts, 
            max_distance, camera_name
        )
    
    samples = []
    
    for sample_token in loader.get_all_sample_tokens():
        if len(samples) >= max_samples:
            break
        
        # Get camera intrinsic
        try:
            camera_intrinsic = loader.get_camera_intrinsic(sample_token, camera_name)
        except Exception as e:
            _LOGGER.warning("Failed to get camera intrinsic: %s", e)
            continue
        
        # Get image path
        image_path = loader.get_camera_image_path(sample_token, camera_name)
        if not image_path.exists():
            continue
        
        # Load image
        with Image.open(image_path) as img:
            img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Get annotations visible in this camera
        annotations = loader.get_annotations_for_sample(sample_token, camera_name)
        
        for ann in annotations:
            if len(samples) >= max_samples:
                break
            
            # Filter by VRU class
            if ann.qa_class not in VRU_CLASSES:
                continue
            
            # Filter by quality
            if ann.visibility < min_visibility:
                continue
            if ann.num_lidar_pts < min_lidar_pts:
                continue
            if ann.distance > max_distance:
                continue
            
            # Project box to image
            try:
                bbox_2d = loader.project_box_to_image(ann.box, camera_intrinsic)
            except Exception as e:
                _LOGGER.warning("Failed to project box: %s", e)
                continue
            
            # Check bounds
            x1, y1, x2, y2 = bbox_2d
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bbox_2d = (x1, y1, x2, y2)
            
            # Skip if too small
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            
            # Save tight bbox from 3D projection before padding
            bbox_2d_tight = bbox_2d
            
            # Crop ROI
            roi_image, adjusted_bbox = crop_roi_from_image(img_array, bbox_2d)
            
            # Skip if too small after padding
            if roi_image.shape[0] < 32 or roi_image.shape[1] < 32:
                continue
            
            samples.append(ROISample(
                annotation_token=ann.token,
                sample_token=sample_token,
                camera_name=camera_name,
                image_path=image_path,
                roi_image=roi_image,
                bbox_2d=adjusted_bbox,
                bbox_2d_tight=bbox_2d_tight,
                gt_class=ann.qa_class,
                category_name=ann.category_name,
                distance=ann.distance,
            ))
    
    _LOGGER.info("Prepared %d ROI samples", len(samples))
    return samples


def _prepare_balanced_samples(
    loader: NuScenesDataLoader,
    max_samples: int,
    min_visibility: int,
    min_lidar_pts: int,
    max_distance: float,
    camera_name: str,
) -> List[ROISample]:
    """Prepare balanced ROI samples across VRU classes.
    
    Collects all valid samples first, then subsamples to get equal
    representation across PEDESTRIAN, CYCLIST, and MOTORCYCLIST.
    """
    # Collect all valid samples by class
    samples_by_class: Dict[str, List[ROISample]] = {
        cls: [] for cls in VRU_CLASSES
    }
    
    for sample_token in loader.get_all_sample_tokens():
        # Get camera intrinsic
        try:
            camera_intrinsic = loader.get_camera_intrinsic(sample_token, camera_name)
        except Exception:
            continue
        
        # Get image path
        image_path = loader.get_camera_image_path(sample_token, camera_name)
        if not image_path.exists():
            continue
        
        # Load image
        with Image.open(image_path) as img:
            img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Get annotations visible in this camera
        annotations = loader.get_annotations_for_sample(sample_token, camera_name)
        
        for ann in annotations:
            # Filter by VRU class
            if ann.qa_class not in VRU_CLASSES:
                continue
            
            # Filter by quality
            if ann.visibility < min_visibility:
                continue
            if ann.num_lidar_pts < min_lidar_pts:
                continue
            if ann.distance > max_distance:
                continue
            
            # Project box to image
            try:
                bbox_2d = loader.project_box_to_image(ann.box, camera_intrinsic)
            except Exception:
                continue
            
            # Check bounds
            x1, y1, x2, y2 = bbox_2d
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bbox_2d = (x1, y1, x2, y2)
            
            # Skip if too small
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            
            # Save tight bbox from 3D projection before padding
            bbox_2d_tight = bbox_2d
            
            # Crop ROI
            roi_image, adjusted_bbox = crop_roi_from_image(img_array, bbox_2d)
            
            if roi_image.shape[0] < 32 or roi_image.shape[1] < 32:
                continue
            
            samples_by_class[ann.qa_class].append(ROISample(
                annotation_token=ann.token,
                sample_token=sample_token,
                camera_name=camera_name,
                image_path=image_path,
                roi_image=roi_image,
                bbox_2d=adjusted_bbox,
                bbox_2d_tight=bbox_2d_tight,
                gt_class=ann.qa_class,
                category_name=ann.category_name,
                distance=ann.distance,
            ))
    
    # Log what we found
    for cls, samples in samples_by_class.items():
        _LOGGER.info("Found %d %s samples", len(samples), cls)
    
    # Balance: take equal samples from each class
    samples_per_class = max_samples // len(VRU_CLASSES)
    balanced_samples = []
    
    for cls in VRU_CLASSES:
        available = samples_by_class[cls]
        n_take = min(len(available), samples_per_class)
        # Shuffle and take
        random.shuffle(available)
        balanced_samples.extend(available[:n_take])
        _LOGGER.info("Selected %d/%d %s samples", n_take, len(available), cls)
    
    # Shuffle final list
    random.shuffle(balanced_samples)
    
    _LOGGER.info("Prepared %d balanced ROI samples", len(balanced_samples))
    return balanced_samples


def create_ghost_box_from_annotation(
    loader: NuScenesDataLoader,
    annotation_token: str,
    sample_token: str,
    camera_name: str,
    shift_config: Dict[str, float],
) -> Optional[GhostBoxSample]:
    """Create a ghost box by shifting an annotation's 2D bounding box in pixel space.
    
    Projects the 3D box to 2D once, then shifts the 2D coordinates directly.
    This maintains the exact box size while changing location.
    
    Args:
        loader: NuScenes data loader
        annotation_token: Token of the annotation to shift
        sample_token: Sample token
        camera_name: Camera to project to
        shift_config: Dict with 'dx', 'dy' (pixels) and 'name'
        
    Returns:
        GhostBoxSample if successful, None if projection fails
    """
    # Get the annotation
    ann_record = loader.nusc.get("sample_annotation", annotation_token)
    
    # Get the original 3D box
    box = Box(
        ann_record["translation"],
        ann_record["size"],
        Quaternion(ann_record["rotation"]),
    )
    
    # Get camera data
    sample = loader.nusc.get("sample", sample_token)
    camera_token = sample["data"][camera_name]
    
    # Transform box to camera frame
    sd = loader.nusc.get("sample_data", camera_token)
    cs = loader.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego_pose = loader.nusc.get("ego_pose", sd["ego_pose_token"])
    
    # Transform: global -> ego -> sensor
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    box.translate(-np.array(cs["translation"]))
    box.rotate(Quaternion(cs["rotation"]).inverse)
    
    # Get camera intrinsic
    camera_intrinsic = loader.get_camera_intrinsic(sample_token, camera_name)
    
    # Check if box is in image
    if not box_in_image(box, camera_intrinsic, (1600, 900)):
        return None
    
    # Project original box to 2D ONCE
    try:
        original_bbox_2d = loader.project_box_to_image(box, camera_intrinsic)
    except Exception as e:
        _LOGGER.warning("Failed to project box: %s", e)
        return None
    
    x1, y1, x2, y2 = original_bbox_2d
    
    # Apply pixel shift to the 2D bbox (keep same width/height)
    dx_pixels = int(shift_config["dx"])
    dy_pixels = int(shift_config["dy"])
    
    shifted_bbox_2d = (
        x1 + dx_pixels,
        y1 + dy_pixels,
        x2 + dx_pixels,
        y2 + dy_pixels,
    )
    
    # Load image
    image_path = loader.get_camera_image_path(sample_token, camera_name)
    with Image.open(image_path) as img:
        img_array = np.array(img)
    
    h, w = img_array.shape[:2]
    xs1, ys1, xs2, ys2 = shifted_bbox_2d
    
    # Check if shifted box is still mostly in image bounds
    if xs2 < 0 or xs1 > w or ys2 < 0 or ys1 > h:
        # Completely out of bounds
        return None
    
    # Clip to image bounds if partially out
    xs1_clipped = max(0, xs1)
    ys1_clipped = max(0, ys1)
    xs2_clipped = min(w, xs2)
    ys2_clipped = min(h, ys2)
    
    # Check if clipping removed too much of the box
    original_width = xs2 - xs1
    original_height = ys2 - ys1
    clipped_width = xs2_clipped - xs1_clipped
    clipped_height = ys2_clipped - ys1_clipped
    
    # If more than 50% is clipped, skip
    if clipped_width < original_width * 0.5 or clipped_height < original_height * 0.5:
        return None
    
    shifted_bbox_2d_clipped = (xs1_clipped, ys1_clipped, xs2_clipped, ys2_clipped)
    
    # Skip if too small after clipping
    if (xs2_clipped - xs1_clipped) < 20 or (ys2_clipped - ys1_clipped) < 20:
        return None
    
    # Crop ROI from the shifted location (NO PADDING - we want exact same size)
    roi_image = img_array[ys1_clipped:ys2_clipped, xs1_clipped:xs2_clipped]
    
    if roi_image.shape[0] < 32 or roi_image.shape[1] < 32:
        return None
    
    # Get original GT class
    original_qa_class = NUSCENES_TO_QA_CLASS.get(ann_record["category_name"], "UNKNOWN")
    
    return GhostBoxSample(
        original_annotation_token=annotation_token,
        sample_token=sample_token,
        camera_name=camera_name,
        image_path=image_path,
        roi_image=roi_image,
        bbox_2d=shifted_bbox_2d_clipped,  # The shifted bbox (same size as original)
        bbox_2d_original=original_bbox_2d,  # Store original for visualization
        shift_type=shift_config["name"],
        shift_vector=(dx_pixels, dy_pixels, 0),
        original_gt_class=original_qa_class,
        distance=0.0,
    )


def prepare_ghost_box_samples(
    loader: NuScenesDataLoader,
    num_samples: int = 10,
    min_visibility: int = 2,
    min_lidar_pts: int = 5,
    max_distance: float = 60.0,
    camera_name: str = "CAM_FRONT",
) -> List[GhostBoxSample]:
    """Prepare ghost box samples by shifting real annotations.
    
    For each sample, picks ONE random shift type to create a unique ghost box.
    This simulates misaligned bounding boxes during labeling.
    
    Args:
        loader: NuScenes data loader
        num_samples: Number of ghost box samples to create
        min_visibility: Minimum visibility for source annotations
        min_lidar_pts: Minimum LiDAR points for source annotations
        max_distance: Maximum distance for source annotations
        camera_name: Which camera to use
        
    Returns:
        List of ghost box samples
    """
    ghost_samples = []
    attempts = 0
    max_attempts = num_samples * 50  # Allow many attempts
    
    for sample_token in loader.get_all_sample_tokens():
        if len(ghost_samples) >= num_samples:
            break
        if attempts >= max_attempts:
            break
        
        # Get annotations for this sample
        annotations = loader.get_annotations_for_sample(sample_token, camera_name)
        
        for ann in annotations:
            if len(ghost_samples) >= num_samples:
                break
            if attempts >= max_attempts:
                break
            
            # Filter by quality
            if ann.visibility < min_visibility:
                continue
            if ann.num_lidar_pts < min_lidar_pts:
                continue
            if ann.distance > max_distance:
                continue
            
            # Use shift types in order: shift_up, shift_down, shift_up, shift_down, ...
            # This gives us predictable shifts for each sample
            shift_index = len(ghost_samples) % len(GHOST_BOX_SHIFTS)
            shift_config = GHOST_BOX_SHIFTS[shift_index]
            
            attempts += 1
            
            # Create ghost box
            ghost_sample = create_ghost_box_from_annotation(
                loader,
                ann.token,
                sample_token,
                camera_name,
                shift_config,
            )
            
            if ghost_sample is not None:
                ghost_samples.append(ghost_sample)
                _LOGGER.info(
                    "Created ghost box %d/%d (shift=%s, original_class=%s)",
                    len(ghost_samples),
                    num_samples,
                    ghost_sample.shift_type,
                    ghost_sample.original_gt_class,
                )
    
    _LOGGER.info(
        "Prepared %d ghost box samples (attempts=%d)",
        len(ghost_samples),
        attempts,
    )
    return ghost_samples

