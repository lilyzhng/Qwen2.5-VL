"""Configuration for VLM-powered Labeling QA System.

Contains prompt templates, class mappings, and generation config.
"""

from typing import Dict, List, Final

# =============================================================================
# Class Definitions
# =============================================================================

SEMANTIC_CLASSES: Final[List[str]] = [
    "PEDESTRIAN",
    "CYCLIST",
    "MOTORCYCLIST",
    "MOTORCYCLE_ONLY",  # Motorcycle without rider
    "BICYCLE_ONLY",     # Bicycle without rider
    "REGULAR_VEHICLE",
    "TRUCK",
    "BUS",
]

# Vulnerable Road User classes - require stricter agreement thresholds
VRU_CLASSES: Final[List[str]] = [
    "PEDESTRIAN",
    "CYCLIST",
    "MOTORCYCLIST",
]

# nuScenes category to QA class mapping
NUSCENES_TO_QA_CLASS: Final[Dict[str, str]] = {
    # Pedestrians
    "human.pedestrian.adult": "PEDESTRIAN",
    "human.pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "human.pedestrian.personal_mobility": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
    "human.pedestrian.stroller": "PEDESTRIAN",
    "human.pedestrian.wheelchair": "PEDESTRIAN",
    # Cyclists
    "vehicle.bicycle": "CYCLIST",
    # Motorcyclists
    "vehicle.motorcycle": "MOTORCYCLIST",
    # Vehicles
    "vehicle.car": "REGULAR_VEHICLE",
    "vehicle.emergency.ambulance": "REGULAR_VEHICLE",
    "vehicle.emergency.police": "REGULAR_VEHICLE",
    "vehicle.construction": "REGULAR_VEHICLE",
    "vehicle.trailer": "REGULAR_VEHICLE",
    # Trucks
    "vehicle.truck": "TRUCK",
    # Buses
    "vehicle.bus.bendy": "BUS",
    "vehicle.bus.rigid": "BUS",
}

# =============================================================================
# Prompt Templates
# =============================================================================

# Forced choice classification prompt - classify first, then we compare to current_label in code
SEMANTIC_PROMPT: Final[str] = """Classify the TARGET object in this image.

Classes:
- PEDESTRIAN: person walking or standing
- CYCLIST: person riding a bicycle
- MOTORCYCLIST: person riding a motorcycle/scooter
- BICYCLE_ONLY: bicycle without rider
- MOTORCYCLE_ONLY: motorcycle without rider
- REGULAR_VEHICLE: car, sedan, SUV
- TRUCK: pickup truck, cargo vehicle
- BUS: passenger bus

Return JSON with:
- "class": exactly one class from the list above
- "evidence": 2-3 visual features supporting your choice

Example:
{{"class": "CYCLIST", "evidence": ["person on bicycle", "pedaling motion"]}}"""

# Visual anchor prefix - prepend to prompt when image has a TARGET box drawn
VISUAL_ANCHOR_PREFIX: Final[str] = """IMPORTANT: Classify ONLY the object inside the green TARGET box. Ignore everything outside the box.

"""

# Two-view prefix - prepend when providing both TARGET and CONTEXT images
TWO_VIEW_PREFIX: Final[str] = """You are given two images:
- IMAGE 1 (TARGET): Tight crop of the object - base your decision on this.
- IMAGE 2 (CONTEXT): Wider view - use only if IMAGE 1 is ambiguous.

Classify the object in IMAGE 1.

"""

# =============================================================================
# Generation Configuration
# =============================================================================

GENERATION_CONFIG: Final[Dict] = {
    # Self-consistency sampling config
    "num_samples": 3,          # Number of samples for self-consistency
    "temperature": 0.8,        # Enough variance to test consistency
    "do_sample": True,         # Required for temperature to take effect
    "top_p": 0.95,             # Nucleus sampling
    "max_new_tokens": 256,     # Max tokens to generate
}

# =============================================================================
# Decision Thresholds
# =============================================================================

def get_decision(agreement_count: int, predicted_class: str) -> str:
    """Map agreement level to decision.
    
    Args:
        agreement_count: Number of samples that agreed (1, 2, or 3)
        predicted_class: The predicted class from majority vote
        
    Returns:
        Decision string: "ACCEPT" or "REVIEW"
    """
    if agreement_count == 3:
        return "ACCEPT"  # High confidence - all samples agree
    elif agreement_count == 2:
        # For critical VRU classes, still send to review
        if predicted_class in VRU_CLASSES:
            return "REVIEW"
        return "ACCEPT"  # Non-critical can accept with 2/3
    else:
        return "REVIEW"  # No majority - human must decide

# =============================================================================
# nuScenes Data Paths
# =============================================================================

NUSCENES_MINI_PATH: Final[str] = "data/v1.0-mini"
NUSCENES_VERSION: Final[str] = "v1.0-mini"

# Camera names in nuScenes
CAMERA_NAMES: Final[List[str]] = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# =============================================================================
# Ghost Box Detection (Experiment 2)
# =============================================================================

GHOST_BOX_PROMPT: Final[str] = """You are checking whether a bounding box is correctly aligned with an object.

Question: Does this bounding box contain a complete, properly framed traffic participant?

Answer with JSON:
- "exists": ONE of {YES, NO, UNCERTAIN}
- "type": if YES, specify the object type (VEHICLE, PEDESTRIAN, CYCLIST, etc.)
- "evidence": 2-3 visual features supporting your choice

Guidelines:
- YES: The box clearly contains a complete, well-framed object
- NO: The box is empty or shows only background (road, sky, buildings)
- UNCERTAIN: The box is MOSTLY empty but shows partial object parts, OR you cannot confidently determine - FLAG FOR REVIEW
- Do NOT hallucinate objects

Examples:
{{"exists": "YES", "type": "VEHICLE", "evidence": ["complete car visible", "wheels and body clearly shown"]}}
{{"exists": "NO", "evidence": ["empty road surface", "no objects present"]}}
{{"exists": "UNCERTAIN", "evidence": ["mostly sky and overpass", "small partial object at edge", "insufficient content for confident labeling"]}}"""

# Ghost box shift strategies (in pixels in image space)
GHOST_BOX_SHIFTS: Final[List[Dict[str, float]]] = [
    {"dx": 0, "dy": -400, "name": "shift_up"},     # Sample 1: Shift 400px up
    {"dx": 0, "dy": 200, "name": "shift_down"},    # Sample 2: Shift 200px down
    {"dx": 0, "dy": -400, "name": "shift_up"},     # Sample 3: Shift 400px up
    {"dx": 350, "dy": 0, "name": "shift_right"},   # Sample 4+: Shift 350px right (if needed)
]

# =============================================================================
# Evaluation Configuration
# =============================================================================

# Distance buckets for range analysis (in meters)
DISTANCE_BUCKETS: Final[List[tuple]] = [
    (0, 20, "0-20m"),
    (20, 40, "20-40m"),
    (40, 60, "40-60m"),
    (60, 100, "60-100m"),
]

