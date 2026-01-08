#!/usr/bin/env python3
"""Visualize crops with GT labels and injected errors.

Creates an annotated grid showing:
- Green border: Clean samples (GT label shown)
- Red border: Corrupted samples (GT → Wrong label shown)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# Standard cell size for uniform grid
CELL_WIDTH = 300
CELL_HEIGHT = 300
TEXT_AREA_HEIGHT = 100
BORDER_WIDTH = 8


def create_annotated_crop(
    image_path: Path,
    gt_class: str,
    category_name: str,
    injected_class: str | None,
    distance: float,
    index: int,
    filename: str = "",
) -> Image.Image:
    """Create an annotated version of a crop with labels below.
    
    All crops are resized to a uniform size with text below.
    
    Args:
        image_path: Path to the crop image
        gt_class: Ground truth class
        category_name: Original nuScenes category
        injected_class: Injected wrong class (None if clean)
        distance: Distance from ego vehicle
        index: Sample index
        filename: Crop filename to display
        
    Returns:
        Annotated PIL Image of uniform size
    """
    # Load and resize crop to fit in cell
    img = Image.open(image_path)
    
    # Resize to fit within CELL_WIDTH x CELL_HEIGHT while maintaining aspect ratio
    img.thumbnail((CELL_WIDTH - 2 * BORDER_WIDTH, CELL_HEIGHT - 2 * BORDER_WIDTH), Image.Resampling.LANCZOS)
    
    # Determine if clean or corrupted
    is_corrupted = injected_class is not None
    border_color = (220, 53, 69) if is_corrupted else (40, 167, 69)  # Red or Green
    bg_color = (255, 240, 240) if is_corrupted else (240, 255, 240)  # Light red or light green
    
    # Create uniform canvas: image area + text area below
    total_height = CELL_HEIGHT + TEXT_AREA_HEIGHT
    canvas = Image.new("RGB", (CELL_WIDTH, total_height), color="white")
    
    # Fill image area with background color
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1], fill=bg_color)
    
    # Draw border around image area
    draw.rectangle(
        [0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1],
        outline=border_color,
        width=BORDER_WIDTH,
    )
    
    # Center the resized image in the cell
    x_offset = (CELL_WIDTH - img.width) // 2
    y_offset = (CELL_HEIGHT - img.height) // 2
    canvas.paste(img, (x_offset, y_offset))
    
    # Try to load a larger font, fall back to default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large
    
    # Text area starts below the image
    text_y = CELL_HEIGHT + 8
    
    # Line 1: Index and status (large, colored)
    if is_corrupted:
        line1 = f"#{index}: CORRUPTED"
        line1_color = (180, 30, 30)
    else:
        line1 = f"#{index}: CLEAN"
        line1_color = (30, 130, 30)
    draw.text((10, text_y), line1, fill=line1_color, font=font_large)
    
    # Line 2: GT class
    text_y += 24
    draw.text((10, text_y), f"GT: {gt_class}", fill=(0, 0, 0), font=font_medium)
    
    # Line 3: Wrong label (if corrupted) or category
    text_y += 20
    if is_corrupted:
        draw.text((10, text_y), f"Wrong: {injected_class}", fill=(180, 30, 30), font=font_medium)
    else:
        # Show shortened category
        short_cat = category_name.split(".")[-1]
        draw.text((10, text_y), f"({short_cat})", fill=(100, 100, 100), font=font_small)
    
    # Line 4: Distance and filename
    text_y += 20
    distance_text = f"Distance: {distance:.1f}m"
    if filename:
        distance_text += f"  |  {filename}"
    draw.text((10, text_y), distance_text, fill=(80, 80, 80), font=font_small)
    
    return canvas


def create_comparison_grid(
    crops_dir: Path,
    output_path: Path,
    cols: int = 5,
) -> None:
    """Create a grid visualization comparing all crops.
    
    All cells are uniform size for easy comparison.
    
    Args:
        crops_dir: Directory containing crops and metadata.json
        output_path: Where to save the grid image
        cols: Number of columns in grid
    """
    # Load metadata
    metadata_path = crops_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Create annotated crops
    annotated = []
    for item in metadata:
        image_path = crops_dir / item["filename"]
        ann_img = create_annotated_crop(
            image_path=image_path,
            gt_class=item["gt_class"],
            category_name=item["category_name"],
            injected_class=item["injected_class"],
            distance=item["distance"],
            index=item["index"],
            filename=item["filename"],
        )
        annotated.append(ann_img)
    
    # Calculate grid dimensions
    rows = (len(annotated) + cols - 1) // cols
    
    # Uniform cell size
    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT + TEXT_AREA_HEIGHT
    padding = 15
    
    # Create grid canvas
    grid_width = cols * cell_width + (cols + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), color=(250, 250, 250))
    
    # Place images
    for i, img in enumerate(annotated):
        row = i // cols
        col = i % cols
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        grid.paste(img, (x, y))
    
    # Save
    grid.save(output_path, quality=95)
    _LOGGER.info("Saved grid visualization to %s", output_path)


def create_separate_views(
    crops_dir: Path,
    output_dir: Path,
) -> None:
    """Create separate visualizations for clean vs corrupted samples.
    
    Args:
        crops_dir: Directory containing crops and metadata.json
        output_dir: Where to save the visualizations
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata_path = crops_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    clean_items = [m for m in metadata if m["injected_class"] is None]
    corrupted_items = [m for m in metadata if m["injected_class"] is not None]
    
    _LOGGER.info("Clean samples: %d, Corrupted samples: %d", 
                 len(clean_items), len(corrupted_items))
    
    # Create annotated images for each
    for item in metadata:
        image_path = crops_dir / item["filename"]
        ann_img = create_annotated_crop(
            image_path=image_path,
            gt_class=item["gt_class"],
            category_name=item["category_name"],
            injected_class=item["injected_class"],
            distance=item["distance"],
            index=item["index"],
            filename=item["filename"],
        )
        
        # Save with prefix based on status
        prefix = "corrupted" if item["injected_class"] else "clean"
        out_name = f"{prefix}_{item['filename']}"
        ann_img.save(output_dir / out_name)
    
    _LOGGER.info("Saved annotated crops to %s", output_dir)


def create_input_cell(
    image_path: Path,
    sample_data: dict,
    index: int,
    img_size: int = 300,
) -> Image.Image:
    """Create INPUT cell: image with clear label info.
    
    Shows:
        - Original GT class
        - If corrupted: what wrong label was injected
    
    Border color:
        - Green = clean (correct label given to VLM)
        - Red = corrupted (wrong label given to VLM)
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((img_size - 20, img_size - 20), Image.Resampling.LANCZOS)
    
    injected_class = sample_data.get("injected_class")
    is_corrupted = injected_class is not None
    border_color = (220, 53, 69) if is_corrupted else (40, 167, 69)
    bg_color = (255, 235, 235) if is_corrupted else (235, 255, 235)
    
    border_width = 6
    text_height = 90  # More space for clearer text
    total_height = img_size + text_height
    
    canvas = Image.new("RGB", (img_size, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Image area with border
    draw.rectangle([0, 0, img_size - 1, img_size - 1], fill=bg_color)
    draw.rectangle([0, 0, img_size - 1, img_size - 1], outline=border_color, width=border_width)
    
    # Center image
    img_x = (img_size - img.width) // 2
    img_y = (img_size - img.height) // 2
    canvas.paste(img, (img_x, img_y))
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_medium = font_title
        font_small = font_title
    
    # Sample number badge
    draw.rectangle([5, 5, 50, 32], fill=(50, 50, 50))
    draw.text((10, 8), f"#{index}", fill=(255, 255, 255), font=font_medium)
    
    # Get filename for later use
    filename = sample_data.get("filename", f"crop_{index-1:04d}.png")
    
    # Text below image - CLEARER LABELING
    y = img_size + 8
    gt_class = sample_data.get("gt_class", "?")
    
    # Line 1: Original GT class (always show)
    draw.text((8, y), f"Original: {gt_class}", fill=(0, 0, 0), font=font_title)
    y += 22
    
    if is_corrupted:
        # Line 2: What we corrupted it to
        draw.text((8, y), f"Corrupted to: {injected_class}", fill=(180, 30, 30), font=font_medium)
        y += 20
        draw.text((8, y), f"{filename}", fill=(100, 100, 100), font=font_small)
    else:
        # Line 2: Clean - correct label given
        draw.text((8, y), "Label: CLEAN (correct)", fill=(30, 130, 30), font=font_medium)
        y += 20
        draw.text((8, y), f"{filename}", fill=(100, 100, 100), font=font_small)
    
    return canvas


def create_output_cell(
    image_path: Path,
    sample_data: dict,
    index: int,
    img_size: int = 300,
) -> Image.Image:
    """Create OUTPUT cell: image with full VLM analysis.
    
    Shows:
        - VLM's prediction
        - Original GT class for comparison
        - Agreement score
        - Evidence (why VLM made this decision)
    
    Border color:
        - Green = VLM correct (prediction matches GT)
        - Red = VLM wrong (prediction doesn't match GT)
    """
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((img_size - 20, img_size - 20), Image.Resampling.LANCZOS)
    
    is_correct = sample_data.get("is_correct", False)
    border_color = (40, 167, 69) if is_correct else (220, 53, 69)
    bg_color = (235, 255, 235) if is_correct else (255, 235, 235)
    
    border_width = 6
    text_height = 150  # More space for VLM analysis
    total_height = img_size + text_height
    
    canvas = Image.new("RGB", (img_size, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Image area with border
    draw.rectangle([0, 0, img_size - 1, img_size - 1], fill=bg_color)
    draw.rectangle([0, 0, img_size - 1, img_size - 1], outline=border_color, width=border_width)
    
    # Center image
    img_x = (img_size - img.width) // 2
    img_y = (img_size - img.height) // 2
    canvas.paste(img, (img_x, img_y))
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_medium = font_title
        font_small = font_title
    
    # Sample number badge
    draw.rectangle([5, 5, 50, 32], fill=(50, 50, 50))
    draw.text((10, 8), f"#{index}", fill=(255, 255, 255), font=font_medium)
    
    # Get filename for later use
    filename = sample_data.get("filename", f"crop_{index-1:04d}.png")
    
    # Text below image - CLEARER VLM ANALYSIS
    y = img_size + 8
    
    pred_class = sample_data.get("pred_class", "?")
    agreement = sample_data.get("agreement", 0)
    decision = sample_data.get("decision", "?")
    gt_class = sample_data.get("gt_class", "?")
    injected_class = sample_data.get("injected_class")
    
    # Line 1: VLM Class Prediction (the actual class VLM chose)
    draw.text((8, y), f"VLM triage: {pred_class}", fill=(0, 0, 0), font=font_title)
    y += 22
    
    # Line 2: Comparison with GT
    if is_correct:
        draw.text((8, y), f"✓ Matches GT: {gt_class}", fill=(30, 130, 30), font=font_medium)
    else:
        draw.text((8, y), f"✗ GT was: {gt_class}", fill=(180, 30, 30), font=font_medium)
    y += 20
    
    # Line 3: Was this a correction case? (only show for corrupted samples)
    if injected_class:
        if pred_class == gt_class:
            draw.text((8, y), f"Fixed error! (was: {injected_class})", fill=(30, 130, 30), font=font_small)
        else:
            draw.text((8, y), f"Failed to fix (was: {injected_class})", fill=(180, 30, 30), font=font_small)
        y += 18
    elif not is_correct:
        # Only show error message for clean samples that were wrongly changed
        draw.text((8, y), "Wrongly changed correct label!", fill=(180, 30, 30), font=font_small)
        y += 18
    # Skip "Confirmed correct label" line for clean samples that are correct
    
    # Line 4: Agreement, Decision, and Filename
    # Decision = ACCEPT (high confidence) or REVIEW (needs human check)
    decision_color = (30, 130, 30) if decision == "ACCEPT" else (180, 120, 0)
    draw.text((8, y), f"Agreement: {agreement}/3 → {decision}  |  {filename}", fill=decision_color, font=font_small)
    y += 16
    
    # Evidence (full display with word wrapping - use full image width)
    evidence = sample_data.get("evidence", [])
    if evidence:
        draw.text((8, y), "Evidence:", fill=(100, 100, 100), font=font_small)
        y += 14
        ev_text = ", ".join(evidence)
        # Use wider text - approximately match image width (~5 pixels per char for small font)
        max_chars = max(70, img_size // 5)
        # Split into multiple lines as needed
        lines = []
        while ev_text:
            if len(ev_text) <= max_chars:
                lines.append(ev_text)
                break
            # Find a good break point (space)
            break_idx = ev_text.rfind(' ', 0, max_chars)
            if break_idx == -1:
                break_idx = max_chars
            lines.append(ev_text[:break_idx])
            ev_text = ev_text[break_idx:].lstrip()
        
        for line in lines[:4]:  # Show up to 4 lines of evidence
            draw.text((8, y), line, fill=(120, 120, 120), font=font_small)
            y += 14
    
    return canvas


def create_two_row_grid(
    crops_dir: Path,
    results_path: Path,
    output_dir: Path,
    cols: int = 3,
    img_size: int = 300,
) -> list[Path]:
    """Create 2-row grid: TOP = inputs, BOTTOM = VLM outputs with analysis.
    
    Layout (3 columns example):
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ INPUT 1 │  │ INPUT 2 │  │ INPUT 3 │
    │ (label) │  │ (label) │  │ (label) │
    └─────────┘  └─────────┘  └─────────┘
    
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │OUTPUT 1 │  │OUTPUT 2 │  │OUTPUT 3 │
    │VLM info │  │VLM info │  │VLM info │
    │evidence │  │evidence │  │evidence │
    └─────────┘  └─────────┘  └─────────┘
    
    Args:
        crops_dir: Directory containing crop images
        results_path: Path to results.json
        output_dir: Where to save images
        cols: Number of columns (samples per page)
        img_size: Size for each image
        
    Returns:
        List of output paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path) as f:
        results = json.load(f)
    
    samples = results.get("samples", [])
    if not samples:
        _LOGGER.error("No samples found in results.json")
        return []
    
    metadata_path = crops_dir / "metadata.json"
    with open(metadata_path) as f:
        crops_meta = json.load(f)
    
    # Determine if this run has corrupted samples
    has_corrupted = any(item.get("injected_class") for item in crops_meta)
    run_type = "corrupted" if has_corrupted else "clean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create input and output cells
    input_cells = []
    output_cells = []
    
    for i, (sample, crop_info) in enumerate(zip(samples, crops_meta)):
        image_path = crops_dir / crop_info["filename"]
        # Merge crop_info into sample so filename is available
        sample_with_meta = {**sample, "filename": crop_info["filename"]}
        input_cells.append(create_input_cell(image_path, sample_with_meta, i + 1, img_size))
        output_cells.append(create_output_cell(image_path, sample_with_meta, i + 1, img_size))
    
    # Split into pages
    output_paths = []
    num_pages = (len(samples) + cols - 1) // cols
    
    input_text_height = 90   # Matches create_input_cell
    output_text_height = 150  # Matches create_output_cell
    padding = 20
    row_gap = 30
    
    for page_idx in range(num_pages):
        start = page_idx * cols
        end = min(start + cols, len(samples))
        page_inputs = input_cells[start:end]
        page_outputs = output_cells[start:end]
        
        # Calculate page size
        actual_cols = len(page_inputs)
        page_width = actual_cols * img_size + (actual_cols + 1) * padding
        page_height = (
            padding +
            (img_size + input_text_height) +  # Top row
            row_gap +
            (img_size + output_text_height) +  # Bottom row
            padding
        )
        
        page = Image.new("RGB", (page_width, page_height), color=(248, 248, 248))
        draw = ImageDraw.Draw(page)
        
        # Load font for row labels
        try:
            font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except (OSError, IOError):
            font_label = ImageFont.load_default()
        
        # Place INPUT row (top)
        y_input = padding
        for i, cell in enumerate(page_inputs):
            x = padding + i * (img_size + padding)
            page.paste(cell, (x, y_input))
        
        # Arrow between rows
        arrow_y = padding + (img_size + input_text_height) + row_gap // 2
        for i in range(actual_cols):
            arrow_x = padding + i * (img_size + padding) + img_size // 2
            draw.polygon([
                (arrow_x - 8, arrow_y - 8),
                (arrow_x + 8, arrow_y - 8),
                (arrow_x, arrow_y + 8),
            ], fill=(120, 120, 120))
        
        # Place OUTPUT row (bottom)
        y_output = padding + (img_size + input_text_height) + row_gap
        for i, cell in enumerate(page_outputs):
            x = padding + i * (img_size + padding)
            page.paste(cell, (x, y_output))
        
        # Save page with timestamp and run type (clean/corrupted)
        if num_pages == 1:
            page_path = output_dir / f"comparison_grid_{run_type}_{timestamp}.png"
        else:
            page_path = output_dir / f"comparison_page_{page_idx + 1:02d}_{run_type}_{timestamp}.png"
        
        page.save(page_path, quality=95)
        output_paths.append(page_path)
        _LOGGER.info("Saved comparison grid to %s", page_path)
    
    return output_paths


def create_side_by_side_pair(
    image_path: Path,
    sample_data: dict,
    index: int,
    img_size: int = 400,
) -> Image.Image:
    """Create a side-by-side comparison: INPUT (left) vs VLM OUTPUT (right).
    
    Left side: Input with label shown to VLM
        - Green border = clean (correct label)
        - Red border = corrupted (wrong injected label)
    
    Right side: VLM result
        - Green border = VLM got it right
        - Red border = VLM got it wrong
    
    Args:
        image_path: Path to the crop image
        sample_data: Dict with gt_class, pred_class, is_correct, evidence, etc.
        index: Sample index
        img_size: Size for each image panel
        
    Returns:
        Side-by-side PIL Image
    """
    # Load and resize crop
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((img_size - 20, img_size - 20), Image.Resampling.LANCZOS)
    
    # Determine input status (clean vs corrupted)
    injected_class = sample_data.get("injected_class")
    is_corrupted = injected_class is not None
    input_border = (220, 53, 69) if is_corrupted else (40, 167, 69)  # Red if corrupted, Green if clean
    input_bg = (255, 235, 235) if is_corrupted else (235, 255, 235)
    
    # Determine output status (VLM correct vs wrong)
    is_correct = sample_data.get("is_correct", False)
    output_border = (40, 167, 69) if is_correct else (220, 53, 69)  # Green if correct, Red if wrong
    output_bg = (235, 255, 235) if is_correct else (255, 235, 235)
    
    # Layout constants
    border_width = 6
    text_height = 120
    arrow_width = 60
    panel_width = img_size
    panel_height = img_size
    total_width = panel_width * 2 + arrow_width
    total_height = panel_height + text_height
    
    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_large = font_title
        font_medium = font_title
        font_small = font_title
    
    # === LEFT PANEL: INPUT ===
    left_x = 0
    draw.rectangle([left_x, 0, left_x + panel_width - 1, panel_height - 1], fill=input_bg)
    draw.rectangle([left_x, 0, left_x + panel_width - 1, panel_height - 1], 
                   outline=input_border, width=border_width)
    
    # Center image in left panel
    img_x = left_x + (panel_width - img.width) // 2
    img_y = (panel_height - img.height) // 2
    canvas.paste(img, (img_x, img_y))
    
    # Input label text (what the VLM was told)
    gt_class = sample_data.get("gt_class", "?")
    if is_corrupted:
        input_label = f"Label: {injected_class}"
        label_color = (180, 30, 30)
    else:
        input_label = f"Label: {gt_class}"
        label_color = (30, 130, 30)
    
    # Draw "INPUT" header
    draw.text((left_x + 10, panel_height + 8), "INPUT", fill=(80, 80, 80), font=font_title)
    draw.text((left_x + 10, panel_height + 35), input_label, fill=label_color, font=font_large)
    draw.text((left_x + 10, panel_height + 58), f"(GT: {gt_class})", fill=(100, 100, 100), font=font_medium)
    
    status_text = "CORRUPTED" if is_corrupted else "CLEAN"
    draw.text((left_x + 10, panel_height + 80), status_text, fill=label_color, font=font_medium)
    
    # === ARROW ===
    arrow_x = panel_width + arrow_width // 2
    arrow_y = panel_height // 2
    # Draw arrow
    draw.polygon([
        (arrow_x - 15, arrow_y - 10),
        (arrow_x + 15, arrow_y),
        (arrow_x - 15, arrow_y + 10),
    ], fill=(100, 100, 100))
    draw.text((panel_width + 10, arrow_y + 20), "VLM", fill=(100, 100, 100), font=font_medium)
    
    # === RIGHT PANEL: VLM OUTPUT ===
    right_x = panel_width + arrow_width
    draw.rectangle([right_x, 0, right_x + panel_width - 1, panel_height - 1], fill=output_bg)
    draw.rectangle([right_x, 0, right_x + panel_width - 1, panel_height - 1], 
                   outline=output_border, width=border_width)
    
    # Center image in right panel
    img_x = right_x + (panel_width - img.width) // 2
    canvas.paste(img, (img_x, img_y))
    
    # Output text
    pred_class = sample_data.get("pred_class", "?")
    agreement = sample_data.get("agreement", 0)
    
    if is_correct:
        result_text = "✓ CORRECT"
        result_color = (30, 130, 30)
    else:
        result_text = "✗ WRONG"
        result_color = (180, 30, 30)
    
    draw.text((right_x + 10, panel_height + 8), "VLM OUTPUT", fill=(80, 80, 80), font=font_title)
    draw.text((right_x + 10, panel_height + 35), f"Prediction: {pred_class}", fill=result_color, font=font_large)
    draw.text((right_x + 10, panel_height + 58), f"Agreement: {agreement}/3", fill=(100, 100, 100), font=font_medium)
    draw.text((right_x + 10, panel_height + 80), result_text, fill=result_color, font=font_medium)
    
    # Sample number in top-left corner
    draw.rectangle([5, 5, 45, 30], fill=(50, 50, 50))
    draw.text((10, 7), f"#{index}", fill=(255, 255, 255), font=font_medium)
    
    return canvas


def create_side_by_side_grid(
    crops_dir: Path,
    results_path: Path,
    output_dir: Path,
    samples_per_page: int = 4,
    img_size: int = 350,
) -> list[Path]:
    """Create side-by-side comparison grids with configurable samples per page.
    
    Args:
        crops_dir: Directory containing crop images
        results_path: Path to results.json from VLM experiment
        output_dir: Where to save the grid images
        samples_per_page: Number of samples per visualization (default 4)
        img_size: Size for each image panel
        
    Returns:
        List of paths to generated images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VLM results
    with open(results_path) as f:
        results = json.load(f)
    
    samples = results.get("samples", [])
    if not samples:
        _LOGGER.error("No samples found in results.json")
        return []
    
    # Load crop metadata to get filenames
    metadata_path = crops_dir / "metadata.json"
    with open(metadata_path) as f:
        crops_meta = json.load(f)
    
    # Create pairs
    pairs = []
    for i, (sample, crop_info) in enumerate(zip(samples, crops_meta)):
        image_path = crops_dir / crop_info["filename"]
        pair = create_side_by_side_pair(image_path, sample, i + 1, img_size=img_size)
        pairs.append(pair)
    
    # Split into pages
    output_paths = []
    num_pages = (len(pairs) + samples_per_page - 1) // samples_per_page
    
    pair_width = img_size * 2 + 60  # Two panels + arrow
    pair_height = img_size + 120   # Panel + text
    padding = 20
    
    for page_idx in range(num_pages):
        start = page_idx * samples_per_page
        end = min(start + samples_per_page, len(pairs))
        page_pairs = pairs[start:end]
        
        # Stack vertically
        page_width = pair_width + 2 * padding
        page_height = len(page_pairs) * (pair_height + padding) + padding
        
        page = Image.new("RGB", (page_width, page_height), color=(248, 248, 248))
        
        for i, pair in enumerate(page_pairs):
            y = padding + i * (pair_height + padding)
            page.paste(pair, (padding, y))
        
        # Save page
        if num_pages == 1:
            page_path = output_dir / "side_by_side_comparison.png"
        else:
            page_path = output_dir / f"side_by_side_page_{page_idx + 1:02d}.png"
        
        page.save(page_path, quality=95)
        output_paths.append(page_path)
        _LOGGER.info("Saved comparison page to %s", page_path)
    
    return output_paths


def create_vlm_analysis_cell(
    image_path: Path,
    sample_data: dict,
    index: int,
) -> Image.Image:
    """Create a cell showing input image + VLM QA results.
    
    Args:
        image_path: Path to the crop image
        sample_data: Dict with gt_class, pred_class, is_correct, evidence, etc.
        index: Sample index
        
    Returns:
        Annotated PIL Image
    """
    # Load and resize crop
    img = Image.open(image_path)
    img.thumbnail((CELL_WIDTH - 2 * BORDER_WIDTH, CELL_HEIGHT - 2 * BORDER_WIDTH), Image.Resampling.LANCZOS)
    
    # Determine colors based on correctness
    is_correct = sample_data.get("is_correct", False)
    border_color = (40, 167, 69) if is_correct else (220, 53, 69)  # Green or Red
    bg_color = (240, 255, 240) if is_correct else (255, 240, 240)
    
    # More text space for VLM analysis
    text_height = 160
    total_height = CELL_HEIGHT + text_height
    canvas = Image.new("RGB", (CELL_WIDTH, total_height), color="white")
    
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1], fill=bg_color)
    draw.rectangle([0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1], outline=border_color, width=BORDER_WIDTH)
    
    # Center image
    x_offset = (CELL_WIDTH - img.width) // 2
    y_offset = (CELL_HEIGHT - img.height) // 2
    canvas.paste(img, (x_offset, y_offset))
    
    # Load fonts
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large
    
    # Text area
    y = CELL_HEIGHT + 6
    
    # Line 1: Status
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    status_color = (30, 130, 30) if is_correct else (180, 30, 30)
    draw.text((8, y), f"#{index}: {status}", fill=status_color, font=font_large)
    y += 20
    
    # Line 2: GT class
    gt_class = sample_data.get("gt_class", "?")
    draw.text((8, y), f"GT: {gt_class}", fill=(0, 0, 0), font=font_medium)
    y += 18
    
    # Line 3: Injected error (if any)
    injected = sample_data.get("injected_class")
    if injected:
        draw.text((8, y), f"Injected: {injected}", fill=(180, 80, 0), font=font_medium)
    else:
        draw.text((8, y), "(clean - no injection)", fill=(120, 120, 120), font=font_small)
    y += 18
    
    # Line 4: VLM Prediction
    pred_class = sample_data.get("pred_class", "?")
    pred_color = (30, 130, 30) if is_correct else (180, 30, 30)
    draw.text((8, y), f"VLM: {pred_class}", fill=pred_color, font=font_medium)
    
    # Agreement on same line
    agreement = sample_data.get("agreement", 0)
    draw.text((150, y), f"({agreement}/3 agree)", fill=(100, 100, 100), font=font_small)
    y += 18
    
    # Line 5: Distance
    distance = sample_data.get("distance", 0)
    draw.text((8, y), f"Distance: {distance:.1f}m", fill=(80, 80, 80), font=font_small)
    y += 16
    
    # Line 6-7: Evidence (wrapped)
    evidence = sample_data.get("evidence", [])
    if evidence:
        ev_text = ", ".join(evidence)
        # Wrap long text
        if len(ev_text) > 45:
            ev_line1 = ev_text[:45]
            ev_line2 = ev_text[45:90] + ("..." if len(ev_text) > 90 else "")
            draw.text((8, y), ev_line1, fill=(100, 100, 100), font=font_small)
            y += 14
            draw.text((8, y), ev_line2, fill=(100, 100, 100), font=font_small)
        else:
            draw.text((8, y), ev_text, fill=(100, 100, 100), font=font_small)
    
    return canvas


def create_vlm_analysis_grid(
    crops_dir: Path,
    results_path: Path,
    output_path: Path,
    cols: int = 5,
) -> None:
    """Create a grid showing VLM QA analysis for all samples.
    
    Args:
        crops_dir: Directory containing crop images
        results_path: Path to results.json from VLM experiment
        output_path: Where to save the grid image
        cols: Number of columns
    """
    # Load VLM results
    with open(results_path) as f:
        results = json.load(f)
    
    samples = results.get("samples", [])
    if not samples:
        _LOGGER.error("No samples found in results.json")
        return
    
    # Load crop metadata to get filenames
    metadata_path = crops_dir / "metadata.json"
    with open(metadata_path) as f:
        crops_meta = json.load(f)
    
    # Create cells
    cells = []
    for i, (sample, crop_info) in enumerate(zip(samples, crops_meta)):
        image_path = crops_dir / crop_info["filename"]
        cell = create_vlm_analysis_cell(image_path, sample, i + 1)
        cells.append(cell)
    
    # Calculate grid dimensions
    rows = (len(cells) + cols - 1) // cols
    cell_width = CELL_WIDTH
    cell_height = CELL_HEIGHT + 160  # Match text height in create_vlm_analysis_cell
    padding = 12
    
    grid_width = cols * cell_width + (cols + 1) * padding
    grid_height = rows * cell_height + (rows + 1) * padding
    grid = Image.new("RGB", (grid_width, grid_height), color=(248, 248, 248))
    
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)
        grid.paste(cell, (x, y))
    
    grid.save(output_path, quality=95)
    _LOGGER.info("Saved VLM analysis grid to %s", output_path)


def draw_dashed_rectangle(draw, bbox, color, dash_length=15, width=4):
    """Draw a dashed rectangle on an ImageDraw object.
    
    Args:
        draw: PIL ImageDraw object
        bbox: Tuple of (x1, y1, x2, y2)
        color: RGB tuple for line color
        dash_length: Length of each dash
        width: Line width
    """
    x1, y1, x2, y2 = bbox
    
    # Top edge
    x = x1
    while x < x2:
        end_x = min(x + dash_length, x2)
        draw.line([(x, y1), (end_x, y1)], fill=color, width=width)
        x += dash_length * 2
    
    # Bottom edge
    x = x1
    while x < x2:
        end_x = min(x + dash_length, x2)
        draw.line([(x, y2), (end_x, y2)], fill=color, width=width)
        x += dash_length * 2
    
    # Left edge
    y = y1
    while y < y2:
        end_y = min(y + dash_length, y2)
        draw.line([(x1, y), (x1, end_y)], fill=color, width=width)
        y += dash_length * 2
    
    # Right edge
    y = y1
    while y < y2:
        end_y = min(y + dash_length, y2)
        draw.line([(x2, y), (x2, end_y)], fill=color, width=width)
        y += dash_length * 2


def visualize_ghost_boxes(
    ghost_crops_dir: Path,
    output_path: Path,
) -> None:
    """Create two-row visualization: top=original crops, bottom=ghost box crops.
    
    Shows the actual cropped regions side-by-side:
    - Top row: Crops of original bounding boxes (showing real objects)
    - Bottom row: Crops of ghost boxes (showing misaligned regions)
    
    Args:
        ghost_crops_dir: Directory containing ghost box crops and metadata
        output_path: Where to save the visualization
    """
    # Load ghost box metadata
    metadata_files = list(ghost_crops_dir.glob('ghost_metadata_*.json'))
    if not metadata_files:
        _LOGGER.error("No ghost metadata found in %s", ghost_crops_dir)
        return
    
    metadata_path = max(metadata_files, key=lambda p: p.stat().st_mtime)
    with open(metadata_path) as f:
        ghost_metadata = json.load(f)
    
    if not ghost_metadata:
        _LOGGER.error("Ghost metadata is empty")
        return
    
    # Load fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = font_title
        font_small = font_title
    
    # Standard cell size
    cell_width = 300
    cell_height = 300
    text_height = 80
    border_width = 6
    
    # Colors
    original_color = (40, 167, 69)  # Green for original
    ghost_color = (220, 53, 69)     # Red for ghost boxes
    
    # Process each ghost sample
    top_cells = []
    bottom_cells = []
    
    for i, item in enumerate(ghost_metadata):
        # Load the ghost crop image from disk (already saved)
        filename = item['filename']
        ghost_crop_path = ghost_crops_dir / filename
        ghost_img = Image.open(ghost_crop_path)
        ghost_crop = np.array(ghost_img)
        
        # Get original bbox and crop from the full image
        bbox_2d_original = tuple(item['bbox_2d_original'])
        
        # Need to load the full image to get original crop
        from .data_prep import NuScenesDataLoader
        data_root = ghost_crops_dir.parent.parent / "v1.0-mini"
        if i == 0:  # Only load once
            loader = NuScenesDataLoader(data_root=str(data_root))
        
        sample_token = item['sample_token']
        camera_name = item['camera_name']
        img_path = loader.get_camera_image_path(sample_token, camera_name)
        full_img = Image.open(img_path)
        img_array = np.array(full_img)
        
        # Crop original bbox region
        x1, y1, x2, y2 = bbox_2d_original
        original_crop = img_array[y1:y2, x1:x2]
        
        # Get ghost bbox for size info
        ghost_bbox = tuple(item['bbox_2d'])
        shift_type = item['shift_type']
        original_gt_class = item['original_gt_class']
        
        # Create TOP cell (original)
        top_cell = Image.new('RGB', (cell_width, cell_height + text_height), 'white')
        draw_top = ImageDraw.Draw(top_cell)
        
        # Resize and paste original crop
        orig_img = Image.fromarray(original_crop)
        orig_img.thumbnail((cell_width - 2*border_width, cell_height - 2*border_width), Image.Resampling.LANCZOS)
        
        # Background and border
        draw_top.rectangle([0, 0, cell_width-1, cell_height-1], fill=(240, 255, 240))
        draw_top.rectangle([0, 0, cell_width-1, cell_height-1], outline=original_color, width=border_width)
        
        # Center image
        img_x = (cell_width - orig_img.width) // 2
        img_y = (cell_height - orig_img.height) // 2
        top_cell.paste(orig_img, (img_x, img_y))
        
        # Text below
        y = cell_height + 10
        draw_top.text((10, y), f"#{i+1} ORIGINAL", fill=original_color, font=font_label)
        y += 22
        draw_top.text((10, y), f"Class: {original_gt_class}", fill=(60, 60, 60), font=font_small)
        y += 18
        draw_top.text((10, y), f"Size: {x2-x1}×{y2-y1}px", fill=(100, 100, 100), font=font_small)
        
        top_cells.append(top_cell)
        
        # Create BOTTOM cell (ghost)
        bottom_cell = Image.new('RGB', (cell_width, cell_height + text_height), 'white')
        draw_bottom = ImageDraw.Draw(bottom_cell)
        
        # Resize and paste ghost crop
        ghost_img_pil = Image.fromarray(ghost_crop)
        ghost_img_pil.thumbnail((cell_width - 2*border_width, cell_height - 2*border_width), Image.Resampling.LANCZOS)
        
        # Background and border
        draw_bottom.rectangle([0, 0, cell_width-1, cell_height-1], fill=(255, 240, 240))
        draw_bottom.rectangle([0, 0, cell_width-1, cell_height-1], outline=ghost_color, width=border_width)
        
        # Center image
        img_x = (cell_width - ghost_img_pil.width) // 2
        img_y = (cell_height - ghost_img_pil.height) // 2
        bottom_cell.paste(ghost_img_pil, (img_x, img_y))
        
        # Text below
        y = cell_height + 10
        draw_bottom.text((10, y), f"#{i+1} GHOST", fill=ghost_color, font=font_label)
        y += 22
        draw_bottom.text((10, y), f"Shift: {shift_type}", fill=(60, 60, 60), font=font_small)
        y += 18
        x1_g, y1_g, x2_g, y2_g = ghost_bbox
        draw_bottom.text((10, y), f"Size: {x2_g-x1_g}×{y2_g-y1_g}px", fill=(100, 100, 100), font=font_small)
        
        bottom_cells.append(bottom_cell)
    
    if not top_cells:
        _LOGGER.error("No valid ghost samples to visualize")
        return
    
    # Create two-row grid
    padding = 20
    title_height = 50
    
    num_cols = len(top_cells)
    cell_total_height = cell_height + text_height
    grid_width = num_cols * cell_width + (num_cols + 1) * padding
    grid_height = title_height + cell_total_height + padding + title_height + cell_total_height + padding * 2
    
    grid = Image.new('RGB', (grid_width, grid_height), color=(250, 250, 250))
    draw_grid = ImageDraw.Draw(grid)
    
    # Title for top row
    title_top = 'TOP ROW: Original Crops (Correct Labels) - Green'
    text_bbox = draw_grid.textbbox((0, 0), title_top, font=font_title)
    text_x = (grid_width - (text_bbox[2] - text_bbox[0])) // 2
    draw_grid.text((text_x, padding), title_top, fill='black', font=font_title)
    
    # Paste top row
    y_top = padding + title_height
    for i, cell in enumerate(top_cells):
        x = padding + i * (cell_width + padding)
        grid.paste(cell, (x, y_top))
    
    # Title for bottom row
    title_bottom = 'BOTTOM ROW: Ghost Box Crops (Misaligned) - Red'
    y_title_bottom = y_top + cell_total_height + padding
    text_bbox = draw_grid.textbbox((0, 0), title_bottom, font=font_title)
    text_x = (grid_width - (text_bbox[2] - text_bbox[0])) // 2
    draw_grid.text((text_x, y_title_bottom), title_bottom, fill='black', font=font_title)
    
    # Paste bottom row
    y_bottom = y_title_bottom + title_height
    for i, cell in enumerate(bottom_cells):
        x = padding + i * (cell_width + padding)
        grid.paste(cell, (x, y_bottom))
    
    # Save
    grid.save(output_path, quality=95)
    _LOGGER.info("Saved ghost box crop visualization to %s", output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize crops with labels")
    parser.add_argument(
        "--crops-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "qa_results" / "crops",
        help="Directory containing crops and metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to crops_dir/annotated)",
    )
    parser.add_argument(
        "--mode",
        choices=["input", "vlm-analysis", "side-by-side", "two-row", "both", "visualize-ghost"],
        default="both",
        help="Visualization mode: 'input', 'vlm-analysis', 'side-by-side', 'two-row' (top=inputs, bottom=outputs), 'both', 'visualize-ghost' (ghost box comparison)",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Path to results.json (for vlm-analysis and side-by-side modes)",
    )
    parser.add_argument(
        "--samples-per-page",
        type=int,
        default=4,
        help="Number of samples per page in side-by-side mode (default: 4)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=350,
        help="Image panel size in side-by-side mode (default: 350)",
    )
    parser.add_argument(
        "--ghost-crops-dir",
        type=Path,
        default=None,
        help="Directory containing ghost box crops (for visualize-ghost mode)",
    )
    parser.add_argument(
        "--full-image",
        type=Path,
        default=None,
        help="Path to full original image (for visualize-ghost mode)",
    )
    parser.add_argument(
        "--original-bbox",
        type=str,
        default=None,
        help="Original bounding box as 'x1,y1,x2,y2' (for visualize-ghost mode)",
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.crops_dir / "annotated"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default results path
    if args.results_json is None:
        args.results_json = args.crops_dir.parent / "results.json"
    
    if args.mode in ["input", "both"]:
        # Create input visualization (original crops with GT/injection info)
        create_separate_views(args.crops_dir, args.output_dir)
        
        # Determine run type and timestamp for filename
        metadata_path = args.crops_dir / "metadata.json"
        with open(metadata_path) as f:
            crops_meta = json.load(f)
        has_corrupted = any(item.get("injected_class") for item in crops_meta)
        run_type = "corrupted" if has_corrupted else "clean"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        grid_path = args.output_dir / f"all_crops_grid_{run_type}_{timestamp}.png"
        create_comparison_grid(args.crops_dir, grid_path, cols=5)
        print(f"Input visualization: {grid_path}")
    
    if args.mode in ["vlm-analysis", "both"]:
        # Create VLM analysis visualization
        if not args.results_json.exists():
            print(f"Warning: results.json not found at {args.results_json}")
            print("Run the experiment first to generate VLM results.")
        else:
            vlm_grid_path = args.output_dir / "vlm_analysis_grid.png"
            create_vlm_analysis_grid(
                args.crops_dir, args.results_json, vlm_grid_path, cols=5
            )
            print(f"VLM analysis visualization: {vlm_grid_path}")
    
    if args.mode == "side-by-side":
        # Create side-by-side comparison (INPUT → VLM OUTPUT)
        if not args.results_json.exists():
            print(f"Error: results.json not found at {args.results_json}")
            print("Run the experiment first to generate VLM results.")
            return
        
        output_paths = create_side_by_side_grid(
            args.crops_dir,
            args.results_json,
            args.output_dir,
            samples_per_page=args.samples_per_page,
            img_size=args.img_size,
        )
        print(f"\nSide-by-side comparison saved:")
        for path in output_paths:
            print(f"  {path}")
    
    if args.mode == "two-row":
        # Create 2-row grid: top=inputs, bottom=outputs with VLM analysis
        if not args.results_json.exists():
            print(f"Error: results.json not found at {args.results_json}")
            print("Run the experiment first to generate VLM results.")
            return
        
        output_paths = create_two_row_grid(
            args.crops_dir,
            args.results_json,
            args.output_dir,
            cols=args.samples_per_page,  # samples_per_page becomes columns
            img_size=args.img_size,
        )
        print(f"\nTwo-row comparison saved:")
        for path in output_paths:
            print(f"  {path}")
    
    if args.mode == "visualize-ghost":
        # Create ghost box comparison visualization
        if args.ghost_crops_dir is None:
            args.ghost_crops_dir = args.crops_dir.parent / "ghost_crops"
        
        if not args.ghost_crops_dir.exists():
            print(f"Error: Ghost crops directory not found at {args.ghost_crops_dir}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / f"ghost_comparison_{timestamp}.png"
        
        visualize_ghost_boxes(
            args.ghost_crops_dir,
            output_path,
        )
        print(f"\nGhost box visualization saved to: {output_path}")
    
    print(f"\nVisualization complete!")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

