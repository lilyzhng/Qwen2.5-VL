import os
import glob
from qwen_vl_utils import process_vision_info
import re
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from IPython.display import Markdown, display

# Get all image files from the samples folder (all camera directories)
samples_folder = "/workspace/Qwen2.5-VL/nuimages-v1.0-mini/samples"
image_files = []

# # Recursively find all .jpg files in all camera subdirectories
# for camera_dir in os.listdir(samples_folder):
    # camera_path = os.path.join(samples_folder, camera_dir)

def analyze_driving_frames(samples_folder, text_prompt, processor, model, max_new_tokens=128):
    """
    Analyze driving frames from a folder using batch inference.
    
    Args:
        samples_folder (str): Path to the folder containing camera subdirectories with images
        text_prompt (str): Text prompt for analysis
        processor: The processor object for handling inputs
        model: The model object for inference
        max_new_tokens (int): Maximum number of new tokens to generate
    
    Returns:
        list: List of output texts from the model
    """
    # Get all image files from the samples folder (all camera directories)
    image_files = []
    
    # Recursively find all .jpg files in all camera subdirectories
    # for camera_dir in os.listdir(samples_folder):
    #     camera_path = os.path.join(samples_folder, camera_dir)
    camera_path = "/workspace/Qwen2.5-VL/nuimages-v1.0-mini/samples/CAM_FRONT"
    if os.path.isdir(camera_path):
        jpg_files = glob.glob(os.path.join(camera_path, "*.jpg"))
        image_files.extend(jpg_files)
    
    # Sort files for consistent ordering
    image_files.sort()
    image_files = image_files[:2]
    
    if not image_files:
        print(f"No image files found in {samples_folder}")
        return []
    
    print(f"Found {len(image_files)} images across all camera directories")
    
    # Create content list with all images
    content = []
    for image_file in image_files:
        content.append({"type": "image", "image": f"file://{image_file}"})
    
    # Add the text prompt
    content.append({
        "type": "text", 
        "text": text_prompt
    })
    
    # Create messages for batch processing
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # Preparation for batch inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_texts




def extract_json_from_response(response):
    """
    Extract JSON data from model response text.
    
    Args:
        response (str): The response text containing JSON
    
    Returns:
        list: Extracted JSON data as Python objects
    """
    json_data = []
    
    # Look for JSON blocks in the response (prioritize code blocks)
    json_patterns = [
        r'```json\s*(\[.*?\])\s*```',  # JSON arrays in code blocks
        r'```json\s*(\{.*?\})\s*```',  # JSON objects in code blocks
        r'(\[(?:[^[\]]*|\[[^\]]*\])*\])',  # JSON arrays (more precise)
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                # Try to parse as JSON
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    # Filter out non-dict items (like strings, numbers)
                    valid_items = [item for item in parsed if isinstance(item, dict)]
                    json_data.extend(valid_items)
                elif isinstance(parsed, dict):
                    json_data.append(parsed)
                return json_data  # Return immediately after first successful parse
            except json.JSONDecodeError:
                continue
    
    return json_data


def visualize_scene_complexity(json_data, base_width=6, base_height=5):
    """
    Display images sorted by scene complexity score with score overlay.
    
    Args:
        json_data (list): List of dictionaries containing frame analysis
        base_width (int): Base width per column
        base_height (int): Base height per row
    """
    if not json_data:
        print("No data to visualize")
        return
    
    # Sort by scene_complexity_score (highest first)
    sorted_data = sorted(json_data, key=lambda x: x['scene_complexity_score'], reverse=True)
    
    # Calculate grid dimensions (fixed 3 columns)
    n_images = len(sorted_data)
    cols = 2
    rows = (n_images + cols - 1) // cols
    
    # Calculate dynamic figure size based on number of rows
    figsize = (cols * base_width, rows * base_height)
    
    # Create figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle different cases for axes
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten() if n_images > 1 else axes
    
    for i, data in enumerate(sorted_data):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        
        # Load and display image
        frame_path = data['frame_path']
        if os.path.exists(frame_path):
            try:
                img = Image.open(frame_path)
                ax.imshow(img)
                
                # Add score overlay at top right
                score = data['scene_complexity_score']
                
                # Create score box
                bbox_props = dict(boxstyle="round,pad=0.3", 
                                facecolor='red' if score >= 4 else 'orange' if score >= 3 else 'yellow' if score >= 2 else 'green',
                                alpha=0.8)
                
                ax.text(0.95, 0.95, f"Score: {score}", 
                       transform=ax.transAxes, 
                       fontsize=14, 
                       fontweight='bold',
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=bbox_props,
                       color='white' if score >= 3 else 'black')
                
                # Add title with key information
                filename = os.path.basename(frame_path)
                title = f"{filename[:20]}...\n{data['traffic_density']} traffic, {data['pedestrian_count']} pedestrians"
                ax.set_title(title, fontsize=10, pad=10)
                
                # Add reasoning at the bottom
                reasoning = data.get('reasoning', 'No reasoning provided')
                # Wrap text to fit in the subplot
                wrapped_reasoning = '\n'.join([reasoning[i:i+80] for i in range(0, len(reasoning), 80)])
                
                ax.text(0.5, 0.02, wrapped_reasoning, 
                       transform=ax.transAxes, 
                       fontsize=8, 
                       verticalalignment='bottom',
                       horizontalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                       wrap=True)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"Error: {os.path.basename(frame_path)}")
        else:
            ax.text(0.5, 0.5, f"Image not found:\n{frame_path}", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"Missing: {os.path.basename(frame_path)}")
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(len(sorted_data), len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Scene Complexity Analysis (Sorted by Score - Highest First)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.15)  # Add bottom margin for reasoning text
    plt.show()


def display_responses(output_texts):
    """Simple function to display model responses as markdown."""
    for i, response in enumerate(output_texts, 1):
        # Skip generic AI responses
        if "I am Qwen" in response or "I am a large language model" in response:
            continue
        
        # Display as markdown
        display(Markdown(response))
        print()

def analyze_frames_batch(frame_paths, text_prompt, processor, model, max_new_tokens=2048):
    """
    Analyze a list of driving frames using batch inference.
    
    Args:
        frame_paths (list): List of image file paths
        text_prompt (str): Text prompt for analysis
        processor: The processor object for handling inputs
        model: The model object for inference
        max_new_tokens (int): Maximum number of new tokens to generate
    
    Returns:
        list: List of output texts from the model
    """
    # Set padding side to left for decoder-only models
    processor.tokenizer.padding_side = 'left'
    
    # Create message with multiple images
    content = []
    for frame_path in frame_paths:
        content.append({"type": "image", "image": frame_path})
    
    # Create the prompt with frame paths included
    frame_list = "\n".join([f"Frame {i+1}: {path}" for i, path in enumerate(frame_paths)])
    full_prompt = f"Frame paths for reference:\n{frame_list}\n\n{text_prompt}"
    
    # Add text prompt
    content.append({"type": "text", "text": full_prompt})
    
    # Create messages for batch processing
    messages = [
        [
            {
                "role": "user",
                "content": content
            }
        ]
    ]
    
    # Preparation for batch inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Batch Inference
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_texts