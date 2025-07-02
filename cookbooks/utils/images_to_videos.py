import cv2
import os
import re
import glob
from pathlib import Path
from typing import List, Tuple
import argparse

def extract_timestamp(filename: str) -> int:
    """Extract timestamp from NuScenes filename format.
    
    Args:
        filename: Filename in format like 'n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385094162404.jpg'
    
    Returns:
        Timestamp as integer
    """
    # Extract timestamp from the end of filename (before .jpg)
    match = re.search(r'__(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    else:
        # Fallback: try to extract any long number sequence
        numbers = re.findall(r'\d{10,}', filename)
        if numbers:
            return int(numbers[-1])
        else:
            raise ValueError(f"Could not extract timestamp from filename: {filename}")

def get_image_size(image_path: str) -> Tuple[int, int]:
    """Get the dimensions of an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    return width, height

def create_video_from_images(image_folder: str, output_path: str, fps: float = 12.0) -> bool:
    """Create a video from a sequence of images.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path for output video file
        fps: Frames per second for the output video
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get all jpg files in the folder
        image_pattern = os.path.join(image_folder, "*.jpg")
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            print(f"No JPG files found in {image_folder}")
            return False
        
        print(f"Found {len(image_files)} images in {os.path.basename(image_folder)}")
        
        # Sort images by timestamp
        try:
            image_files.sort(key=lambda x: extract_timestamp(os.path.basename(x)))
        except ValueError as e:
            print(f"Error sorting images by timestamp: {e}")
            # Fallback to alphabetical sorting
            image_files.sort()
        
        # Get dimensions from first image
        width, height = get_image_size(image_files[0])
        print(f"Video dimensions: {width}x{height}")
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}")
            return False
        
        # Process each image
        for i, image_path in enumerate(image_files):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # Resize image if dimensions don't match (shouldn't happen but just in case)
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            video_writer.write(img)
            
            # Progress update every 10 frames
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")
        
        # Release everything
        video_writer.release()
        print(f"Successfully created video: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating video from {image_folder}: {str(e)}")
        return False

def process_nuscenes_sequences(base_folder: str, output_folder: str, fps: float = 12.0):
    """Process all NuScenes image sequences in subfolders.
    
    Args:
        base_folder: Path to the folder containing sequence subfolders
        output_folder: Path to save output videos
        fps: Frames per second for output videos
    """
    base_path = Path(base_folder)
    output_path = Path(output_folder)
    
    if not base_path.exists():
        print(f"Error: Base folder does not exist: {base_folder}")
        return
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all subfolders
    subfolders = [f for f in base_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in {base_folder}")
        return
    
    print(f"Found {len(subfolders)} sequence folders to process")
    successful = 0
    failed = 0
    
    for subfolder in subfolders:
        print(f"\nProcessing: {subfolder.name}")
        
        # Create output video filename
        video_name = f"{subfolder.name}.mp4"
        video_path = output_path / video_name
        
        # Skip if video already exists
        if video_path.exists():
            print(f"Video already exists, skipping: {video_path}")
            continue
        
        # Create video from images
        if create_video_from_images(str(subfolder), str(video_path), fps):
            successful += 1
        else:
            failed += 1
    
    print(f"\n=== Summary ===")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(subfolders)}")

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Convert NuScenes image sequences to videos")
    parser.add_argument(
        "--input", 
        default="/workspace/Qwen2.5-VL/nuscenes_mini/sweeps/CAM_FRONT",
        help="Input folder containing sequence subfolders"
    )
    parser.add_argument(
        "--output", 
        default="/workspace/Qwen2.5-VL/nuscenes_mini/videos",
        help="Output folder for videos"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=12.0,
        help="Frames per second for output videos"
    )
    
    args = parser.parse_args()
    
    print("NuScenes Image Sequence to Video Converter")
    print("=" * 50)
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print(f"FPS: {args.fps}")
    print()
    
    process_nuscenes_sequences(args.input, args.output, args.fps)

if __name__ == "__main__":
    main()
