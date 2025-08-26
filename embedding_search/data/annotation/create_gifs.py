#!/usr/bin/env python3
"""
Script to create GIFs from videos listed in parquet files and update the parquet files with GIF paths.
"""

import pandas as pd
import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GifGenerator:
    def __init__(self, gif_width=320, gif_fps=10, gif_duration=None):
        """
        Initialize GIF generator with default parameters.
        
        Args:
            gif_width (int): Width of the generated GIF (height will be auto-calculated)
            gif_fps (int): Frames per second for the GIF
            gif_duration (float): Duration in seconds (None = full video)
        """
        self.gif_width = gif_width
        self.gif_fps = gif_fps
        self.gif_duration = gif_duration
        
    def create_gif_from_video(self, video_path, gif_path, overwrite=False):
        """
        Create a GIF from a video file using ffmpeg.
        
        Args:
            video_path (str): Path to the input video file
            gif_path (str): Path where the GIF should be saved
            overwrite (bool): Whether to overwrite existing GIF files
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            return False
            
        if os.path.exists(gif_path) and not overwrite:
            logger.info(f"GIF already exists, skipping: {gif_path}")
            return True
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        
        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps={self.gif_fps},scale={self.gif_width}:-1:flags=lanczos',
                '-c:v', 'gif',
                '-y' if overwrite else '-n',  # -y to overwrite, -n to not overwrite
            ]
            
            # Add duration limit if specified
            if self.gif_duration:
                cmd.extend(['-t', str(self.gif_duration)])
                
            cmd.append(gif_path)
            
            # Run ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully created GIF: {gif_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating GIF for {video_path}: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating GIF for {video_path}: {e}")
            return False

def get_gif_path(video_path, gif_base_dir):
    """
    Generate the corresponding GIF path for a video file.
    
    Args:
        video_path (str): Path to the video file
        gif_base_dir (str): Base directory for GIF files
        
    Returns:
        str: Path where the GIF should be saved
    """
    video_path = Path(video_path)
    
    # Create a relative path structure that mirrors the video structure
    # Extract the relevant part of the path after 'videos/'
    path_parts = video_path.parts
    
    # Find the index of 'videos' in the path
    try:
        videos_index = path_parts.index('videos')
        relative_parts = path_parts[videos_index + 1:]  # Everything after 'videos/'
    except ValueError:
        # If 'videos' not found, use the filename
        relative_parts = (video_path.name,)
    
    # Create GIF path with .gif extension
    gif_filename = video_path.stem + '.gif'
    gif_path = Path(gif_base_dir) / Path(*relative_parts[:-1]) / gif_filename
    
    return str(gif_path)

def process_parquet_file(parquet_path, gif_base_dir, gif_generator, overwrite=False):
    """
    Process a parquet file to generate GIFs and update with GIF paths.
    
    Args:
        parquet_path (str): Path to the parquet file
        gif_base_dir (str): Base directory for GIF files
        gif_generator (GifGenerator): GIF generator instance
        overwrite (bool): Whether to overwrite existing GIF files
        
    Returns:
        pd.DataFrame: Updated dataframe with GIF paths
    """
    logger.info(f"Processing parquet file: {parquet_path}")
    
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    logger.info(f"Found {len(df)} videos to process")
    
    # Create a new column for GIF paths
    gif_paths = []
    successful_gifs = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating GIFs"):
        video_path = row['sensor_video_file']
        gif_path = get_gif_path(video_path, gif_base_dir)
        
        # Try to create the GIF
        success = gif_generator.create_gif_from_video(video_path, gif_path, overwrite)
        
        if success:
            gif_paths.append(gif_path)
            successful_gifs += 1
        else:
            gif_paths.append(None)  # Mark as failed
    
    # Add the GIF paths column to the dataframe
    df['gif_file'] = gif_paths
    
    logger.info(f"Successfully created {successful_gifs}/{len(df)} GIFs")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create GIFs from videos in parquet files')
    parser.add_argument('--main-parquet', 
                       default='/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/main_input_mini.parquet',
                       help='Path to main input parquet file')
    parser.add_argument('--query-parquet',
                       default='/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/query_input_path.parquet', 
                       help='Path to query input parquet file')
    parser.add_argument('--gif-dir',
                       default='/Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/gifs',
                       help='Base directory for generated GIFs')
    parser.add_argument('--gif-width', type=int, default=320,
                       help='Width of generated GIFs (default: 320)')
    parser.add_argument('--gif-fps', type=int, default=10,
                       help='Frames per second for GIFs (default: 10)')
    parser.add_argument('--gif-duration', type=float, default=None,
                       help='Duration limit for GIFs in seconds (default: full video)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing GIF files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually creating GIFs')
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg is not available. Please install ffmpeg first.")
        logger.error("You can install it with: brew install ffmpeg (on macOS)")
        return 1
    
    # Initialize GIF generator
    gif_generator = GifGenerator(
        gif_width=args.gif_width,
        gif_fps=args.gif_fps,
        gif_duration=args.gif_duration
    )
    
    # Process both parquet files
    files_to_process = [
        (args.main_parquet, 'main_input_mini_with_gifs.parquet'),
        (args.query_parquet, 'query_input_path_with_gifs.parquet')
    ]
    
    for input_file, output_filename in files_to_process:
        if not os.path.exists(input_file):
            logger.warning(f"Parquet file does not exist: {input_file}")
            continue
            
        if args.dry_run:
            logger.info(f"DRY RUN: Would process {input_file}")
            df = pd.read_parquet(input_file)
            for idx, row in df.iterrows():
                video_path = row['sensor_video_file']
                gif_path = get_gif_path(video_path, args.gif_dir)
                logger.info(f"Would create: {video_path} -> {gif_path}")
        else:
            # Process the file
            updated_df = process_parquet_file(input_file, args.gif_dir, gif_generator, args.overwrite)
            
            # Save the updated parquet file
            output_path = os.path.join(os.path.dirname(input_file), output_filename)
            updated_df.to_parquet(output_path, index=False)
            logger.info(f"Updated parquet file saved to: {output_path}")
    
    logger.info("GIF generation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
