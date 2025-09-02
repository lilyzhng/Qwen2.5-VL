#!/usr/bin/env python3
"""
Script to create GIFs from videos or frame zip files listed in parquet files and update the parquet files with GIF paths.

Supports both:
- sensor_video_file: MP4/AVI/MOV video files (processed with ffmpeg)
- sensor_frame_zip: ZIP files containing image frames (processed with PIL)

The script automatically detects the input type and uses the appropriate processing method.
"""

import pandas as pd
import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
import zipfile
import tempfile
import cv2
import numpy as np
from PIL import Image

# Get project root directory
def get_project_root():
    """Get the project root directory (embedding_search folder)."""
    current_dir = Path(__file__).parent.absolute()
    # Go up from data/annotation to project root
    return current_dir.parent.parent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GifGenerator:
    def __init__(self, gif_width=320, gif_fps=5, gif_duration=None):
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
        
    def create_gif_from_input(self, input_path, gif_path, overwrite=False):
        """
        Create a GIF from either a video file or a zip file containing frames.
        Supports zip fragment syntax (e.g., file.zip#subfolder).
        
        Args:
            input_path (str): Path to the input video file or zip file (supports zip#subfolder)
            gif_path (str): Path where the GIF should be saved
            overwrite (bool): Whether to overwrite existing GIF files
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Handle zip fragment syntax (file.zip#subfolder)
        actual_file_path = input_path
        is_zip_fragment = False
        
        if '#' in input_path:
            # Extract the actual zip file path
            actual_file_path = input_path.split('#')[0]
            is_zip_fragment = True
            
        if not os.path.exists(actual_file_path):
            logger.error(f"Input file does not exist: {actual_file_path}")
            return False
            
        if os.path.exists(gif_path) and not overwrite:
            logger.info(f"GIF already exists, skipping: {gif_path}")
            return True
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        
        # Determine input type and process accordingly
        # Check if it's a zip file (either by extension or fragment syntax)
        if is_zip_fragment or Path(actual_file_path).suffix.lower() == '.zip':
            return self._create_gif_from_zip(input_path, gif_path, overwrite)
        else:
            return self._create_gif_from_video(input_path, gif_path, overwrite)
    
    def _create_gif_from_video(self, video_path, gif_path, overwrite=False):
        """
        Create a GIF from a video file using ffmpeg.
        
        Args:
            video_path (str): Path to the input video file
            gif_path (str): Path where the GIF should be saved
            overwrite (bool): Whether to overwrite existing GIF files
            
        Returns:
            bool: True if successful, False otherwise
        """
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
            logger.info(f"Successfully created GIF from video: {gif_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating GIF from video {video_path}: {e}")
            logger.error(f"ffmpeg stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating GIF from video {video_path}: {e}")
            return False
    
    def _create_gif_from_zip(self, zip_path, gif_path, overwrite=False):
        """
        Create a GIF from a zip file containing image frames.
        Supports zip fragment syntax (e.g., file.zip#subfolder).
        
        Args:
            zip_path (str): Path to the input zip file (supports zip#subfolder)
            gif_path (str): Path where the GIF should be saved
            overwrite (bool): Whether to overwrite existing GIF files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse zip path and optional subfolder
            actual_zip_path = zip_path
            subfolder = None
            
            if '#' in zip_path:
                actual_zip_path, subfolder = zip_path.split('#', 1)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract frames from zip
                with zipfile.ZipFile(actual_zip_path, 'r') as zip_file:
                    # Get image files, filtering out system files and by subfolder if specified
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                    all_files = zip_file.namelist()
                    
                    # Filter by subfolder if specified
                    if subfolder:
                        subfolder_prefix = subfolder + '/' if not subfolder.endswith('/') else subfolder
                        filtered_files = [f for f in all_files if f.startswith(subfolder_prefix)]
                    else:
                        filtered_files = all_files
                    
                    # Find image files
                    image_files = [f for f in filtered_files 
                                  if Path(f).suffix.lower() in image_extensions and not f.startswith('__MACOSX/')]
                    
                    if not image_files:
                        logger.error(f"No image files found in zip: {zip_path}")
                        return False
                    
                    # Extract image files
                    for img_file in image_files:
                        zip_file.extract(img_file, temp_path)
                    
                    # Get paths of extracted files and sort them
                    extracted_files = []
                    for img_file in image_files:
                        extracted_path = temp_path / img_file
                        if extracted_path.exists():
                            extracted_files.append(extracted_path)
                    
                    extracted_files.sort()
                    
                    if not extracted_files:
                        logger.error(f"No frames could be extracted from zip: {zip_path}")
                        return False
                    
                    # Load frames and create GIF using PIL
                    frames = []
                    target_height = None
                    
                    for frame_path in extracted_files:
                        try:
                            # Load image
                            img = Image.open(frame_path)
                            
                            # Convert to RGB if necessary
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Resize to target width while maintaining aspect ratio
                            original_width, original_height = img.size
                            target_width = self.gif_width
                            target_height = int(original_height * target_width / original_width)
                            
                            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                            frames.append(img_resized)
                            
                        except Exception as e:
                            logger.warning(f"Failed to process frame {frame_path}: {e}")
                            continue
                    
                    if not frames:
                        logger.error(f"No frames could be processed from zip: {zip_path}")
                        return False
                    
                    # Calculate frame duration in milliseconds
                    frame_duration = int(1000 / self.gif_fps)
                    
                    # Limit number of frames if duration is specified
                    if self.gif_duration:
                        max_frames = int(self.gif_duration * self.gif_fps)
                        if len(frames) > max_frames:
                            frames = frames[:max_frames]
                    
                    # Save as GIF
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=frame_duration,
                        loop=0,
                        optimize=True
                    )
                    
                    logger.info(f"Successfully created GIF from zip ({len(frames)} frames): {gif_path}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error creating GIF from zip {zip_path}: {e}")
            return False
    
    # Keep backward compatibility
    def create_gif_from_video(self, video_path, gif_path, overwrite=False):
        """Backward compatibility method - redirects to create_gif_from_input."""
        return self.create_gif_from_input(video_path, gif_path, overwrite)

def get_gif_path(input_path, gif_base_dir):
    """
    Generate the corresponding GIF path for a video file or zip file.
    Supports zip fragment syntax (e.g., file.zip#subfolder).
    
    Args:
        input_path (str): Path to the video file or zip file (supports zip#subfolder)
        gif_base_dir (str): Base directory for GIF files
        
    Returns:
        str: Path where the GIF should be saved
    """
    # Handle zip fragment syntax - remove fragment for path processing
    actual_path = input_path.split('#')[0] if '#' in input_path else input_path
    input_path = Path(actual_path)
    
    # Simply use the filename with .gif extension directly in gif_base_dir
    gif_filename = input_path.stem + '.gif'
    gif_path = Path(gif_base_dir) / gif_filename
    
    return str(gif_path)

def process_parquet_file(parquet_path, gif_base_dir, gif_generator, overwrite=False):
    """
    Process a parquet file to generate GIFs and update with GIF paths.
    Supports both sensor_video_file and sensor_frame_zip columns.
    
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
    logger.info(f"Found {len(df)} entries to process")
    
    # Check for both sensor_video_file and sensor_frame_zip columns
    video_column = None
    frame_column = None
    
    if 'sensor_video_file' in df.columns:
        video_column = 'sensor_video_file'
        logger.info(f"Found sensor_video_file column")
    if 'sensor_frame_zip' in df.columns:
        frame_column = 'sensor_frame_zip'
        logger.info(f"Found sensor_frame_zip column")
        
    if not video_column and not frame_column:
        raise ValueError(f"Neither 'sensor_video_file' nor 'sensor_frame_zip' columns found in parquet file. Available columns: {list(df.columns)}")
    
    # Create a new column for GIF paths
    gif_paths = []
    successful_gifs = 0
    skipped_entries = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating GIFs"):
        # Determine input path - prioritize video files, then frame files
        input_path = None
        input_type = None
        
        if video_column and pd.notna(row[video_column]) and row[video_column]:
            input_path = row[video_column]
            input_type = "video"
        elif frame_column and pd.notna(row[frame_column]) and row[frame_column]:
            input_path = row[frame_column]
            input_type = "frame"
        
        if not input_path:
            logger.warning(f"No valid input path found for row {idx}")
            gif_paths.append(None)
            skipped_entries += 1
            continue
        
        # Generate GIF path
        gif_path = get_gif_path(input_path, gif_base_dir)
        
        # Try to create the GIF using the enhanced method
        success = gif_generator.create_gif_from_input(input_path, gif_path, overwrite)
        
        if success:
            gif_paths.append(gif_path)
            successful_gifs += 1
            logger.debug(f"Created GIF from {input_type}: {input_path} -> {gif_path}")
        else:
            gif_paths.append(None)  # Mark as failed
    
    # Add the GIF paths column to the dataframe
    df['gif_file'] = gif_paths
    
    logger.info(f"Successfully created {successful_gifs}/{len(df)} GIFs")
    if skipped_entries > 0:
        logger.warning(f"Skipped {skipped_entries} entries due to missing input paths")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Create GIFs from videos in parquet files')
    project_root = get_project_root()
    parser.add_argument('--input-parquet', 
                       default=str(project_root / "data" / "unified_input_path.parquet"),
                       help='Path to unified input parquet file')
    parser.add_argument('--gif-dir',
                       default=str(project_root / "data" / "gifs_new"),
                       help='Base directory for generated GIFs')
    parser.add_argument('--gif-width', type=int, default=320,
                       help='Width of generated GIFs (default: 320)')
    parser.add_argument('--gif-fps', type=int, default=5,
                       help='Frames per second for GIFs (default: 5)')
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
    
    # Process the unified parquet file
    input_file = args.input_parquet
    
    if not os.path.exists(input_file):
        logger.error(f"Parquet file does not exist: {input_file}")
        return 1
        
    if args.dry_run:
        logger.info(f"DRY RUN: Would process {input_file}")
        df = pd.read_parquet(input_file)
        
        # Check available columns
        video_column = 'sensor_video_file' if 'sensor_video_file' in df.columns else None
        frame_column = 'sensor_frame_zip' if 'sensor_frame_zip' in df.columns else None
        
        for idx, row in df.iterrows():
            # Determine input path
            input_path = None
            input_type = None
            
            if video_column and pd.notna(row[video_column]) and row[video_column]:
                input_path = row[video_column]
                input_type = "video"
            elif frame_column and pd.notna(row[frame_column]) and row[frame_column]:
                input_path = row[frame_column]
                input_type = "frame"
            
            if input_path:
                gif_path = get_gif_path(input_path, args.gif_dir)
                logger.info(f"Would create GIF from {input_type}: {input_path} -> {gif_path}")
            else:
                logger.warning(f"No valid input path found for row {idx}")
    else:
        # Process the file
        updated_df = process_parquet_file(input_file, args.gif_dir, gif_generator, args.overwrite)
        
        # Save back to the same input file (update in place)
        updated_df.to_parquet(input_file, index=False)
        logger.info(f"Updated parquet file saved to: {input_file}")
    
    logger.info("GIF generation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
