"""
Video Clip Preprocessor for dividing long clips into standardized segments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import logging
import math
import subprocess
import os

from .embedder import get_video_duration, estimate_frames_duration, _is_zip_path, _parse_zip_path
import zipfile

logger = logging.getLogger(__name__)


class ClipPreprocessor:
    """
    Preprocessor to divide long video clips into standardized segments.
    """
    
    def __init__(self, target_duration: float = 20.0, min_segment_duration: float = 5.0, create_video_segments: bool = True, fps: float = 30.0, output_video_dir: str = None):
        """
        Initialize the clip preprocessor.
        
        Args:
            target_duration: Target duration for each segment in seconds (default: 20.0)
            min_segment_duration: Minimum duration for the last segment in seconds (default: 5.0)
            create_video_segments: Whether to create actual video segment files (default: True)
            fps: Frames per second for video generation from frames (default: 30.0)
            output_video_dir: Directory to save generated video segments (default: same as source)
        """
        self.target_duration = target_duration
        self.min_segment_duration = min_segment_duration
        self.create_video_segments = create_video_segments
        self.fps = fps
        self.output_video_dir = output_video_dir
    
    def get_input_duration(self, input_path: Union[str, Path]) -> float:
        """
        Get duration of input file (video, frame folder, or zip file).
        
        Args:
            input_path: Path to the input file
            
        Returns:
            Duration in seconds
        """
        try:
            if _is_zip_path(input_path):
                return self._get_zip_duration(input_path)
            
            path = Path(input_path)
            if path.is_file():
                # Assume it's a video file
                duration = get_video_duration(path)
                return duration if duration is not None else 0.0
            elif path.is_dir():
                # Frame folder - estimate based on frame count
                return self._get_folder_duration(path)
            else:
                logger.warning(f"Unknown input type: {input_path}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting duration for {input_path}: {e}")
            return 0.0
    
    def _get_zip_duration(self, zip_path_input: Union[str, Path], fps: float = 30.0) -> float:
        """Estimate duration of zip file containing frames assuming default FPS."""
        zip_path, subfolder = _parse_zip_path(zip_path_input)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                all_files = zip_file.namelist()
                
                # Filter by subfolder if specified
                if subfolder:
                    subfolder_prefix = subfolder + '/' if not subfolder.endswith('/') else subfolder
                    filtered_files = [f for f in all_files if f.startswith(subfolder_prefix)]
                else:
                    filtered_files = all_files
                
                # Count image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                image_files = [f for f in filtered_files 
                              if Path(f).suffix.lower() in image_extensions and not f.startswith('__MACOSX/')]
                
                return estimate_frames_duration(len(image_files), fps)
        except Exception as e:
            logger.warning(f"Error getting zip duration for {zip_path}: {e}")
            return 0.0
    
    def _get_folder_duration(self, folder_path: Path, fps: float = 30.0) -> float:
        """Estimate duration of frame folder assuming default FPS."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        return estimate_frames_duration(len(image_files), fps)
    
    def calculate_segments(self, duration: float) -> List[Dict[str, float]]:
        """
        Calculate how to divide a clip into segments.
        
        Args:
            duration: Total duration of the clip in seconds
            
        Returns:
            List of segments with start and end times
        """
        if duration <= self.target_duration:
            # No need to divide
            return [{"start": 0.0, "end": duration}]
        
        segments = []
        current_time = 0
        duration_int = int(duration)
        
        while current_time < duration_int:
            segment_end = min(current_time + int(self.target_duration), duration_int)
            remaining_duration = duration_int - segment_end
            
            # If the remaining duration is less than min_segment_duration,
            # extend the current segment to include it
            if 0 < remaining_duration < int(self.min_segment_duration):
                segment_end = duration_int
            
            segments.append({
                "start": current_time,
                "end": segment_end
            })
            
            current_time = segment_end
            
            # Break if we've reached the end
            if segment_end >= duration_int:
                break
        
        return segments
    
    def create_video_segment(self, input_video_path: Union[str, Path], output_video_path: Union[str, Path], 
                           start_time: float, end_time: float) -> bool:
        """
        Create a video segment using FFmpeg.
        
        Args:
            input_video_path: Path to the input video file
            output_video_path: Path to the output video segment
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output file already exists
            if output_path.exists():
                logger.info(f"Video segment already exists: {output_path}")
                return True
            
            duration = end_time - start_time
            
            # FFmpeg command to extract video segment with precise timing
            cmd = [
                'ffmpeg',
                '-i', str(input_video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # Re-encode video for precise timing
                '-c:a', 'aac',      # Re-encode audio for precise timing
                '-preset', 'fast',  # Fast encoding preset
                '-crf', '23',       # Good quality setting
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output file if it exists
                str(output_video_path)
            ]
            
            logger.info(f"Creating video segment: {input_video_path} -> {output_video_path} ({start_time}s-{end_time}s)")
            
            # Run FFmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info(f"Successfully created video segment: {output_video_path}")
                return True
            else:
                logger.error(f"FFmpeg failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating video segment {output_video_path}: {e}")
            return False
    
    def generate_segment_video_path(self, original_video_path: Union[str, Path], start_time: float, end_time: float) -> str:
        """
        Generate the path for a video segment file.
        
        Args:
            original_video_path: Original video file path (can be zip#subfolder format)
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Path for the segment video file (always .mp4 format)
        """
        # Handle zip#subfolder format
        original_path_str = str(original_video_path)
        if '#' in original_path_str:
            # For zip files, use the zip filename as base
            zip_path, subfolder = original_path_str.split('#', 1)
            original_path = Path(zip_path)
        else:
            original_path = Path(original_video_path)
        
        # Format times as 4-digit integers (e.g., 0000, 0020)
        start_str = f"{int(start_time):04d}"
        end_str = f"{int(end_time):04d}"
        
        # Create new filename: original_name_start_end.mp4 (always mp4 output)
        stem = original_path.stem
        new_filename = f"{stem}_{start_str}_{end_str}.mp4"
        
        # Use custom output directory if specified, otherwise same directory as original
        if self.output_video_dir:
            output_dir = Path(self.output_video_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return str(output_dir / new_filename)
        else:
            return str(original_path.parent / new_filename)
    
    def create_video_from_frames_zip(self, zip_path: Union[str, Path], output_video_path: Union[str, Path], 
                                   start_time: float, end_time: float) -> bool:
        """
        Create a video segment from frames stored in a zip file using FFmpeg.
        
        Args:
            zip_path: Path to the zip file containing frames (supports zip#subfolder syntax)
            output_video_path: Path to the output video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import tempfile
            import zipfile
            
            # Ensure output directory exists
            output_path = Path(output_video_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if output file already exists
            if output_path.exists():
                logger.info(f"Video segment already exists: {output_path}")
                return True
            
            # Parse zip path and subfolder
            zip_path_str = str(zip_path)
            if '#' in zip_path_str:
                actual_zip_path, subfolder = zip_path_str.split('#', 1)
            else:
                actual_zip_path = zip_path_str
                subfolder = None
            
            # Calculate frame range based on timing and FPS
            start_frame = int(start_time * self.fps)
            end_frame = int(end_time * self.fps)
            
            # Create temporary directory for extracted frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract frames from zip
                with zipfile.ZipFile(actual_zip_path, 'r') as zip_file:
                    all_files = zip_file.namelist()
                    
                    # Filter by subfolder if specified
                    if subfolder:
                        subfolder_prefix = subfolder + '/' if not subfolder.endswith('/') else subfolder
                        filtered_files = [f for f in all_files if f.startswith(subfolder_prefix)]
                    else:
                        filtered_files = all_files
                    
                    # Filter for image files and sort them
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                    image_files = [f for f in filtered_files 
                                  if Path(f).suffix.lower() in image_extensions and not f.startswith('__MACOSX/')]
                    image_files.sort()
                    
                    # Select frames for the specified time range
                    selected_frames = image_files[start_frame:end_frame]
                    
                    if not selected_frames:
                        logger.error(f"No frames found in range {start_frame}-{end_frame} for {zip_path}")
                        return False
                    
                    # Extract selected frames to temp directory using original filenames
                    extracted_filenames = []
                    for frame_file in selected_frames:
                        # Use the original filename from the zip
                        original_filename = Path(frame_file).name
                        frame_path = temp_path / original_filename
                        
                        with zip_file.open(frame_file) as src, open(frame_path, 'wb') as dst:
                            dst.write(src.read())
                        
                        extracted_filenames.append(original_filename)
                
                # Create a text file listing all frame files for FFmpeg concat demuxer
                concat_file = temp_path / "frame_list.txt"
                with open(concat_file, 'w') as f:
                    for filename in extracted_filenames:
                        f.write(f"file '{filename}'\n")
                        f.write(f"duration {1/self.fps}\n")  # Duration per frame
                    # Add the last frame again to ensure proper duration
                    if extracted_filenames:
                        f.write(f"file '{extracted_filenames[-1]}'\n")
                
                cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', str(concat_file),
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    str(output_video_path)
                ]
                
                logger.info(f"Creating video from frames: {zip_path} -> {output_video_path} ({start_time}s-{end_time}s)")
                
                # Run FFmpeg command
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    logger.info(f"Successfully created video from frames: {output_video_path}")
                    return True
                else:
                    logger.error(f"FFmpeg failed with return code {result.returncode}")
                    logger.error(f"FFmpeg stderr: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating video from frames {output_video_path}: {e}")
            return False
    
    def generate_segment_slice_id(self, original_slice_id: str, start_time: float, end_time: float) -> str:
        """
        Generate a new slice_id for a segment.
        
        Args:
            original_slice_id: Original slice ID
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            New slice_id with format: <original_slice_id>_<start>_<end>
        """
        # Format times as 4-digit integers (e.g., 0020, 0040)
        start_str = f"{int(start_time):04d}"
        end_str = f"{int(end_time):04d}"
        
        # Remove file extension if present
        base_id = original_slice_id
        if '.' in base_id:
            base_id = base_id.rsplit('.', 1)[0]
        
        return f"{base_id}_{start_str}_{end_str}"
    
    def preprocess_parquet(self, parquet_path: Union[str, Path], output_path: Union[str, Path] = None) -> pd.DataFrame:
        """
        Preprocess a parquet file to divide long clips into segments.
        
        Args:
            parquet_path: Path to input parquet file
            output_path: Path to save processed parquet file (optional)
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Preprocessing parquet file: {parquet_path}")
        
        # Load the original parquet file
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} entries from parquet file")
        
        processed_rows = []
        
        for _, row in df.iterrows():
            original_slice_id = row['slice_id']
            
            # Get the input file path (check both sensor_video_file and sensor_frame_zip)
            input_path = None
            input_type = None
            if 'sensor_video_file' in row and pd.notna(row['sensor_video_file']):
                input_path = row['sensor_video_file']
                input_type = 'video'
            elif 'sensor_frame_zip' in row and pd.notna(row['sensor_frame_zip']):
                input_path = row['sensor_frame_zip']
                input_type = 'frame_zip'
            
            if input_path is None:
                logger.warning(f"No input path found for slice_id: {original_slice_id}")
                # Keep the original row as-is
                processed_rows.append(row.to_dict())
                continue
            
            # Get duration of the input
            duration = self.get_input_duration(input_path)
            
            if duration <= 0:
                logger.warning(f"Invalid duration ({duration}s) for {original_slice_id}, keeping original")
                processed_rows.append(row.to_dict())
                continue
            
            # Calculate segments
            segments = self.calculate_segments(duration)
            
            logger.info(f"Dividing {original_slice_id} ({duration:.1f}s) into {len(segments)} segments")
            
            # Create new rows for each segment
            for segment in segments:
                new_row = row.to_dict().copy()
                
                # Generate new slice_id
                new_slice_id = self.generate_segment_slice_id(
                    original_slice_id, 
                    segment["start"], 
                    segment["end"]
                )
                
                # Update the row with segment information
                new_row['slice_id'] = new_slice_id
                new_row['span_start'] = int(segment["start"])
                new_row['span_end'] = int(segment["end"])
                
                # Handle segmentation based on input type
                if input_type == 'video':
                    # Handle video file segmentation
                    original_video_path = row['sensor_video_file']
                    
                    # Generate segment video path
                    segment_video_path = self.generate_segment_video_path(
                        original_video_path,
                        segment["start"],
                        segment["end"]
                    )
                    
                    # Always update the sensor_video_file path to the segmented path
                    new_row['sensor_video_file'] = segment_video_path
                    logger.info(f"Updated sensor_video_file: {original_video_path} -> {segment_video_path}")
                    
                    # Create the actual video segment if enabled
                    if self.create_video_segments:
                        success = self.create_video_segment(
                            original_video_path,
                            segment_video_path,
                            segment["start"],
                            segment["end"]
                        )
                        if not success:
                            logger.warning(f"Failed to create video segment: {segment_video_path}")
                
                elif input_type == 'frame_zip':
                    # Handle frame zip segmentation - generate MP4 from frames first
                    original_zip_path = row['sensor_frame_zip']
                    
                    # Generate segment video path (convert to MP4)
                    segment_video_path = self.generate_segment_video_path(
                        original_zip_path,
                        segment["start"],
                        segment["end"]
                    )
                    
                    # Update to use sensor_video_file instead of sensor_frame_zip for the segments
                    new_row['sensor_video_file'] = segment_video_path
                    if 'sensor_frame_zip' in new_row:
                        del new_row['sensor_frame_zip']  # Remove frame zip column
                    logger.info(f"Converting frame zip to video: {original_zip_path} -> {segment_video_path}")
                    
                    # Create video from frames if enabled
                    if self.create_video_segments:
                        success = self.create_video_from_frames_zip(
                            original_zip_path,
                            segment_video_path,
                            segment["start"],
                            segment["end"]
                        )
                        if not success:
                            logger.warning(f"Failed to create video from frames: {segment_video_path}")
                
                processed_rows.append(new_row)
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(processed_rows)
        
        logger.info(f"Preprocessing complete: {len(df)} → {len(processed_df)} entries")
        logger.info(f"Expansion ratio: {len(processed_df) / len(df):.2f}x")
        
        # Save to output file if specified
        if output_path:
            processed_df.to_parquet(output_path, index=False)
            logger.info(f"Saved processed parquet to: {output_path}")
        
        return processed_df
    
    def get_preprocessing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about the preprocessing results.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with statistics
        """
        # Calculate durations
        df_copy = df.copy()
        df_copy['duration'] = df_copy['span_end'] - df_copy['span_start']
        
        stats = {
            'total_entries': len(df_copy),
            'duration_stats': {
                'mean': df_copy['duration'].mean(),
                'median': df_copy['duration'].median(),
                'min': df_copy['duration'].min(),
                'max': df_copy['duration'].max(),
                'std': df_copy['duration'].std()
            },
            'duration_distribution': df_copy['duration'].value_counts().to_dict()
        }
        
        return stats


def preprocess_clips_cli(input_parquet: str, output_parquet: str = None, target_duration: float = 20.0, create_video_segments: bool = True, output_video_dir: str = None):
    """
    CLI function to preprocess clips.
    
    Args:
        input_parquet: Path to input parquet file
        output_parquet: Path to output parquet file (defaults to input_processed.parquet)
        target_duration: Target duration for segments in seconds
        create_video_segments: Whether to create actual video segment files
        output_video_dir: Directory to save generated video segments (default: same as source)
    """
    if output_parquet is None:
        input_path = Path(input_parquet)
        output_parquet = str(input_path.parent / f"{input_path.stem}_processed{input_path.suffix}")
    
    preprocessor = ClipPreprocessor(target_duration=target_duration, create_video_segments=create_video_segments, output_video_dir=output_video_dir)
    processed_df = preprocessor.preprocess_parquet(input_parquet, output_parquet)
    
    # Print statistics
    stats = preprocessor.get_preprocessing_stats(processed_df)
    print(f"\n📊 Preprocessing Statistics:")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Duration stats:")
    print(f"  Mean: {stats['duration_stats']['mean']:.1f}s")
    print(f"  Median: {stats['duration_stats']['median']:.1f}s")
    print(f"  Range: {stats['duration_stats']['min']:.1f}s - {stats['duration_stats']['max']:.1f}s")
    print(f"\nDuration distribution:")
    for duration, count in sorted(stats['duration_distribution'].items()):
        print(f"  {duration:.1f}s: {count} clips")
    
    return processed_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess video clips into standardized segments")
    parser.add_argument("input_parquet", help="Path to input parquet file")
    parser.add_argument("--output", "-o", help="Path to output parquet file")
    parser.add_argument("--target-duration", "-t", type=float, default=20.0, 
                       help="Target duration for segments in seconds (default: 20.0)")
    parser.add_argument("--no-video-segments", action="store_true", 
                       help="Skip creating actual video segment files (only update metadata)")
    parser.add_argument("--output-video-dir", help="Directory to save generated video segments (default: same as source)")
    
    args = parser.parse_args()
    
    create_video_segments = not args.no_video_segments
    preprocess_clips_cli(args.input_parquet, args.output, args.target_duration, create_video_segments, args.output_video_dir)
