import os
import math
import hashlib
import requests

from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")

def get_video_frames(video_path, num_frames=128, fps=None, cache_dir='.cache'):
    """
    Extract frames from a video.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract (ignored if fps is provided)
        fps (float, optional): Target frames per second. If provided, calculates num_frames 
                              based on video duration (fps * video_duration)
        cache_dir (str): Directory to store cached frames and timestamps
    
    Returns:
        tuple: (video_file_path, frames, timestamps)
    """
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    # If fps is provided, we need to calculate num_frames first to determine cache file names
    if fps is not None:
        vr_temp = VideoReader(video_file_path, ctx=cpu(0))
        total_frames = len(vr_temp)
        video_fps = vr_temp.get_avg_fps()
        video_duration = total_frames / video_fps
        calculated_num_frames = int(fps * video_duration)
        print(f"Video info: {total_frames} frames, {video_fps:.2f} fps, {video_duration:.2f}s duration")
        print(f"Calculated num_frames for {fps} fps: {calculated_num_frames}")
        # Use calculated num_frames for cache file naming and extraction
        effective_num_frames = calculated_num_frames
        cache_suffix = f"{fps}fps"
    else:
        effective_num_frames = num_frames
        cache_suffix = f"{num_frames}_frames"

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{cache_suffix}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{cache_suffix}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        print(f"Loaded {len(frames)} frames from cache")
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if fps is not None:
        # Use the pre-calculated effective_num_frames
        actual_num_frames = effective_num_frames
    else:
        actual_num_frames = num_frames

    # Ensure we don't request more frames than available
    actual_num_frames = min(actual_num_frames, total_frames)
    
    indices = np.linspace(0, total_frames - 1, num=actual_num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    print(f"Extracted {len(frames)} frames and saved to cache")
    return video_file_path, frames, timestamps


def get_video_frames(video_path, num_frames=None, fps=None, cache_dir='.cache'):
    """
    Extract frames from a video.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract (ignored if fps is provided)
        fps (float, optional): Target frames per second. If provided, calculates num_frames 
                              based on video duration (fps * video_duration)
        cache_dir (str): Directory to store cached frames and timestamps
    
    Returns:
        tuple: (video_file_path, frames, timestamps)
    """
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    # If fps is provided, we need to calculate num_frames first to determine cache file names
    if fps is not None:
        vr_temp = VideoReader(video_file_path, ctx=cpu(0))
        total_frames = len(vr_temp)
        video_fps = vr_temp.get_avg_fps()
        video_duration = total_frames / video_fps
        calculated_num_frames = int(fps * video_duration)
        print(f"Video info: {total_frames} frames, {video_fps:.2f} fps, {video_duration:.2f}s duration")
        print(f"Calculated num_frames for {fps} fps: {calculated_num_frames}")
        # Use calculated num_frames for cache file naming and extraction
        effective_num_frames = calculated_num_frames
        cache_suffix = f"{fps}fps"
    else:
        effective_num_frames = num_frames
        cache_suffix = f"{num_frames}_frames"

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{cache_suffix}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{cache_suffix}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        print(f"Loaded {len(frames)} frames from cache")
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)
    
    if fps is not None:
        # Use the pre-calculated effective_num_frames
        actual_num_frames = effective_num_frames
    else:
        actual_num_frames = num_frames

    # Ensure we don't request more frames than available
    actual_num_frames = min(actual_num_frames, total_frames)
    
    indices = np.linspace(0, total_frames - 1, num=actual_num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    print(f"Extracted {len(frames)} frames and saved to cache")
    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def display_frames_by_time(frames, timestamps, time_duration, num_columns=4):
    """
    Display video frames within a specified time window in a grid.
    
    Args:
        frames (np.ndarray): Array of video frames from get_video_frames
        timestamps (np.ndarray): Array of frame timestamps from get_video_frames
        time_duration (str): Time range in format "MM:SS.ss - MM:SS.ss" (e.g., "00:15.00 - 00:20.00")
        num_columns (int): Number of columns in the grid (default: 4)
    
    Returns:
        PIL.Image: Grid image containing frames from the specified time window
    """
    # Debug: Print shapes to understand the structure
    print(f"timestamps shape: {timestamps.shape}")
    print(f"timestamps[0] shape: {timestamps[0].shape if hasattr(timestamps[0], 'shape') else 'no shape'}")
    print(f"timestamps[0]: {timestamps[0]}")
    
    # Parse time duration
    def parse_time(time_str):
        """Convert MM:SS.ss format to seconds"""
        parts = time_str.strip().split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    # Extract start and end times
    start_time_str, end_time_str = time_duration.split(' - ')
    start_time = parse_time(start_time_str)
    end_time = parse_time(end_time_str)
    
    # Handle both 1D and 2D timestamp arrays
    if timestamps.ndim == 2:
        # If timestamps is 2D, use the first timestamp of each frame
        timestamps_for_filtering = timestamps[:, 0]
    else:
        timestamps_for_filtering = timestamps
    
    # Filter frames within the time window
    mask = (timestamps_for_filtering >= start_time) & (timestamps_for_filtering <= end_time)
    
    # Use np.where to get indices, then index explicitly
    valid_indices = np.where(mask)[0]
    filtered_frames = frames[valid_indices]
    filtered_timestamps = timestamps_for_filtering[valid_indices]
    
    if len(filtered_frames) == 0:
        print(f"No frames found in time range {time_duration}")
        return None
    
    print(f"Found {len(filtered_frames)} frames in time range {time_duration}")
    print(f"Timestamps: {filtered_timestamps[0]:.2f}s - {filtered_timestamps[-1]:.2f}s")
    
    # Create and display the image grid
    grid_image = create_image_grid(filtered_frames, num_columns=num_columns)
    
    # Display the grid using IPython display
    display(grid_image)
    
    # return grid_image
