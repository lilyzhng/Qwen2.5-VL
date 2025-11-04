"""Utility functions for VLM judge in ALFA Curate."""

import logging
from pathlib import Path
from typing import List, Optional, Final

import cv2
import numpy as np
import numpy.typing as npt

from autonomy.perception.datasets.active_learning.alfa_curate.utils import SearchResult
from kits.scalex.dataset.manifest import Manifest
from kits.scalex.hpc.tiered_file_system import tiered_filesystem

_LOGGER: Final = logging.getLogger(__name__)


def load_video_frames(
    video_path: str,
    start_time_ns: int,
    end_time_ns: int,
    desired_fps: float = 1.0,
    max_frames: int = 8,
) -> List[npt.NDArray[np.uint8]]:
    """Load video frames for a specific time segment.
    
    Args:
        video_path: Path to the video file (can be local or s3://).
        start_time_ns: Start time in nanoseconds.
        end_time_ns: End time in nanoseconds.
        desired_fps: Desired frame rate for sampling (frames per second).
        max_frames: Maximum number of frames to return.
        
    Returns:
        List of frames.
    """
    frames = []
    
    # Handle remote paths (s3://, lakefs://, etc.)
    if video_path.startswith(("s3://", "lakefs://")):
        fs = tiered_filesystem()
        temp_path = f"/tmp/{Path(video_path).name}"
        fs.get_file(video_path, temp_path)
        video_path = temp_path
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _LOGGER.warning("Failed to open video: %s", video_path)
            return frames
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        
        start_time_sec = start_time_ns / 1e9
        end_time_sec = end_time_ns / 1e9
        duration_sec = end_time_sec - start_time_sec
        
        frame_interval = 1.0 / desired_fps
        num_frames = min(int(duration_sec * desired_fps), max_frames)
        
        if num_frames <= 0:
            num_frames = 1
        
        for i in range(num_frames):
            timestamp_sec = start_time_sec + (i * duration_sec / num_frames)
            frame_number = int(timestamp_sec * video_fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB (OpenCV loads as BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
    except Exception as e:
        _LOGGER.error("Error loading frames from %s: %s", video_path, e)
    
    return frames


def get_video_path_for_slice(
    search_result: SearchResult,
    manifest: Manifest,
) -> Optional[str]:
    """Get the video file path for a search result.
    
    Args:
        search_result: The search result containing slice_id and camera info.
        manifest: The log slices manifest to look up paths.
        
    Returns:
        Path to the video file, or None if not found.
    """

    slice_id = search_result.slice_id
    

    for reference in manifest.key_to_data_file.values():
        if slice_id in reference.path or slice_id == reference.key:
            return reference.physical_address
    
    _LOGGER.warning("Could not find video path for slice_id: %s", slice_id)
    return None


def load_frames_for_search_result(
    search_result: SearchResult,
    manifest: Manifest,
    desired_fps: float = 1.0,
    max_frames: int = 8,
) -> List[npt.NDArray[np.uint8]]:
    """Load video frames for a SearchResult.
    
    Args:
        search_result: The search result to load frames for.
        manifest: The log slices manifest to look up paths.
        desired_fps: Desired frame rate for sampling.
        max_frames: Maximum number of frames to return.
        
    Returns:
        List of frames as HWC uint8 numpy arrays.
    """
    video_path = get_video_path_for_slice(search_result, manifest)
    
    if video_path is None:
        _LOGGER.warning("No video path found for slice_id: %s", search_result.slice_id)
        return []
    
    return load_video_frames(
        video_path,
        search_result.segment_start_ns,
        search_result.segment_end_ns,
        desired_fps=desired_fps,
        max_frames=max_frames,
    )

