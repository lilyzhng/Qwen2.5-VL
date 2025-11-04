"""Unit tests for VLM Judge utilities."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from autonomy.perception.datasets.active_learning.alfa_curate.utils import SearchResult
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils import (
    load_video_frames,
    get_video_path_for_slice,
    load_frames_for_search_result,
)
from kits.scalex.dataset.manifest import FileReference, GenericRef, Manifest
from kits.scalex.dataset.stage import Stage
from platforms.lakefs.client import LakefsRef


def test_load_video_frames_local() -> None:
    """Test loading frames from local video."""
    video_path = "/tmp/test_video.mp4"
    
    with patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0  # 30 fps
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.side_effect = lambda frame, _: frame
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1
        mock_cv2.COLOR_BGR2RGB = 4
        
        frames = load_video_frames(
            video_path,
            start_time_ns=0,
            end_time_ns=2_000_000_000,
            desired_fps=1.0,
            max_frames=8,
        )
        
        assert len(frames) >= 1
        mock_cv2.VideoCapture.assert_called_once_with(video_path)


def test_load_video_frames_remote() -> None:
    """Test loading frames from S3/remote video."""
    video_path = "s3://bucket/video.mp4"
    
    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils.tiered_filesystem") as mock_fs,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils.cv2") as mock_cv2,
    ):
        mock_filesystem = MagicMock()
        mock_fs.return_value = mock_filesystem
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.side_effect = lambda frame, _: frame
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_POS_FRAMES = 1
        mock_cv2.COLOR_BGR2RGB = 4
        
        frames = load_video_frames(
            video_path,
            start_time_ns=0,
            end_time_ns=1_000_000_000,
            desired_fps=1.0,
            max_frames=8,
        )
        
        # Should download file first
        mock_filesystem.get_file.assert_called_once()


def test_load_video_frames_failed_open() -> None:
    """Test handling of video open failure."""
    with patch("autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils.cv2") as mock_cv2:
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        
        frames = load_video_frames("/nonexistent.mp4", 0, 1_000_000_000)
        assert frames == []


def test_get_video_path_for_slice() -> None:
    """Test getting video path from search result."""
    search_result = SearchResult(
        row_id="slice123_segment_0_1000000000_camera_front",
        score=0.9,
    )
    
    file_ref = FileReference(
        path="s3://bucket/slice123.mp4",
        checksum="abc123",
        physical_address="s3://bucket/data/slice123.mp4",
    )
    
    manifest = Manifest(
        stage=Stage("test", "v1"),
        ref=GenericRef(LakefsRef("repo", "branch", "path")),
        key_to_data_file={"slice123": file_ref},
    )
    
    video_path = get_video_path_for_slice(search_result, manifest)
    assert video_path == "s3://bucket/data/slice123.mp4"


def test_get_video_path_not_found() -> None:
    """Test video path lookup when slice not found."""
    search_result = SearchResult(
        row_id="missing_slice_segment_0_1000000000_camera_front",
        score=0.9,
    )
    
    manifest = Manifest(
        stage=Stage("test", "v1"),
        ref=GenericRef(LakefsRef("repo", "branch", "path")),
        key_to_data_file={},
    )
    
    video_path = get_video_path_for_slice(search_result, manifest)
    assert video_path is None


def test_load_frames_for_search_result() -> None:
    """Test complete workflow of loading frames for a search result."""
    search_result = SearchResult(
        row_id="slice456_segment_1000000000_2000000000_camera_rear",
        score=0.85,
    )
    
    file_ref = FileReference(
        path="s3://bucket/slice456.mp4",
        checksum="def456",
        physical_address="s3://bucket/data/slice456.mp4",
    )
    
    manifest = Manifest(
        stage=Stage("test", "v1"),
        ref=GenericRef(LakefsRef("repo", "branch", "path")),
        key_to_data_file={"slice456": file_ref},
    )
    
    mock_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
    
    with patch(
        "autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils.load_video_frames"
    ) as mock_load:
        mock_load.return_value = mock_frames
        
        frames = load_frames_for_search_result(search_result, manifest)
        
        assert len(frames) == 4
        mock_load.assert_called_once()


def test_load_frames_no_video_found() -> None:
    """Test loading frames when video path not found."""
    search_result = SearchResult(
        row_id="missing_slice_segment_0_1000000000_camera_front",
        score=0.7,
    )
    
    manifest = Manifest(
        stage=Stage("test", "v1"),
        ref=GenericRef(LakefsRef("repo", "branch", "path")),
        key_to_data_file={},
    )
    
    frames = load_frames_for_search_result(search_result, manifest)
    assert frames == []

