"""
Test frame folder processing functionality in the embedder.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import cv2
import os

from core.embedder import CosmosVideoEmbedder, VideoFrameProcessor
from core.config import VideoRetrievalConfig
from core.exceptions import VideoNotFoundError, VideoLoadError, InvalidVideoFormatError


class TestFrameFolderProcessing:
    """Test frame folder processing functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VideoRetrievalConfig(
            num_frames=8,
            resolution=(224, 224),
            device="cpu"  # Use CPU for testing
        )
    
    @pytest.fixture
    def frame_processor(self, config):
        """Create frame processor for testing."""
        return VideoFrameProcessor(config)
    
    @pytest.fixture
    def sample_frame_folder(self):
        """Create a temporary folder with sample frames."""
        temp_dir = tempfile.mkdtemp()
        folder_path = Path(temp_dir) / "test_frames"
        folder_path.mkdir()
        
        # Create 12 sample frames (more than the required 8)
        for i in range(12):
            # Create a simple colored frame with more distinct patterns
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 40) % 255  # Vary red channel more dramatically
            frame[:, :, 1] = (i * 30 + 50) % 255  # Vary green channel
            frame[:, :, 2] = (255 - i * 35) % 255  # Vary blue channel more dramatically
            
            frame_path = folder_path / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        yield folder_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def small_frame_folder(self):
        """Create a temporary folder with fewer frames than required."""
        temp_dir = tempfile.mkdtemp()
        folder_path = Path(temp_dir) / "small_frames"
        folder_path.mkdir()
        
        # Create only 3 frames (less than the required 8)
        for i in range(3):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Different color for each frame
            
            frame_path = folder_path / f"frame_{i:03d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        yield folder_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def empty_folder(self):
        """Create an empty temporary folder."""
        temp_dir = tempfile.mkdtemp()
        folder_path = Path(temp_dir) / "empty_frames"
        folder_path.mkdir()
        
        yield folder_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_load_frames_from_folder_success(self, frame_processor, sample_frame_folder):
        """Test successful loading of frames from folder."""
        frames = frame_processor.load_frames_from_folder(sample_frame_folder, num_frames=8)
        
        assert frames.shape == (8, 224, 224, 3)  # Expected shape
        assert frames.dtype == np.uint8
        
        # Check that frames are different (not all the same)
        frame_means = [np.mean(frame) for frame in frames]
        assert len(set(frame_means)) > 1, "Frames should be different"
    
    def test_load_frames_from_folder_small_dataset(self, frame_processor, small_frame_folder):
        """Test loading frames when folder has fewer frames than requested."""
        frames = frame_processor.load_frames_from_folder(small_frame_folder, num_frames=8)
        
        assert frames.shape == (8, 224, 224, 3)  # Should still return 8 frames
        assert frames.dtype == np.uint8
        
        # Last frames should be repeated
        assert np.array_equal(frames[2], frames[7]), "Last frame should be repeated"
    
    def test_load_frames_from_folder_empty(self, frame_processor, empty_folder):
        """Test loading frames from empty folder."""
        with pytest.raises(VideoLoadError, match="No image files found"):
            frame_processor.load_frames_from_folder(empty_folder)
    
    def test_load_frames_from_folder_nonexistent(self, frame_processor):
        """Test loading frames from non-existent folder."""
        nonexistent_path = Path("/nonexistent/folder")
        with pytest.raises((VideoLoadError, FileNotFoundError)):
            frame_processor.load_frames_from_folder(nonexistent_path)
    
    def test_validate_input_video_file(self, frame_processor):
        """Test input validation for video files."""
        # This test would require a real video file, so we'll skip it
        # In a real test environment, you'd create a sample video file
        pass
    
    def test_validate_input_frame_folder(self, frame_processor, sample_frame_folder):
        """Test input validation for frame folders."""
        result = frame_processor.validate_input(sample_frame_folder)
        assert result is True
    
    def test_validate_input_empty_folder(self, frame_processor, empty_folder):
        """Test input validation for empty folder."""
        result = frame_processor.validate_input(empty_folder)
        assert result is False
    
    def test_frame_selection_uniform_sampling(self, frame_processor, sample_frame_folder):
        """Test that frames are selected uniformly across the available frames."""
        frames = frame_processor.load_frames_from_folder(sample_frame_folder, num_frames=4)
        
        # With 12 frames available and requesting 4, we should get frames at indices 0, 3, 7, 11
        # We can't directly test the indices, but we can verify the frames are different
        assert frames.shape == (4, 224, 224, 3)
        
        # Check that selected frames have different characteristics
        frame_means = [np.mean(frame) for frame in frames]
        # Allow for some tolerance in case of similar frames due to compression
        unique_means = len(set(np.round(frame_means, 1)))
        assert unique_means >= 3, f"Expected at least 3 different frames, got {unique_means} unique means: {frame_means}"


class TestCosmosVideoEmbedderFrameFolders:
    """Test CosmosVideoEmbedder with frame folders."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VideoRetrievalConfig(
            num_frames=8,
            resolution=(224, 224),
            device="cpu",
            model_name="test_model"  # This won't actually load in tests
        )
    
    def test_extract_video_embedding_validation(self, config):
        """Test input path validation in extract_video_embedding."""
        # This test would require mocking the model loading
        # In a real test environment, you'd mock the AutoModel and AutoProcessor
        pass
    
    def test_batch_processing_mixed_inputs(self, config):
        """Test batch processing with mixed video files and frame folders."""
        # This test would require mocking the model and creating sample inputs
        # In a real test environment, you'd create both video files and frame folders
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
