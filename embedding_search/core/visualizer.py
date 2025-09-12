"""
Results Visualizer for displaying search results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging
from datetime import datetime
import zipfile
import tempfile
from PIL import Image

logger = logging.getLogger(__name__)


def _is_zip_path(input_path: Union[str, Path]) -> bool:
    """Check if input path is a zip file, including paths with fragments like file.zip#subfolder."""
    path_str = str(input_path)
    
    # Handle fragment syntax (file.zip#subfolder)
    if '#' in path_str:
        actual_path = path_str.split('#')[0]
        return Path(actual_path).suffix.lower() == '.zip'
    
    # Handle regular zip file
    return Path(path_str).suffix.lower() == '.zip'


def _parse_zip_path(input_path: Union[str, Path]) -> tuple[Path, str]:
    """Parse zip path and return (zip_file_path, subfolder_or_None)."""
    path_str = str(input_path)
    
    if '#' in path_str:
        zip_path_str, subfolder = path_str.split('#', 1)
        return Path(zip_path_str), subfolder
    
    return Path(path_str), None


class VideoResultsVisualizer:
    """Visualize video search results with thumbnails and similarity scores."""
    
    def __init__(self, thumbnail_size: tuple = (480, 270)):
        """
        Initialize the visualizer.
        
        Args:
            thumbnail_size: Size for video thumbnails (width, height) - default 16:9 aspect ratio
        """
        self.thumbnail_size = thumbnail_size
    
    def extract_thumbnail(self, video_path: Union[str, Path], frame_position: float = 0.5) -> np.ndarray:
        """
        Extract a thumbnail from a video file or zip file containing frames.
        
        Args:
            video_path: Path to the video file or zip file (supports zip#subfolder syntax)
            frame_position: Position in video to extract frame (0.0 to 1.0) - ignored for zip files
            
        Returns:
            Thumbnail image as numpy array
        """
        # Handle zip files (sensor_frame_zip)
        if _is_zip_path(video_path):
            return self._extract_thumbnail_from_zip(video_path)
        
        # Handle regular video files
        video_path = Path(video_path)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get frame at specified position
        frame_idx = int(total_frames * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Create placeholder if frame extraction fails
            frame = self._create_placeholder_thumbnail("No Preview")
        else:
            # Convert BGR to RGB and resize with high-quality interpolation
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.thumbnail_size, interpolation=cv2.INTER_LANCZOS4)
        
        return frame
    
    def _extract_thumbnail_from_zip(self, zip_path_input: Union[str, Path]) -> np.ndarray:
        """
        Extract thumbnail from the first frame in a zip file.
        
        Args:
            zip_path_input: Path to zip file (supports zip#subfolder syntax)
            
        Returns:
            Thumbnail image as numpy array
        """
        try:
            # Parse zip path and subfolder
            zip_path, subfolder = _parse_zip_path(zip_path_input)
            
            if not zip_path.exists():
                logger.warning(f"Zip file not found: {zip_path}")
                return self._create_placeholder_thumbnail("Zip Not Found")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                # Get all files from zip
                all_files = zip_file.namelist()
                
                # Filter files by subfolder if specified
                if subfolder:
                    subfolder_prefix = subfolder + '/' if not subfolder.endswith('/') else subfolder
                    filtered_files = [f for f in all_files if f.startswith(subfolder_prefix)]
                else:
                    filtered_files = all_files
                
                # Find image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                image_files = [f for f in filtered_files 
                              if Path(f).suffix.lower() in image_extensions and not f.startswith('__MACOSX/')]
                
                if not image_files:
                    logger.warning(f"No image files found in zip: {zip_path}")
                    return self._create_placeholder_thumbnail("No Images")
                
                # Sort to get the first frame (typically frame_0000.jpg or similar)
                image_files.sort()
                first_image = image_files[0]
                
                # Extract and load the first image
                with zip_file.open(first_image) as img_file:
                    img_data = img_file.read()
                    
                # Convert to PIL Image using BytesIO
                from io import BytesIO
                pil_image = Image.open(BytesIO(img_data))
                
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Resize to thumbnail size
                pil_image = pil_image.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                thumbnail_array = np.array(pil_image)
                
                logger.debug(f"Extracted thumbnail from zip {zip_path.name}, first frame: {first_image}")
                return thumbnail_array
                
        except Exception as e:
            logger.warning(f"Failed to extract thumbnail from zip {zip_path_input}: {e}")
            return self._create_placeholder_thumbnail("Error")
    
    def _create_placeholder_thumbnail(self, message: str = "No Preview") -> np.ndarray:
        """Create a placeholder thumbnail with a message."""
        frame = np.zeros((*self.thumbnail_size[::-1], 3), dtype=np.uint8)
        cv2.putText(frame, message, (10, self.thumbnail_size[1]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def visualize_video_search_results(self, query_video_path: Union[str, Path], 
                                     results: List[Dict], 
                                     save_path: Optional[Union[str, Path]] = None,
                                     show_interactive: bool = False,
                                     keep_open: bool = False) -> Path:
        """
        Visualize video-to-video search results.
        
        Args:
            query_video_path: Path to the query video
            results: List of search results
            save_path: Optional path to save the visualization
            show_interactive: If True, display the plot interactively
            keep_open: If True, keep the matplotlib window open (requires manual close)
            
        Returns:
            Path where visualization was saved
        """
        # Create figure
        num_results = len(results)
        fig_width = 4 * (num_results + 1)
        fig = plt.figure(figsize=(fig_width, 8))
        
        # Add title
        fig.suptitle("Video Similarity Search Results", fontsize=16, fontweight='bold')
        
        # Plot query video
        ax_query = plt.subplot(2, num_results + 1, 1)
        query_thumb = self.extract_thumbnail(query_video_path)
        ax_query.imshow(query_thumb)
        ax_query.set_title("Query Video\n" + Path(query_video_path).name, fontsize=12)
        ax_query.axis('off')
        
        # Add arrow
        ax_arrow = plt.subplot(2, num_results + 1, num_results + 2)
        ax_arrow.annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.8),
                         arrowprops=dict(arrowstyle='->', lw=3, color='green'))
        ax_arrow.set_xlim(0, 1)
        ax_arrow.set_ylim(0, 1)
        ax_arrow.axis('off')
        ax_arrow.text(0.5, 0.5, 'Similar Videos', ha='center', va='center', 
                     fontsize=14, fontweight='bold', color='green')
        
        # Plot results
        for i, result in enumerate(results):
            ax = plt.subplot(2, num_results + 1, num_results + 3 + i)
            
            # Extract thumbnail
            try:
                thumb = self.extract_thumbnail(result['video_path'])
                ax.imshow(thumb)
            except Exception as e:
                logger.error(f"Error loading thumbnail for {result['slice_id']}: {e}")
                ax.text(0.5, 0.5, "Error loading\nthumbnail", ha='center', va='center')
            
            # Add title with rank and similarity score
            title = f"Rank {result['rank']}\n{result['slice_id']}\nScore: {result['similarity_score']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"video_search_results_{timestamp}.png")
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Handle display options
        if show_interactive or keep_open:
            plt.show()
            if not keep_open:
                # Show briefly then close automatically
                plt.pause(3)
                plt.close()
            # If keep_open=True, user must manually close the window
        else:
            plt.close()
        
        logger.info(f"Visualization saved to {save_path}")
        return save_path
    
    def visualize_text_search_results(self, query_text: str, 
                                    results: List[Dict], 
                                    save_path: Optional[Union[str, Path]] = None,
                                    show_interactive: bool = False,
                                    keep_open: bool = False) -> Path:
        """
        Visualize text-to-video search results.
        
        Args:
            query_text: Text query used for search
            results: List of search results
            save_path: Optional path to save the visualization
            show_interactive: If True, display the plot interactively
            keep_open: If True, keep the matplotlib window open (requires manual close)
            
        Returns:
            Path where visualization was saved
        """
        # Create figure
        num_results = len(results)
        fig_width = 4 * num_results
        fig = plt.figure(figsize=(fig_width, 6))
        
        # Add title with query text
        fig.suptitle(f'Text-to-Video Search Results\nQuery: "{query_text}"', 
                    fontsize=16, fontweight='bold')
        
        # Plot results
        for i, result in enumerate(results):
            ax = plt.subplot(1, num_results, i + 1)
            
            # Extract thumbnail
            try:
                thumb = self.extract_thumbnail(result['video_path'])
                ax.imshow(thumb)
            except Exception as e:
                logger.error(f"Error loading thumbnail for {result['slice_id']}: {e}")
                ax.text(0.5, 0.5, "Error loading\nthumbnail", ha='center', va='center')
            
            # Add title with rank and similarity score
            title = f"Rank {result['rank']}\n{result['slice_id']}\nScore: {result['similarity_score']:.3f}"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"text_search_results_{timestamp}.png")
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Handle display options
        if show_interactive or keep_open:
            plt.show()
            if not keep_open:
                # Show briefly then close automatically
                plt.pause(3)
                plt.close()
            # If keep_open=True, user must manually close the window
        else:
            plt.close()
        
        logger.info(f"Visualization saved to {save_path}")
        return save_path
    
    def create_video_grid(self, video_paths: List[Union[str, Path]], 
                         grid_size: tuple = (3, 3),
                         save_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a grid visualization of multiple videos.
        
        Args:
            video_paths: List of video paths to include in grid
            grid_size: Grid dimensions (rows, cols)
            save_path: Optional path to save the visualization
            
        Returns:
            Path where visualization was saved
        """
        rows, cols = grid_size
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, ax in enumerate(axes_flat):
            if i < len(video_paths):
                try:
                    thumb = self.extract_thumbnail(video_paths[i])
                    ax.imshow(thumb)
                    ax.set_title(Path(video_paths[i]).name, fontsize=8)
                except Exception as e:
                    logger.error(f"Error loading video {video_paths[i]}: {e}")
                    ax.text(0.5, 0.5, "Error", ha='center', va='center')
            
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(f"video_grid_{timestamp}.png")
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
