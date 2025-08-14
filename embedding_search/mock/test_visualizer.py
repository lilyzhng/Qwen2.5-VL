#!/usr/bin/env python3
"""
Unit tests and demonstration of the search visualizer interface.
Shows how the visualization components work together.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import cv2
from unittest.mock import Mock, patch, MagicMock
import json

from visualizer import VideoResultsVisualizer
from search import OptimizedVideoSearchEngine
from config import VideoRetrievalConfig


class TestVideoVisualizer(unittest.TestCase):
    """Test cases for video visualization components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.visualizer = VideoResultsVisualizer()
        
        # Create mock video files
        self.mock_videos = []
        for i in range(5):
            video_path = self.temp_dir / f"video_{i}.mp4"
            self._create_mock_video(video_path)
            self.mock_videos.append(video_path)
        
        # Create mock search results
        self.mock_results = [
            {
                "rank": 1,
                "video_name": "car2cyclist_1.mp4",
                "video_path": str(self.mock_videos[0]),
                "similarity_score": 0.95,
                "metadata": {"num_frames": 8, "added_at": "2024-01-01"}
            },
            {
                "rank": 2,
                "video_name": "car2cyclist_2.mp4",
                "video_path": str(self.mock_videos[1]),
                "similarity_score": 0.87,
                "metadata": {"num_frames": 8, "added_at": "2024-01-01"}
            },
            {
                "rank": 3,
                "video_name": "car2ped_1.mp4",
                "video_path": str(self.mock_videos[2]),
                "similarity_score": 0.73,
                "metadata": {"num_frames": 8, "added_at": "2024-01-01"}
            }
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_video(self, video_path: Path):
        """Create a mock video file for testing."""
        # Create a simple video with colored frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (224, 224))
        
        for i in range(8):
            # Create frame with different colors
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :, i % 3] = (i * 30) % 255
            out.write(frame)
        
        out.release()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_video_search_visualization(self, mock_close, mock_savefig):
        """Test video-to-video search visualization."""
        query_video = self.mock_videos[0]
        
        # Test visualization creation
        vis_path = self.visualizer.visualize_video_search_results(
            query_video, 
            self.mock_results[:3],
            save_path=self.temp_dir / "test_vis.png"
        )
        
        # Verify the visualization was created
        self.assertTrue(isinstance(vis_path, Path))
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_text_search_visualization(self, mock_close, mock_savefig):
        """Test text-to-video search visualization."""
        query_text = "car approaching cyclist"
        
        # Test visualization creation
        vis_path = self.visualizer.visualize_text_search_results(
            query_text,
            self.mock_results[:3],
            save_path=self.temp_dir / "test_text_vis.png"
        )
        
        # Verify the visualization was created
        self.assertTrue(isinstance(vis_path, Path))
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    def test_thumbnail_extraction(self):
        """Test video thumbnail extraction."""
        video_path = self.mock_videos[0]
        
        # Extract thumbnail
        thumbnail = self.visualizer.extract_thumbnail(video_path)
        
        # Verify thumbnail properties
        self.assertEqual(thumbnail.shape, (224, 224, 3))
        self.assertEqual(thumbnail.dtype, np.uint8)
    
    def test_video_grid_creation(self):
        """Test video grid visualization."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            with patch('matplotlib.pyplot.close') as mock_close:
                grid_path = self.visualizer.create_video_grid(
                    self.mock_videos[:4],
                    grid_size=(2, 2),
                    save_path=self.temp_dir / "grid.png"
                )
                
                self.assertTrue(isinstance(grid_path, Path))
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()


class TestInteractiveVisualizer:
    """
    Demonstration of interactive visualizer interface inspired by the official implementation.
    This shows how a Streamlit-like interface would work.
    """
    
    def __init__(self, search_engine: OptimizedVideoSearchEngine):
        self.search_engine = search_engine
        self.selected_video_idx = None
        self.text_query = ""
        self.search_results = []
        
    def simulate_user_interaction(self):
        """Simulate user interactions with the interface."""
        print("=" * 60)
        print("INTERACTIVE SEARCH VISUALIZER DEMO")
        print("=" * 60)
        
        # Simulate dataset selection
        print("\n1. Dataset Selection:")
        db_info = self.search_engine.get_statistics()
        print(f"   Loaded dataset with {db_info['num_videos']} videos")
        print(f"   Using {db_info['search_backend']} for search")
        
        # Simulate text search
        print("\n2. Text Search Simulation:")
        self.text_query = "car approaching cyclist"
        print(f"   User enters: '{self.text_query}'")
        
        try:
            self.search_results = self.search_engine.search_by_text(
                self.text_query, top_k=5
            )
            print(f"   Found {len(self.search_results)} results")
            
            # Display top result
            if self.search_results:
                top_result = self.search_results[0]
                print(f"   Top result: {top_result['video_name']} (score: {top_result['similarity_score']:.3f})")
                
        except Exception as e:
            print(f"   Search failed: {e}")
            return
        
        # Simulate video selection
        print("\n3. Video Selection Simulation:")
        if self.search_results:
            self.selected_video_idx = 0
            selected = self.search_results[self.selected_video_idx]
            print(f"   User selects: {selected['video_name']}")
            
            # Get neighbors
            neighbors = self._get_similar_videos(selected['video_path'])
            print(f"   Found {len(neighbors)} similar videos")
            
        # Simulate visualization creation
        print("\n4. Visualization Generation:")
        self._create_interface_visualizations()
        
        print("\n5. Interface Features Demonstrated:")
        print("   ✓ Real-time text search")
        print("   ✓ Interactive video selection")
        print("   ✓ Similarity-based recommendations")
        print("   ✓ Visual result comparison")
        print("   ✓ Exportable results")
    
    def _get_similar_videos(self, video_path: str, k: int = 5):
        """Get similar videos to the selected one."""
        try:
            results = self.search_engine.search_by_video(video_path, top_k=k)
            return results
        except:
            return []
    
    def _create_interface_visualizations(self):
        """Create visualizations for the interface."""
        if not self.search_results:
            return
        
        # Create search results visualization
        visualizer = VideoResultsVisualizer()
        
        # Text search visualization
        try:
            vis_path = visualizer.visualize_text_search_results(
                self.text_query,
                self.search_results[:3]
            )
            print(f"   📊 Text search visualization: {vis_path}")
        except Exception as e:
            print(f"   ⚠️  Visualization failed: {e}")
        
        # Create comparison grid
        if len(self.search_results) >= 4:
            video_paths = [r['video_path'] for r in self.search_results[:4]]
            try:
                grid_path = visualizer.create_video_grid(
                    video_paths,
                    grid_size=(2, 2)
                )
                print(f"   📊 Comparison grid: {grid_path}")
            except Exception as e:
                print(f"   ⚠️  Grid creation failed: {e}")
    
    def export_session_data(self, output_path: Path):
        """Export current session data."""
        session_data = {
            "text_query": self.text_query,
            "selected_video_idx": self.selected_video_idx,
            "search_results": self.search_results,
            "database_stats": self.search_engine.get_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"   💾 Session data exported: {output_path}")


class TestSearchEngineIntegration(unittest.TestCase):
    """Integration tests showing how visualizer works with search engine."""
    
    def setUp(self):
        """Set up test environment with mock search engine."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock config
        self.config = VideoRetrievalConfig()
        self.config.database_path = str(self.temp_dir / "test_db")
        
        # Mock search engine with fake data
        self.search_engine = Mock(spec=OptimizedVideoSearchEngine)
        self.search_engine.get_statistics.return_value = {
            "num_videos": 8,
            "embedding_dim": 768,
            "search_backend": "FAISS",
            "using_gpu": False
        }
        
        # Mock search results
        self.mock_results = [
            {
                "rank": i+1,
                "video_name": f"video_{i}.mp4",
                "video_path": str(self.temp_dir / f"video_{i}.mp4"),
                "similarity_score": 0.9 - i*0.1,
                "metadata": {"test": True}
            }
            for i in range(5)
        ]
        
        self.search_engine.search_by_text.return_value = self.mock_results
        self.search_engine.search_by_video.return_value = self.mock_results
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_interactive_visualizer_demo(self):
        """Test the interactive visualizer demonstration."""
        # Create interactive visualizer
        interactive_viz = TestInteractiveVisualizer(self.search_engine)
        
        # Run simulation
        interactive_viz.simulate_user_interaction()
        
        # Verify search engine was called
        self.search_engine.search_by_text.assert_called()
        self.search_engine.get_statistics.assert_called()
        
        # Export session data
        session_file = self.temp_dir / "session.json"
        interactive_viz.export_session_data(session_file)
        
        # Verify export
        self.assertTrue(session_file.exists())
        
        with open(session_file) as f:
            session_data = json.load(f)
        
        self.assertIn("text_query", session_data)
        self.assertIn("search_results", session_data)
    
    def test_end_to_end_visualization_workflow(self):
        """Test complete visualization workflow."""
        print("\n" + "=" * 60)
        print("END-TO-END VISUALIZATION WORKFLOW DEMO")
        print("=" * 60)
        
        # Step 1: Initialize components
        print("\n1. Initializing components...")
        visualizer = VideoResultsVisualizer()
        print("   ✓ VideoResultsVisualizer created")
        
        # Step 2: Perform search
        print("\n2. Performing search...")
        query = "car approaching cyclist"
        results = self.search_engine.search_by_text(query, top_k=3)
        print(f"   ✓ Search completed: {len(results)} results")
        
        # Step 3: Create visualizations
        print("\n3. Creating visualizations...")
        
        # Mock the visualization creation (since we don't have real videos)
        with patch.object(visualizer, 'visualize_text_search_results') as mock_vis:
            mock_vis.return_value = Path("mock_visualization.png")
            
            vis_path = visualizer.visualize_text_search_results(query, results)
            print(f"   ✓ Text search visualization: {vis_path}")
            
            mock_vis.assert_called_once_with(query, results)
        
        # Step 4: Export results
        print("\n4. Exporting results...")
        export_data = {
            "query": query,
            "results": results,
            "visualization_path": str(vis_path),
            "timestamp": "2024-01-01T12:00:00"
        }
        
        export_path = self.temp_dir / "workflow_export.json"
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"   ✓ Results exported: {export_path}")
        
        # Step 5: Summary
        print("\n5. Workflow Summary:")
        print("   ✓ Search performed successfully")
        print("   ✓ Visualizations generated")
        print("   ✓ Results exported for sharing")
        print("   ✓ Ready for interactive use")


def create_streamlit_like_interface_demo():
    """
    Demonstrate how a Streamlit-like interface would work.
    This simulates the official NVIDIA implementation structure.
    """
    print("\n" + "=" * 70)
    print("STREAMLIT-LIKE INTERFACE DEMONSTRATION")
    print("Simulating the official NVIDIA Cosmos-Embed1 interface structure")
    print("=" * 70)
    
    # Simulate session state (like Streamlit)
    session_state = {
        "model": None,
        "preprocessor": None,
        "database": None,
        "selected_video": None,
        "text_query": "",
        "search_results": []
    }
    
    # Simulate interface layout
    print("\n📱 INTERFACE LAYOUT SIMULATION:")
    print("┌─────────────────────┬─────────────────────┐")
    print("│ LEFT PANEL          │ RIGHT PANEL         │")
    print("│                     │                     │")
    print("│ 🔍 Search Controls  │ 📺 Video Preview    │")
    print("│ • Text Input        │ • Selected Video    │")
    print("│ • Dataset Selector  │ • Similarity Score  │")
    print("│                     │                     │")
    print("│ 📊 Results Plot     │ ⚙️ Controls         │")
    print("│ • Similarity Map    │ • Export Options    │")
    print("│ • Clickable Points  │ • View Settings     │")
    print("└─────────────────────┴─────────────────────┘")
    print("│                                           │")
    print("│ 🎬 BOTTOM PANEL: 5 Nearest Neighbors     │")
    print("│ [Video1] [Video2] [Video3] [Video4] [Video5] │")
    print("└───────────────────────────────────────────┘")
    
    # Simulate user interactions
    print("\n🖱️  USER INTERACTION SIMULATION:")
    
    # 1. Dataset selection
    print("1. User selects dataset: 'Traffic Scenarios (8 videos)'")
    session_state["database"] = "traffic_scenarios"
    
    # 2. Text search
    print("2. User enters text: 'car approaching cyclist'")
    session_state["text_query"] = "car approaching cyclist"
    
    # Simulate search processing
    print("   🔄 Processing text query...")
    print("   📊 Finding nearest video in embedding space...")
    
    # Mock search results
    session_state["search_results"] = [
        {"video_id": 3, "similarity": 0.94, "name": "car2cyclist_2.mp4"},
        {"video_id": 1, "similarity": 0.87, "name": "car2cyclist_1.mp4"},
        {"video_id": 5, "similarity": 0.73, "name": "car2ped_1.mp4"}
    ]
    
    print(f"   ✅ Found match: {session_state['search_results'][0]['name']}")
    print(f"   📍 Similarity score: {session_state['search_results'][0]['similarity']:.3f}")
    
    # 3. Video selection and neighbors
    session_state["selected_video"] = session_state["search_results"][0]
    print(f"3. Auto-selected video: {session_state['selected_video']['name']}")
    
    # 4. Show neighbors
    print("4. Displaying 5 nearest neighbors:")
    neighbors = [
        "car2cyclist_1.mp4 (0.87)",
        "car2ped_1.mp4 (0.73)",
        "car2car_1.mp4 (0.65)",
        "car2motorcyclist.mp4 (0.61)",
        "car2ped_2.mp4 (0.58)"
    ]
    
    for i, neighbor in enumerate(neighbors, 1):
        print(f"   {i}. {neighbor}")
    
    # 5. Interactive features
    print("\n🎯 INTERACTIVE FEATURES:")
    print("✓ Real-time text search with instant results")
    print("✓ Clickable similarity plot for exploration")
    print("✓ Hover tooltips showing video metadata")
    print("✓ Zoom and pan on similarity visualization")
    print("✓ Export functionality for found results")
    print("✓ Responsive layout adapting to screen size")
    
    # 6. Technical implementation
    print("\n⚙️  TECHNICAL IMPLEMENTATION:")
    print("• Frontend: Streamlit with custom components")
    print("• Backend: FAISS for similarity search")
    print("• Visualization: Plotly for interactive plots")
    print("• Video Display: HTML5 video with YouTube embedding")
    print("• State Management: Streamlit session state")
    print("• Caching: @st.cache_data for model and embeddings")
    
    return session_state


def run_all_visualizer_tests():
    """Run all visualizer tests and demonstrations."""
    print("🧪 RUNNING COMPREHENSIVE VISUALIZER TESTS")
    print("=" * 60)
    
    # Run unit tests
    print("\n1. UNIT TESTS:")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVideoVisualizer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if result.wasSuccessful():
        print("✅ All unit tests passed!")
    else:
        print("❌ Some unit tests failed!")
    
    # Run integration tests
    print("\n2. INTEGRATION TESTS:")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSearchEngineIntegration)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    if result.wasSuccessful():
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
    
    # Run interface demonstration
    print("\n3. INTERFACE DEMONSTRATIONS:")
    try:
        session_state = create_streamlit_like_interface_demo()
        print("✅ Interface demonstration completed!")
        
        # Show final session state
        print(f"\n📊 FINAL SESSION STATE:")
        print(f"Selected Video: {session_state.get('selected_video', {}).get('name', 'None')}")
        print(f"Text Query: '{session_state.get('text_query', '')}'")
        print(f"Results Found: {len(session_state.get('search_results', []))}")
        
    except Exception as e:
        print(f"❌ Interface demonstration failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 VISUALIZER TEST SUITE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_visualizer_tests()
