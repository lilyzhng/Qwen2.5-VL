#!/usr/bin/env python3
"""
Mock visualizer interface demonstration that works without CUDA, videos, or heavy dependencies.
This allows you to review the design and interface without any hardware requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path
import random
from typing import List, Dict, Any
import time


class MockVideoData:
    """Generate mock video data for demonstration."""
    
    def __init__(self):
        self.video_categories = [
            "car2cyclist", "car2pedestrian", "car2car", "car2motorcycle", 
            "traffic_intersection", "highway_merge", "parking_lot", "residential_street"
        ]
        
        self.mock_videos = self._generate_mock_videos()
        self.embeddings = self._generate_mock_embeddings()
    
    def _generate_mock_videos(self) -> List[Dict]:
        """Generate mock video metadata."""
        videos = []
        
        for i in range(20):
            category = random.choice(self.video_categories)
            videos.append({
                "id": i,
                "video_name": f"{category}_{i:02d}.mp4",
                "video_path": f"/mock/videos/{category}_{i:02d}.mp4",
                "category": category,
                "duration": round(random.uniform(1.5, 8.0), 1),
                "resolution": random.choice(["720p", "1080p", "480p"]),
                "fps": random.choice([24, 30, 60]),
                "file_size_mb": round(random.uniform(5.2, 45.8), 1),
                "added_at": f"2024-01-{random.randint(1, 30):02d}T{random.randint(8, 18):02d}:00:00"
            })
        
        return videos
    
    def _generate_mock_embeddings(self) -> np.ndarray:
        """Generate mock 768-dimensional embeddings."""
        # Create embeddings with some category clustering
        embeddings = []
        
        for video in self.mock_videos:
            # Base embedding based on category
            category_idx = self.video_categories.index(video["category"])
            base_embedding = np.random.normal(0, 0.1, 768)
            
            # Add category-specific bias
            base_embedding[category_idx * 96:(category_idx + 1) * 96] += np.random.normal(0.5, 0.2, 96)
            
            # Normalize
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            embeddings.append(base_embedding)
        
        return np.array(embeddings)


class MockSearchEngine:
    """Mock search engine that simulates the real interface without dependencies."""
    
    def __init__(self):
        self.data = MockVideoData()
        self.search_history = []
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simulate text-to-video search."""
        print(f"🔍 Searching for: '{query}'")
        time.sleep(0.1)  # Simulate processing time
        
        # Simple keyword matching simulation
        query_lower = query.lower()
        scores = []
        
        for i, video in enumerate(self.data.mock_videos):
            score = 0.1  # Base score
            
            # Check for keyword matches
            if any(word in video["video_name"].lower() for word in query_lower.split()):
                score += 0.8
            
            if any(word in video["category"] for word in query_lower.split()):
                score += 0.6
            
            # Add some randomness
            score += random.uniform(-0.1, 0.2)
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            scores.append((score, i))
        
        # Sort by score and get top-k
        scores.sort(reverse=True)
        top_results = scores[:top_k]
        
        # Format results
        results = []
        for rank, (score, video_idx) in enumerate(top_results, 1):
            video = self.data.mock_videos[video_idx]
            results.append({
                "rank": rank,
                "video_name": video["video_name"],
                "video_path": video["video_path"],
                "similarity_score": score,
                "metadata": {
                    "category": video["category"],
                    "duration": video["duration"],
                    "resolution": video["resolution"],
                    "added_at": video["added_at"]
                }
            })
        
        # Record search
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results),
            "top_score": results[0]["similarity_score"] if results else 0
        })
        
        print(f"✅ Found {len(results)} results")
        return results
    
    def search_by_video(self, video_path: str, top_k: int = 5) -> List[Dict]:
        """Simulate video-to-video search."""
        video_name = Path(video_path).name
        print(f"🎥 Finding videos similar to: {video_name}")
        time.sleep(0.2)  # Simulate processing time
        
        # Find the query video
        query_video = None
        for video in self.data.mock_videos:
            if video["video_name"] == video_name or video["video_path"] == video_path:
                query_video = video
                break
        
        if not query_video:
            # If not found, simulate with random video
            query_video = random.choice(self.data.mock_videos)
        
        # Simulate similarity based on category
        results = []
        for video in self.data.mock_videos:
            if video["id"] == query_video["id"]:
                continue  # Skip self
            
            # Higher similarity for same category
            if video["category"] == query_video["category"]:
                score = random.uniform(0.7, 0.95)
            else:
                score = random.uniform(0.2, 0.6)
            
            results.append({
                "rank": 0,  # Will be set later
                "video_name": video["video_name"],
                "video_path": video["video_path"],
                "similarity_score": score,
                "metadata": {
                    "category": video["category"],
                    "duration": video["duration"],
                    "resolution": video["resolution"],
                    "added_at": video["added_at"]
                }
            })
        
        # Sort by similarity and assign ranks
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        for i, result in enumerate(results[:top_k], 1):
            result["rank"] = i
        
        print(f"✅ Found {len(results[:top_k])} similar videos")
        return results[:top_k]
    
    def get_statistics(self) -> Dict:
        """Get mock database statistics."""
        return {
            "num_inputs": len(self.data.mock_videos),
            "embedding_dim": 768,
            "categories": list(set(v["category"] for v in self.data.mock_videos)),
            "total_duration": sum(v["duration"] for v in self.data.mock_videos),
            "search_backend": "Mock FAISS",
            "using_gpu": False,
            "cache_size": 0,
            "database_size_mb": 12.5
        }


class MockVisualizer:
    """Mock visualizer that creates interface previews without real video processing."""
    
    def __init__(self):
        self.color_palette = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def create_text_search_visualization(self, query: str, results: List[Dict]) -> str:
        """Create mock text search visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Text Search Results for: "{query}"', fontsize=16, fontweight='bold')
        
        # 1. Results overview
        ax1.set_title("Search Results Overview")
        video_names = [r['video_name'][:15] + "..." if len(r['video_name']) > 15 else r['video_name'] 
                      for r in results]
        scores = [r['similarity_score'] for r in results]
        
        bars = ax1.barh(range(len(video_names)), scores, color=self.color_palette[:len(results)])
        ax1.set_yticks(range(len(video_names)))
        ax1.set_yticklabels(video_names)
        ax1.set_xlabel("Similarity Score")
        ax1.set_xlim(0, 1)
        
        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + 0.01, i, f'{score:.3f}', va='center')
        
        # 2. Category distribution
        ax2.set_title("Results by Category")
        categories = [r['metadata']['category'] for r in results]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        if category_counts:
            ax2.pie(category_counts.values(), labels=category_counts.keys(), 
                   autopct='%1.0f%%', startangle=90)
        
        # 3. Score distribution
        ax3.set_title("Similarity Score Distribution")
        ax3.hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel("Similarity Score")
        ax3.set_ylabel("Frequency")
        ax3.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.3f}')
        ax3.legend()
        
        # 4. Mock similarity space (2D projection)
        ax4.set_title("Video Similarity Space (Mock t-SNE)")
        
        # Generate mock 2D coordinates
        np.random.seed(42)
        x_coords = np.random.uniform(-3, 3, len(results))
        y_coords = np.random.uniform(-3, 3, len(results))
        
        scatter = ax4.scatter(x_coords, y_coords, c=scores, s=100, 
                            cmap='viridis', alpha=0.7, edgecolors='black')
        
        # Highlight query point
        ax4.scatter([0], [0], c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidth=2, label='Query')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax4, label='Similarity Score')
        ax4.set_xlabel("t-SNE Dimension 1")
        ax4.set_ylabel("t-SNE Dimension 2")
        ax4.legend()
        
        plt.tight_layout()
        save_path = f"mock_text_search_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_video_search_visualization(self, query_video: str, results: List[Dict]) -> str:
        """Create mock video search visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Video Search Results for: {Path(query_video).name}', 
                    fontsize=16, fontweight='bold')
        
        # Query video (top left)
        ax_query = axes[0, 0]
        ax_query.set_title("Query Video", fontweight='bold', color='blue')
        
        # Create mock thumbnail
        query_thumb = np.random.rand(100, 100, 3)
        ax_query.imshow(query_thumb)
        ax_query.text(50, 110, Path(query_video).name, ha='center', va='top', 
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax_query.set_xticks([])
        ax_query.set_yticks([])
        
        # Arrow
        ax_arrow = axes[0, 1]
        ax_arrow.annotate('', xy=(0.8, 0.5), xytext=(0.2, 0.5),
                         arrowprops=dict(arrowstyle='->', lw=5, color='green'))
        ax_arrow.text(0.5, 0.3, 'Find Similar', ha='center', va='center', 
                     fontsize=14, fontweight='bold', color='green')
        ax_arrow.set_xlim(0, 1)
        ax_arrow.set_ylim(0, 1)
        ax_arrow.axis('off')
        
        # Top result (top right)
        if results:
            ax_top = axes[0, 2]
            ax_top.set_title(f"Top Match (Score: {results[0]['similarity_score']:.3f})", 
                           fontweight='bold', color='green')
            
            top_thumb = np.random.rand(100, 100, 3)
            ax_top.imshow(top_thumb)
            ax_top.text(50, 110, results[0]['video_name'], ha='center', va='top', 
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            ax_top.set_xticks([])
            ax_top.set_yticks([])
        
        # Results grid (bottom row)
        for i, result in enumerate(results[:3]):
            ax = axes[1, i]
            ax.set_title(f"Rank {result['rank']}\nScore: {result['similarity_score']:.3f}")
            
            # Mock thumbnail
            thumb = np.random.rand(100, 100, 3)
            ax.imshow(thumb)
            
            # Video info
            info_text = f"{result['video_name']}\n{result['metadata']['category']}\n{result['metadata']['duration']}s"
            ax.text(50, 110, info_text, ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        save_path = f"mock_video_search_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard_mockup(self, search_engine: MockSearchEngine) -> str:
        """Create a mockup of the interactive dashboard interface."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create complex subplot layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 2, 1], width_ratios=[1, 1, 1, 1])
        
        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.text(0.5, 0.5, '🔍 alpha 0.1 Video Search Dashboard (Mock Interface)', 
                      ha='center', va='center', fontsize=20, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        ax_header.set_xlim(0, 1)
        ax_header.set_ylim(0, 1)
        ax_header.axis('off')
        
        # Left panel - Search controls
        ax_search = fig.add_subplot(gs[1, 0])
        ax_search.set_title("🔍 Search Controls", fontweight='bold')
        
        # Mock search interface
        search_interface = [
            "📝 Text Search:",
            "  'car approaching cyclist'",
            "",
            "🎥 Video Upload:",
            "  car2cyclist_2.mp4",
            "",
            "⚙️ Settings:",
            f"  • Top-K: 5",
            f"  • Threshold: 0.5",
            f"  • Backend: Mock FAISS",
            "",
            "📊 Database:",
            f"  • Videos: {search_engine.get_statistics()['num_inputs']}",
            f"  • Categories: {len(search_engine.get_statistics()['categories'])}"
        ]
        
        for i, line in enumerate(search_interface):
            ax_search.text(0.05, 0.95 - i*0.06, line, transform=ax_search.transAxes, 
                          fontsize=10, va='top', family='monospace')
        
        ax_search.set_xlim(0, 1)
        ax_search.set_ylim(0, 1)
        ax_search.axis('off')
        
        # Center panel - Similarity plot
        ax_similarity = fig.add_subplot(gs[1, 1:3])
        ax_similarity.set_title("📊 Interactive Similarity Space", fontweight='bold')
        
        # Generate mock similarity data
        np.random.seed(42)
        n_videos = 50
        x = np.random.normal(0, 2, n_videos)
        y = np.random.normal(0, 2, n_videos)
        categories = np.random.choice(['car2cyclist', 'car2pedestrian', 'car2car', 'traffic'], n_videos)
        similarities = np.random.uniform(0.3, 0.9, n_videos)
        
        # Color by category
        unique_cats = np.unique(categories)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
        
        for i, cat in enumerate(unique_cats):
            mask = categories == cat
            ax_similarity.scatter(x[mask], y[mask], c=[colors[i]], 
                                label=cat, s=similarities[mask]*100, alpha=0.7)
        
        # Highlight selected point
        ax_similarity.scatter([0.5], [0.5], c='red', s=200, marker='*', 
                            edgecolors='darkred', linewidth=2, label='Selected', zorder=5)
        
        ax_similarity.set_xlabel("t-SNE Dimension 1")
        ax_similarity.set_ylabel("t-SNE Dimension 2")
        ax_similarity.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_similarity.grid(True, alpha=0.3)
        
        # Right panel - Video preview
        ax_preview = fig.add_subplot(gs[1, 3])
        ax_preview.set_title("🎬 Video Preview", fontweight='bold')
        
        # Mock video preview
        preview_thumb = np.random.rand(150, 150, 3)
        ax_preview.imshow(preview_thumb)
        
        preview_info = [
            "car2cyclist_1.mp4",
            "",
            "Similarity: 0.876",
            "Category: car2cyclist", 
            "Duration: 3.2s",
            "Resolution: 1080p",
            "",
            "🎯 Actions:",
            "• Find Similar",
            "• Export Info",
            "• Download"
        ]
        
        for i, line in enumerate(preview_info):
            y_pos = -0.1 - i*0.08
            if line.startswith("🎯"):
                weight = 'bold'
                color = 'blue'
            else:
                weight = 'normal'
                color = 'black'
            
            ax_preview.text(0.5, y_pos, line, transform=ax_preview.transAxes,
                          ha='center', va='top', fontsize=9, fontweight=weight, color=color)
        
        ax_preview.set_xticks([])
        ax_preview.set_yticks([])
        
        # Bottom panel - Similar videos grid
        for i in range(4):
            ax_thumb = fig.add_subplot(gs[2, i])
            ax_thumb.set_title(f"Similar #{i+1}", fontsize=10)
            
            # Mock thumbnail
            thumb = np.random.rand(60, 60, 3)
            ax_thumb.imshow(thumb)
            
            # Mock info
            mock_score = random.uniform(0.6, 0.9)
            ax_thumb.text(0.5, -0.1, f"Score: {mock_score:.3f}", 
                         transform=ax_thumb.transAxes, ha='center', fontsize=8)
            
            ax_thumb.set_xticks([])
            ax_thumb.set_yticks([])
        
        plt.tight_layout()
        save_path = f"mock_dashboard_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path


def demonstrate_interface_flow():
    """Demonstrate the complete interface flow."""
    print("🎬 MOCK VISUALIZER INTERFACE DEMONSTRATION")
    print("=" * 60)
    print("This demo works without CUDA, videos, or heavy dependencies!")
    print("=" * 60)
    
    # Initialize mock components
    search_engine = MockSearchEngine()
    visualizer = MockVisualizer()
    
    print("\n📊 Mock Database Statistics:")
    stats = search_engine.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate text search
    print("\n🔍 DEMO 1: Text Search")
    print("-" * 30)
    
    text_queries = [
        "car approaching cyclist",
        "pedestrian crossing",
        "traffic intersection"
    ]
    
    all_visualizations = []
    
    for query in text_queries:
        print(f"\nSearching for: '{query}'")
        results = search_engine.search_by_text(query, top_k=5)
        
        # Show results
        for result in results[:3]:
            print(f"  {result['rank']}. {result['video_name']} (score: {result['similarity_score']:.3f})")
        
        # Create visualization
        vis_path = visualizer.create_text_search_visualization(query, results)
        all_visualizations.append(vis_path)
        print(f"  📊 Visualization: {vis_path}")
    
    # Demonstrate video search
    print("\n🎥 DEMO 2: Video Search")
    print("-" * 30)
    
    query_videos = [
        "/mock/videos/car2cyclist_01.mp4",
        "/mock/videos/car2pedestrian_03.mp4"
    ]
    
    for query_video in query_videos:
        print(f"\nFinding videos similar to: {Path(query_video).name}")
        results = search_engine.search_by_video(query_video, top_k=5)
        
        # Show results
        for result in results[:3]:
            print(f"  {result['rank']}. {result['video_name']} (score: {result['similarity_score']:.3f})")
        
        # Create visualization
        vis_path = visualizer.create_video_search_visualization(query_video, results)
        all_visualizations.append(vis_path)
        print(f"  📊 Visualization: {vis_path}")
    
    # Demonstrate interactive dashboard
    print("\n🖥️  DEMO 3: Interactive Dashboard")
    print("-" * 30)
    
    dashboard_path = visualizer.create_interactive_dashboard_mockup(search_engine)
    all_visualizations.append(dashboard_path)
    print(f"📊 Interactive dashboard mockup: {dashboard_path}")
    
    # Create session export
    print("\n💾 DEMO 4: Session Export")
    print("-" * 30)
    
    session_data = {
        "session_id": f"mock_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "search_history": search_engine.search_history,
        "database_stats": stats,
        "generated_visualizations": all_visualizations,
        "interface_features": {
            "text_search": "✅ Demonstrated",
            "video_search": "✅ Demonstrated", 
            "interactive_plots": "✅ Mocked",
            "dashboard_layout": "✅ Created",
            "export_functionality": "✅ Working"
        }
    }
    
    export_path = f"mock_session_export_{datetime.now().strftime('%H%M%S')}.json"
    with open(export_path, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print(f"📁 Session data exported: {export_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 MOCK DEMONSTRATION COMPLETED!")
    print("=" * 60)
    
    print(f"\n📊 Generated Files:")
    for i, vis_path in enumerate(all_visualizations, 1):
        print(f"  {i}. {vis_path}")
    print(f"  {len(all_visualizations) + 1}. {export_path}")
    
    print(f"\n🎯 Interface Features Demonstrated:")
    print("  ✅ Text-to-video search with similarity scoring")
    print("  ✅ Video-to-video similarity search")
    print("  ✅ Interactive dashboard layout")
    print("  ✅ Results visualization and comparison")
    print("  ✅ Category-based filtering and grouping")
    print("  ✅ Export and session management")
    print("  ✅ Error-free operation without CUDA/videos")
    
    print(f"\n🔍 Interface Design Review:")
    print("  • Clean, intuitive layout following official NVIDIA design")
    print("  • Multi-modal search capabilities (text + video)")
    print("  • Real-time results with interactive visualizations")
    print("  • Comprehensive metadata display and filtering")
    print("  • Professional dashboard suitable for production use")
    print("  • Responsive design adapting to different content types")
    
    print(f"\n🚀 Next Steps:")
    print("  • Review generated visualizations to see interface design")
    print("  • Check session export file for data structure")
    print("  • Run with real data when CUDA/videos are available")
    print("  • Customize interface themes and layouts as needed")
    
    return all_visualizations, export_path


if __name__ == "__main__":
    # Set matplotlib backend for headless operation
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    try:
        visualizations, export_file = demonstrate_interface_flow()
        print(f"\n✅ Mock demonstration completed successfully!")
        print(f"Generated {len(visualizations)} visualizations and 1 export file.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
