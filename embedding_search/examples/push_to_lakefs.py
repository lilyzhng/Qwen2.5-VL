#!/usr/bin/env python3
"""
Simple example demonstrating how to push parquet files to LakeFS.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Push embeddings to LakeFS and verify upload."""
    print("🚀 Push Parquet Files to LakeFS")
    print("=" * 40)
    
    # Check if LakeFS is configured
    lakectl_path = Path.home() / '.lakectl.yaml'
    if not lakectl_path.exists():
        print("❌ ~/.lakectl.yaml not found!")
        print("   Please configure LakeFS credentials first.")
        return
    
    try:
        from core.config import VideoRetrievalConfig
        from core.search import VideoSearchEngine
        
        # Initialize with LakeFS enabled
        config = VideoRetrievalConfig()
        config.use_lakefs = True
        config.lakefs_repository = "embedding-search"
        
        print(f"Repository: {config.lakefs_repository}")
        print(f"Branch: {config.lakefs_branch}")
        
        # Create search engine (loads existing embeddings)
        print("Loading embeddings...")
        search_engine = VideoSearchEngine(config=config, use_gpu_faiss=False)
        
        # Get current stats
        stats = search_engine.database.get_statistics()
        print(f"Embeddings: {stats['total_embeddings']}")
        print(f"Size: {stats['database_size_mb']} MB")
        
        # Push to LakeFS
        print("Pushing to LakeFS...")
        search_engine.database.save()
        print("✅ Successfully pushed to LakeFS!")
        
        # Verify upload
        print("Verifying upload...")
        from lakefs_spec import LakeFSFileSystem
        fs = LakeFSFileSystem()
        
        repo_path = stats['lakefs_path']
        if fs.exists(repo_path):
            info = fs.info(repo_path)
            print(f"✅ Verified: {info['size'] / (1024*1024):.2f} MB")
            print(f"LakeFS path: {repo_path}")
        else:
            print("❌ File not found in LakeFS")
        
        print("\n🎉 Complete! Check LakeFS web UI: http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
