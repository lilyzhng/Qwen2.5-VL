#!/usr/bin/env python3
"""
Example script demonstrating LakeFS integration for embedding storage.

This script shows how to:
1. Configure LakeFS settings
2. Initialize the search engine with LakeFS backend
3. Build and store embeddings in LakeFS
4. Search using LakeFS-stored embeddings
"""

import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.config import VideoRetrievalConfig
from core.search import VideoSearchEngine

def main():
    """Demonstrate LakeFS integration."""
    
    print("Using LakeFS credentials from ~/.lakectl.yaml...")
    print("Make sure you have ~/.lakectl.yaml configured with your LakeFS credentials.")
    
    # Configure via config object (credentials read from ~/.lakectl.yaml)
    config = VideoRetrievalConfig(
        use_lakefs=True,
        lakefs_repository="embedding-search",
        lakefs_branch="main",
        lakefs_embeddings_path="data/unified_embeddings.parquet",
        lakefs_input_path="data/unified_input_path.parquet"
        # No need to set credentials - they're read from ~/.lakectl.yaml
    )
    
    print(f"LakeFS Configuration:")
    print(f"  Repository: {config.lakefs_repository}")
    print(f"  Branch: {config.lakefs_branch}")
    print(f"  Embeddings Path: {config.lakefs_embeddings_path}")
    
    try:
        # Initialize search engine with LakeFS backend
        print("\nInitializing VideoSearchEngine with LakeFS backend...")
        search_engine = VideoSearchEngine(config=config, use_gpu_faiss=False)
        
        # The search engine will automatically:
        # 1. Try to load existing embeddings from LakeFS
        # 2. Fall back to local cache if LakeFS is unavailable
        # 3. Create empty database if neither exists
        
        print("✅ Successfully initialized LakeFS backend!")
        
        # Build embeddings database (if you have input data)
        # Uncomment this if you have a unified_input_path.parquet file
        # print("\nBuilding embeddings database...")
        # search_engine.build_database_from_parquet(force_rebuild=False)
        
        # Get database statistics
        stats = search_engine.database.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Storage Backend: {stats.get('storage_backend', 'Local')}")
        print(f"  Total Embeddings: {stats.get('total_embeddings', 0)}")
        print(f"  LakeFS Repository: {stats.get('lakefs_repository', 'N/A')}")
        print(f"  LakeFS Path: {stats.get('lakefs_path', 'N/A')}")
        
        # Example search (if you have embeddings)
        # Uncomment this if you have built the database
        # print("\nPerforming example search...")
        # results = search_engine.search_by_text("car approaching cyclist", top_k=3)
        # print(f"Found {len(results)} results")
        
    except ImportError as e:
        if "lakefs-spec" in str(e):
            print("❌ Error: lakefs-spec not installed. Install with:")
            print("   pip install lakefs-spec")
        else:
            print(f"❌ Import error: {e}")
    except ConnectionError as e:
        print(f"❌ LakeFS connection error: {e}")
        print("   Please check your LakeFS endpoint and credentials")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()