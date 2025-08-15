# 🚀 Enhanced Query Video System Guide

## Overview

The enhanced query video system provides **instant similarity search** using pre-computed embeddings, eliminating the need for real-time video processing during user interactions.

## 🏗️ Architecture

### Two-Tier Database System

1. **Database Videos** (Large Dataset)
   - Location: `/videos/video_database/`
   - Purpose: Main corpus of videos to search through
   - Storage: Optimized Parquet format with FAISS indexing

2. **Query Videos** (Pre-cached User Input)
   - Location: `/videos/user_input/`
   - Purpose: Frequently used query videos with pre-computed embeddings
   - Storage: SQLite metadata + NumPy embedding files
   - Features: Instant lookup, access tracking, cache management

### Key Components

- **`SafeVideoDatabase`**: Enhanced with filename-based lookup
- **`QueryVideoCache`**: Persistent cache with SQLite + file storage
- **`QueryDatabaseManager`**: Combines database and cache management
- **`OptimizedVideoSearchEngine`**: Updated with pre-computed query support

## 🚀 Quick Start

### 1. Build Database Videos (One-time)
```bash
# Build main video database
python main.py build --video-dir videos/video_database/

# Verify database
python main.py info
```

### 2. Build Query Cache (One-time)
```bash
# Pre-compute embeddings for query videos
python main.py build-query --query-dir videos/user_input/

# Check query cache status
python main.py query-info

# List available query videos
python main.py list-queries
```

### 3. Search Operations

#### A) Instant Search (Recommended)
```bash
# Uses pre-computed embeddings (instant results)
python main.py search --query-filename car2cyclist_2.mp4 --visualize
```

#### B) Traditional Video Search
```bash
# Processes video in real-time
python main.py search --query-video videos/user_input/car2cyclist_2.mp4 --visualize
```

#### C) Text Search
```bash
# Text-based semantic search
python main.py search --query-text "car approaching cyclist" --visualize
```

## 🌐 Web Interface Usage

### 1. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

### 2. Video Search Interface

The sidebar now includes **two search options**:

#### ⚡ Quick Search
- **Purpose**: Uses pre-computed embeddings
- **Speed**: Instant results
- **Requirement**: Video must be in query cache
- **Fallback**: Automatically falls back to real-time processing if needed

#### 🎥 Full Search  
- **Purpose**: Processes video in real-time
- **Speed**: Slower but works with any video
- **Use Case**: Testing new videos or when cache miss occurs

### 3. Enhanced Statistics

The sidebar displays:
- **Database Stats**: Main video corpus information
- **Query Cache Stats**: Pre-computed embedding statistics
  - Cached videos count
  - Cache size
  - Usage statistics

## 📊 Performance Benefits

### Before (Real-time Processing)
```
User selects video → Extract 8 frames → NVIDIA Cosmos inference → 
Normalize → FAISS search → Results
⏱️ Time: ~2-5 seconds per query
```

### After (Pre-computed)
```
User selects video → SQLite lookup → FAISS search → Results
⏱️ Time: ~50-100ms per query (50x faster!)
```

## 🔧 Advanced Usage

### Cache Management

```bash
# Force rebuild query cache
python main.py build-query --force-rebuild

# View detailed cache statistics
python main.py query-info

# List all cached query videos
python main.py list-queries
```

### Programmatic Access

```python
from search import OptimizedVideoSearchEngine

# Initialize engine
engine = OptimizedVideoSearchEngine()

# Build query cache
stats = engine.build_query_database("videos/user_input/")
print(f"Processed: {stats['processed']} videos")

# Instant search by filename
results = engine.search_by_filename("car2cyclist_2.mp4")

# Get available query videos
available = engine.get_query_videos_list()
print(f"Available query videos: {available}")
```

### Database Operations

```python
from query_cache import QueryDatabaseManager

# Initialize manager
manager = QueryDatabaseManager()

# Check if video is cached
if manager.query_cache.is_cached("video.mp4"):
    embedding = manager.get_query_embedding("video.mp4")
    
# Get cache statistics  
stats = manager.get_statistics()
print(f"Cache size: {stats['cache']['cache_size_mb']:.1f} MB")
```

## 🗂️ File Structure

```
embedding_search/
├── 📁 videos/
│   ├── 📁 user_input/          # Query videos
│   │   ├── car2cyclist_2.mp4
│   │   └── ...
│   └── 📁 video_database/      # Database videos
│       ├── car2car_1.mp4
│       └── ...
├── 📁 query_cache/             # Query cache storage
│   ├── query_cache.db          # SQLite metadata
│   └── 📁 embeddings/          # NumPy embedding files
│       ├── car2cyclist_2_embedding.npy
│       └── ...
├── video_embeddings.json       # Database metadata
├── video_embeddings.npy        # Database embeddings
├── video_embeddings.parquet    # Optimized database storage
└── query_embeddings.json       # Query database metadata
```

## 🎯 Use Cases

### 1. Interactive Demo/Research
- **Scenario**: Researchers testing different query videos
- **Benefit**: Instant results for iterative exploration
- **Setup**: Pre-compute all potential query videos

### 2. Production Video Search
- **Scenario**: Users searching through large video corpus
- **Benefit**: Consistent fast response times
- **Setup**: Cache frequently used query patterns

### 3. Batch Analysis
- **Scenario**: Processing multiple queries programmatically
- **Benefit**: Massive time savings for repeated queries
- **Setup**: Build comprehensive query cache

## 🔍 Troubleshooting

### Query Video Not Found
```bash
# Check if video is in cache
python main.py list-queries

# Add video to cache
python main.py build-query --query-dir path/to/video/dir/
```

### Cache Miss in Streamlit
- **Issue**: "Pre-computed embedding not found"
- **Solution**: Use "🎥 Full Search" or rebuild query cache
- **Prevention**: Build query cache before using web interface

### Performance Issues
```bash
# Check cache statistics
python main.py query-info

# Verify FAISS backend is being used
python main.py info
```

### Database Corruption
```bash
# Rebuild from scratch
python main.py build --force-rebuild
python main.py build-query --force-rebuild
```

## 📈 Monitoring & Optimization

### Performance Metrics
- **Cache Hit Rate**: Check via `query-info`
- **Search Time**: Monitor via application logs
- **Memory Usage**: SQLite cache is memory-efficient

### Optimization Tips
1. **Pre-compute Popular Queries**: Build cache for frequently used videos
2. **Monitor Access Patterns**: Use `query-info` to see usage statistics
3. **Clean Orphaned Files**: Automatic cleanup during cache operations
4. **Regular Rebuilds**: Rebuild cache when query videos change

## 🔄 Migration from Old System

### Existing Users
1. **Backup**: Save existing databases
2. **Update**: Pull latest code
3. **Rebuild**: Run `build-query` for instant search
4. **Verify**: Test both quick and full search modes

### Configuration Updates
- No config changes required
- All paths remain the same
- Automatic fallback to old behavior if needed

---

## 🎉 Summary

The enhanced query system provides:
- ⚡ **50x faster** query video searches
- 🎯 **Seamless fallback** to real-time processing
- 📊 **Usage analytics** and cache management
- 🔄 **Backward compatibility** with existing workflows
- 🌐 **Enhanced web interface** with dual search modes

**Next Steps**: 
1. Build your query cache: `python main.py build-query`
2. Try instant search: `python main.py search --query-filename your_video.mp4`
3. Explore the enhanced web interface: `streamlit run streamlit_app.py`
