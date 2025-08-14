# NVIDIA Cosmos Video Retrieval System - Project Structure

## Core Components

### Main Implementation Files
- **`main.py`** - Main CLI interface (updated with all improvements)
- **`video_embedder.py`** - Video embedding extraction using Cosmos model
- **`video_database.py`** - Safe database management with JSON + numpy storage
- **`video_search.py`** - Video search engine with extensible architecture
- **`video_search_optimized.py`** - Optimized search engine with FAISS
- **`video_visualizer.py`** - Result visualization and comparison

### Configuration and Base Classes
- **`config.py`** - Configuration management system
- **`config.yaml`** - Sample configuration file
- **`base.py`** - Abstract base classes for extensibility
- **`exceptions.py`** - Custom exception hierarchy
- **`optimizations.py`** - Performance optimizations (FAISS, caching, etc.)

### Interface and Demonstrations
- **`streamlit_app.py`** - Production Streamlit web interface
- **`mock_streamlit_app.py`** - Mock interface for testing without CUDA
- **`mock_interface.html`** - Standalone HTML demo interface
- **`mock_visualizer_demo.py`** - Comprehensive mock demonstration
- **`test_visualizer.py`** - Unit tests for visualization components

### Utilities and Migration
- **`run_demo.py`** - Unified launcher for all demonstrations
- **`migrate.py`** - Migration tool from v1 to current implementation
- **`benchmark_optimizations.py`** - Performance benchmarking tools

### Documentation
- **`README.md`** - Main documentation (updated)
- **`IMPROVEMENTS.md`** - Detailed improvement documentation
- **`OPTIMIZATIONS_SUMMARY.md`** - Performance optimization summary
- **`VISUALIZER_GUIDE.md`** - Comprehensive visualizer guide
- **`PROJECT_STRUCTURE.md`** - This file

### Dependencies
- **`requirements.txt`** - Python package requirements

### Data Directory
- **`videos/`** - Video data directory
  - **`user_input/`** - Query videos
  - **`video_database/`** - Video collection for database

## Key Features Implemented

### ✅ Core Functionality
- Video embedding extraction with batch processing
- Safe JSON + numpy database storage
- Multi-modal search (text and video)
- Interactive web interface
- Comprehensive error handling

### ✅ Performance Optimizations
- FAISS for efficient similarity search
- Batch processing for video embeddings
- Embedding caching with LRU eviction
- Parquet format support for large datasets

### ✅ Visualization and Interface
- Interactive Streamlit web interface
- Static matplotlib visualizations
- Mock interface for testing without CUDA
- Comprehensive unit tests

### ✅ Developer Experience
- Modular, extensible architecture
- Configuration management
- Migration tools
- Comprehensive documentation
- Multiple demo modes

## Quick Start Commands

```bash
# Check dependencies
python run_demo.py check

# Run all demonstrations
python run_demo.py all

# Launch web interface (requires streamlit)
python run_demo.py web

# Launch mock interface (no dependencies)
streamlit run mock_streamlit_app.py

# Build database and search
python main.py build
python main.py search --query-text "car approaching cyclist"

# Run performance benchmarks
python benchmark_optimizations.py
```

## Architecture Highlights

1. **Extensible Design**: Abstract base classes allow easy swapping of components
2. **Configuration-Driven**: YAML/JSON configuration files for customization
3. **Error Resilient**: Comprehensive exception handling and graceful degradation
4. **Performance Optimized**: FAISS, caching, and batch processing
5. **Well Tested**: Unit tests and mock interfaces for validation
6. **Production Ready**: Streamlit interface with real-time search capabilities

## Migration from Previous Versions

If you have code from earlier versions:
1. Use `migrate.py` to convert old databases
2. Update imports to remove `_v2` suffixes
3. Use new configuration system with `config.yaml`
4. Leverage new optimizations in `video_search_optimized.py`

This clean architecture provides a solid foundation for video retrieval applications using NVIDIA's Cosmos-Embed1 model.
