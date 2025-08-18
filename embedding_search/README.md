# NVIDIA Cosmos Video Embedding Search System

A high-performance video similarity search system using NVIDIA Cosmos embeddings with GPU acceleration.

## Project Structure

```
embedding_search/
├── app/                    # Streamlit application and web interface
│   ├── streamlit_app.py   # Main Streamlit application
│   └── mock/              # Mock interfaces for testing
├── benchmarks/            # Performance benchmarking scripts
│   ├── example_benchmark.py
│   ├── inference_benchmark.py
│   └── performance_test.py
├── core/                  # Core functionality modules
│   ├── base.py           # Abstract base classes
│   ├── config.py         # Configuration management
│   ├── database.py       # Database operations
│   ├── embedder.py       # Video embedding extraction
│   ├── exceptions.py     # Custom exceptions
│   ├── optimizations.py  # FAISS and performance optimizations
│   ├── query_cache.py    # Query caching system
│   ├── search.py         # Main search engine
│   └── visualizer.py     # Results visualization
├── data/                  # Data storage
│   ├── videos/           # Video files
│   │   ├── video_database/  # Reference video database
│   │   └── user_input/      # User query videos
│   ├── query_cache/      # Cached query embeddings
│   └── *.parquet         # Embedding databases
├── docs/                  # Documentation
├── models/               # Model download scripts
├── scripts/              # Utility scripts
│   ├── launch_streamlit.sh  # Launch web interface
│   ├── main.py             # CLI interface
│   └── run_*.sh            # Helper scripts
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the web interface:**
   ```bash
   cd scripts
   ./launch_streamlit.sh
   ```

3. **Or use the CLI:**
   ```bash
   cd scripts
   python main.py build --video-dir ../data/videos/video_database
   python main.py search --query-filename car2cyclist_2.mp4
   ```

## Key Features

- GPU-accelerated similarity search using FAISS
- Real-time video embedding extraction with NVIDIA Cosmos
- Query caching for instant results
- Web interface with visualization
- Batch processing support
- Comprehensive error handling

## Documentation

See the `docs/` directory for detailed documentation:
- [Installation Guide](docs/INSTALLATION_GUIDE.md)
- [Query System Guide](docs/QUERY_SYSTEM_GUIDE.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Optimizations Summary](docs/OPTIMIZATIONS_SUMMARY.md)

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## License

[Add license information here]
