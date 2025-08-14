# ALFA 0.1 - Similarity Search

Advanced embedding visualization and video similarity search system.

## 📁 Project Structure

```
nvidia_cosmos/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
├── main.py                     # Command-line interface
├── streamlit_app.py           # Web interface
│
├── 🔧 Core Components
├── base.py                     # Base classes and interfaces
├── config.py                   # Configuration management
├── exceptions.py               # Custom exceptions
├── search.py                   # Optimized search engine
├── embedder.py                 # Video embedding functionality
├── database.py                 # Database management
├── visualizer.py               # Results visualization
├── optimizations.py            # Performance optimizations
├── advanced_visualization.py   # Advanced visualization methods
│
├── 📊 Mock Implementation
├── mock/
│   ├── mock_interface.html     # Interactive web demo
│   ├── mock_streamlit_app.py   # Mock Streamlit app
│   ├── mock_visualizer_demo.py # Mock visualization demo
│   └── test_visualizer.py      # Test scripts
│
├── 🛠️ Utilities
├── utils/
│   ├── migrate.py              # Database migration tools
│   ├── performance_test.py     # Performance testing
│   └── run_demo.py             # Demo runner
│
├── 📚 Documentation
├── instructions/
│   ├── README.md               # Main documentation
│   ├── VISUALIZATION_COMPARISON.md  # Visualization methods
│   ├── VISUALIZER_GUIDE.md     # Usage guide
│   ├── PROJECT_STRUCTURE.md    # Project structure
│   ├── IMPROVEMENTS.md         # Improvement history
│   ├── OPTIMIZATIONS_SUMMARY.md # Performance optimizations
│   ├── CLEANUP_SUMMARY.md      # Code cleanup notes
│   └── TSNE_ALTERNATIVES_SUMMARY.md # t-SNE alternatives
│
└── 🎬 Video Data
    └── videos/
        ├── user_input/         # Query videos
        └── video_database/     # Database videos
```

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Build Database
```bash
python main.py build --video-dir videos/video_database/
```

### 3. Run Search
```bash
# Text search
python main.py search --query-text "car approaching cyclist" --visualize

# Video search  
python main.py search --query-video videos/user_input/car2cyclist_2.mp4 --visualize
```

### 4. Web Interface
```bash
streamlit run streamlit_app.py
```

### 5. Interactive Demo
Open `mock/mock_interface.html` in your browser for a full-featured demo.

## 🎯 Key Features

### Search Capabilities
- **Text Search**: Natural language video queries
- **Video Search**: Find similar videos using video input
- **Advanced Filtering**: Similarity thresholds and top-K results

### Visualization Methods
- **UMAP**: Best balance of speed and quality ⭐
- **PCA**: Fastest method for quick visualization 🚀
- **TriMAP**: Optimal for large datasets 🔥
- **t-SNE**: Legacy support ⚠️
- **3D UMAP**: Three-dimensional exploration 🎯
- **Similarity Heatmap**: Direct similarity matrix 📊

### Interface Options
- **Command Line**: Full-featured CLI with extensive options
- **Streamlit Web App**: Interactive web interface
- **Mock Interface**: Professional demo with advanced features

## 🔧 Configuration

Edit `config.yaml` to customize:
- Video database paths
- Embedding model settings  
- Search parameters
- Performance optimizations

## 📊 Performance

The system includes several optimizations:
- **FAISS Integration**: Fast similarity search
- **Batch Processing**: Efficient video processing
- **Embedding Caching**: Reduced computation time
- **Optimized Database**: Safe, fast storage format

## 🧪 Testing

Run performance tests:
```bash
python utils/performance_test.py
```

Run demo:
```bash
python utils/run_demo.py
```

## 📚 Documentation

Detailed documentation is available in the `instructions/` directory:
- **Usage Guide**: Complete usage instructions
- **Visualization Guide**: Explanation of visualization methods
- **Performance Guide**: Optimization techniques
- **Development Guide**: Code structure and architecture

## 🎬 Mock Interface

The `mock/mock_interface.html` provides a complete demonstration of the system capabilities without requiring actual model inference. Features include:

- **Interactive Visualization**: Switch between visualization methods
- **Real-time Search**: Text and video query simulation
- **Advanced UI**: Professional interface with modern design
- **Query Highlighting**: Visual query vector and top-K results
- **Responsive Design**: Works on desktop and mobile

## 🔄 Migration

If upgrading from previous versions:
```bash
python utils/migrate.py --old-db old_database.pkl --new-db new_database/
```

## 📈 Development

The codebase is organized for maintainability:
- **Core Logic**: Clean separation of concerns
- **Mock Implementation**: Safe testing without models
- **Utilities**: Supporting tools and scripts
- **Documentation**: Comprehensive guides

---

**ALFA 0.1** - Advanced embedding visualization and video similarity search
