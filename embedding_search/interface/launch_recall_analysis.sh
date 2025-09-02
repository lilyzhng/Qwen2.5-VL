#!/bin/bash

# Get the absolute path of this script, no matter where it's called from
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment if needed
if [[ "$CONDA_DEFAULT_ENV" != "qwen_env" ]]; then
    echo "🔄 Activating qwen_env conda environment..."
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
    conda activate qwen_env
fi

# Set environment variables for OpenMP compatibility
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🎯 Launching Recall Analysis Dashboard..."
echo "=========================================="
echo "📁 Project root: ${PROJECT_ROOT}"

# Change to project root to ensure relative paths work correctly
cd "${PROJECT_ROOT}"

# Check if annotation file exists
if [ ! -f "data/annotation/video_annotation.csv" ]; then
    echo "❌ Error: Annotation file not found: data/annotation/video_annotation.csv"
    echo "Please ensure the annotation file exists before running the analysis."
    exit 1
fi

# Check if embeddings database exists
if [ ! -f "data/unified_embeddings.parquet" ]; then
    echo "❌ Error: Embeddings database not found: data/unified_embeddings.parquet"
    echo "Please build the embeddings database first:"
    echo "  python interface/main.py build --input-path data/unified_input_path.parquet"
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""
echo "🚀 Starting Streamlit app..."
echo "📊 The dashboard will be available at: http://localhost:8502"
echo ""
echo "Features available:"
echo "  📊 Overall Performance - Comprehensive recall metrics and charts"
echo "  🏷️ Keyword Analysis - Analyze specific keywords and categories"
echo "  🎬 Video Triaging - Individual video analysis and search testing"
echo "  📈 Custom Evaluation - Run custom evaluations with your parameters"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="

# Launch Streamlit app on port 8502 (different from main app)
streamlit run "${PROJECT_ROOT}/interface/recall_analysis_app.py" --server.port 8502 --server.address localhost
