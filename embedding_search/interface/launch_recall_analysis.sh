#!/bin/bash

# Launch Recall Analysis Streamlit App
# This script starts the recall evaluation analysis dashboard

echo "🎯 Launching Recall Analysis Dashboard..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "interface/recall_analysis_app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: interface/recall_analysis_app.py"
    exit 1
fi

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
streamlit run interface/recall_analysis_app.py --server.port 8502 --server.address localhost
