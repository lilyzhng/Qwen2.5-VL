#!/bin/bash
# Launch script for Streamlit app with proper environment setup

echo "🚀 Launching ALFA 0.1 - Streamlit Video Search Interface"
echo "=================================================="

# Fix OpenMP library conflict
export KMP_DUPLICATE_LIB_OK=TRUE

# Activate virtual environment
echo "Activating qwen_venv..."
source /Users/lilyzhang/Desktop/Qwen2.5-VL/qwen_venv/bin/activate

# Navigate to embedding_search directory
cd /Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search

echo "Starting Streamlit app..."
echo "📱 App will be available at: http://localhost:8501"
echo "💡 Press Ctrl+C to stop the app"
echo ""

# Launch Streamlit
streamlit run streamlit_app.py
