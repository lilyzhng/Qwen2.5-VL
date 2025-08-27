#!/bin/bash
"""
Launch the video annotation Streamlit app.
"""

echo "🎬 Starting Video Annotation App..."
echo "📍 Make sure you're in the qwen_env environment"

# # Activate conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate qwen_env

# Change to project directory
cd /Users/lilyzhang/Desktop/Qwen2.5-VL/embedding_search/data/annotation

# Launch Streamlit app
echo "🚀 Launching Streamlit app on http://localhost:8502"
echo "📝 Use this app to annotate your 20 videos with ground truth categories"
echo ""

streamlit run annotation_app.py --server.port 8502
