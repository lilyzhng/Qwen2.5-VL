#!/bin/bash
"""
Launch the video annotation Streamlit app.
"""

echo "🎬 Starting Video Annotation App..."

# Launch Streamlit app
echo "🚀 Launching Streamlit app on http://localhost:8502"
echo "📝 Use this app to annotate your 20 videos with ground truth categories"
echo ""

streamlit run data/annotation/annotation_app.py --server.port 8502

# bash data/annotation/launch_annotation_app.sh