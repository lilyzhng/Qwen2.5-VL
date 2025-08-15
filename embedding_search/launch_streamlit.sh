#!/bin/bash

# Launch Streamlit with OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🚀 Launching ALFA 0.1 - Enhanced Similarity Search Interface..."
echo "🔧 OpenMP fix applied for stable operation"
echo "📝 Open http://localhost:8501 in your browser"
echo ""

streamlit run streamlit_app.py