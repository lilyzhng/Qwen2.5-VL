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

# Launch Streamlit with OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🚀 Launching ALFA 0.1 - Enhanced Similarity Search Interface..."
echo "🔧 OpenMP fix applied for stable operation"
echo "📁 Project root: ${PROJECT_ROOT}"
echo "📝 Open http://localhost:8501 in your browser"
echo ""

# Change to project root to ensure relative paths work correctly
cd "${PROJECT_ROOT}"

# Run streamlit
streamlit run "${PROJECT_ROOT}/interface/streamlit_app.py"