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

echo "🚀 Launching ALFA 0.1 - Embedding Builder..."
echo "📁 Project root: ${PROJECT_ROOT}"
echo "📊 Building embeddings from unified input parquet file..."
echo ""

# Change to project root to ensure relative paths work correctly
cd "${PROJECT_ROOT}"

# Check if input file exists
INPUT_FILE="data/unified_input_path.parquet"
if [ ! -f "${INPUT_FILE}" ]; then
    echo "❌ Error: Input file not found: ${INPUT_FILE}"
    echo "Please ensure the unified input parquet file exists."
    exit 1
fi

echo "✅ Input file found: ${INPUT_FILE}"
echo "🔧 Starting embedding extraction process..."
echo ""

# Run the build command
python interface/main.py build --input-path "${INPUT_FILE}"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Embedding extraction completed successfully!"
    echo "📊 Database is now ready for similarity search"
    echo ""
    echo "Next steps:"
    echo "  🔍 Run similarity search: python interface/main.py search --query-video <path_to_video>"
    echo "  🌐 Launch web interface: ./interface/launch_streamlit.sh"
    echo "  📈 Run recall analysis: ./interface/launch_recall_analysis.sh"
else
    echo ""
    echo "❌ Embedding extraction failed!"
    echo "Please check the logs above for error details."
    exit 1
fi
