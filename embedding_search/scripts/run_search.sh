#!/bin/bash

# Get the absolute path of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda environment if needed
if [[ "$CONDA_DEFAULT_ENV" != "qwen_env" ]]; then
    echo "🔄 Activating qwen_env conda environment..."
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
    conda activate qwen_env
fi

# Script to run search commands with OpenMP fix
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🔧 Setting OpenMP environment variable to avoid library conflicts..."
echo "🔍 Running search command with filename: $1"

# Change to script directory to run main.py
cd "${SCRIPT_DIR}"
python main.py search --query-filename "$1" "${@:2}"
