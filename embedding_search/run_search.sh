#!/bin/bash

# Script to run search commands with OpenMP fix
# This addresses the "OMP: Error #15" issue with multiple OpenMP runtimes

export KMP_DUPLICATE_LIB_OK=TRUE

echo "🔧 Setting OpenMP environment variable to avoid library conflicts..."
echo "🔍 Running search command with filename: $1"

python main.py search --query-filename "$1" "${@:2}"
