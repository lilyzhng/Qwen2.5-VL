#!/bin/bash
# Fix for OpenMP library conflict and run the video search with visualization

export KMP_DUPLICATE_LIB_OK=TRUE

# Run the search command with proper environment
python main.py "$@"
