#!/bin/bash

# Launch script for create_gifs.py
# This script creates GIFs from videos or frame zip files listed in parquet files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${BLUE}🎬 GIF Creation Tool${NC}"
echo "========================================"

# Change to project root
cd "${PROJECT_ROOT}"

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --input-parquet PATH    Path to input parquet file"
    echo "  -o, --gif-dir PATH          Directory for generated GIFs"
    echo "  -w, --gif-width WIDTH       Width of generated GIFs (default: 320)"
    echo "  -f, --gif-fps FPS           Frames per second (default: 5)"
    echo "  -d, --gif-duration SEC      Duration limit in seconds"
    echo "  --overwrite                 Overwrite existing GIF files"
    echo "  --dry-run                   Show what would be done without creating GIFs"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Use default settings"
    echo "  $0 --dry-run                # Preview what would be processed"
    echo "  $0 --overwrite              # Overwrite existing GIFs"
    echo "  $0 -w 480 -f 10             # Higher resolution and frame rate"
}

# Default values
INPUT_PARQUET=""
GIF_DIR=""
GIF_WIDTH=""
GIF_FPS=""
GIF_DURATION=""
OVERWRITE=""
DRY_RUN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-parquet)
            INPUT_PARQUET="$2"
            shift 2
            ;;
        -o|--gif-dir)
            GIF_DIR="$2"
            shift 2
            ;;
        -w|--gif-width)
            GIF_WIDTH="$2"
            shift 2
            ;;
        -f|--gif-fps)
            GIF_FPS="$2"
            shift 2
            ;;
        -d|--gif-duration)
            GIF_DURATION="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check if ffmpeg is available
echo -e "${BLUE}🔍 Checking dependencies...${NC}"
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ ffmpeg is not installed${NC}"
    echo "Please install ffmpeg first:"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi
echo -e "${GREEN}✅ ffmpeg is available${NC}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ conda is not available${NC}"
    exit 1
fi

# Activate environment
echo -e "${BLUE}🐍 Activating qwen_env environment...${NC}"
eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda activate qwen_env

if [[ "$CONDA_DEFAULT_ENV" != "qwen_env" ]]; then
    echo -e "${RED}❌ Failed to activate qwen_env environment${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Environment activated: $CONDA_DEFAULT_ENV${NC}"

# Build command arguments
CMD_ARGS=()

if [[ -n "$INPUT_PARQUET" ]]; then
    CMD_ARGS+=("--input-parquet" "$INPUT_PARQUET")
fi

if [[ -n "$GIF_DIR" ]]; then
    CMD_ARGS+=("--gif-dir" "$GIF_DIR")
fi

if [[ -n "$GIF_WIDTH" ]]; then
    CMD_ARGS+=("--gif-width" "$GIF_WIDTH")
fi

if [[ -n "$GIF_FPS" ]]; then
    CMD_ARGS+=("--gif-fps" "$GIF_FPS")
fi

if [[ -n "$GIF_DURATION" ]]; then
    CMD_ARGS+=("--gif-duration" "$GIF_DURATION")
fi

if [[ -n "$OVERWRITE" ]]; then
    CMD_ARGS+=("$OVERWRITE")
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD_ARGS+=("$DRY_RUN")
fi

# Show command that will be executed
echo -e "${BLUE}🚀 Executing command:${NC}"
echo "python data/annotation/create_gifs.py ${CMD_ARGS[*]}"
echo ""

# Execute the Python script
echo -e "${YELLOW}⏳ Running GIF creation...${NC}"
python data/annotation/create_gifs.py "${CMD_ARGS[@]}"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}🎉 GIF creation completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ GIF creation failed${NC}"
    exit 1
fi
