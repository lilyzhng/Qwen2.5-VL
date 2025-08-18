#!/bin/bash

# Convenience launcher script that can be run from anywhere
# Simply executes the actual launch script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Execute the launch script
exec "${SCRIPT_DIR}/scripts/launch_streamlit.sh"
