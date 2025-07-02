import os
import shutil
import re
from pathlib import Path

def organize_frames_by_prefix(source_dir):
    """
    Organize image frames by their prefix into separate folders.
    
    Args:
        source_dir (str): Path to the directory containing the image frames
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    
    # Dictionary to store prefix -> list of files mapping
    prefix_files = {}
    
    # Pattern to extract prefix from filename
    # Matches everything before the last double underscore followed by a timestamp
    pattern = r'^(.+__)(\d+)\.jpg$'
    
    print(f"Scanning files in: {source_dir}")
    
    # Scan all files in the source directory
    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == '.jpg':
            filename = file_path.name
            match = re.match(pattern, filename)
            
            if match:
                prefix = match.group(1).rstrip('_')  # Remove trailing underscores
                timestamp = match.group(2)
                
                if prefix not in prefix_files:
                    prefix_files[prefix] = []
                prefix_files[prefix].append(file_path)
            else:
                print(f"Warning: Could not extract prefix from '{filename}'")
    
    print(f"\nFound {len(prefix_files)} unique prefixes:")
    for prefix, files in prefix_files.items():
        print(f"  - {prefix}: {len(files)} files")
    
    # Create folders and move files
    for prefix, files in prefix_files.items():
        # Create folder with the prefix name
        folder_path = source_path / prefix
        folder_path.mkdir(exist_ok=True)
        
        print(f"\nProcessing prefix: {prefix}")
        
        # Move files to the folder
        for file_path in files:
            destination = folder_path / file_path.name
            try:
                shutil.move(str(file_path), str(destination))
                print(f"  Moved: {file_path.name}")
            except Exception as e:
                print(f"  Error moving {file_path.name}: {e}")
    
    print(f"\nOrganization complete! Files organized into {len(prefix_files)} folders.")

def main():
    """Main function to run the script."""
    # Default source directory
    source_directory = "/workspace/Qwen2.5-VL/nuscenes_mini/sweeps/CAM_FRONT"
    
    print("Frame Organization Script")
    print("=" * 50)
    
    # Allow user to specify a different directory if needed
    user_input = input(f"Press Enter to use default directory:\n{source_directory}\nOr enter a different path: ").strip()
    
    if user_input:
        source_directory = user_input
    
    # Run the organization
    organize_frames_by_prefix(source_directory)

if __name__ == "__main__":
    main()
