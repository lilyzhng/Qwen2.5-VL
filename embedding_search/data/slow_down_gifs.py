#!/usr/bin/env python3
"""
Simple script to slow down GIF files from a parquet file.
Reads gif_file column from parquet and saves slowed-down GIFs to output directory.
"""

import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse

def slow_down_gif(input_gif_path, output_gif_path, factor=2.0):
    """
    Slow down a GIF by increasing frame duration and save to output path.
    
    Args:
        input_gif_path (str): Path to the input GIF file
        output_gif_path (str): Path to save the slowed GIF
        factor (float): Slowdown factor (2.0 = half speed, 3.0 = third speed)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with Image.open(input_gif_path) as img:
            if not getattr(img, "is_animated", False):
                print(f"Skipping non-animated file: {input_gif_path}")
                return False
            
            frames = []
            durations = []
            
            # Extract all frames and their durations
            for frame_idx in range(img.n_frames):
                img.seek(frame_idx)
                duration = img.info.get('duration', 100)  # Default 100ms if not specified
                durations.append(int(duration * factor))
                frames.append(img.copy())
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_gif_path), exist_ok=True)
            
            # Save the slowed down GIF
            frames[0].save(
                output_gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=img.info.get('loop', 0),
                optimize=True
            )
            return True
            
    except Exception as e:
        print(f"Error processing {input_gif_path}: {e}")
        return False

def process_parquet_gifs(parquet_path, output_dir, slowdown_factor=2.0):
    """
    Process all GIF files listed in a parquet file and save to output directory.
    
    Args:
        parquet_path (str): Path to the parquet file
        output_dir (str): Output directory for slowed GIFs
        slowdown_factor (float): Factor to slow down GIFs
    
    Returns:
        tuple: (successful_count, total_count)
    """
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        return 0, 0
    
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    if 'gif_file' not in df.columns:
        print(f"No 'gif_file' column found in {parquet_path}")
        print(f"Available columns: {list(df.columns)}")
        return 0, 0
    
    # Get all non-null GIF paths
    gif_paths = df['gif_file'].dropna().tolist()
    
    if not gif_paths:
        print(f"No GIF files found in {parquet_path}")
        return 0, 0
    
    print(f"Found {len(gif_paths)} GIF files in {parquet_path}")
    print(f"Output directory: {output_dir}")
    
    successful = 0
    
    for gif_path in tqdm(gif_paths, desc="Processing GIFs"):
        if os.path.exists(gif_path):
            # Create output path maintaining the same filename
            gif_filename = os.path.basename(gif_path)
            output_gif_path = os.path.join(output_dir, gif_filename)
            
            if slow_down_gif(gif_path, output_gif_path, slowdown_factor):
                successful += 1
                print(f"✓ {gif_filename}")
            else:
                print(f"✗ Failed: {gif_filename}")
        else:
            print(f"✗ File not found: {gif_path}")
    
    return successful, len(gif_paths)

def main():
    parser = argparse.ArgumentParser(description='Slow down GIFs from a parquet file')
    parser.add_argument('parquet_file', 
                       help='Path to parquet file containing gif_file column')
    parser.add_argument('output_dir',
                       help='Output directory for slowed GIFs')
    parser.add_argument('--factor', '-f', type=float, default=2.0,
                       help='Slowdown factor (default: 2.0 = half speed)')
    
    args = parser.parse_args()
    
    # Process the parquet file
    successful, total = process_parquet_gifs(args.parquet_file, args.output_dir, args.factor)
    
    print(f"\n=== Summary ===")
    print(f"Processed {successful}/{total} GIFs successfully")
    print(f"Slowdown factor: {args.factor}x")
    print(f"Output directory: {args.output_dir}")
    
    if successful > 0:
        print("✓ Done! Your GIFs are now slower.")
    else:
        print("✗ No GIFs were processed successfully.")

if __name__ == "__main__":
    main()
