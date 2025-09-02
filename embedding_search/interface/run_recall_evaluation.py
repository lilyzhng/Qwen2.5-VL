#!/usr/bin/env python3
"""
Recall Evaluation Runner Script.

This script runs the recall evaluation framework and prints the results.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from core.evaluate import run_recall_evaluation, print_recall_results


def main():
    """Main function to run recall evaluation."""
    print("🎯 Video Search Recall Evaluation")
    print("=" * 50)
    
    # Check if annotation file exists
    annotation_path = project_root / "data" / "annotation" / "unified_annotation.csv"
    if not annotation_path.exists():
        print(f"❌ Annotation file not found: {annotation_path}")
        print("Please ensure the unified_annotation.csv file exists.")
        return 1
    
    try:
        print("🚀 Running recall evaluation...")
        results = run_recall_evaluation()
        print_recall_results(results)
        
        print("\n✅ Recall evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Recall evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
