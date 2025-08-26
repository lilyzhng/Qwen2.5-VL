#!/usr/bin/env python3
"""
Keyword-Specific Recall Evaluation Script.

This script allows running recall evaluation for specific keywords or categories.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from core.evaluate import (
    run_text_to_video_evaluation, 
    run_video_to_video_evaluation,
    GroundTruthProcessor
)


def print_text_to_video_results(results, keywords):
    """Print text-to-video evaluation results."""
    print(f"\n📝 TEXT-TO-VIDEO RECALL EVALUATION")
    print("=" * 50)
    
    if keywords:
        print(f"Keywords: {', '.join(keywords)}")
    
    print(f"\n📊 Overall Results:")
    for metric, value in results['average_recalls'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"  Total queries: {results['total_queries']}")
    
    # Print keyword breakdown if available
    if 'keyword_breakdown' in results:
        print(f"\n🏷️  Keyword-Specific Results:")
        for keyword, data in results['keyword_breakdown'].items():
            recalls = data['recalls']
            relevant_count = data['relevant_count']
            print(f"\n  {keyword} ({relevant_count} relevant videos):")
            print(f"    R@1={recalls.get('recall@1', 0):.3f}, R@3={recalls.get('recall@3', 0):.3f}, R@5={recalls.get('recall@5', 0):.3f}")
            
            # Show top retrieved videos
            retrieved_ids = data.get('retrieved_ids', [])
            if retrieved_ids:
                print(f"    Top retrieved: {', '.join(retrieved_ids[:3])}")


def print_video_to_video_results(results, keywords):
    """Print video-to-video evaluation results."""
    print(f"\n🎬 VIDEO-TO-VIDEO RECALL EVALUATION")
    print("=" * 50)
    
    if keywords:
        print(f"Filter keywords: {', '.join(keywords)}")
    
    print(f"\n📊 Overall Results:")
    for metric, value in results['average_recalls'].items():
        print(f"  {metric}: {value:.3f}")
    
    print(f"  Total queries: {results['total_queries']}")
    
    # Show some example queries
    if 'detailed_results' in results and results['detailed_results']:
        print(f"\n🎯 Example Query Results:")
        for i, result in enumerate(results['detailed_results'][:3]):  # Show first 3
            query_video = result['query_video']
            query_keywords = result['query_keywords']
            recalls = result['recalls']
            print(f"\n  Query {i+1}: {query_video}")
            print(f"    Keywords: {', '.join(query_keywords)}")
            print(f"    R@1={recalls.get(1, 0):.3f}, R@3={recalls.get(3, 0):.3f}, R@5={recalls.get(5, 0):.3f}")


def list_available_keywords():
    """List all available keywords in the annotation file."""
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if not annotation_path.exists():
        print(f"❌ Annotation file not found: {annotation_path}")
        return
    
    try:
        processor = GroundTruthProcessor(str(annotation_path))
        
        print("📋 AVAILABLE KEYWORDS")
        print("=" * 30)
        
        # Group by semantic categories
        for group_name, categories in processor.semantic_groups.items():
            print(f"\n🏷️  {group_name.upper()}:")
            for category in categories:
                if category in processor.keyword_to_videos:
                    video_count = len(processor.keyword_to_videos[category])
                    print(f"  - {category} ({video_count} videos)")
        
        # Show other keywords not in semantic groups
        all_semantic_keywords = set()
        for categories in processor.semantic_groups.values():
            all_semantic_keywords.update(categories)
        
        other_keywords = set(processor.keyword_to_videos.keys()) - all_semantic_keywords
        if other_keywords:
            print(f"\n🏷️  OTHER:")
            for keyword in sorted(other_keywords):
                video_count = len(processor.keyword_to_videos[keyword])
                print(f"  - {keyword} ({video_count} videos)")
        
        print(f"\nTotal keywords: {len(processor.keyword_to_videos)}")
        print(f"Total videos: {len(processor.video_to_keywords)}")
        
    except Exception as e:
        print(f"❌ Error loading keywords: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run keyword-specific recall evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available keywords
  python run_keyword_evaluation.py --list-keywords
  
  # Text-to-video evaluation for specific keywords
  python run_keyword_evaluation.py --mode text --keywords urban highway
  
  # Video-to-video evaluation for car2pedestrian videos
  python run_keyword_evaluation.py --mode video --keywords car2pedestrian
  
  # Evaluate intersection scenarios
  python run_keyword_evaluation.py --mode both --keywords intersection crosswalk
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['text', 'video', 'both'],
        default='both',
        help='Evaluation mode: text (text-to-video), video (video-to-video), or both'
    )
    
    parser.add_argument(
        '--keywords', '-k',
        nargs='+',
        help='Keywords to evaluate (space-separated)'
    )
    
    parser.add_argument(
        '--list-keywords', '-l',
        action='store_true',
        help='List all available keywords and exit'
    )
    
    parser.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        default=[1, 3, 5],
        help='K values for Recall@K evaluation (default: 1 3 5)'
    )
    
    args = parser.parse_args()
    
    # List keywords and exit
    if args.list_keywords:
        list_available_keywords()
        return 0
    
    # Validate keywords
    if not args.keywords:
        print("❌ Please specify keywords with --keywords or use --list-keywords to see available options")
        return 1
    
    print(f"🎯 Keyword-Specific Recall Evaluation")
    print("=" * 50)
    print(f"Keywords: {', '.join(args.keywords)}")
    print(f"K values: {args.k_values}")
    print(f"Mode: {args.mode}")
    
    try:
        # Run text-to-video evaluation
        if args.mode in ['text', 'both']:
            print(f"\n🚀 Running text-to-video evaluation...")
            t2v_results = run_text_to_video_evaluation(
                keywords=args.keywords,
                k_values=args.k_values
            )
            print_text_to_video_results(t2v_results, args.keywords)
        
        # Run video-to-video evaluation
        if args.mode in ['video', 'both']:
            print(f"\n🚀 Running video-to-video evaluation...")
            v2v_results = run_video_to_video_evaluation(
                keywords=args.keywords,
                k_values=args.k_values
            )
            print_video_to_video_results(v2v_results, args.keywords)
        
        print(f"\n✅ Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
