#!/usr/bin/env python3
"""
Recall Evaluation Framework for Video Search System.

This module implements comprehensive recall measurement using the annotated ground truth
from video_annotation.csv to evaluate the performance of video-to-video and text-to-video
search capabilities.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging

# Handle imports for both module and standalone execution
try:
    from .search import VideoSearchEngine
    from .config import VideoRetrievalConfig
    from .database import ParquetVectorDatabase
except ImportError:
    # Standalone execution
    from search import VideoSearchEngine
    from config import VideoRetrievalConfig
    from database import ParquetVectorDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundTruthProcessor:
    """Process and manage ground truth annotations for recall evaluation."""
    
    def __init__(self, annotation_csv_path: str):
        """
        Initialize ground truth processor.
        
        Args:
            annotation_csv_path: Path to the video annotation CSV file
        """
        self.annotation_path = Path(annotation_csv_path)
        self.annotations_df = None
        self.keyword_to_videos = defaultdict(set)
        self.video_to_keywords = {}
        self.semantic_groups = {}
        
        self._load_annotations()
        self._build_mappings()
    
    def _load_annotations(self):
        """Load annotations from CSV file."""
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        
        self.annotations_df = pd.read_csv(self.annotation_path)
        logger.info(f"Loaded {len(self.annotations_df)} annotations from {self.annotation_path}")
    
    def _build_mappings(self):
        """Build keyword-to-video and video-to-keyword mappings."""
        for _, row in self.annotations_df.iterrows():
            slice_id = row['slice_id']
            keywords_str = row['keywords']
            
            # Parse keywords (comma-separated, may have quotes)
            keywords = [kw.strip().strip('"') for kw in keywords_str.split(',')]
            
            self.video_to_keywords[slice_id] = set(keywords)
            
            for keyword in keywords:
                self.keyword_to_videos[keyword].add(slice_id)
        
        # Build semantic groups
        self._build_semantic_groups()
        
        logger.info(f"Built mappings for {len(self.video_to_keywords)} videos and {len(self.keyword_to_videos)} keywords")
    
    def _build_semantic_groups(self):
        """Build semantic groups for related concepts."""
        # Define semantic groupings
        interaction_types = ['car2pedestrian', 'car2cyclist', 'car2motorcyclist', 'car2car']
        environments = ['urban', 'highway', 'freeway', 'intersection', 'crosswalk']
        conditions = ['night', 'daytime', 'rain', 'parking', 'tunnel']
        actions = ['turning_left', 'turning_right', 'lane_merge']
        critical_objects = ['bicyclist', 'motorcyclist', 'parked bicycle', 'parked motorcycle', 'truck', 'pedestrian']
        
        self.semantic_groups = {
            'interactions': interaction_types,
            'environments': environments,
            'conditions': conditions,
            'actions': actions,
            'critical_objects': critical_objects
        }
    
    def get_relevant_videos(self, query_video_id: str, include_self: bool = False) -> Set[str]:
        """
        Get videos that should be considered relevant for a given query video.
        A video is considered relevant if it contains ALL keywords of the query video.
        
        Args:
            query_video_id: ID of the query video
            include_self: Whether to include the query video itself
            
        Returns:
            Set of relevant video IDs
        """
        if query_video_id not in self.video_to_keywords:
            return set()
        
        query_keywords = set(self.video_to_keywords[query_video_id])
        relevant_videos = set()
        
        # Find videos that contain ALL keywords of the query video
        for video_id, video_keywords in self.video_to_keywords.items():
            video_keywords_set = set(video_keywords)
            # Check if query keywords are a subset of video keywords
            if query_keywords.issubset(video_keywords_set):
                relevant_videos.add(video_id)
        
        if not include_self:
            relevant_videos.discard(query_video_id)
        
        return relevant_videos
    
    def get_relevant_videos_for_text(self, query_text: str) -> Set[str]:
        """
        Get videos relevant for a text query.
        
        Args:
            query_text: Text query (keyword or phrase)
            
        Returns:
            Set of relevant video IDs
        """
        # Normalize query text
        query_text = query_text.strip().lower()
        
        relevant_videos = set()
        
        # Direct keyword match
        if query_text in self.keyword_to_videos:
            relevant_videos.update(self.keyword_to_videos[query_text])
        
        # More precise partial matching to avoid false positives
        # Only match if the query is a meaningful substring (not just contained within)
        for keyword in self.keyword_to_videos:
            keyword_lower = keyword.lower()
            
            # Skip if exact match already handled
            if keyword_lower == query_text:
                continue
                
            # Allow partial matches only in specific cases:
            # 1. Query is at the start of keyword (e.g., "car" matches "car2pedestrian")
            # 2. Query is at the end of keyword (e.g., "pedestrian" matches "car2pedestrian") 
            # 3. Query is separated by underscore/number (e.g., "car" matches "car_merge")
            if (keyword_lower.startswith(query_text + '_') or 
                keyword_lower.startswith(query_text + '2') or
                keyword_lower.endswith('_' + query_text) or
                keyword_lower.endswith('2' + query_text) or
                ('_' + query_text + '_' in keyword_lower) or
                ('2' + query_text + '2' in keyword_lower)):
                relevant_videos.update(self.keyword_to_videos[keyword])
        
        return relevant_videos
    
    def get_category_videos(self, category: str) -> Set[str]:
        """Get all videos belonging to a specific category."""
        return self.keyword_to_videos.get(category, set())
    
    def get_all_keywords(self) -> List[str]:
        """Get all unique keywords in the dataset."""
        return list(self.keyword_to_videos.keys())
    
    def get_video_info(self, video_id: str) -> Dict[str, Any]:
        """Get detailed information about a video."""
        if video_id not in self.video_to_keywords:
            return {}
        
        row = self.annotations_df[self.annotations_df['slice_id'] == video_id].iloc[0]
        return {
            'slice_id': video_id,
            'video_path': row['video_path'],
            'gif_path': row['gif_path'],
            'keywords': list(self.video_to_keywords[video_id])
        }


class RecallEvaluator:
    """Evaluate recall performance of the video search system."""
    
    def __init__(self, search_engine: VideoSearchEngine, ground_truth: GroundTruthProcessor):
        """
        Initialize recall evaluator.
        
        Args:
            search_engine: Video search engine to evaluate
            ground_truth: Ground truth processor
        """
        self.search_engine = search_engine
        self.ground_truth = ground_truth
        self.results_cache = {}
        
        # Load pre-computed embeddings database for efficient evaluation
        self.embeddings_df = None
        self._load_embeddings_database()
    
    def _load_embeddings_database(self):
        """Load the pre-computed embeddings database."""
        try:
            embeddings_path = self.search_engine.database.database_path
            if embeddings_path.exists():
                self.embeddings_df = pd.read_parquet(embeddings_path)
                self.embeddings_df = self.embeddings_df.set_index('slice_id', drop=False)
                logger.info(f"Loaded {len(self.embeddings_df)} pre-computed embeddings from {embeddings_path}")
            else:
                logger.warning(f"Embeddings database not found: {embeddings_path}")
        except Exception as e:
            logger.error(f"Failed to load embeddings database: {e}")
    
    def _get_precomputed_embedding(self, slice_id: str) -> Optional[np.ndarray]:
        """Get pre-computed embedding for a video by slice_id."""
        if self.embeddings_df is None or slice_id not in self.embeddings_df.index:
            return None
        
        try:
            embedding_data = self.embeddings_df.loc[slice_id, 'embedding']
            if isinstance(embedding_data, list):
                return np.array(embedding_data, dtype=np.float32)
            return embedding_data
        except Exception as e:
            logger.error(f"Error retrieving embedding for {slice_id}: {e}")
            return None
    
    def _search_by_precomputed_embedding(self, query_slice_id: str, top_k: int) -> List[Dict]:
        """
        Perform search using pre-computed embedding directly from the database.
        
        Args:
            query_slice_id: Slice ID of the query video
            top_k: Number of results to return
            
        Returns:
            Search results
        """
        query_embedding = self._get_precomputed_embedding(query_slice_id)
        if query_embedding is None:
            raise ValueError(f"No pre-computed embedding found for {query_slice_id}")
        
        # Use the search engine's internal search method with the embedding
        return self.search_engine._search_by_embedding(
            query_embedding, 
            top_k, 
            exclude_slice_id=query_slice_id
        )
    
    def evaluate_video_to_video_recall(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluate video-to-video recall performance.
        
        Args:
            k_values: List of K values for Recall@K evaluation
            
        Returns:
            Dictionary containing recall metrics
        """
        logger.info("Evaluating video-to-video recall...")
        
        all_recalls = {k: [] for k in k_values}
        detailed_results = []
        
        # Get all annotated videos
        annotated_videos = list(self.ground_truth.video_to_keywords.keys())
        
        for query_video_id in annotated_videos:
            try:
                # Get ground truth relevant videos
                relevant_videos = self.ground_truth.get_relevant_videos(query_video_id, include_self=False)
                
                if len(relevant_videos) == 0:
                    logger.warning(f"No relevant videos found for {query_video_id}")
                    continue
                
                # Get video info
                video_info = self.ground_truth.get_video_info(query_video_id)
                
                # Use pre-computed embedding for faster evaluation
                max_k = max(k_values)
                try:
                    search_results = self._search_by_precomputed_embedding(
                        query_video_id, 
                        top_k=max_k
                    )
                except ValueError:
                    # Fallback to video file search if pre-computed embedding not available
                    logger.warning(f"No pre-computed embedding for {query_video_id}, using video file")
                    query_video_path = video_info['video_path']
                    search_results = self.search_engine.search_by_video(
                        query_video_path, 
                        top_k=max_k,
                        use_cache=True
                    )
                
                # Extract retrieved video IDs
                retrieved_ids = [result['slice_id'] for result in search_results]
                
                # Calculate recall for each K
                recalls_for_query = {}
                for k in k_values:
                    retrieved_k = set(retrieved_ids[:k])
                    relevant_retrieved = retrieved_k.intersection(relevant_videos)
                    # Fixed: Recall@K = relevant_found_in_top_k / k
                    recall_k = len(relevant_retrieved) / k
                    recalls_for_query[k] = recall_k
                    all_recalls[k].append(recall_k)
                
                detailed_results.append({
                    'query_video': query_video_id,
                    'query_keywords': video_info['keywords'],
                    'relevant_count': len(relevant_videos),
                    'retrieved_ids': retrieved_ids,
                    'recalls': recalls_for_query
                })
                
            except Exception as e:
                logger.error(f"Error evaluating {query_video_id}: {e}")
                continue
        
        # Calculate average recalls
        avg_recalls = {}
        for k in k_values:
            if all_recalls[k]:
                avg_recalls[f'recall@{k}'] = np.mean(all_recalls[k])
            else:
                avg_recalls[f'recall@{k}'] = 0.0
        
        return {
            'average_recalls': avg_recalls,
            'detailed_results': detailed_results,
            'total_queries': len(detailed_results)
        }
    
    def evaluate_text_to_video_recall(self, k_values: List[int] = [1, 3, 5], quality_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Evaluate text-to-video recall performance.
        
        Args:
            k_values: List of K values for Recall@K evaluation
            quality_threshold: Minimum similarity score to consider results valid
            
        Returns:
            Dictionary containing recall metrics
        """
        logger.info("Evaluating text-to-video recall...")
        
        all_recalls = {k: [] for k in k_values}
        detailed_results = []
        filtered_queries = 0  # Track queries filtered out due to low quality
        
        # Test with all unique keywords
        keywords_to_test = self.ground_truth.get_all_keywords()
        
        for keyword in keywords_to_test:
            try:
                # Get ground truth relevant videos
                relevant_videos = self.ground_truth.get_relevant_videos_for_text(keyword)
                
                if len(relevant_videos) == 0:
                    continue
                
                # Perform text search
                max_k = max(k_values)
                search_results = self.search_engine.search_by_text(
                    keyword,
                    top_k=max_k
                )
                
                # Apply quality threshold filtering
                if quality_threshold > 0.0:
                    # Check if top result meets quality threshold
                    if not search_results or search_results[0].get('similarity_score', search_results[0].get('similarity', 0)) < quality_threshold:
                        filtered_queries += 1
                        logger.debug(f"Filtered out query '{keyword}' due to low similarity (< {quality_threshold})")
                        continue
                    
                    # Filter all results by quality threshold
                    search_results = [r for r in search_results if r.get('similarity_score', r.get('similarity', 0)) >= quality_threshold]
                
                # Extract retrieved video IDs
                retrieved_ids = [result['slice_id'] for result in search_results]
                
                # Calculate recall for each K
                recalls_for_query = {}
                for k in k_values:
                    retrieved_k = set(retrieved_ids[:k])
                    relevant_retrieved = retrieved_k.intersection(relevant_videos)
                    # Standard Recall@K = relevant_found_in_top_k / k
                    recall_k = len(relevant_retrieved) / k
                    recalls_for_query[k] = recall_k
                    all_recalls[k].append(recall_k)
                
                detailed_results.append({
                    'query_text': keyword,
                    'relevant_count': len(relevant_videos),
                    'retrieved_ids': retrieved_ids,
                    'recalls': recalls_for_query
                })
                
            except Exception as e:
                logger.error(f"Error evaluating text query '{keyword}': {e}")
                continue
        
        # Calculate average recalls
        avg_recalls = {}
        for k in k_values:
            if all_recalls[k]:
                avg_recalls[f'recall@{k}'] = np.mean(all_recalls[k])
            else:
                avg_recalls[f'recall@{k}'] = 0.0
        
        return {
            'average_recalls': avg_recalls,
            'detailed_results': detailed_results,
            'total_queries': len(detailed_results),
            'filtered_queries': filtered_queries,
            'quality_threshold': quality_threshold
        }
    
    def evaluate_category_specific_recall(self, k_values: List[int] = [1, 3, 5]) -> Dict[str, Any]:
        """
        Evaluate recall performance for specific categories.
        
        Args:
            k_values: List of K values for Recall@K evaluation
            
        Returns:
            Dictionary containing category-specific recall metrics
        """
        logger.info("Evaluating category-specific recall...")
        
        category_results = {}
        
        # Evaluate for each semantic group
        for group_name, categories in self.ground_truth.semantic_groups.items():
            group_results = {}
            
            for category in categories:
                if category not in self.ground_truth.keyword_to_videos:
                    continue
                
                category_videos = self.ground_truth.get_category_videos(category)
                if len(category_videos) < 2:  # Need at least 2 videos for meaningful recall
                    continue
                
                category_recalls = {k: [] for k in k_values}
                
                # Use each video in category as query
                for query_video_id in category_videos:
                    try:
                        relevant_videos = category_videos - {query_video_id}  # Exclude self
                        
                        max_k = max(k_values)
                        try:
                            search_results = self._search_by_precomputed_embedding(
                                query_video_id,
                                top_k=max_k
                            )
                        except ValueError:
                            # Fallback to video file search
                            video_info = self.ground_truth.get_video_info(query_video_id)
                            query_video_path = video_info['video_path']
                            search_results = self.search_engine.search_by_video(
                                query_video_path,
                                top_k=max_k,
                                use_cache=True
                            )
                        
                        retrieved_ids = [result['slice_id'] for result in search_results]
                        
                        for k in k_values:
                            retrieved_k = set(retrieved_ids[:k])
                            relevant_retrieved = retrieved_k.intersection(relevant_videos)
                            # Standard Recall@K = relevant_found_in_top_k / k
                            recall_k = len(relevant_retrieved) / k
                            category_recalls[k].append(recall_k)
                    
                    except Exception as e:
                        logger.error(f"Error in category {category}, video {query_video_id}: {e}")
                        continue
                
                # Calculate average recall for this category
                avg_category_recalls = {}
                for k in k_values:
                    if category_recalls[k]:
                        avg_category_recalls[f'recall@{k}'] = np.mean(category_recalls[k])
                    else:
                        avg_category_recalls[f'recall@{k}'] = 0.0
                
                group_results[category] = {
                    'average_recalls': avg_category_recalls,
                    'video_count': len(category_videos),
                    'query_count': len(category_recalls[k_values[0]])
                }
            
            category_results[group_name] = group_results
        
        return category_results
    
    def generate_comprehensive_report(self, k_values: List[int] = [1, 3, 5], quality_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Generate a comprehensive recall evaluation report.
        
        Args:
            k_values: List of K values for evaluation
            
        Returns:
            Complete evaluation report
        """
        logger.info("Generating comprehensive recall evaluation report...")
        
        report = {
            'evaluation_config': {
                'k_values': k_values,
                'annotation_file': str(self.ground_truth.annotation_path),
                'total_annotated_videos': len(self.ground_truth.video_to_keywords),
                'total_keywords': len(self.ground_truth.keyword_to_videos)
            }
        }
        
        # Video-to-video recall
        try:
            v2v_results = self.evaluate_video_to_video_recall(k_values)
            report['video_to_video'] = v2v_results
        except Exception as e:
            logger.error(f"Video-to-video evaluation failed: {e}")
            report['video_to_video'] = {'error': str(e)}
        
        # Text-to-video recall
        try:
            t2v_results = self.evaluate_text_to_video_recall(k_values, quality_threshold)
            report['text_to_video'] = t2v_results
        except Exception as e:
            logger.error(f"Text-to-video evaluation failed: {e}")
            report['text_to_video'] = {'error': str(e)}
        
        # Category-specific recall
        try:
            cat_results = self.evaluate_category_specific_recall(k_values)
            report['category_specific'] = cat_results
        except Exception as e:
            logger.error(f"Category-specific evaluation failed: {e}")
            report['category_specific'] = {'error': str(e)}
        
        return report


def run_recall_evaluation(annotation_csv_path: str = None, search_engine: VideoSearchEngine = None, quality_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Run complete recall evaluation and return results.
    
    Args:
        annotation_csv_path: Path to annotation CSV file (defaults to project annotation file)
        search_engine: Optional search engine instance (creates new if None)
        
    Returns:
        Complete evaluation report
    """
    if annotation_csv_path is None:
        # Default to project annotation file
        annotation_csv_path = str(project_root / "data" / "annotation" / "video_annotation.csv")
    
    if search_engine is None:
        config = VideoRetrievalConfig()
        search_engine = VideoSearchEngine(config=config)
    
    # Initialize ground truth processor
    ground_truth = GroundTruthProcessor(annotation_csv_path)
    
    # Initialize evaluator
    evaluator = RecallEvaluator(search_engine, ground_truth)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(k_values=[1, 3, 5], quality_threshold=quality_threshold)
    
    return report


def run_text_to_video_evaluation(keywords: List[str] = None, k_values: List[int] = [1, 3, 5], 
                                annotation_csv_path: str = None, search_engine: VideoSearchEngine = None,
                                quality_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Run text-to-video recall evaluation for specific keywords.
    
    Args:
        keywords: List of keywords to evaluate (if None, evaluates all keywords)
        k_values: List of K values for Recall@K evaluation
        annotation_csv_path: Path to annotation CSV file
        search_engine: Optional search engine instance
        
    Returns:
        Text-to-video evaluation results
    """
    if annotation_csv_path is None:
        annotation_csv_path = str(project_root / "data" / "annotation" / "video_annotation.csv")
    
    if search_engine is None:
        config = VideoRetrievalConfig()
        search_engine = VideoSearchEngine(config=config)
    
    ground_truth = GroundTruthProcessor(annotation_csv_path)
    evaluator = RecallEvaluator(search_engine, ground_truth)
    
    # Filter keywords if specified
    if keywords is not None:
        # Temporarily modify the ground truth to only include specified keywords
        original_keyword_to_videos = evaluator.ground_truth.keyword_to_videos.copy()
        filtered_keywords = {k: v for k, v in original_keyword_to_videos.items() if k in keywords}
        evaluator.ground_truth.keyword_to_videos = filtered_keywords
    
    try:
        results = evaluator.evaluate_text_to_video_recall(k_values, quality_threshold)
        
        # Add keyword-specific breakdown
        if keywords is not None:
            keyword_breakdown = {}
            for keyword in keywords:
                if keyword in original_keyword_to_videos:
                    relevant_videos = evaluator.ground_truth.get_relevant_videos_for_text(keyword)
                    if len(relevant_videos) > 0:
                        try:
                            max_k = max(k_values)
                            search_results = search_engine.search_by_text(keyword, top_k=max_k)
                            
                            # Apply quality threshold filtering
                            if quality_threshold > 0.0:
                                # Check if top result meets quality threshold
                                if not search_results or search_results[0].get('similarity_score', search_results[0].get('similarity', 0)) < quality_threshold:
                                    # Skip this keyword due to low quality
                                    continue
                                
                                # Filter all results by quality threshold
                                search_results = [r for r in search_results if r.get('similarity_score', r.get('similarity', 0)) >= quality_threshold]
                            
                            retrieved_ids = [result['slice_id'] for result in search_results]
                            
                            keyword_recalls = {}
                            for k in k_values:
                                retrieved_k = set(retrieved_ids[:k])
                                relevant_retrieved = retrieved_k.intersection(relevant_videos)
                                # Standard Recall@K = relevant_found_in_top_k / k
                                recall_k = len(relevant_retrieved) / k
                                keyword_recalls[f'recall@{k}'] = recall_k
                            
                            keyword_breakdown[keyword] = {
                                'recalls': keyword_recalls,
                                'relevant_count': len(relevant_videos),
                                'retrieved_ids': retrieved_ids[:max(k_values)],
                                'quality_threshold': quality_threshold
                            }
                        except Exception as e:
                            logger.error(f"Error evaluating keyword '{keyword}': {e}")
            
            results['keyword_breakdown'] = keyword_breakdown
        
        return results
    
    finally:
        # Restore original keywords
        if keywords is not None:
            evaluator.ground_truth.keyword_to_videos = original_keyword_to_videos


def run_video_to_video_evaluation(keywords: List[str] = None, k_values: List[int] = [1, 3, 5],
                                 annotation_csv_path: str = None, search_engine: VideoSearchEngine = None) -> Dict[str, Any]:
    """
    Run video-to-video recall evaluation for videos with specific keywords.
    
    Args:
        keywords: List of keywords to filter videos by (if None, evaluates all videos)
        k_values: List of K values for Recall@K evaluation
        annotation_csv_path: Path to annotation CSV file
        search_engine: Optional search engine instance
        
    Returns:
        Video-to-video evaluation results
    """
    if annotation_csv_path is None:
        annotation_csv_path = str(project_root / "data" / "annotation" / "video_annotation.csv")
    
    if search_engine is None:
        config = VideoRetrievalConfig()
        search_engine = VideoSearchEngine(config=config)
    
    ground_truth = GroundTruthProcessor(annotation_csv_path)
    evaluator = RecallEvaluator(search_engine, ground_truth)
    
    # Filter videos by keywords if specified
    target_videos = None
    if keywords is not None:
        target_videos = set()
        for keyword in keywords:
            target_videos.update(ground_truth.keyword_to_videos.get(keyword, set()))
        target_videos = list(target_videos)
    else:
        target_videos = list(ground_truth.video_to_keywords.keys())
    
    logger.info(f"Evaluating video-to-video recall for {len(target_videos)} videos with keywords: {keywords}")
    
    all_recalls = {k: [] for k in k_values}
    detailed_results = []
    
    for query_video_id in target_videos:
        try:
            # Get ground truth relevant videos
            relevant_videos = ground_truth.get_relevant_videos(query_video_id, include_self=False)
            
            if len(relevant_videos) == 0:
                logger.warning(f"No relevant videos found for {query_video_id}")
                continue
            
            # Get video info
            video_info = ground_truth.get_video_info(query_video_id)
            
            # Use pre-computed embedding for faster evaluation
            max_k = max(k_values)
            try:
                search_results = evaluator._search_by_precomputed_embedding(
                    query_video_id, 
                    top_k=max_k
                )
            except ValueError:
                # Fallback to video file search if pre-computed embedding not available
                logger.warning(f"No pre-computed embedding for {query_video_id}, using video file")
                query_video_path = video_info['video_path']
                search_results = search_engine.search_by_video(
                    query_video_path, 
                    top_k=max_k,
                    use_cache=True
                )
            
            # Extract retrieved video IDs
            retrieved_ids = [result['slice_id'] for result in search_results]
            
            # Calculate recall for each K
            recalls_for_query = {}
            for k in k_values:
                retrieved_k = set(retrieved_ids[:k])
                relevant_retrieved = retrieved_k.intersection(relevant_videos)
                recall_k = len(relevant_retrieved) / len(relevant_videos)
                recalls_for_query[k] = recall_k
                all_recalls[k].append(recall_k)
            
            detailed_results.append({
                'query_video': query_video_id,
                'query_keywords': video_info['keywords'],
                'relevant_count': len(relevant_videos),
                'retrieved_ids': retrieved_ids,
                'recalls': recalls_for_query
            })
            
        except Exception as e:
            logger.error(f"Error evaluating {query_video_id}: {e}")
            continue
    
    # Calculate average recalls
    avg_recalls = {}
    for k in k_values:
        if all_recalls[k]:
            avg_recalls[f'recall@{k}'] = np.mean(all_recalls[k])
        else:
            avg_recalls[f'recall@{k}'] = 0.0
    
    return {
        'average_recalls': avg_recalls,
        'detailed_results': detailed_results,
        'total_queries': len(detailed_results),
        'filter_keywords': keywords
    }


def print_recall_results(results: Dict[str, Any]):
    """
    Print recall evaluation results in a formatted way.
    
    Args:
        results: Results dictionary from run_recall_evaluation
    """
    print("\n📊 RECALL EVALUATION RESULTS")
    print("=" * 50)
    
    # Print configuration
    if 'evaluation_config' in results:
        config = results['evaluation_config']
        print(f"\n📋 Configuration:")
        print(f"  Total annotated videos: {config['total_annotated_videos']}")
        print(f"  Total keywords: {config['total_keywords']}")
        print(f"  K values: {config['k_values']}")
    
    # Print video-to-video results
    if 'video_to_video' in results and 'average_recalls' in results['video_to_video']:
        print(f"\n🎬 Video-to-Video Recall:")
        v2v = results['video_to_video']
        for metric, value in v2v['average_recalls'].items():
            print(f"  {metric}: {value:.3f}")
        print(f"  Total queries: {v2v['total_queries']}")
    
    # Print text-to-video results
    if 'text_to_video' in results and 'average_recalls' in results['text_to_video']:
        print(f"\n📝 Text-to-Video Recall:")
        t2v = results['text_to_video']
        for metric, value in t2v['average_recalls'].items():
            print(f"  {metric}: {value:.3f}")
        print(f"  Total queries: {t2v['total_queries']}")
    
    # Print category-specific results
    if 'category_specific' in results:
        print(f"\n🏷️  Category-Specific Recall:")
        for group_name, group_data in results['category_specific'].items():
            print(f"\n  {group_name.upper()}:")
            for category, cat_data in group_data.items():
                if 'average_recalls' in cat_data:
                    recalls = cat_data['average_recalls']
                    video_count = cat_data.get('video_count', 0)
                    query_count = cat_data.get('query_count', 0)
                    print(f"    {category} ({video_count} videos, {query_count} queries):")
                    print(f"      R@1={recalls.get('recall@1', 0):.3f}, R@3={recalls.get('recall@3', 0):.3f}, R@5={recalls.get('recall@5', 0):.3f}")


if __name__ == '__main__':
    print("🎯 Video Search Recall Evaluation")
    print("=" * 50)
    
    # Check if annotation file exists
    annotation_path = project_root / "data" / "annotation" / "video_annotation.csv"
    if not annotation_path.exists():
        print(f"❌ Annotation file not found: {annotation_path}")
        print("Please ensure the video_annotation.csv file exists.")
        sys.exit(1)
    
    try:
        print("🚀 Running recall evaluation...")
        results = run_recall_evaluation()
        print_recall_results(results)
        
    except Exception as e:
        print(f"❌ Recall evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
