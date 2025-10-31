"""Score fusion for CLIP-style embeddings."""

import logging
from typing import Final

from autonomy.perception.datasets.active_learning.alfa_curate.generate_prompt_variants import VariedPrompts
from autonomy.perception.datasets.active_learning.alfa_curate.utils import SearchResult


_LOGGER: Final = logging.getLogger(__name__)


def fuse_scores(
    expanded: VariedPrompts,
    results: dict[str, list[SearchResult]],
) -> dict[str, float]:
    """
    Fuse scores from multiple prompt variants using simple averaging.
    
    All variants are treated equally. Adds diversity bonus for videos 
    matching multiple variants (CLIP ensemble effect).
    """
    video_scores: dict[str, float] = {}
    video_counts: dict[str, int] = {}
    
    # Accumulate scores from all variants (all treated equally)
    for variant_text, search_results in results.items():
        for result in search_results:
            vid = result.row_id
            video_scores[vid] = video_scores.get(vid, 0.0) + result.similarity
            video_counts[vid] = video_counts.get(vid, 0) + 1
    
    # Average scores and apply diversity bonus
    fused = {}
    for vid, total_score in video_scores.items():
        # Simple average across all variants that matched this video
        base = total_score / video_counts[vid]
        
        # Diversity bonus: videos matching multiple variants get boost (max +15%)
        # This rewards CLIP ensemble agreement
        bonus = min(0.05 * (video_counts[vid] - 1), 0.15)
        fused[vid] = min(1.0, base + bonus)
    
    _LOGGER.info("Fused %d videos from %d variants", len(fused), len(results))
    return fused
