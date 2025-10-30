"""Simple prompt expansion for Cosmos CLIP-style embeddings."""

import logging
from dataclasses import dataclass
from typing import Final


_LOGGER: Final = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant."""
    text: str
    is_negative: bool = False


@dataclass
class VariedPrompts:
    """Collection of prompt variants."""
    original: str
    positive_variants: list[PromptVariant]
    negative_variants: list[PromptVariant]


def _generate_template_variants(query: str) -> list[str]:
    """
    Generate template-based variations.
    
    Templates rephrase the query in different natural ways that CLIP understands.
    All templates work universally for any query.
    """
    templates = [
        f"video of {query}",
        f"footage of {query}",
        f"scene showing {query}",
        f"clip of {query}",
    ]
    return templates


# Compact synonym dictionary for common driving/video terms
SYNONYMS: Final[dict[str, list[str]]] = {
    "person": ["pedestrian", "individual"],
    "people": ["pedestrians", "individuals"],
    "car": ["vehicle", "automobile"],
    "truck": ["vehicle"],
    "running": ["jogging", "moving quickly"],
    "walking": ["moving"],
    "crossing": ["traversing", "going across"],
    "turning": ["rotating"],
    "road": ["street", "roadway"],
    "street": ["road"],
    "intersection": ["junction", "crossroad"],
    "camera obstruction": ["dirty lens", "camera blocked"],
}


def generate_prompt(query: str, num_variants: int = 6) -> VariedPrompts:
    """
    Expand query into variants optimized for CLIP models.
    
    Generates variants from multiple sources:
    - Template rephrasing: "video of X", "footage of X", etc.
    - Synonym substitution: word replacements from dictionary
    
    All variants have equal weight (0.9) - they're just different ways to express
    the same concept. Original query has weight 1.0 as baseline.
    
    Args:
        query: Original query string
        num_variants: Number of variants to generate (default: 6)
                     More variants = better recall but slower
    
    Returns:
        VariedPrompts with original + variants
    """
    all_variants = []
    words = query.split()
    
    # Generate template variants (universal, work for any query)
    template_variants = _generate_template_variants(query)
    all_variants.extend(template_variants)
    
    # Generate synonym variants (if words are in dictionary)
    synonym_variants = []
    for i, word in enumerate(words):
        if word.lower() in SYNONYMS:
            for syn in SYNONYMS[word.lower()]:
                new_words = words.copy()
                new_words[i] = syn
                synonym_variants.append(" ".join(new_words))
    all_variants.extend(synonym_variants)
    
    # Remove duplicates
    seen = set([query.lower()])
    unique_variants = []
    for variant_text in all_variants:
        if variant_text.lower() not in seen:
            seen.add(variant_text.lower())
            unique_variants.append(variant_text)
    
    # Build final variant list: original + top N variants
    final_variants = [PromptVariant(query)]  # Original first
    final_variants.extend([
        PromptVariant(v) for v in unique_variants[:num_variants]
    ])
    
    # Generic negatives for CLIP
    negatives = [
        PromptVariant("empty scene", is_negative=True),
        PromptVariant("static view", is_negative=True),
    ]
    
    return VariedPrompts(query, final_variants, negatives)


def add_synonyms(word: str, synonyms: list[str]) -> None:
    """Add custom synonyms to dictionary."""
    SYNONYMS[word.lower()] = [s.lower() for s in synonyms]
