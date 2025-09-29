from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import lancedb
import numpy as np
import numpy.typing as npt

from autonomy.perception.datasets.features.cosmos.infer import Cosmos
from kits.scalex.dataset.instances.lance_dataset import read_database_path
from platforms.lakefs.client import LakeFS


# Optional environment overrides for operational flexibility
_ENV_DB_PATH = os.getenv("ALFA_LANCEDB_PATH")
_ENV_REPO = os.getenv("ALFA_LANCEDB_REPO")
_ENV_TABLE = os.getenv("ALFA_LANCEDB_TABLE")


@lru_cache(maxsize=1)
def get_lakefs() -> LakeFS:
    return LakeFS()


@lru_cache(maxsize=8)
def load_table(branch: str) -> Any:
    """Open LanceDB table for the given branch.

    Priority:
    1) ALFA_LANCEDB_PATH (path to LanceDB dir) + ALFA_LANCEDB_TABLE
    2) ALFA_LANCEDB_REPO + ALFA_LANCEDB_TABLE via LakeFS read_database_path
    """
    table_name = _ENV_TABLE or "cosmos"  # sensible default; override via env

    if _ENV_DB_PATH:
        db = lancedb.connect(_ENV_DB_PATH)
        return db.open_table(table_name)

    if not _ENV_REPO:
        raise ValueError(
            "Missing ALFA_LANCEDB_REPO or ALFA_LANCEDB_PATH. Set env vars to locate the LanceDB database."
        )

    db_path = read_database_path(_ENV_REPO, branch, get_lakefs())
    db = lancedb.connect(db_path)
    return db.open_table(table_name)


@lru_cache(maxsize=2)
def load_model(model_size: str) -> Cosmos:
    # Align with streamlit app behavior: do not load model from LakeFS
    return Cosmos(model_size=model_size, load_model_from_lakefs=False)


def text_to_embedding(query: str, model_size: str) -> npt.NDArray[np.float32]:
    model = load_model(model_size)
    return model.text_embedding(query)


def distance_to_similarity(distance: float) -> float:
    """Convert LanceDB distance to similarity in [0, 1].

    Mirrors logic used in the Streamlit app to keep parity.
    """
    return max(0.0, 1.0 - (float(distance) / 2.0))


def parse_row_id(row_id: str) -> tuple[str, str, str, str]:
    """Parse a slice row_id into components: (base_video_id, start_ns, end_ns, camera_name).

    If the id does not contain segment metadata, returns (row_id, "", "", "").
    """
    if "_segment_" not in row_id:
        return row_id, "", "", ""

    base_row_id, remainder = row_id.split("_segment_", 1)
    parts = remainder.split("_", 2)
    start_ns = parts[0] if len(parts) > 0 else ""
    end_ns = parts[1] if len(parts) > 1 else ""
    camera_name = parts[2] if len(parts) > 2 else ""
    return base_row_id, start_ns, end_ns, camera_name


def deduplicate_by_base_slice_id(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the best (lowest _distance) hit per base video id.

    Returns results sorted by ascending _distance.
    """
    best_by_base: dict[str, dict[str, Any]] = {}
    best_dist: dict[str, float] = {}

    for r in results:
        row_id = str(r.get("row_id", ""))
        if not row_id:
            continue
        base_video_id, _, _, _ = parse_row_id(row_id)
        dist = float(r.get("_distance", float("inf")))

        prev = best_dist.get(base_video_id)
        if prev is None or dist < prev:
            best_dist[base_video_id] = dist
            best_by_base[base_video_id] = r

    deduped = list(best_by_base.values())
    deduped.sort(key=lambda x: float(x.get("_distance", float("inf"))))
    return deduped


def run_text_query(branch: str, query: str, top_k: int, model_size: str) -> list[dict[str, Any]]:
    """Search for text query in Cosmos embeddings using LanceDB.
    
    Args:
        branch: LanceDB branch to search
        query: Text query to search for
        top_k: Number of top results to return
        model_size: Cosmos model size for text embedding
        
    Returns:
        List of search results with row_id and _distance fields
    """
    table = load_table(branch)
    vec = text_to_embedding(query, model_size)
    return table.search(vec, vector_column_name="embedding").limit(top_k).to_list()
