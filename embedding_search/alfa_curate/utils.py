from functools import lru_cache
from typing import Any, Final

import lancedb
import numpy as np
import numpy.typing as npt

from autonomy.perception.datasets.features.cosmos.config import COSMOS_MODEL_SIZE
from autonomy.perception.datasets.features.cosmos.infer import Cosmos
from kits.scalex.dataset.instances.lance_dataset import read_database_path
from platforms.lakefs.client import LakeFS

REPO: Final = "sensing--features--cosmos-index"
BRANCH: Final = "main"
TABLE: Final = "data"
MODEL_SIZE: Final = COSMOS_MODEL_SIZE


@lru_cache(maxsize=1)
def load_table(branch: str, repo: str = REPO, table_name: str = TABLE) -> Any:
    """Loads the lancedb table."""
    lakefs = LakeFS()
    db_path = read_database_path(repo, branch, lakefs)
    db = lancedb.connect(db_path)
    return db.open_table(table_name)


@lru_cache(maxsize=2)
def load_model(model_size: str) -> Cosmos:
    return Cosmos(model_size=model_size, load_model_from_lakefs=False)


def text_to_embedding(query: str, model_size: str) -> npt.NDArray[np.float32]:
    model = load_model(model_size)
    return model.text_embedding(query)


def distance_to_similarity(distance: float) -> float:
    """Convert LanceDB distance to similarity in [0, 1]."""
    return max(0.0, 1.0 - (float(distance) / 2.0))


def parse_row_id(row_id: str) -> tuple[str, str, str, str]:
    """Parse a slice row_id into components: (base_video_id, start_ns, end_ns, camera_name)."""
    if "_segment_" not in row_id:
        return row_id, "", "", ""

    base_row_id, remainder = row_id.split("_segment_", 1)
    parts = remainder.split("_", 2)
    start_ns = parts[0] if len(parts) > 0 else ""
    end_ns = parts[1] if len(parts) > 1 else ""
    camera_name = parts[2] if len(parts) > 2 else ""
    return base_row_id, start_ns, end_ns, camera_name


def deduplicate_by_base_slice_id(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the best (lowest _distance) hit per base video id."""
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


def run_text_query(branch: str, query: str, top_k: int, model_size: str = MODEL_SIZE) -> list[dict[str, Any]]:
    """Search for text query in Cosmos embeddings using LanceDB."""
    table = load_table(branch)
    vec = text_to_embedding(query, model_size)
    return table.search(vec, vector_column_name="embedding").limit(top_k).to_list()
