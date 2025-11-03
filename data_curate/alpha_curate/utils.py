"""Utility functions for ALFA based data selection strategy."""


from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Final

import fsspec
import lancedb
import numpy as np
import numpy.typing as npt

from autonomy.perception.datasets.features.cosmos.infer import Cosmos
from kits.scalex.dataset.constants import SLICE_ID
from kits.scalex.dataset.instances.lance_dataset import read_database_path
from kits.scalex.dataset.stage_str import get_manifest_from_stage_str
from kits.scalex.io_v2.read_table import read_arrow_table_with_retries
from platforms.lakefs.client import LakeFS

ROW_ID_COLUMN: Final = "row_id"
SENSOR_NAME_COLUMN: Final = "sensor_name"
DISTANCE_COLUMN: Final = "_distance"
EMBEDDING_COLUMN: Final = "embedding"
IDENTIFIERS_COLUMN: Final = "identifiers"
LOGAPPS_METADATA_COLUMN: Final = "logapps_metadata"
IMAGE_PATHS_COLUMN: Final = "image_paths"


@dataclass
class SearchResult:
    """Result from text-to-video similarity search.

    Similar to EmbeddedVideo with distance
    (autonomy/perception/datasets/features/cosmos_index/generate_cosmos_index.py).
    """

    row_id: str
    sensor_name: str
    distance: float

    slice_id: str = field(init=False)
    segment_start_ns: int = field(init=False)
    segment_end_ns: int = field(init=False)
    camera_name: str = field(init=False)

    def __post_init__(self) -> None:
        """Post init processing."""
        if "_segment_" not in self.row_id:
            raise ValueError(f"Invalid row_id format: {self.row_id}")

        base_row_id, remainder = self.row_id.split("_segment_", 1)

        # Split remainder - only split on first 2 underscores
        parts = remainder.split("_", 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid row_id format: {self.row_id}")

        self.slice_id = base_row_id
        self.segment_start_ns = int(parts[0])
        self.segment_end_ns = int(parts[1])
        self.camera_name = parts[2]

    @property
    def similarity(self) -> float:
        """Convert l2 distance 0~sqrt(2) to similarity score 0~1."""
        return l2_distance_to_similarity(self.distance)


@lru_cache(maxsize=1)
def load_table(repo: str, branch: str, table_name: str) -> Any:
    """Loads the lancedb table."""

    lakefs = LakeFS()
    db_path = read_database_path(repo, branch, lakefs)
    db = lancedb.connect(db_path)
    return db.open_table(table_name)


@lru_cache(maxsize=2)
def load_model(model_size: str) -> Cosmos:
    """Load and cache Cosmos model."""
    return Cosmos(model_size=model_size, load_model_from_lakefs=False)


def text_to_embedding(query: str, model_size: str) -> npt.NDArray[np.float32]:
    """Generate text embedding."""
    model = load_model(model_size)
    return model.text_embedding(query)


def run_text_query(
    table: Any,
    query: str,
    model_size: str,
    upper_bound_distance: float | None = None,
    camera_names: Sequence[str] = (),
    limit: int = 100_000,
) -> list[SearchResult]:
    """Search for text query in Cosmos embeddings using LanceDB."""
    vec = text_to_embedding(query, model_size)
    columns = [ROW_ID_COLUMN, SENSOR_NAME_COLUMN, DISTANCE_COLUMN]
    cameras_string = "', '".join(name for name in camera_names)
    where_clause = "TRUE" if not camera_names else f"{SENSOR_NAME_COLUMN} IN ('{cameras_string}')"
    results = (
        table.search(vec, vector_column_name=EMBEDDING_COLUMN)
        .distance_range(upper_bound=upper_bound_distance)
        .where(where_clause)
        .limit(limit)
        .select(columns)
        .to_list()
    )
    return sorted(
        [
            SearchResult(row_id=r[ROW_ID_COLUMN], sensor_name=r[SENSOR_NAME_COLUMN], distance=r[DISTANCE_COLUMN])
            for r in results
        ],
        key=lambda x: x.similarity,
        reverse=True,
    )


def l2_distance_to_similarity(distance: float) -> float:
    """Convert LanceDB l2 distance to similarity in [0, 1]."""
    return max(0.0, 1.0 - (float(distance) / 2.0))


def similarity_to_l2_distance(similarity: float) -> float:
    """Convert similarity in [0, 1] to LanceDB l2 distance."""
    return 2.0 * (1.0 - similarity)


def deduplicate_by_base_slice(results: list[SearchResult]) -> list[SearchResult]:
    """Keep only the best (highest similarity) hit per base slice id."""
    dedup_by_base: dict[str, SearchResult] = {}

    for result in results:
        if not result.row_id:
            continue
        base_slice_id = result.slice_id
        similarity = result.similarity

        if base_slice_id not in dedup_by_base or similarity > dedup_by_base[base_slice_id].similarity:
            dedup_by_base[base_slice_id] = result

    deduped = list(dedup_by_base.values())
    deduped.sort(key=lambda x: x.similarity, reverse=True)
    return deduped


def get_slice_ids_to_exclude(
    filter_reference: str, lakefs: LakeFS, read_filesystem: fsspec.AbstractFileSystem
) -> set[str]:
    """Get the IDs of slices that should be excluded from the dataset because they are labeled or from the track.

    Args:
        filter_reference: Reference string for the filter manifest.
        lakefs: LakeFS client instance.
        read_filesystem: Filesystem to read data from.

    Returns:
        A set of slice IDs to exclude.
    """
    exclude_slices_manifest = get_manifest_from_stage_str(filter_reference, lakefs)
    references = list(exclude_slices_manifest.key_to_data_file.values())
    assert len(references) == 1, "One parquet file is expected for this dataset."
    exclude_slices_table = read_arrow_table_with_retries(
        references[0].physical_address, read_filesystem, filters=None, cache=False, columns=[SLICE_ID]
    )
    exclude_slice_ids = set(exclude_slices_table.column(SLICE_ID).to_pylist())
    return exclude_slice_ids
