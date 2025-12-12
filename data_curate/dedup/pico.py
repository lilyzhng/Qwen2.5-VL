import logging
from typing import Any, Final, Optional, cast

import faiss
import numpy as np
import numpy.typing as npt
import pyarrow as pa

from autonomy.perception.datasets.active_learning.alfa_curate.utils import load_table
from autonomy.perception.datasets.human_labels.pico.config import HumanLabelsPicoConfig
from kits.scalex.dataset.constants import EMBEDDING, FRAMES_KEY, IDENTIFIERS, SLICE_ID, START_NS, TIMESTAMP_NS
from kits.scalex.dataset.index.index_writer import get_chunks
from kits.scalex.dataset.instances.parquet_dataset import ParquetDataset
from kits.scalex.dataset.stage_str import get_stage_and_reference
from kits.scalex.pipeline.ray.map import map_remote_to_args
from platforms.lakefs.client import LakeFS

_LOGGER: Final = logging.getLogger(__name__)

PICO_ROW_WINDOW_SEC = 0.5


# NOTE: Old VRU preservation functions removed (build_vru_frame_index, check_embedding_has_vru)
# These were part of the pre-computation approach where we built a VRU index upfront.
# New approach: VRU rescue happens per-cluster in parallel using rescue_vru_frames_from_excluded()
# See rescue_vru_frames_from_excluded() below for the current implementation.

def rescue_preserved_classes_from_excluded(
    excluded_indices: set[int],
    embeddings_table: pa.Table,
    gold_dataset: ParquetDataset,
    config: HumanLabelsPicoConfig,
) -> set[int]:
    """Check excluded embeddings for preserved objects and rescue them."""
    if not excluded_indices:
        return set()

    _LOGGER.info(f"Checking {len(excluded_indices)} excluded embeddings for preserved objects...")

    preserved_classes_lower = {cls.lower() for cls in config.preserved_classes}
    rescued_indices = set()

    identifiers = embeddings_table.column(IDENTIFIERS).to_pylist()
    start_ns_list = embeddings_table.column(START_NS).to_pylist()
    end_ns_list = embeddings_table.column(END_NS).to_pylist()  # ← ADD THIS
    slice_to_embeddings: dict[str, list[tuple[int, int, int]]] = {}  # ← Now includes end_ns

    for idx in excluded_indices:
        if idx >= embeddings_table.num_rows or idx < 0:
            raise ValueError(f"Invalid embedding index {idx}: out of range [0, {embeddings_table.num_rows})")

        identifier = identifiers[idx]
        if not identifier or SLICE_ID not in identifier:
            raise ValueError(f"Missing SLICE_ID in identifiers for embedding {idx}")

        slice_id = identifier.get(SLICE_ID)
        start_ns = start_ns_list[idx]
        end_ns = end_ns_list[idx]  # ← ADD THIS

        if slice_id not in slice_to_embeddings:
            slice_to_embeddings[slice_id] = []
        slice_to_embeddings[slice_id].append((idx, start_ns, end_ns))  # ← Include end_ns

    _LOGGER.info(f"Excluded embeddings span {len(slice_to_embeddings)} slices")

    def process_slice(slice_data: tuple[str, list[tuple[int, int, int]]]) -> set[int]:
        slice_id, embedding_infos = slice_data
        local_rescued = set()
        rows = gold_dataset.get_rows(ids=[slice_id])
        if rows.num_rows == 0:
            return local_rescued

        row = rows.slice(0, 1)
        frames_column = row.column(FRAMES_KEY)
        if not frames_column:
            return local_rescued

        frames = frames_column[0].as_py()
        if not frames:
            return local_rescued

        # Collect all timestamps with preserved objects
        preserved_obj_timestamps_in_slice = set()
        for frame in frames:
            cuboids = frame.get("cuboids", [])
            if cuboids and any(cuboid.get("label_class", "").lower() in preserved_classes_lower for cuboid in cuboids):
                preserved_obj_timestamps_in_slice.add(frame.get(TIMESTAMP_NS))

        if not preserved_obj_timestamps_in_slice:
            return local_rescued

        _LOGGER.info(
            f"Found {len(preserved_obj_timestamps_in_slice)} frames with preserved objects in slice {slice_id}"
        )

        # Check if any excluded embedding's time window contains a preserved object
        for embedding_idx, start_ns, end_ns in embedding_infos:
            for preserved_ts in preserved_obj_timestamps_in_slice:
                if start_ns <= preserved_ts <= end_ns:  # ← TIME WINDOW CHECK
                    local_rescued.add(embedding_idx)
                    break  # This embedding is rescued, no need to check more timestamps

        _LOGGER.info(f"Rescued {len(local_rescued)} embeddings from slice {slice_id}")
        return local_rescued

    rescue_tasks = [
        (slice_id, embedding_infos)
        for slice_id, embedding_infos in slice_to_embeddings.items()
    ]

    results = map_remote_to_args(
        process_slice,
        rescue_tasks,
        disable_ray=config.disable_ray,
        dynamically_adjust_memory=True,
    )

    for result in results:
        if isinstance(result, Exception):
            raise result
        rescued_indices.update(result)

    _LOGGER.info(
        f"Rescue complete: rescued {len(rescued_indices)} out of {len(excluded_indices)} "
        f"excluded embeddings ({100*len(rescued_indices)/len(excluded_indices) if excluded_indices else 0:.1f}%)"
    )

    return rescued_indices


def _find_reference_pico_row(
    current_idx: int,
    timestamps: list[Optional[float]],
    temporal_window_size: float,
    valid_data: list,
) -> Optional[int]:
    """Find reference pico row - simple and efficient."""
    if timestamps[current_idx] is None:
        return None
    
    target_time = timestamps[current_idx] - (temporal_window_size * 1e9)
    
    valid_indices = []
    valid_timestamps = []
    for j in range(current_idx):
        if valid_data[j] is not None and timestamps[j] is not None:
            valid_indices.append(j)
            valid_timestamps.append(timestamps[j])
    
    if len(valid_timestamps) == 0:
        return None
    
    # Compute all distances at once (vectorized!)
    distances = np.abs(np.array(valid_timestamps) - target_time)
    
    # Find index of minimum distance with O(n) time.
    nearest_idx = np.argmin(distances)
    
    return valid_indices[nearest_idx]


def _compute_temporal_velocities(
    pico_embeddings: list[Optional[npt.NDArray[np.float32]]],
    pico_timestamps: list[Optional[float]],
    temporal_window_size: float,
) -> list[Optional[float]]:
    """Compute temporal embedding velocities for pico rows.
    
    Velocity is calculated as: embedding_distance / actual_time_difference
    where actual_time_difference is the time between the current pico row's embedding
    timestamp and the reference pico row's embedding timestamp.
    
    Args:
        pico_embeddings: Representative embeddings for each pico row in the slice
        pico_timestamps: Embedding timestamps for each pico row (nanoseconds)
        temporal_window_size: Comparison window in seconds (used to find reference row)
    
    Returns:
        List of temporal velocities for each pico row
    """
    temporal_velocities = []
    
    for i in range(len(pico_embeddings)):
        if pico_embeddings[i] is None or pico_timestamps[i] is None:
            temporal_velocities.append(None)
            continue
        
        # Find reference pico row approximately temporal_window_size seconds ago
        ref_idx = _find_reference_pico_row(i, pico_timestamps, temporal_window_size, pico_embeddings)
        
        if ref_idx is not None:
            # Velocity = Δembedding_distance / actual_time_diff
            emb_distance = np.linalg.norm(pico_embeddings[i] - pico_embeddings[ref_idx])
            actual_time_diff_sec = (pico_timestamps[i] - pico_timestamps[ref_idx]) / 1e9  # Convert ns to seconds
            if actual_time_diff_sec > 0:
                velocity = float(emb_distance / actual_time_diff_sec)
                temporal_velocities.append(velocity)
            else:
                # Same timestamp, undefined velocity
                temporal_velocities.append(None)
        else:
            temporal_velocities.append(None)
    
    return temporal_velocities


def _compute_temporal_accelerations(
    temporal_velocities: list[Optional[float]],
    pico_timestamps: list[Optional[float]],
    temporal_window_size: float,
) -> list[Optional[float]]:
    """Compute temporal embedding accelerations for pico rows.
    
    Acceleration is calculated as: velocity_difference / actual_time_difference
    where actual_time_difference is the time between the current pico row's embedding
    timestamp and the reference pico row's embedding timestamp.
    
    Args:
        temporal_velocities: Temporal velocities for each pico row
        pico_timestamps: Embedding timestamps for each pico row (nanoseconds)
        temporal_window_size: Comparison window in seconds (used to find reference row)
    
    Returns:
        List of temporal accelerations for each pico row
    """
    temporal_accelerations = []
    
    for i in range(len(temporal_velocities)):
        if temporal_velocities[i] is None or pico_timestamps[i] is None:
            temporal_accelerations.append(None)
            continue
        
        # Find reference pico row approximately temporal_window_size seconds ago
        ref_idx = _find_reference_pico_row(i, pico_timestamps, temporal_window_size, temporal_velocities)
        
        if ref_idx is not None:
            # Acceleration = Δvelocity / actual_time_diff
            velocity_diff = temporal_velocities[i] - temporal_velocities[ref_idx]
            actual_time_diff_sec = (pico_timestamps[i] - pico_timestamps[ref_idx]) / 1e9  # Convert ns to seconds
            if actual_time_diff_sec > 0:
                acceleration = float(velocity_diff / actual_time_diff_sec)
                temporal_accelerations.append(acceleration)
            else:
                # Same timestamp, undefined acceleration
                temporal_accelerations.append(None)
        else:
            temporal_accelerations.append(None)
    
    return temporal_accelerations


def _pico_row_representative_embedding(
    frame_timestamps: list[int],
    embeddings_timestamps: npt.NDArray[np.int64],
    embeddings_array: npt.NDArray[np.float32],
) -> tuple[Optional[npt.NDArray[np.float32]], Optional[float]]:
    """Get representative embedding for a pico row using a hybrid approach.
    
    This function adapts to different embedding frequencies:
    - Dense embeddings (multiple per pico row): Uses mean pooling
    - Sparse embeddings (fewer than pico rows): Uses nearest neighbor
    
    Strategy:
    1. Find all embeddings that fall within the pico row's time window
    2. If found → Use mean pooling of those embeddings (optimal for dense embeddings)
    3. If not found → Use nearest embedding in time (handles sparse embeddings)
    
    This hybrid approach is future-proof and works optimally regardless of whether
    embeddings are generated every 50 frames, 10 frames, 5 frames, etc.
    
    Args:
        frame_timestamps: List of frame timestamps (nanoseconds) in the pico row
        embeddings_timestamps: Array of all embedding timestamps (nanoseconds)
        embeddings_array: Array of all embeddings, aligned with embeddings_timestamps
    
    Returns:
        Tuple of (representative_embedding, mean_timestamp) or (None, None) if no embeddings available
    """
    if len(frame_timestamps) == 0 or len(embeddings_array) == 0:
        return None, None
    
    # Get pico row time window bounds
    min_pico_time = min(frame_timestamps)
    max_pico_time = max(frame_timestamps)
    
    # Find embeddings that fall within the pico row's time window
    within_window_mask = (embeddings_timestamps >= min_pico_time) & (embeddings_timestamps <= max_pico_time)
    within_window_indices = np.where(within_window_mask)[0]
    
    if len(within_window_indices) > 0:
        # Dense embeddings case: Use mean pooling of embeddings within the pico row
        window_embeddings = embeddings_array[within_window_indices]
        window_timestamps = embeddings_timestamps[within_window_indices]
        
        mean_embedding = np.mean(window_embeddings, axis=0)
        mean_timestamp = float(np.mean(window_timestamps))
        
        return mean_embedding, mean_timestamp
    else:
        # Sparse embeddings case: Use nearest neighbor
        pico_center_timestamp = np.mean(frame_timestamps)
        time_diffs = np.abs(embeddings_timestamps - pico_center_timestamp)
        nearest_idx = np.argmin(time_diffs)
        
        return embeddings_array[nearest_idx], float(embeddings_timestamps[nearest_idx])


def compute_embedding_change_rates(
    table: pa.Table,
    embedding_table: pa.Table,
    temporal_window_size: float = 15.0,
) -> pa.Table:
    """Compute embedding change rates (velocity and acceleration) for each pico row.

    Uses a hybrid approach that adapts to any embedding frequency:
    - If embeddings exist within pico row → mean pooling (optimal for dense embeddings)
    - If no embeddings in pico row → nearest neighbor (handles sparse embeddings)
    
    This works optimally whether embeddings are every 50 frames, 10 frames, 5 frames, etc.
    
    Two temporal windows are involved:
    1. Pico row window (~0.5s): Used to find/pool embeddings for each row
    2. Comparison window (temporal_window_size): Used for computing embedding velocity/acceleration
    
    Steps:
    1. Assign each pico row a representative embedding (hybrid approach)
    2. Find reference pico row approximately temporal_window_size seconds in the past
    3. Embedding Velocity = Δembedding_distance / actual_time_difference
    4. Embedding Acceleration = Δvelocity / actual_time_difference
    
    Note: temporal_window_size is used to find the reference pico row, but velocity/acceleration
    are calculated using the ACTUAL time difference between embedding timestamps.
    
    Note: Assumes embedding_table is already filtered to relevant slices (done in transform_table).

    Args:
        table: Pico table with 'slice_id' column and frame timestamps
        embedding_table: Table containing embeddings (any frequency: sparse or dense), 
                        pre-filtered to relevant slices
        temporal_window_size: Comparison window in seconds for velocity/acceleration (default 15s)

    Returns:
        Table with 'embedding_velocity' and 'embedding_acceleration' columns added
    """

    embeddings_array = get_embedding_array_from_table(embedding_table)
    embeddings_timestamps = embedding_table.column(START_NS).to_numpy().astype(np.int64)

    slice_ids = table.column(SLICE_ID).to_pylist()
    frames = table.column(FRAMES_KEY)

    # Get per slice groups of pico rows
    groups: dict[str, list[int]] = {}
    for pico_idx, slice_id in enumerate(slice_ids):
        if slice_id not in groups:
            groups[slice_id] = []
        groups[slice_id].append(pico_idx)

    all_velocities = [None] * len(table)
    all_accelerations = [None] * len(table)

    for slice_id, pico_indices in groups.items():
        pico_embeddings = []
        pico_timestamps = []

        for pico_idx in pico_indices:
            # Get pico row frames and timestamps
            row_frames = frames[pico_idx].as_py()
            frame_timestamps = [frame[TIMESTAMP_NS] for frame in row_frames]

            # Get representative embedding for this pico row (hybrid approach)
            representative_embedding, embedding_timestamp = _pico_row_representative_embedding(
                frame_timestamps, embeddings_timestamps, embeddings_array
            )
            
            pico_embeddings.append(representative_embedding)
            pico_timestamps.append(embedding_timestamp)

        # Compute temporal velocities for pico rows
        temporal_velocities = _compute_temporal_velocities(
            pico_embeddings, pico_timestamps, temporal_window_size
        )

        # Compute temporal accelerations for pico rows
        temporal_accelerations = _compute_temporal_accelerations(
            temporal_velocities, pico_timestamps, temporal_window_size
        )
        
        # Update global output lists
        for i, pico_idx in enumerate(pico_indices):
            all_velocities[pico_idx] = temporal_velocities[i]
            all_accelerations[pico_idx] = temporal_accelerations[i]

    table = table.append_column("embedding_velocity", pa.array(all_velocities, type=pa.float64()))
    table = table.append_column("embedding_acceleration", pa.array(all_accelerations, type=pa.float64()))

    return table


def temporal_subsample_by_change_rate(
    table: pa.Table,
    embedding_table: pa.Table,
    use_acceleration: bool = True,
    temporal_window_size: float = 15.0,
    diversity_threshold: float = 0.05,
) -> pa.Table:
    """Subsample pico rows based on embedding change rates.

    Since embeddings are sparse (e.g., every 50 frames), each pico row is assigned
    the nearest embedding in time rather than pooling multiple embeddings.
    
    Two temporal windows:
    1. Pico row window (~0.5s): Each row uses its center timestamp to find nearest embedding
    2. Comparison window (temporal_window_size): Window for computing velocity/acceleration
    
    Strategy:
    - Compute embedding_velocity and embedding_acceleration using temporal_window_size
    - Keep rows where change rate >= diversity_threshold
    - Drop rows where change rate < diversity_threshold
    - Always keep rows where change rate is None (first rows in each slice)

    Args:
        table: Table with 'slice_id' column
        embedding_table: Table containing sparse embeddings (e.g., every 50 frames)
        use_acceleration: If True, filter based on embedding_acceleration. If False, filter based on embedding_velocity.
        temporal_window_size: Comparison window for velocity/acceleration in seconds (default 15s)
        diversity_threshold: Minimum change rate to keep a row. Rows below this threshold are dropped.
                           Default: 0.05
    Returns:
        Subsampled table with 'embedding_velocity' and 'embedding_acceleration' columns added
    """
    # Step 1: Compute embedding change rates (velocity and acceleration) for each pico row
    table = compute_embedding_change_rates(
        table, 
        embedding_table,
        temporal_window_size=temporal_window_size
    )
    
    # Step 2: Filter based on change rate threshold
    metric_name = "embedding_acceleration" if use_acceleration else "embedding_velocity"
    embedding_change_rates = table.column(metric_name).to_pylist()
    
    # Keep rows where change_rate >= threshold or change_rate is None
    keep_indices = []
    kept_count = 0
    dropped_count = 0
    
    for idx, change_rate in enumerate(embedding_change_rates):
        if change_rate is None or change_rate >= diversity_threshold:
            keep_indices.append(idx)
            kept_count += 1
        else:
            dropped_count += 1
    
    if table.num_rows > 0:
        percentage = len(keep_indices) / table.num_rows * 100
        _LOGGER.info(
            f"Threshold-based filtering by {metric_name} (threshold={diversity_threshold}): "
            f"{table.num_rows} → {len(keep_indices)} pico rows ({percentage:.1f}%). "
            f"Kept: {kept_count}, Dropped: {dropped_count}"
        )
    else:
        _LOGGER.info("No pico rows to filter (table is empty)")

    if len(keep_indices) == 0:
        # Return empty table with the same schema
        return table
    return table.take(keep_indices)


def _filter_frame_groups(grouped: pa.Array, include: list[int], exclude: list[int]) -> pa.Array:
    """Filter frame groups based on include and exclude timestamps.

    The function looks at the first timestamp (using the timestamp_ns field) of each group in the grouped array and
    calculates the distance to the closest timestamp in the include and exclude lists. If the closest include timestamp
    is less than or equal to the closest exclude timestamp, the group is kept; otherwise, it is filtered out.

    Args:
        grouped: A PyArrow Array of frame groups, where each group is an array of frames. Each frame should have a
            `timestamp_ns` field. Groups do not need to be in sorted order, but the frames within each group should be
            sorted by timestamp. The timestamp used for comparison is the timestamp of the first frame in each group.
        include: A list of timestamps to include. These timestamps do not need to be sorted.
        exclude: A list of timestamps to exclude. These timestamps do not need to be sorted.


    Returns:
        A PyArrow Array containing only the groups that meet the inclusion criteria.
    """
    if not include and not exclude:
        return grouped

    include_indices = []
    for j in range(len(grouped)):
        first_timestamp = grouped[j][0][TIMESTAMP_NS].as_py()
        include_dists = [abs(first_timestamp - include_timestamp) for include_timestamp in include]
        best_include = min(include_dists) if include_dists else np.inf
        exclude_dists = [abs(first_timestamp - exclude_timestamp) for exclude_timestamp in exclude]
        best_exclude = min(exclude_dists) if exclude_dists else np.inf
        if best_include <= best_exclude:
            include_indices.append(j)
    if not include_indices:
        return pa.array([])
    return grouped.take(include_indices)


def filter_embeddings_by_slice_ids(
    embeddings_table: pa.Table, 
    slice_ids: set[str]
) -> pa.Table:
    """Filter embeddings table to only include specified slice_ids.
    
    This function filters the embeddings table to only include rows where the
    slice_id in the IDENTIFIERS column matches one of the provided slice_ids.
    This is useful to avoid loading all embeddings when only processing a subset of slices.
    
    Args:
        embeddings_table: Table with IDENTIFIERS column containing slice_id information
        slice_ids: Set of slice_ids to keep
        
    Returns:
        Filtered table containing only embeddings for specified slices
    """
    if len(slice_ids) == 0 or embeddings_table.num_rows == 0:
        # Return empty table with same schema
        return embeddings_table.slice(0, 0)
    
    identifiers = embeddings_table.column(IDENTIFIERS).to_pylist()
    mask = [row[SLICE_ID] in slice_ids for row in identifiers]
    return embeddings_table.filter(pa.array(mask, type=pa.bool_()))


def transform_table(
    table: pa.Table,
    stride: int,
    frames_per_row: int,
    include_timestamps: pa.Array,
    exclude_timestamps: pa.Array,
    config: HumanLabelsPicoConfig,
    embeddings_table: Optional[pa.Table] = None,
) -> pa.Table:
    """Transform the input table by splitting frames into groups, applying stride, and filtering.


    This function will take a table of N rows and transform it into a table of N / frames_per_row, where each row
    contains frames_per_row frames.  The function will also reduce the number of output rows based on the stride or
    include/exclude timestamps.
      - stride is used to only keep every Nth group of frames.
      - include_timestamps and exclude_timestamps are used to filter out groups of frames based on their timestamps. If
        both include and exclude are empty for a given row, no filtering is applied.


    Args:
        table: A PyArrow Table containing a "frames" column, where each entry is a list of frames. Each frame should
            have a `timestamp_ns` field.
        stride: An integer representing the stride to apply when selecting groups of frames. For example, a stride of 3
            will keep every 3rd group of frames.
        frames_per_row: An integer representing the number of frames to include in each output row.
        include_timestamps: A PyArrow Array of lists of timestamps to include for each row. This array should be the
            same length as table and contain lists of timestamps that roughly correspond to the frames in each row.
        exclude_timestamps: A PyArrow Array of lists of timestamps to exclude for each row. This array should be the
            same length as table and contain lists of timestamps that roughly correspond to the frames in each row.
        config: The configuration object containing parameters for temporal subsampling.
        embeddings_table: Optional pre-loaded embeddings table. Required if config.apply_temporal_subsampling is True.
            For distributed processing (e.g., Ray), load once, filter to relevant slices, and pass via object store.
        use_acceleration: If True, filter based on embedding_acceleration. If False, filter based on embedding_velocity.

    Returns:
        A PyArrow Table containing the transformed rows with filtered frame groups.
    """

    frames = table.column(FRAMES_KEY)
    if len(frames) != len(include_timestamps) or len(frames) != len(exclude_timestamps):
        raise ValueError(
            "Length of frames, include_timestamps, and exclude_timestamps must be the same, got "
            f"{len(frames)}, {len(include_timestamps)}, {len(exclude_timestamps)}"
        )

    # Validate that embeddings_table is provided when temporal subsampling is enabled
    if config.apply_temporal_subsampling and embeddings_table is None:
        raise ValueError(
            "embeddings_table is required when apply_temporal_subsampling is True. "
        )

    without_frames = table.drop_columns([FRAMES_KEY])

    for i in range(table.num_rows):
        flat_frames = frames.slice(i, 1).combine_chunks().values
        if len(flat_frames) == 0:
            continue

        chunked_frames = list(get_chunks(flat_frames, frames_per_row))
        if stride > 1:
            chunked_frames = chunked_frames[::stride]
        grouped = pa.array(chunked_frames, frames.type)

        include = include_timestamps[i].as_py()
        exclude = exclude_timestamps[i].as_py()
        grouped = _filter_frame_groups(grouped, include, exclude)

        if len(grouped) == 0:
            continue

        full_table = without_frames.take([i] * len(grouped)).append_column(FRAMES_KEY, grouped)

        # Subsample based on embedding change rates
        if config.apply_temporal_subsampling and embeddings_table is not None:
            # Filter embeddings to only include the slice(s) in full_table
            # Extract slice_ids from the IDENTIFIERS column
            identifiers = full_table.column(IDENTIFIERS).to_pylist()
            slice_ids = {row[SLICE_ID] for row in identifiers}
            filtered_embeddings = filter_embeddings_by_slice_ids(embeddings_table, slice_ids)
            
            full_table = temporal_subsample_by_change_rate(
                full_table,
                filtered_embeddings,
                use_acceleration=config.use_acceleration,
                temporal_window_size=config.temporal_window_size,
                diversity_thresholds=config.diversity_thresholds,
                stride_multipliers=config.stride_multipliers
            )

        yield full_table


def get_embeddings_table(
    features_dinov2_index_reference: str, lakefs: LakeFS, embedding_column: str = EMBEDDING, table_name: str = "data"
) -> pa.Table:
    """Load the embeddings table from the DINOv2 index dataset.

    This function assumes that the embeddings table is a LanceDB table referenced by LakeFS. The table should have
    an identifiers column containing slice and frame information, a start_ns column for timestamps, and a column
    containing the embeddings.

    Args:
        features_dinov2_index_reference: The LakeFS stage and reference for the DINOv2 index dataset.
        lakefs: The LakeFS client to use for loading the dataset.
        embedding_column: The name of the column containing the embeddings.
        table_name: The name of the table to load.


    Returns:
        A PyArrow Table containing the embeddings.
    """
    dino_stage, dino_reference = get_stage_and_reference(features_dinov2_index_reference, lakefs)
    dino_table = load_table(dino_stage.repo, dino_reference.commit, table_name)

    num_rows = dino_table.count_rows()
    _LOGGER.info("Index has %s rows.", num_rows)

    embeddings_table = dino_table.search().select([embedding_column, IDENTIFIERS, START_NS]).to_arrow()
    return embeddings_table


def _build_slice_timestamp_map(
    identifiers_and_timestamps: list[dict[str, Any]], indices: set[int]
) -> dict[str, list[int]]:
    """Build a map from slice_id to list of timestamps for the given indices.

    Args:
        identifiers_and_timestamps: A list of dictionaries containing identifiers and timestamps.
        indices: A set of indices to process.

    Returns:
        A dictionary mapping slice_id to a list of timestamps.
    """
    result_map: dict[str, list[int]] = {}
    for index in indices:
        row = identifiers_and_timestamps[index]
        slice_id = row[IDENTIFIERS][SLICE_ID]
        if slice_id not in result_map:
            result_map[slice_id] = []
        result_map[slice_id].append(row[START_NS])
    return result_map

def get_include_exclude_maps(
    embeddings_table: pa.Table, include: set[int], exclude: set[int]
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Get include/exclude maps for the embeddings table.


    Args:
        embeddings_table: A PyArrow Table containing the embeddings.
        include: A set of indices to include.
        exclude: A set of indices to exclude.


    Returns:
        A dictionary mapping row indices to inclusion status.
    """
    identifiers_and_timestamps = embeddings_table.select([IDENTIFIERS, START_NS]).to_pylist()
    include_map = _build_slice_timestamp_map(identifiers_and_timestamps, include)
    exclude_map = _build_slice_timestamp_map(identifiers_and_timestamps, exclude)
    return include_map, exclude_map

def get_embedding_array_from_table(
    table_with_embeddings: pa.Table, embedding_column: str = EMBEDDING
) -> npt.NDArray[np.float32]:
    """Convert a column of embeddings to a numpy array.


    This function assumes that the embeddings are stored in a fixed-size tensor format in the specified column.


    Args:
        table_with_embeddings: A PyArrow Table containing the embeddings.
        embedding_column: The name of the column containing the embeddings.


    Returns:
        A numpy array of shape (num_embeddings, embedding_dimension) containing the embeddings.
    """
    num_rows = table_with_embeddings.num_rows
    if num_rows == 0:
        # Return empty array with proper shape for empty tables
        return np.array([], dtype=np.float32).reshape(0, 0)
    combined = table_with_embeddings.column(embedding_column).combine_chunks()
    embeddings_np = combined.storage.values.to_numpy(zero_copy_only=True)
    embeddings_np = embeddings_np.reshape(num_rows, -1).astype(np.float32)
    return cast(npt.NDArray[np.float32], embeddings_np)

def get_cluster_assignments(
    points: npt.NDArray[np.float32], num_clusters: int, num_iterations: int = 30
) -> npt.NDArray[np.int64]:
    """Get cluster assignments for the given points using k-means clustering.


    Args:
        points: A numpy array of shape (num_points, num_dimensions) representing the data points.
        num_clusters: An integer representing the number of clusters.
        num_iterations: An integer representing the number of iterations to run the k-means algorithm.


    Returns:
        A numpy array of shape (num_points,) containing the cluster assignments for each point.
    """
    kmeans = faiss.Kmeans(points.shape[1], num_clusters, niter=num_iterations, verbose=True)
    kmeans.train(points)
    _, indices = kmeans.index.search(points, 1)
    return cast(npt.NDArray[np.int64], indices.flatten())

def deduplicate_cluster(
    indices: npt.NDArray[np.int64], 
    embeddings: npt.NDArray[np.float32],
    config: HumanLabelsPicoConfig,
    embeddings_table: Optional[pa.Table] = None,
    gold_dataset: Optional[ParquetDataset] = None,
) -> tuple[set[int], set[int]]:
    """Deduplicate embeddings within a cluster and rescue VRU frames in parallel.

    This function does the following:
        1. Computes pairwise distances and deduplicates based on similarity threshold
        2. If VRU preservation is enabled, checks excluded frames for VRU content
        3. Rescues any VRU frames from excluded back to included
        
    By doing VRU rescue per-cluster, we achieve parallelization via Ray instead of
    sequential processing in the main function.

    Args:
        indices: A numpy array of shape (num_embeddings_in_cluster,) containing the original indices of the embeddings.
        embeddings: A numpy array of shape (num_embeddings_in_cluster, embedding_dimension) containing the embeddings.
        config: Configuration object with deduplication parameters.
        embeddings_table: Optional embeddings table needed for VRU checking.
        gold_dataset: Optional gold dataset for loading frames with annotations.

    Returns:
        A tuple of (included_indices, excluded_indices) sets containing the original indices.
    """
    if len(embeddings) == 0:
        return set(), set()
    if len(embeddings) != len(indices):
        raise ValueError(f"Length of embeddings {len(embeddings)} must match length of indices {len(indices)}")

    # Step 1: Run deduplication
    num_points, dim = embeddings.shape
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    distances, _ = index.search(embeddings, min(num_points, config.nearest_k_points + 1))
    distances = np.sqrt(distances[:, 1:]).mean(axis=1)  # Remove self-distance

    included_indices = set()
    excluded_indices = set()
    
    threshold = config.embedding_dedupe_threshold
    while distances.min() < np.inf:
        min_index = distances.argmin()
        point_distances, point_indices = index.search(embeddings[min_index : min_index + 1], num_points)
        point_distances, point_indices = np.sqrt(point_distances.ravel()), point_indices.ravel()

        close_indices = point_indices[point_distances <= threshold]

        distances[min_index] = np.inf
        included_indices.add(indices[min_index])

        if not close_indices.size:
            continue
        distances[close_indices] = np.inf

        for i in close_indices:
            if i == min_index:
                continue
            excluded_indices.add(indices[i])
    
    # Step 2: Rescue VRU frames from excluded (per-cluster, runs in parallel via Ray)
    vru_preservation_enabled = (
        config.enable_vru_preservation and
        embeddings_table is not None and 
        gold_dataset is not None
    )
    
    if vru_preservation_enabled and excluded_indices:
        rescued_indices = rescue_preserved_classes_from_excluded(
            excluded_indices,
            embeddings_table,
            gold_dataset,
            config.vru_classes,
        )
        
        if rescued_indices:
            included_indices.update(rescued_indices)
            excluded_indices.difference_update(rescued_indices)
            _LOGGER.debug(
                f"Cluster VRU rescue: rescued {len(rescued_indices)} out of "
                f"{len(excluded_indices) + len(rescued_indices)} excluded frames"
            )

    return included_indices, excluded_indices


def deduplicate_embeddings(
    cluster_assignments: npt.NDArray[np.int64],
    embeddings: npt.NDArray[np.float32],
    config: HumanLabelsPicoConfig,
    embeddings_table: Optional[pa.Table] = None,
    gold_dataset: Optional[ParquetDataset] = None,
) -> tuple[set[int], set[int]]:
    """Deduplicate embeddings based on a similarity threshold with parallel VRU rescue.

    Each cluster is processed in parallel via Ray. If VRU preservation is enabled,
    each cluster rescues its own VRU frames from the excluded set, achieving
    parallelization instead of sequential processing.

    Args:
        cluster_assignments: A numpy array of shape (num_embeddings,) containing the cluster assignment for each
            embedding.
        embeddings: A numpy array of shape (num_embeddings, embedding_dimension) containing the embeddings.
        config: Configuration object with deduplication parameters.
        embeddings_table: Optional embeddings table needed for VRU checking.
        gold_dataset: Optional gold dataset for loading frames with annotations.

    Returns:
        A tuple of (included_indices, excluded_indices) sets.
    """
    deduplication_tasks = []
    for cluster_id in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        print("Cluster size:", len(cluster_indices))

        if cluster_indices.size > 0:
            deduplication_tasks.append(
                (cluster_indices, cluster_embeddings, config, embeddings_table, gold_dataset)
            )
        else:
            _LOGGER.info("Cluster %d is empty.", cluster_id)

    # Helper function since map_remote_to_args requires a single argument function.
    def deduplicate_embeddings_in_cluster(
        input_data: tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], HumanLabelsPicoConfig,
                         Optional[pa.Table], Optional[ParquetDataset]]
    ) -> tuple[set[int], set[int]]:
        return deduplicate_cluster(*input_data)

    included, excluded = set(), set()
    for result in map_remote_to_args(
        deduplicate_embeddings_in_cluster,
        deduplication_tasks,
        disable_ray=config.disable_ray,
        dynamically_adjust_memory=True,
    ):
        if isinstance(result, Exception):
            raise result
        incremental_include, incremental_exclude = result
        included.update(incremental_include)
        excluded.update(incremental_exclude)

    return included, excluded

def dedupe_and_get_include_and_exclude_maps(
    config: HumanLabelsPicoConfig, lakefs: LakeFS
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Run deduplication based on the provided configuration and LakeFS client.

    This function will get the embeddings table from the reference passed with the config, cluster the embeddings,
    and then deduplicate them based on the specified threshold. If VRU preservation is enabled, frames containing
    VRU objects will be preserved during deduplication.

    Args:
        config: The configuration object containing parameters for deduplication.
        lakefs: The LakeFS client used to access data.

    Returns:
        A tuple containing two dictionaries:
        - include_map: A dictionary mapping string keys to lists of integers representing included embeddings.
        - exclude_map: A dictionary mapping string keys to lists of integers representing excluded embeddings.
    """
    _LOGGER.info("Loading embeddings.")
    embedding_table = get_embeddings_table(config.features_dinov2_index_reference, lakefs)
    embeddings_array = get_embedding_array_from_table(embedding_table)
    
    # Load gold dataset if VRU preservation is enabled (needed for parallel processing)
    gold_dataset = None
    if config.enable_vru_preservation:
        _LOGGER.info("VRU preservation enabled. Loading gold dataset for parallel VRU rescue...")
        
        annotations_ref = config.vru_annotations_reference or config.human_labels_gold_reference
        if not annotations_ref:
            raise ValueError("VRU preservation enabled but no annotations reference provided")
        
        gold_stage, gold_reference = get_stage_and_reference(annotations_ref, lakefs)
        from kits.scalex.dataset.constants import ROW_ID
        gold_dataset = ParquetDataset(gold_stage, gold_reference.commit, row_id_column=ROW_ID)
        
        if not config.vru_classes:
            raise ValueError("VRU preservation enabled but vru_classes is empty")
        
        _LOGGER.info(f"VRU classes to preserve: {config.vru_classes}")
    
    if config.testing_limit_clustering_points:
        embeddings_array = embeddings_array[: config.testing_limit_clustering_points]

    _LOGGER.info("Loaded %d embeddings. Assigning to %d clusters.", len(embeddings_array), config.num_kmeans_clusters)
    cluster_assignments = get_cluster_assignments(
        embeddings_array, num_clusters=config.num_kmeans_clusters, num_iterations=config.kmeans_iterations
    )
    
    # Run deduplication with parallel VRU rescue per cluster
    if config.enable_vru_preservation:
        _LOGGER.info(
            "Assigned clusters. Deduplicating embeddings with parallel VRU rescue per cluster "
            "(each cluster rescues its own VRU frames via Ray)."
        )
    else:
        _LOGGER.info("Assigned clusters. Deduplicating embeddings.")
    
    include, exclude = deduplicate_embeddings(
        cluster_assignments, 
        embeddings_array,
        config,
        embeddings_table=embedding_table if config.enable_vru_preservation else None,
        gold_dataset=gold_dataset,
    )
    _LOGGER.info("<<<Deduplicated embeddings. Included %d, excluded %d.>>>", len(include), len(exclude))
    
    include_map, exclude_map = get_include_exclude_maps(embedding_table, include, exclude)
    _LOGGER.info("Generated include and exclude maps.")
    return include_map, exclude_map
