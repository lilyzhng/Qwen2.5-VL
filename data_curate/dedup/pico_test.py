from typing import Final
from unittest.mock import MagicMock
import sys
import numpy as np
import pyarrow as pa
import pytest

# Define constants that would normally come from kits.scalex.dataset.constants
EMBEDDING = "embedding"
FRAMES_KEY = "frames"
IDENTIFIERS = "identifiers"
ROW_ID = "row_id"
SLICE_ID = "slice_id"
START_NS = "start_ns"
TIMESTAMP_NS = "timestamp_ns"

# Create a mock module for constants with the actual values
constants_mock = MagicMock()
constants_mock.EMBEDDING = EMBEDDING
constants_mock.FRAMES_KEY = FRAMES_KEY
constants_mock.IDENTIFIERS = IDENTIFIERS
constants_mock.ROW_ID = ROW_ID
constants_mock.SLICE_ID = SLICE_ID
constants_mock.START_NS = START_NS
constants_mock.TIMESTAMP_NS = TIMESTAMP_NS

# Mock get_chunks function for index_writer
def mock_get_chunks(array, chunk_size):
    """Mock implementation of get_chunks."""
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size]

index_writer_mock = MagicMock()
index_writer_mock.get_chunks = mock_get_chunks

# Mock map_remote_to_args function
def mock_map_remote_to_args(func, args_list, disable_ray=False, dynamically_adjust_memory=False):
    """Mock implementation that just applies function to each argument."""
    for args in args_list:
        try:
            yield func(args)
        except Exception as e:
            yield e

ray_map_mock = MagicMock()
ray_map_mock.map_remote_to_args = mock_map_remote_to_args

# Mock the external dependencies that don't exist in this codebase
sys.modules['autonomy'] = MagicMock()
sys.modules['autonomy.perception'] = MagicMock()
sys.modules['autonomy.perception.datasets'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels.pico'] = MagicMock()
sys.modules['autonomy.perception.datasets.human_labels.pico.config'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.alfa_curate'] = MagicMock()
sys.modules['autonomy.perception.datasets.active_learning.alfa_curate.utils'] = MagicMock()
sys.modules['kits'] = MagicMock()
sys.modules['kits.scalex'] = MagicMock()
sys.modules['kits.scalex.dataset'] = MagicMock()
sys.modules['kits.scalex.dataset.constants'] = constants_mock
sys.modules['kits.scalex.dataset.index'] = MagicMock()
sys.modules['kits.scalex.dataset.index.index_writer'] = index_writer_mock
sys.modules['kits.scalex.dataset.stage_str'] = MagicMock()
sys.modules['kits.scalex.pipeline'] = MagicMock()
sys.modules['kits.scalex.pipeline.ray'] = MagicMock()
sys.modules['kits.scalex.pipeline.ray.map'] = ray_map_mock
sys.modules['platforms'] = MagicMock()
sys.modules['platforms.lakefs'] = MagicMock()
sys.modules['platforms.lakefs.client'] = MagicMock()

from pico import (  # noqa: E402
    deduplicate_cluster,
    deduplicate_embeddings,
    get_embedding_array_from_table,
    get_include_exclude_maps,
    transform_table,
    _find_reference_pico_row,
    _compute_temporal_velocities,
    _compute_temporal_accelerations,
    _pico_row_mean_pooling,
    compute_embedding_change_rates,
    temporal_subsample_by_change_rate,
)

TSTAMP: Final = "timestamp_ns"


class HumanLabelsPicoConfig:
    def __init__(self, apply_temporal_subsampling=False, features_dinov2_index_reference=None, 
                 testing_limit_clustering_points=None, num_kmeans_clusters=10, 
                 kmeans_iterations=30, embedding_dedupe_threshold=0.5, disable_ray=True):
        self.apply_temporal_subsampling = apply_temporal_subsampling
        self.features_dinov2_index_reference = features_dinov2_index_reference
        self.testing_limit_clustering_points = testing_limit_clustering_points
        self.num_kmeans_clusters = num_kmeans_clusters
        self.kmeans_iterations = kmeans_iterations
        self.embedding_dedupe_threshold = embedding_dedupe_threshold
        self.disable_ray = disable_ray


def get_example_temporal_data():
    """Helper function to create test data for temporal functions."""
    timestamps = [0.0, 5e9, 10e9, 15e9, 20e9, 25e9]  # 0s, 5s, 10s, 15s, 20s, 25s in nanoseconds
    embeddings = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([6.0, 8.0], dtype=np.float32),
        np.array([9.0, 12.0], dtype=np.float32),
        np.array([12.0, 16.0], dtype=np.float32),
        np.array([15.0, 20.0], dtype=np.float32),
    ]
    return timestamps, embeddings


def get_example_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                ROW_ID: 0,
                FRAMES_KEY: [{TSTAMP: 0}, {TSTAMP: 10}, {TSTAMP: 20}, {TSTAMP: 30}, {TSTAMP: 40}, {TSTAMP: 50}],
            },
            {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 20}, {TSTAMP: 30}, {TSTAMP: 40}, {TSTAMP: 50}, {TSTAMP: 60}]},
            {ROW_ID: 2, FRAMES_KEY: []},
        ]
    )
    
def test_transform_table() -> None:
    table = get_example_table()
    config = HumanLabelsPicoConfig(apply_temporal_subsampling=False)
    lakefs = MagicMock()
    result = pa.concat_tables(
        list(
            transform_table(
                table,
                stride=1,
                frames_per_row=2,
                include_timestamps=pa.array([[], [], []]),
                exclude_timestamps=pa.array([[], [], []]),
                config=config,
                lakefs=lakefs,
            )
        )
    )
    assert result.to_pylist() == [
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 0}, {TSTAMP: 10}]},
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 20}, {TSTAMP: 30}]},
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 40}, {TSTAMP: 50}]},
        {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 20}, {TSTAMP: 30}]},
        {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 40}, {TSTAMP: 50}]},
        {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 60}]},
    ]
    result = pa.concat_tables(
        list(
            transform_table(
                table,
                stride=2,
                frames_per_row=2,
                include_timestamps=pa.array([[], [], []]),
                exclude_timestamps=pa.array([[], [], []]),
                config=config,
                lakefs=lakefs,
            )
        )
    )
    assert result.to_pylist() == [
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 0}, {TSTAMP: 10}]},
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 40}, {TSTAMP: 50}]},
        {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 20}, {TSTAMP: 30}]},
        {ROW_ID: 1, FRAMES_KEY: [{TSTAMP: 60}]},
    ]
def test_transform_table_invalid() -> None:
    table = get_example_table()
    config = HumanLabelsPicoConfig(apply_temporal_subsampling=False)
    lakefs = MagicMock()
    with pytest.raises(ValueError):
        list(
            transform_table(
                table,
                stride=1,
                frames_per_row=2,
                include_timestamps=pa.array([[], []]),
                exclude_timestamps=pa.array([[], [], []]),
                config=config,
                lakefs=lakefs,
            )
        )
    with pytest.raises(ValueError):
        list(
            transform_table(
                table,
                stride=1,
                frames_per_row=2,
                include_timestamps=pa.array([[], [], []]),
                exclude_timestamps=pa.array([[], []]),
                config=config,
                lakefs=lakefs,
            )
        )
def test_transform_table_include_exclude() -> None:
    table = get_example_table()
    config = HumanLabelsPicoConfig(apply_temporal_subsampling=False)
    lakefs = MagicMock()
    result = pa.concat_tables(
        list(
            transform_table(
                table,
                stride=1,
                frames_per_row=2,
                include_timestamps=pa.array([[5, 11, 51], [], []]),
                exclude_timestamps=pa.array([[12], [0], []]),
                config=config,
                lakefs=lakefs,
            )
        )
    )
    assert result.to_pylist() == [
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 0}, {TSTAMP: 10}]},
        {ROW_ID: 0, FRAMES_KEY: [{TSTAMP: 40}, {TSTAMP: 50}]},
    ]
    assert not list(
        transform_table(
            table,
            stride=1,
            frames_per_row=2,
            include_timestamps=pa.array([[], [], []]),
            exclude_timestamps=pa.array([[12], [0], []]),
            config=config,
            lakefs=lakefs,
        )
    )
def test_get_include_exclude_maps() -> None:
    """Test get_include_exclude_maps function."""
    embeddings_table = pa.Table.from_pylist(
        [
            {IDENTIFIERS: {SLICE_ID: "slice_1"}, START_NS: 1000},
            {IDENTIFIERS: {SLICE_ID: "slice_1"}, START_NS: 2000},
            {IDENTIFIERS: {SLICE_ID: "slice_2"}, START_NS: 3000},
            {IDENTIFIERS: {SLICE_ID: "slice_2"}, START_NS: 4000},
            {IDENTIFIERS: {SLICE_ID: "slice_3"}, START_NS: 5000},
        ]
    )
    include_map, exclude_map = get_include_exclude_maps(embeddings_table, {0, 2, 4}, {1, 3})
    assert include_map == {"slice_1": [1000], "slice_2": [3000], "slice_3": [5000]}
    assert exclude_map == {"slice_1": [2000], "slice_2": [4000]}
    include_map, exclude_map = get_include_exclude_maps(embeddings_table, set(), set())
    assert include_map == {}
    assert exclude_map == {}
    include_map, exclude_map = get_include_exclude_maps(embeddings_table, {0, 1, 2}, {3})
    assert include_map == {"slice_1": [1000, 2000], "slice_2": [3000]}
    assert exclude_map == {"slice_2": [4000]}
def test_get_embedding_array_from_table() -> None:
    """Test get_embedding_array_from_table function."""
    # Create a table with fixed-size list embeddings
    embeddings_list = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]
    table = pa.Table.from_pydict(
        {EMBEDDING: embeddings_list},
        schema=pa.schema([pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3]))]),
    )
    result = get_embedding_array_from_table(table, EMBEDDING)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    assert result.shape == (4, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_almost_equal(result, expected)

def test_deduplicate_cluster() -> None:
    """Test deduplicate_cluster function."""
    indices = np.array([10, 11, 22, 33, 1, 2, 3, 4, 5], dtype=np.int64)
    embeddings = np.array(
        [[0.0, 0.0], [3, 4], [5, 12], [1, 1], [0, 100], [0, 101], [0, 102], [0, 101.2], [0, 101.1]],
        dtype=np.float32,
    )
    included, excluded = deduplicate_cluster(indices, embeddings, 0.0, nearest_k_points=5)
    assert included == {10, 11, 22, 33, 1, 2, 3, 4, 5}
    assert not excluded
    included, excluded = deduplicate_cluster(indices, embeddings, 1.5, nearest_k_points=5)
    assert included == {11, 22, 33, 5}
    assert excluded == {10, 1, 2, 3, 4}

def test_deduplicate_embeddings() -> None:
    """Test deduplicate_embeddings function."""
    cluster_assignments = np.array([12, 12, 12, 12, 1, 1, 1, 1, 1], dtype=np.int64)
    embeddings = np.array(
        [[0.0, 0.0], [3, 4], [5, 12], [1, 1], [0, 100], [0, 101], [0, 102], [0, 101.2], [0, 101.1]],
        dtype=np.float32,
    )
    included, excluded = deduplicate_embeddings(
        cluster_assignments, embeddings, threshold=1.5, nearest_k_points=5, disable_ray=True
    )
    assert included == {1, 2, 3, 8}
    assert excluded == {0, 4, 5, 6, 7}


def test_find_reference_pico_row() -> None:
    """Test _find_reference_pico_row function."""
    timestamps, embeddings = get_example_temporal_data()
    temporal_window_size = 10.0  # 10 seconds
    
    # Normal case - find reference within temporal window (20s -> 10s)
    ref_idx = _find_reference_pico_row(4, timestamps, temporal_window_size, embeddings)
    assert ref_idx == 2, f"Expected to find reference at index 2 (10s ago from 20s), got {ref_idx}"
    
    # Test with different window size (15s window: 25s -> 10s)
    ref_idx = _find_reference_pico_row(5, timestamps, 15.0, embeddings)
    assert ref_idx == 2, f"With 15s window from 25s, expected reference at 10s (index 2), got {ref_idx}"
    
    # Test with None values
    timestamps_with_none = [0.0, 5e9, None, 15e9, 20e9]
    valid_data_with_none = embeddings[:5]
    ref_idx = _find_reference_pico_row(2, timestamps_with_none, temporal_window_size, valid_data_with_none)
    assert ref_idx is None, f"None timestamp at current index should return None, got {ref_idx}"
    
    # Test with gaps in valid data - should skip invalid entries
    valid_data_partial = [embeddings[0], None, embeddings[2], embeddings[3], embeddings[4], embeddings[5]]
    ref_idx = _find_reference_pico_row(4, timestamps, temporal_window_size, valid_data_partial)
    assert ref_idx == 2, f"Should skip None at index 1 and find valid reference at index 2, got {ref_idx}"


def test_compute_temporal_velocities() -> None:
    """Test _compute_temporal_velocities function."""

    pico_embeddings = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float32),  # Distance from [0,0] is 5.0
        np.array([6.0, 8.0], dtype=np.float32),  # Distance from [3,4] is 5.0
        np.array([9.0, 12.0], dtype=np.float32), # Distance from [6,8] is 5.0
    ]
    pico_timestamps = [0.0, 10e9, 20e9, 30e9]  # 0s, 10s, 20s, 30s
    temporal_window_size = 10.0
    
    velocities = _compute_temporal_velocities(pico_embeddings, pico_timestamps, temporal_window_size)
    
    assert abs(velocities[1] - 0.5) < 0.01, f"Expected velocity ~0.5 (dist 5.0 / 10s), got {velocities[1]}"
    assert abs(velocities[2] - 0.5) < 0.01, f"Expected constant velocity ~0.5, got {velocities[2]}"
    
    static_embeddings = [np.array([1.0, 2.0], dtype=np.float32)] * 4
    velocities = _compute_temporal_velocities(static_embeddings, pico_timestamps, temporal_window_size)
    for i, v in enumerate(velocities[1:], 1):
        assert v is not None, f"Expected velocity at index {i} for static embeddings, got None"
        assert abs(v) < 0.01, f"Static embeddings of no change should have near-zero velocity, got {v} at index {i}"
    
    pico_embeddings_with_none = [
        np.array([0.0, 0.0], dtype=np.float32),
        None,
        np.array([6.0, 8.0], dtype=np.float32),
        np.array([9.0, 12.0], dtype=np.float32),
    ]
    velocities = _compute_temporal_velocities(pico_embeddings_with_none, pico_timestamps, temporal_window_size)
    assert velocities[0] is None, "First element should have None velocity"
    assert velocities[1] is None, "None embedding should result in None velocity"


def test_compute_temporal_accelerations() -> None:
    """Test _compute_temporal_accelerations function."""
    pico_timestamps = [0.0, 10e9, 20e9, 30e9, 40e9]
    temporal_window_size = 10.0
    
    temporal_velocities_constant = [None, 0.5, 0.5, 0.5, 0.5]
    accelerations = _compute_temporal_accelerations(temporal_velocities_constant, pico_timestamps, temporal_window_size)
    assert abs(accelerations[2]) < 0.01, f"Constant velocity should have ~0 acceleration, got {accelerations[2]}"
    assert abs(accelerations[3]) < 0.01, f"Constant velocity should have ~0 acceleration, got {accelerations[3]}"
    
    temporal_velocities_increasing = [None, 0.5, 1.0, 1.5, 2.0]
    accelerations = _compute_temporal_accelerations(temporal_velocities_increasing, pico_timestamps, temporal_window_size)
    assert accelerations[0] is None, "First element should have None acceleration"
    assert accelerations[1] is None, "First velocity should have no reference"
    assert accelerations[2] > 0, f"Increasing velocity should have positive acceleration, got {accelerations[2]}"
    assert abs(accelerations[2] - 0.05) < 0.01, f"Expected acceleration ~0.05 ((1.0-0.5)/10), got {accelerations[2]}"
    assert abs(accelerations[3] - 0.05) < 0.01, f"Expected constant acceleration ~0.05, got {accelerations[3]}"
    
    # With None values
    temporal_velocities_with_none = [None, 0.5, None, 1.5, 2.0]
    accelerations = _compute_temporal_accelerations(temporal_velocities_with_none, pico_timestamps, temporal_window_size)
    assert accelerations[0] is None, "First element should have None acceleration"
    assert accelerations[1] is None, "First velocity should have no reference"
    assert accelerations[2] is None, "None velocity should result in None acceleration"


def test_pico_row_mean_pooling() -> None:
    """Test _pico_row_mean_pooling function."""
    timestamp_to_index = {1000: 0, 2000: 1, 3000: 2, 4000: 3, 5000: 4}
    embeddings_array = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
    ], dtype=np.float32)
    
    frame_timestamps = [1000, 2000, 3000, 4000]
    mean_embedding, mean_timestamp = _pico_row_mean_pooling(frame_timestamps, timestamp_to_index, embeddings_array)
    assert mean_embedding is not None, "Multiple frames should produce mean embedding"
    assert mean_timestamp is not None, "Multiple frames should produce mean timestamp"
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    np.testing.assert_array_almost_equal(mean_embedding, expected_mean, 
                                        err_msg=f"Expected mean embedding {expected_mean}, got {mean_embedding}")
    assert mean_timestamp == 2500.0, f"Expected mean timestamp 2500.0 (avg of 1000,2000,3000,4000), got {mean_timestamp}"
    
    # Single frame - should return that frame's embedding
    mean_embedding, mean_timestamp = _pico_row_mean_pooling([1000], timestamp_to_index, embeddings_array)
    assert mean_embedding is not None, "Single frame should produce embedding"
    np.testing.assert_array_almost_equal(mean_embedding, embeddings_array[0],
                                        err_msg="Single frame should return its own embedding")
    assert mean_timestamp == 1000.0, f"Single frame should return its own timestamp, got {mean_timestamp}"


def test_compute_embedding_change_rates() -> None:
    """Test compute_embedding_change_rates function."""

    pico_table = pa.Table.from_pylist([
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: 0}, {TIMESTAMP_NS: 1000}]},
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: 10000000000}, {TIMESTAMP_NS: 10000001000}]},
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: 20000000000}, {TIMESTAMP_NS: 20000001000}]},
    ])
    
    embeddings_list = [
        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # t=0, t=1000
        [2.0, 0.0, 0.0], [2.0, 0.0, 0.0],  # t=10s
        [3.0, 0.0, 0.0], [3.0, 0.0, 0.0],  # t=20s
    ]
    embedding_table = pa.Table.from_pydict(
        {
            EMBEDDING: embeddings_list,
            START_NS: [0, 1000, 10000000000, 10000001000, 20000000000, 20000001000],
        },
        schema=pa.schema([
            pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3])),
            pa.field(START_NS, pa.int64()),
        ]),
    )
    
    result_table = compute_embedding_change_rates(pico_table, embedding_table, temporal_window_size=15.0)
    
    assert "embedding_velocity" in result_table.column_names, "Result should have 'embedding_velocity' column"
    assert "embedding_acceleration" in result_table.column_names, "Result should have 'embedding_acceleration' column"
    assert result_table.num_rows == 3, f"Expected 3 rows in result, got {result_table.num_rows}"
    
    velocities = result_table.column("embedding_velocity").to_pylist()
    accelerations = result_table.column("embedding_acceleration").to_pylist()
    
    assert velocities[0] is None, "First pico row should have None velocity (no reference)"
    assert accelerations[0] is None, "First pico row should have None acceleration"


def test_compute_embedding_change_rates_multiple_slices() -> None:
    """Test compute_embedding_change_rates with multiple slices."""
    pico_table = pa.Table.from_pylist([
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: 0}]},
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: 10000000000}]},
        {SLICE_ID: "slice_2", FRAMES_KEY: [{TIMESTAMP_NS: 0}]},
        {SLICE_ID: "slice_2", FRAMES_KEY: [{TIMESTAMP_NS: 10000000000}]},
    ])
    
    embeddings_list = [
        [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],  # slice_1
        [5.0, 0.0, 0.0], [6.0, 0.0, 0.0],  # slice_2
    ]
    embedding_table = pa.Table.from_pydict(
        {
            EMBEDDING: embeddings_list,
            START_NS: [0, 10000000000, 0, 10000000000],
        },
        schema=pa.schema([
            pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3])),
            pa.field(START_NS, pa.int64()),
        ]),
    )
    
    result_table = compute_embedding_change_rates(pico_table, embedding_table, temporal_window_size=15.0)
    
    assert result_table.num_rows == 4, f"Expected 4 rows (2 per slice), got {result_table.num_rows}"
    velocities = result_table.column("embedding_velocity").to_pylist()
    
    # First row of each slice should have None velocity
    assert velocities[0] is None, "First row of slice_1 should have None velocity"
    assert velocities[2] is None, "First row of slice_2 should have None velocity (independent processing)"
    # Second row of each slice should have velocity
    assert velocities[1] is not None, "Second row of slice_1 should have velocity, got None"
    assert velocities[3] is not None, "Second row of slice_2 should have velocity, got None"



def test_temporal_subsample_by_change_rate() -> None:
    """Test temporal_subsample_by_change_rate function."""
    # Create pico table with gradual changes
    pico_table = pa.Table.from_pylist([
        {SLICE_ID: "slice_1", FRAMES_KEY: [{TIMESTAMP_NS: i * 1000000000}]} 
        for i in range(20)
    ])
    
    # Gradually changing embeddings: acceleration increases from 0.004 to 0.062
    # With threshold=0.02, early frames should be dropped
    embeddings_list = [[float(i), 0.0, 0.0] for i in range(20)]
    embedding_table = pa.Table.from_pydict(
        {
            EMBEDDING: embeddings_list,
            START_NS: [i * 1000000000 for i in range(20)],
        },
        schema=pa.schema([
            pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3])),
            pa.field(START_NS, pa.int64()),
        ]),
    )
    
    # Test acceleration-based filtering with threshold
    result_table = temporal_subsample_by_change_rate(
        pico_table,
        embedding_table,
        use_acceleration=True,
        temporal_window_size=15.0,
        diversity_threshold=0.02,
    )
    
    assert result_table.num_rows <= pico_table.num_rows, \
        f"Subsampled rows ({result_table.num_rows}) should be <= original ({pico_table.num_rows})"
    assert result_table.num_rows > 0, "Should keep at least some rows after subsampling"
    assert "embedding_velocity" in result_table.column_names, "Result should include velocity column"
    assert "embedding_acceleration" in result_table.column_names, "Result should include acceleration column"
    
    # Check that high-acceleration frames are kept
    accelerations = result_table.column("embedding_acceleration").to_pylist()
    non_none_accels = [a for a in accelerations if a is not None]
    assert all(a >= 0.02 for a in non_none_accels), \
        f"All kept frames (except None) should have acceleration >= 0.02, got {non_none_accels}"
    
    # Test velocity-based filtering
    result_table_velocity = temporal_subsample_by_change_rate(
        pico_table,
        embedding_table,
        use_acceleration=False,
        temporal_window_size=15.0,
        diversity_threshold=0.5,
    )
    
    assert result_table_velocity.num_rows <= pico_table.num_rows, \
        f"Velocity-based subsampled rows ({result_table_velocity.num_rows}) should be <= original ({pico_table.num_rows})"
    assert result_table_velocity.num_rows > 0, "Velocity-based filtering should keep at least some rows"
    
    # Check that high-velocity frames are kept
    velocities = result_table_velocity.column("embedding_velocity").to_pylist()
    non_none_vels = [v for v in velocities if v is not None]
    assert all(v >= 0.5 for v in non_none_vels), \
        f"All kept frames (except None) should have velocity >= 0.5"


def test_temporal_subsample_empty_table() -> None:
    """Test temporal_subsample_by_change_rate with empty table."""
    pico_table = pa.Table.from_pydict(
        {
            SLICE_ID: [],
            FRAMES_KEY: [],
        },
        schema=pa.schema([
            pa.field(SLICE_ID, pa.string()),
            pa.field(FRAMES_KEY, pa.list_(pa.struct([pa.field(TIMESTAMP_NS, pa.int64())]))),
        ]),
    )
    embedding_table = pa.Table.from_pydict(
        {
            EMBEDDING: [],
            START_NS: [],
        },
        schema=pa.schema([
            pa.field(EMBEDDING, pa.fixed_shape_tensor(pa.float32(), [3])),
            pa.field(START_NS, pa.int64()),
        ]),
    )
    
    result_table = temporal_subsample_by_change_rate(
        pico_table,
        embedding_table,
        use_acceleration=True,
    )
    
    assert result_table.num_rows == 0, f"Empty input should return empty table, got {result_table.num_rows} rows"
    assert "embedding_velocity" in result_table.column_names, "Empty table should still have velocity column"
    assert "embedding_acceleration" in result_table.column_names, "Empty table should still have acceleration column"
