"""Unit tests for ALFA curate data selection strategy."""

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import fsspec
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given
from hypothesis import strategies as st

from autonomy.perception.datasets.active_learning.alfa_curate.config import AlfaCurateConfig
from autonomy.perception.datasets.active_learning.alfa_curate.data_types import PromptConfig, ScenarioConfig
from autonomy.perception.datasets.active_learning.alfa_curate.generate_alfa_curate import select_slices_for_scenario
from autonomy.perception.datasets.active_learning.alfa_curate.utils import (
    SearchResult,
    _combine_results,
    _maybe_run_sql_query,
    _maybe_run_text_query,
    deduplicate_by_base_slice,
    get_slice_ids_to_exclude,
    l2_distance_to_similarity,
    load_table,
    run_text_query,
    similarity_to_l2_distance,
)
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.config import VLMJudgeConfig
from kits.scalex.dataset.manifest import FileReference, GenericRef, Manifest
from kits.scalex.dataset.stage import Stage
from platforms.lakefs.client import LakefsRef


def test_search_result() -> None:
    """Test SearchResult dataclass."""
    with pytest.raises(ValueError):
        SearchResult(row_id="invalid_row_id", score=l2_distance_to_similarity(0.5))

    with pytest.raises(ValueError):
        SearchResult(row_id="invalid_row_id", score=l2_distance_to_similarity(0.5))

    result = SearchResult(row_id="row1_segment_1000000000_1000000010_camera1", score=l2_distance_to_similarity(0.5))

    assert result.slice_id == "row1"
    assert result.segment_start_ns == 1000000000
    assert result.segment_end_ns == 1000000010
    assert result.camera_name == "camera1"
    assert result.score == pytest.approx(1 - 0.5 / 2)


def data() -> list[dict[str, Any]]:
    return [
        {
            "row_id": "row1_segment_0_10_camera1",
            "sensor_name": "camera1",
            "embedding": np.array(
                [0.1, 0.1, 0.1],
            ),
        },
        {
            "row_id": "row1_segment_100_1000_camera1",
            "sensor_name": "camera1",
            "embedding": np.array(
                [0.3, -0.3, 0.3],
            ),
        },
        {
            "row_id": "row2_segment_0_10_camera2",
            "sensor_name": "camera2",
            "embedding": np.array(
                [0.2, 0.1, 0.9],
            ),
        },
    ]


def write_database(db_path: str) -> lancedb.table.LanceTable:
    db = lancedb.connect(str(db_path))
    return db.create_table("data", pa.Table.from_pylist(data()))


def test_load_table(tmp_path: Path) -> None:
    """Test load_table function."""

    # Create a temporary LanceDB with a test table
    db_path = os.fspath(tmp_path / "test_db")
    write_database(db_path)

    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.utils.read_database_path") as mock_read_path,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.utils.LakeFS"),
    ):
        mock_read_path.return_value = db_path
        table = load_table("repo", "branch", "data")
        assert table.count_rows() == len(data())


def test_maybe_run_text_query(tmp_path: Path) -> None:
    """Test run_text_query function."""

    # Create a temporary LanceDB with a test table
    db_path = os.fspath(tmp_path / "test_db")
    table = write_database(db_path)

    assert _maybe_run_text_query(table, "", "Cosmos-Embed1-448p", upper_bound_distance=1.0) is None

    with patch(
        "autonomy.perception.datasets.active_learning.alfa_curate.utils.text_to_embedding"
    ) as mock_text_to_embedding:
        mock_text_to_embedding.return_value = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        results = _maybe_run_text_query(table, "test query", "Cosmos-Embed1-448p", upper_bound_distance=0.1)
        assert results is not None
        assert len(results) == 1
        assert [r.row_id for r in results] == ["row1_segment_0_10_camera1"]

        results = _maybe_run_text_query(table, "test query", "Cosmos-Embed1-448p", upper_bound_distance=1.0)
        assert results is not None
        assert len(results) == 3
        assert [r.row_id for r in results] == [
            "row1_segment_0_10_camera1",
            "row1_segment_100_1000_camera1",
            "row2_segment_0_10_camera2",
        ]

        results = run_text_query(table, "test query", "Cosmos-Embed1-448p", upper_bound_distance=1.0, limit=2)
        assert len(results) == 2
        assert [r.row_id for r in results] == ["row1_segment_0_10_camera1", "row1_segment_100_1000_camera1"]

        results = run_text_query(
            table, "test query", "Cosmos-Embed1-448p", upper_bound_distance=1.0, limit=2, camera_names=["camera2"]
        )
        assert len(results) == 1
        assert [r.row_id for r in results] == ["row2_segment_0_10_camera2"]


def test_deduplicate_by_slice_id() -> None:
    """Test deduplicate_by_base_slice function."""
    results = [
        SearchResult(row_id="row1_segment_0_10_camera1", score=0.75),
        SearchResult(row_id="row1_segment_100_1000_camera1", score=0.85),
        SearchResult(row_id="row2_segment_0_10_camera2", score=0.8),
        SearchResult(row_id="row2_segment_20_30_camera2", score=0.9),
    ]

    deduplicated = list(deduplicate_by_base_slice(results))
    assert len(deduplicated) == 2
    assert {r.row_id for r in deduplicated} == {"row1_segment_100_1000_camera1", "row2_segment_20_30_camera2"}


@given(st.floats(min_value=0.0, max_value=2.0))
def test_l2_distance_functions(distance: float) -> None:
    """Test l2_distance_to_similarity and similarity_to_l2_distance functions."""
    sim = l2_distance_to_similarity(distance)
    d_converted = similarity_to_l2_distance(sim)
    assert d_converted == pytest.approx(distance)


def test_get_slice_ids_to_exclude(tmp_path: Path) -> None:
    """Test get_slice_ids_to_exclude function."""
    file_path = os.fspath(tmp_path / "test_manifest.parquet")
    table = pa.Table.from_pylist(
        [
            {"slice_id": "slice1"},
            {"slice_id": "slice2"},
            {"slice_id": "slice3"},
        ]
    )
    pq.write_table(table, file_path)
    manifest = Manifest(
        Stage("foo", "bar"),
        GenericRef(LakefsRef("repo", "branch", "path")),
        [FileReference("s3://a-b-c/branch/data/foo.parquet", "checksum", file_path)],
    )

    with patch(
        "autonomy.perception.datasets.active_learning.alfa_curate.utils.get_manifest_from_stage_str"
    ) as mock_get:
        mock_get.return_value = manifest
        slice_ids = get_slice_ids_to_exclude("foo", MagicMock(), fsspec.filesystem("file"))
        assert slice_ids == {"slice1", "slice2", "slice3"}


def test_maybe_run_sql_query() -> None:
    """Test _maybe_run_sql_query function with None/empty SQL query."""
    # Test with None/empty SQL query
    slice_ids, scores = _maybe_run_sql_query(None)
    assert slice_ids is None
    assert scores is None

    slice_ids, scores = _maybe_run_sql_query("")
    assert slice_ids is None
    assert scores is None


@pytest.mark.parametrize(
    "query,df_data,expected_slice_ids,expected_scores",
    [
        # Only slice_id column
        (
            "query1",
            {"slice_id": ["slice1", "slice2", "slice3"]},
            ["slice1", "slice2", "slice3"],
            None,
        ),
        # Query is cached
        (
            "query1",
            {"slice_id": ["slice1"]},
            ["slice1", "slice2", "slice3"],
            None,
        ),
        # Both slice_id and slice_score columns
        (
            "query2",
            {"slice_id": ["slice1", "slice2"], "slice_score": [0.8, 0.9]},
            ["slice1", "slice2"],
            [0.8, 0.9],
        ),
    ],
)
def test_maybe_run_sql_query_with_valid_data(
    query: str,
    df_data: dict[str, list[str] | list[float]],
    expected_slice_ids: list[str],
    expected_scores: list[float] | None,
) -> None:
    """Test _maybe_run_sql_query function with valid DataFrames."""
    with patch("autonomy.perception.datasets.active_learning.alfa_curate.utils.load_bq_client") as mock_load_bq:
        mock_bq_client = MagicMock()
        mock_load_bq.return_value = mock_bq_client

        df = pd.DataFrame(df_data)
        mock_bq_client.query.return_value.to_dataframe.return_value = df

        slice_ids, scores = _maybe_run_sql_query(query)

        assert slice_ids == expected_slice_ids
        assert scores == expected_scores


def test_maybe_run_sql_query_missing_slice_id() -> None:
    """Test _maybe_run_sql_query function with missing slice_id column."""
    with patch("autonomy.perception.datasets.active_learning.alfa_curate.utils.load_bq_client") as mock_load_bq:
        mock_bq_client = MagicMock()
        mock_load_bq.return_value = mock_bq_client

        df = pd.DataFrame({"other_column": ["value1", "value2"]})
        mock_bq_client.query.return_value.to_dataframe.return_value = df

        with pytest.raises(ValueError, match="SQL query must return a column named 'slice_id'"):
            _maybe_run_sql_query("SELECT other_column FROM table")


@pytest.mark.parametrize(
    "prompt_results,sql_slice_ids,maybe_sql_scores,multiply_prompt_and_sql_scores,expected_slice_ids,expected_scores",
    [
        # Case 1: Only prompt results (no SQL filter)
        (
            [
                SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
                SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
            ],
            None,
            None,
            False,
            ["slice2", "slice1"],
            [0.9, 0.8],
        ),
        # Case 2: Only SQL results (no prompt)
        (None, ["slice1", "slice2", "slice3"], None, False, ["slice1", "slice2", "slice3"], [1.0, 1.0, 1.0]),
        # Case 3: SQL results with scores (no prompt)
        (None, ["slice1", "slice2"], [0.7, 0.85], False, ["slice2", "slice1"], [0.85, 0.7]),
        # Case 4: Both prompt and SQL results (intersection, no multiplication)
        (
            [
                SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
                SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
                SearchResult(row_id="slice3_segment_0_10_cam1", score=0.7),
            ],
            ["slice1", "slice3"],
            None,
            False,
            ["slice1", "slice3"],
            [0.8, 0.7],
        ),
        # Case 5: Both prompt and SQL results with scores (intersection, WITH multiplication)
        (
            [
                SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
                SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
                SearchResult(row_id="slice3_segment_0_10_cam1", score=0.7),
            ],
            ["slice1", "slice3"],
            [2.0, 3.0],
            True,
            ["slice3", "slice1"],  # slice3 has higher multiplied score: 0.7 * 3.0 = 2.1 > 0.8 * 2.0 = 1.6
            [2.1, 1.6],
        ),
        # Case 6: Both prompt and SQL results with scores (intersection, multiplication disabled)
        (
            [
                SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
                SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
                SearchResult(row_id="slice3_segment_0_10_cam1", score=0.7),
            ],
            ["slice1", "slice3"],
            [2.0, 3.0],
            False,
            ["slice1", "slice3"],  # Original prompt scores retained
            [0.8, 0.7],
        ),
        # Case 7: No results from either source
        (None, None, None, False, [], []),
        # Case 8: Empty prompt results, no SQL
        ([], None, None, False, [], []),
        # Case 9: Prompt results with no matching SQL slice IDs
        (
            [
                SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
                SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
            ],
            ["slice3", "slice4"],
            None,
            False,
            [],
            [],
        ),
    ],
)
def test_combine_results(
    prompt_results: list[SearchResult] | None,
    sql_slice_ids: list[str] | None,
    maybe_sql_scores: list[float] | None,
    multiply_prompt_and_sql_scores: bool,
    expected_slice_ids: list[str],
    expected_scores: list[float],
) -> None:
    """Test _combine_results function with various input combinations."""
    results = _combine_results(prompt_results, sql_slice_ids, maybe_sql_scores, multiply_prompt_and_sql_scores)

    assert len(results) == len(expected_slice_ids)
    assert [r.slice_id for r in results] == expected_slice_ids
    for r, expected_score in zip(results, expected_scores):
        assert r.score == pytest.approx(expected_score)


def test_run_text_query_with_mocked_helpers() -> None:
    """Test run_text_query with mocked _maybe_run_text_query and _maybe_run_sql_query."""
    mock_table = MagicMock()

    text_query_results = [
        SearchResult(row_id="slice1_segment_0_10_cam1", score=0.8),
        SearchResult(row_id="slice2_segment_0_10_cam1", score=0.9),
        SearchResult(row_id="slice3_segment_0_10_cam1", score=0.7),
    ]
    sql_slice_ids = ["slice1", "slice3"]
    sql_scores = None

    with (
        patch(
            "autonomy.perception.datasets.active_learning.alfa_curate.utils._maybe_run_text_query"
        ) as mock_text_query,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.utils._maybe_run_sql_query") as mock_sql_query,
    ):
        mock_text_query.return_value = text_query_results
        mock_sql_query.return_value = (sql_slice_ids, sql_scores)

        results = run_text_query(
            mock_table,
            "test query",
            "Cosmos-Embed1-448p",
            upper_bound_distance=0.5,
            camera_names=["camera1"],
            limit=100,
            sql_query="SELECT * FROM table",
        )

        # Verify the helper functions were called correctly
        mock_text_query.assert_called_once_with(mock_table, "test query", "Cosmos-Embed1-448p", 0.5, ["camera1"], 100)
        mock_sql_query.assert_called_once_with("SELECT * FROM table")

        # Verify results - should only return intersection of text and SQL results
        assert len(results) == 2
        assert [r.slice_id for r in results] == ["slice1", "slice3"]



def test_select_slices_for_scenario() -> None:
    """Test select_slices_for_scenario with VLM judge enabled."""
    # Setup config with VLM judge enabled
    vlm_config = VLMJudgeConfig(
        enable_vlm_judge=True,
        max_candidates_for_vlm=50,
        vlm_confidence_threshold=0.6,
    )
    config = AlfaCurateConfig(
        log_slices_silver_reference="sensing--log-slices--silver/main",
        repo="sensing--features--cosmos-index",
        vlm_judge=vlm_config,
    )
    
    scenario = ScenarioConfig(
        name="test_scenario",
        prompts=[
            PromptConfig(
                prompt="vehicle at intersection",
                camera_names=["camera_front"],
                similarity_threshold=0.7,
            ),
        ],
    )
    
    # Mock initial search results
    initial_results = [
        SearchResult(row_id="slice1_segment_0_1000000000_camera_front", sensor_name="camera_front", distance=0.3),
        SearchResult(row_id="slice2_segment_0_1000000000_camera_front", sensor_name="camera_front", distance=0.4),
    ]
    
    # Mock VLM-filtered results
    filtered_results = [initial_results[0]]
    
    with (
        patch("autonomy.perception.datasets.active_learning.alfa_curate.generate_alfa_curate.load_table") as mock_load_table,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.generate_alfa_curate.run_text_query") as mock_run_query,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.generate_alfa_curate.deduplicate_by_base_slice") as mock_dedup,
        patch("autonomy.perception.datasets.active_learning.alfa_curate.generate_alfa_curate.apply_vlm_judge") as mock_vlm_judge,
    ):
        mock_table = MagicMock()
        mock_load_table.return_value = mock_table
        mock_run_query.return_value = initial_results
        mock_dedup.return_value = initial_results
        mock_vlm_judge.return_value = filtered_results
        
        # Call the function
        results = select_slices_for_scenario(config, scenario)
        
        # Verify VLM judge was applied
        assert len(results) == 1
        assert results == filtered_results
        mock_vlm_judge.assert_called_once_with(
            results=initial_results,
            scenario=scenario,
            config=config,
        )



