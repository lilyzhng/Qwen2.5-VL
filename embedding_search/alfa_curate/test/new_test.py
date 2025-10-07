from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from autonomy.perception.datasets.active_learning.blis_miner.config import StrategyConfig
from autonomy.perception.datasets.active_learning.blis_miner.slice_scorer import SliceScorer, get_curvature_over_thresh
from autonomy.perception.datasets.log_slices.data_model import LogSlice
from autonomy.perception.datasets.log_slices.logapps_metadata.tests.logapps_metadata_test_support import (
    sample_logapps_metadata,
)
from autonomy.perception.datasets.log_slices.tests.log_slices_test_support import log_slices_row
from kits.ml.datasets.geometry import Point3D, Pose, Quaternion


@pytest.fixture(name="config")
def get_config() -> StrategyConfig:
    """Get a default config for testing."""
    return StrategyConfig()


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert planar yaw (rad) about +Z into Quaternion(w, x, y, z)."""
    half = 0.5 * yaw
    return Quaternion(float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def _slice_to_pose_sequence(log_slice: LogSlice) -> dict[int, dict[str, float]]:
    pose_sequence = {}

    for frame in log_slice.frames:
        if frame.cuboids_scene_se3_vehicle is not None and frame.cuboids_tov_ns is not None:
            # unpack to make api consistent
            pose = {
                "position_x_m": frame.cuboids_scene_se3_vehicle.translation.x_m,
                "position_y_m": frame.cuboids_scene_se3_vehicle.translation.y_m,
                "position_z_m": frame.cuboids_scene_se3_vehicle.translation.z_m,
            }
            pose_sequence[int(frame.cuboids_tov_ns)] = pose
    return pose_sequence


def _generate_valid_log_slice(frames_count: int, curvature: Optional[tuple[float, float]] = None) -> LogSlice:
    """Generate a LogSlice for testing.

    Args:
        frames_count: Number of frames.
        curvature: Optional (k_start, k_end) planar curvature values (1/m). When provided,
            curvature is linearly interpolated over frames and integrated to produce
            a 2D trajectory. If None, a straight line along +X is produced (legacy behavior).

    Returns:
        LogSlice with the requested number of frames and poses, linearly interpolated curvature if provided.

    Notes:
        Uses unit arc-length step (ds = 1.0) per frame. Heading is integrated as:
            theta_{i} = theta_{i-1} + k_{i-1} * ds
        Position increments:
            x_{i} = x_{i-1} + cos(theta_{i}) * ds
            y_{i} = y_{i-1} + sin(theta_{i}) * ds
    """
    log_slice = log_slices_row(frames_count)
    log_slice.logapps_metadata = sample_logapps_metadata(platform_config_version="M-1")

    if frames_count == 0:
        return log_slice

    if curvature is None or frames_count < 2:
        # Straight line fallback
        for index, frame in enumerate(log_slice.frames):
            frame.cuboids_scene_se3_vehicle = Pose(
                rotation=Quaternion(1.0, 0.0, 0.0, 0.0),
                translation=Point3D(float(index), 0.0, 0.0),
            )
            frame.cuboids_tov_ns = np.int64(index * 1e8)
        return log_slice

    k_start, k_end = curvature
    k_vals = np.linspace(k_start, k_end, frames_count)  # curvature at each frame
    ds = 1.0

    x = 0.0
    y = 0.0
    theta = 0.0

    # Frame 0
    frame0 = log_slice.frames[0]
    frame0.cuboids_scene_se3_vehicle = Pose(
        rotation=Quaternion(1.0, 0.0, 0.0, 0.0),
        translation=Point3D(x, y, 0.0),
    )
    frame0.cuboids_tov_ns = np.int64(0)

    for i in range(1, frames_count):
        # Integrate heading using previous curvature (piecewise constant per step), just an euler spiral with euler integration
        theta += k_vals[i - 1] * ds
        x += np.cos(theta) * ds
        y += np.sin(theta) * ds

        frame = log_slice.frames[i]
        frame.cuboids_scene_se3_vehicle = Pose(
            rotation=_yaw_to_quaternion(theta),
            translation=Point3D(x, y, 0.0),
        )
        frame.cuboids_tov_ns = np.int64(i * 1e8)

    return log_slice


def test_curvature_helper_straight(config: StrategyConfig) -> None:
    """Test the curvature threshold helper function."""

    valid_log_slice = _generate_valid_log_slice(frames_count=200)

    pose_sequence = _slice_to_pose_sequence(valid_log_slice)

    assert sum(get_curvature_over_thresh(pose_sequence, config)) == 0


def test_curvature_helper_clothoid(config: StrategyConfig) -> None:
    """Test the curvature threshold helper function."""

    valid_log_slice = _generate_valid_log_slice(frames_count=100, curvature=(0.0, 0.1))

    pose_sequence = _slice_to_pose_sequence(valid_log_slice)
    # we expect at least 25 frames to be above the 0.05 rad/m threshold, but NOT HALF as the simplifier is trying to _reduce_ the curvature
    assert sum(get_curvature_over_thresh(pose_sequence, config)) > 25


def test_curvature_helper_sterk(config: StrategyConfig) -> None:
    """Test the curvature threshold helper function when the vehicle is stationary."""

    valid_log_slice = _generate_valid_log_slice(frames_count=100, curvature=(0.0, 0.1))

    pose_sequence = {}

    for frame in valid_log_slice.frames:
        if frame.cuboids_scene_se3_vehicle is not None and frame.cuboids_tov_ns is not None:
            # unpack to make api consistent
            pose = {
                "position_x_m": 0.0,
                "position_y_m": 0.0,
                "position_z_m": 0.0,
            }
            pose_sequence[int(frame.cuboids_tov_ns)] = pose
    # min displacement not met, so no curvature above threshold by definition as stationary points are weird.
    assert sum(get_curvature_over_thresh(pose_sequence, config)) == 0


def test_process_slice_valid() -> None:
    """Tests that the process_slice method returns a score if provided valid data."""
    config = StrategyConfig()
    scorer = SliceScorer(config)

    mock_data_model_reader = MagicMock()

    # Confirm that None is returned if the data is invalid
    invalid_log_slice = log_slices_row(frames_count=0)
    mock_data_model_reader.read_data_models.return_value = [invalid_log_slice]
    assert scorer.process_slice(mock_data_model_reader) is None

    # Confirm that a valid score is returned if the data is valid
    valid_log_slice = _generate_valid_log_slice(frames_count=200, curvature=(0.0, 0.1))
    mock_data_model_reader.read_data_models.return_value = [valid_log_slice]
    assert scorer.process_slice(mock_data_model_reader) is not None
