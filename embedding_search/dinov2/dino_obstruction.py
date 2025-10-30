"""Implementation of DINOv2 based obstruction classification active learning selection algorithm."""

import logging
import os
from typing import Final, Optional

import numpy as np
import numpy.typing as npt
import ray

from autonomy.perception.datasets.active_learning.camera_obstruction_dinov2.config import StrategyConfig
from autonomy.perception.datasets.active_learning.framework.lakefs_support import HUMAN_LABELS_NAME
from autonomy.perception.datasets.active_learning.framework.slice_scorer_base import (
    DataModelReader,
    SimpleSliceScorerBase,
)
from autonomy.perception.datasets.features.dinov2.infer import EmbeddedImage
from autonomy.perception.datasets.features.dinov2_index.index import IndexWithMetadata
from autonomy.perception.datasets.unified.gold.data_model import Unified
from kits.ml.file_io.data_path import data_path_base
from platforms.lakefs.client import LakeFS

_LOGGER: Final = logging.getLogger(__name__)

_DINOV2_INDEX_LAKEFS_REPO: Final = "sensing--features--dinov2-index"

# A small epsilon to avoid division by zero
EPSILON: Final = 1e-6


@ray.remote
class RemoteIndex:
    """Ray actor that caches the DINOv2 FAISS index from LakeFS and answers nearest‑neighbour queries."""

    def __init__(self) -> None:
        """Initialize the object."""
        lakefs = LakeFS()

        _LOGGER.info("Downloading index files from lakefs. This may take several minutes, please wait...")
        self.index_files_cache = data_path_base() / "dinov2_index_cache"
        self.index_files_cache.mkdir(exist_ok=True, parents=True)
        for index_file in lakefs.list_objects(_DINOV2_INDEX_LAKEFS_REPO, "main", prefix="data"):
            _, _, filename = index_file.path.partition("/data/")

            filepath = self.index_files_cache / filename
            if filepath.exists():
                _LOGGER.info("File '%s' already cached.", filename)
                continue

            resp = lakefs._lakefs.objects.get_object(
                repository=_DINOV2_INDEX_LAKEFS_REPO,
                ref="main",
                path="data/" + filename,
            )
            content = resp.read()
            with open(os.fspath(filepath), "wb") as f:
                f.write(content)

        _LOGGER.info("Done downloading index files.  Instantiating index . . .")

        self.index = IndexWithMetadata[EmbeddedImage].from_saved(EmbeddedImage, os.fspath(self.index_files_cache))
        _LOGGER.info("Index Instantiated")

    def query(
        self, features: npt.NDArray[np.float32], num_neighbors: int, nprobe: int = 8
    ) -> tuple[npt.NDArray[np.float32], list[list[EmbeddedImage | None]]]:
        """Query the index."""
        return self.index.query(features=features, num_neighbors=num_neighbors, nprobe=nprobe)


class SliceScorer(SimpleSliceScorerBase[StrategyConfig]):
    """Implementation of the slice scorer."""

    _shared_idx_actor = None

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize the Slice Scorer."""
        super().__init__(config)
        if SliceScorer._shared_idx_actor is None:
            SliceScorer._shared_idx_actor = RemoteIndex.options(  # type: ignore[attr-defined]
                max_concurrency=self.config.index_actor_concurrency
            ).remote()
        self.remote_index = SliceScorer._shared_idx_actor
        self._num_voting_neighbors = self.config.num_voting_neighbors
        self._min_labeled = min(self.config.min_voters, self._num_voting_neighbors)
        if self.config.min_voters > self._num_voting_neighbors:
            _LOGGER.warning(
                "min_voters (%d) > num_voting_neighbors (%d); clamping to %d",
                self.config.min_voters,
                self._num_voting_neighbors,
                self._min_labeled,
            )
        self._target_camera_name = self.config.target_camera_name.lower()

    def _image_obstruction_severity(
        self,
        unified_rows: list[Unified],
        frame_id: str,
        sensor_name: str,
        image_id: str,
    ) -> Optional[float]:
        """Determine the obstruction “severity” for one exact image.

        The image is identified by the `(frame_id, sensor_name, image_id)` triple.

        Args:
            unified_rows: All `Unified` objects that belong to the neighbour slice.
            frame_id: The `frame.frame_id` from the FAISS neighbour’s metadata.
            sensor_name: The camera name (`camera.sensor_name`) of the neighbour image.
            image_id: The unique `camera.image_id` of the neighbour image.

        Returns:
            The obstruction severity for the image, or None if the image is not found. 1.0 means at least one opaque
            obstruction bounding box is present, 0.5 means at least one translucent obstruction bounding box is present,
            and 0.0 means no obstructions are present. If the image is not found, returns None.
        """
        for u in unified_rows:
            for fr in u.frames:
                if fr.frame_id != frame_id:
                    continue
                if not fr.cameras:
                    continue
                for cam in fr.cameras:
                    if cam.sensor_name != sensor_name or cam.image_id != image_id:
                        continue
                    if not cam.labels:
                        return 0.0
                    sev = 0.0
                    for box in cam.labels:
                        cls = (box.label_class or "").upper()
                        if cls == "OPAQUECAMERAOBSTRUCTION":
                            return 1.0  # early‑exit, highest severity
                        if cls == "TRANSLUCENTCAMERAOBSTRUCTION":
                            sev = 0.5  # keep looking in case opaque exists
                    return sev  # transluc or clear
        return None

    def _gather_labelled_neighbours(
        self,
        nn_flat: list[tuple[float, EmbeddedImage]],
        query_slice_ids: set[str],
    ) -> list[tuple[float, float]]:
        """Retrieve (distance, obstruction‑severity) pairs for up to self._num_voting_neighbors images.

        Args:
            nn_flat: A flattened list of tuples (distance, metadata) produced by the FAISS query, sorted arbitrarily.
            query_slice_ids: Slice‑IDs that belong to the slice being scored. Any neighbour whose
                meta.identifiers.slice_id is in this set is skipped to avoid self votes.

        Returns:
            A list containing at most self._num_voting_neighbors tuples (distance, severity).
        """
        labelled_nn: list[tuple[float, float]] = []  # (distance, severity)
        seen_keys: set[tuple[str, str, str]] = set()  # (slice_id, frame_id, image_id) to avoid dupes

        for dist, meta in sorted(nn_flat, key=lambda x: x[0]):  # nearest first
            if meta.sensor_name.lower() != self._target_camera_name:
                continue
            slice_id = meta.identifiers.slice_id
            if slice_id is None:
                continue
            if slice_id in query_slice_ids:
                continue
            key = (slice_id, meta.frame_id, meta.image_id)
            if key in seen_keys:
                continue
            if self.ray_datasets is None:
                raise ValueError("RayDatasets handle is missing; scorer cannot look up neighbour labels")
            unified_rows = self.ray_datasets.read_data_models(HUMAN_LABELS_NAME, slice_id, Unified)
            if not unified_rows:  # neighbour slice not labeled -> skip
                continue

            severity = self._image_obstruction_severity(
                unified_rows,
                frame_id=meta.frame_id,
                sensor_name=meta.sensor_name,
                image_id=meta.image_id,
            )
            if severity is None:  # Could not locate the exact image
                continue

            seen_keys.add(key)
            labelled_nn.append((dist, severity))
            if len(labelled_nn) == self._num_voting_neighbors:
                break

        return labelled_nn

    def process_slice(self, data_model_reader: DataModelReader) -> Optional[float]:
        """Process slice to determine its active learning score.

        This method reads input DataModels using the provided DataModelReader and computes a slice score. The
        DataModelReader is configured to return all records that correspond to the slice ID being processed. This
        arrangement means that in some cases (unlabeled log slices, human labeled slices) it will return a list with a
        single value. In other cases where the schema corresponds to multiple rows per slice (Hulk UM, any UM output)
        it will return a list of frames that correspond to the slice. The available input datasets are defined in the
        `input_assets` arg of the `active_learning_asset_ray` bazel rule. Valid values can be found in the config dict
        in autonomy/perception/datasets/active_learning/framework/lakefs_support.py. The corresponding DataModel class
        should also be provided to enable reading the data into the correct type.

        The method should return a score for the slice, or None if a score cannot be computed. Lower values have higher
        priority, and it's only the relative values of scores that are important.

        Args:
            data_model_reader: Provides access to input DataModels defined in the bazel target for this strategy.

        Returns:
            Score for this slice, or None if a score cannot be computed. Note that lower values have higher priority.
        """
        dinov2_features_list = data_model_reader.read_data_models("features_dinov2", EmbeddedImage)
        if not dinov2_features_list:
            _LOGGER.error("No DINOv2 features for slice %s", data_model_reader.id)
            return None

        embeddings = [
            np.asarray(r.embedding, dtype=np.float32)
            for r in dinov2_features_list
            if getattr(r, "embedding", None) is not None and r.sensor_name.lower() == self._target_camera_name
        ]
        if not embeddings:
            _LOGGER.warning(
                "Row ID %s: no %s embeddings present",
                data_model_reader.id,
                self._target_camera_name,
            )
            return None

        # Usually one slice id
        query_slice_ids: set[str] = {
            r.identifiers.slice_id
            for r in dinov2_features_list
            if r.sensor_name.lower() == self._target_camera_name and r.identifiers.slice_id is not None
        }

        if not query_slice_ids:
            _LOGGER.warning(
                "Row ID %s: missing slice_id for camera %s",
                data_model_reader.id,
                self._target_camera_name,
            )
            return None

        distances, metadata = ray.get(
            self.remote_index.query.remote(
                np.vstack(embeddings).astype(np.float32, copy=False),
                num_neighbors=self.config.num_faiss_query_neighbors,
            )
        )

        nn_flat: list[tuple[float, EmbeddedImage]] = [
            (float(distances[q_idx, k]), metadata[q_idx][k])
            for q_idx in range(distances.shape[0])
            for k in range(self.config.num_faiss_query_neighbors)
            if metadata[q_idx][k] is not None
        ]

        labelled_nn = self._gather_labelled_neighbours(nn_flat, query_slice_ids)

        # Require at least _min_labeled before scoring
        if len(labelled_nn) < self._min_labeled:
            return None

        # Distance‑weighted severity, slice score in [-1.0, 0.0]
        weights = np.array(
            [min(1.0 / (EPSILON + d), self.config.max_weight) for d, _ in labelled_nn],
            dtype=np.float32,
        )
        values = np.array([v for _, v in labelled_nn], dtype=np.float32)

        mean_severity = float(np.dot(weights, values) / weights.sum())
        score = float(np.clip(mean_severity, 0.0, 1.0))
        threshold = self.config.obstruction_prob_threshold
        if score < threshold:
            return None

        return -score
