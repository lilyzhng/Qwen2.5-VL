"""Main entrypoint to generate the stage."""

import logging
from dataclasses import asdict
from typing import Final

import click
import pyarrow as pa
import ray

from autonomy.perception.datasets.human_labels.pico.ingredients import (
    dedupe_and_get_include_and_exclude_maps,
    get_embeddings_table,
    transform_table,
)
from autonomy.perception.datasets.human_labels.pico.pico_config import HumanLabelsPicoConfig
from autonomy.perception.datasets.human_labels.pico.stage import STAGE_DEFINITION
from kits.ml.test_resources.unified.data_model import Unified
from kits.scalex.dataset.bluemind.command import initialize_ray
from kits.scalex.dataset.constants import ROW_ID
from kits.scalex.dataset.generate_support import get_config_for_local_run
from kits.scalex.dataset.index.index_tables import DEFAULT_INDEX_FIELDS
from kits.scalex.dataset.index.index_writer import get_chunks
from kits.scalex.dataset.instances.parquet_dataset import ParquetDataset, ParquetDatasetWriter
from kits.scalex.dataset.interface.parquet_writer import random_path_generator
from kits.scalex.dataset.orchestration import Materialization
from kits.scalex.dataset.stage_str import get_stage_and_reference
from kits.scalex.dataset.stage_support import get_branch_to_use, update_config_for_user
from kits.scalex.ray.constants import PYARROW
from lat_click_util import dataclass_option
from platforms.lakefs.client import LakeFS

_LOGGER: Final = logging.getLogger(__name__)


_INCLUDE_TIMESTAMPS: Final = "include_timestamps"
_EXCLUDE_TIMESTAMPS: Final = "exclude_timestamps"


def load_references(
    row: pa.Table,
    gold_dataset_reference: "ray.ObjectRef[ParquetDataset]",
    embeddings_table_reference: "ray.ObjectRef[pa.Table] | None",
    stride: int,
    frames_per_row: int,
    config: HumanLabelsPicoConfig,
) -> pa.Table:
    """Load the reference from the row dictionary.

    Args:
        row: A dictionary with a "reference" key.
        gold_dataset_reference: A reference to the gold dataset.
        embeddings_table_reference: Optional reference to the embeddings table for temporal subsampling.
        stride: The stride to use when transforming the table.
        frames_per_row: The number of frames per pico row.
        config: The configuration object containing parameters.

    Returns:
        The same dictionary with the "reference" key replaced by the actual Reference object.
    """
    lakefs = LakeFS()
    dataset: ParquetDataset = ray.get(gold_dataset_reference)
    
    # Retrieve embeddings from Ray object store if provided
    embeddings_table = ray.get(embeddings_table_reference) if embeddings_table_reference else None
    
    table = dataset.get_rows(ids=row.column("id").to_pylist())
    include_timestamps = row.column(_INCLUDE_TIMESTAMPS)
    exclude_timestamps = row.column(_EXCLUDE_TIMESTAMPS)

    return transform_table(
        table, stride, frames_per_row, 
        include_timestamps, exclude_timestamps, 
        config, lakefs, embeddings_table
    )


def write_batch(batch: pa.Table, writer: ParquetDatasetWriter[Unified]) -> pa.Table:
    """Write the batch using the writer."""
    writer.add_rows(batch)
    return pa.Table.from_pylist([{"written": True}])


def generate_human_labels_pico(config: HumanLabelsPicoConfig) -> None:
    """Generate the human labels pico dataset.

    Args:
        config: an instance of a dataclass with the config parameters.
    """
    lakefs = LakeFS()
    config = update_config_for_user(config)
    if config.local_run:
        config = get_config_for_local_run(config, ["human_labels_gold_reference"])
    branch_to_use = get_branch_to_use(STAGE_DEFINITION.stage, config, lakefs, _LOGGER)

    _LOGGER.info("Using branch for materialization output: %s", branch_to_use)

    gold_stage, gold_reference = get_stage_and_reference(config.human_labels_gold_reference, lakefs)
    gold_dataset = ParquetDataset(gold_stage, gold_reference.commit, row_id_column=ROW_ID)
    gold_manifest = gold_dataset.get_manifest()
    _LOGGER.info("Loaded gold dataset with %d rows.", len(gold_manifest))
    gold_dataset.cache_index_in_memory()
    gold_dataset_reference = ray.put(gold_dataset)

    if config.limit:
        gold_manifest = gold_manifest[: config.limit]
        _LOGGER.info("Limited to %d references.", len(gold_manifest))

    include_map: dict[str, list[int]] = {}
    exclude_map: dict[str, list[int]] = {}
    if config.embedding_dedupe_threshold > 0:
        include_map, exclude_map = dedupe_and_get_include_and_exclude_maps(config, lakefs)

    # Load embeddings table once for temporal subsampling and put in Ray object store
    embeddings_table_ref = None
    if config.apply_temporal_subsampling and config.features_dinov2_index_reference:
        _LOGGER.info("Loading embeddings table for temporal subsampling...")
        embeddings_table = get_embeddings_table(config.features_dinov2_index_reference, lakefs)
        embeddings_table_ref = ray.put(embeddings_table)
        _LOGGER.info("Embeddings table loaded and stored in Ray object store.")

    with (
        Materialization(
            stage_definition=STAGE_DEFINITION,
            main_branch=branch_to_use,
            input_manifests=[],
            lakefs=lakefs,
            incremental=False,  # Incremental processing is handled by the ParquetDatasetWriter.
            delete_bulk_files=True,
        ) as materialization,
        ParquetDatasetWriter(
            materialization.uris,
            Unified,
            STAGE_DEFINITION.code_version,
            lineage_mode=None,
            index_columns=DEFAULT_INDEX_FIELDS,
            file_namer=random_path_generator,
            writer_kwargs={"row_group_size": 50},
        ) as writer,
    ):
        # The number of files per chunk should be multiplied by the stide since the stride will reduce the dataset size.
        num_files_per_chunk = config.num_rows_per_partition * config.stride
        for partition in get_chunks(gold_manifest, num_files_per_chunk):
            dataset_data = []
            for reference in partition:
                reference_dict = asdict(reference)  # Convert the reference dataclass to a dictionary to pass to Ray.
                include = include_map.get(reference.id, [])
                exclude = exclude_map.get(reference.id, [])
                if (include_map and exclude_map) and (not include and not exclude):
                    _LOGGER.info("Skipping reference %s because missing from include/exclude maps.", reference.id)
                    continue
                reference_dict[_INCLUDE_TIMESTAMPS] = include
                reference_dict[_EXCLUDE_TIMESTAMPS] = exclude
                dataset_data.append(reference_dict)

            dataset = ray.data.from_items(dataset_data)
            dataset = dataset.map_batches(
                load_references,
                batch_format=PYARROW,
                batch_size=2,
                fn_args=(gold_dataset_reference, embeddings_table_ref, config.stride, config.frames_per_row, config),
            )
            dataset = dataset.materialize()
            dataset = dataset.random_shuffle()
            dataset = dataset.map_batches(
                write_batch, batch_format=PYARROW, batch_size=config.final_rows_per_file, fn_args=(writer,)
            )
            dataset.materialize()
            if config.testing_stop_after_one_partition:
                _LOGGER.info("Stopping after one partition as per testing flag.")
                return

command = click.command()(dataclass_option("config")(generate_human_labels_pico))

if __name__ == "__main__":
    initialize_ray(initialize_hpc_filesystem=False)
    command()