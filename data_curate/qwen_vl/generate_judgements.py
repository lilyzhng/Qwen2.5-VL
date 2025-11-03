"""Calculate QwenVL video judgments for the log slices dataset."""

import logging
from typing import Final

from autonomy.perception.datasets.features.qwen_vl.config import QwenVLJudgeConfig
from autonomy.perception.datasets.features.qwen_vl.infer import HF_CACHE_DIR, HF_HOME, QwenVLJudge
from autonomy.perception.datasets.features.qwen_vl.stage import STAGE_DEFINITION
from autonomy.perception.datasets.features.dinov2.generate_embedded_image_dataset import generate_embedded_dataset
from autonomy.perception.datasets.features.dinov2.infer import DatasetVideoEmbedder, EmbeddedVideo
from kits.scalex.dataset.bluemind.command import initialize_ray
from kits.scalex.dataset.constants import ROW_ID
from kits.scalex.dataset.instances.parquet_dataset import ParquetDataset
from kits.scalex.dataset.lakefs_branches import get_branch_for_env_and_user
from kits.scalex.dataset.orchestration import Materialization
from kits.scalex.dataset.stage_str import get_stage_and_reference
from kits.scalex.dataset.stage_support import build_command, update_config_for_user
from kits.scalex.ray.constants import NUM_CPUS, NUM_GPUS
from platforms.lakefs.client import LakeFS

_LOGGER: Final = logging.getLogger(__name__)


def generate_qwenvl_judgments(config: QwenVLJudgeConfig) -> None:
    """Generate QwenVL video judgments for the log slices dataset."""
    lakefs = LakeFS()
    config = update_config_for_user(config)
    silver_stage, silver_reference = get_stage_and_reference(config.log_slices_silver_reference, lakefs)
    log_slices_dataset = ParquetDataset(silver_stage, silver_reference.commit, row_id_column=ROW_ID)

    branch_to_use = get_branch_for_env_and_user(
        STAGE_DEFINITION.stage.repo, config.branch, config.env, config.user, lakefs, _LOGGER
    )
    _LOGGER.info("Using branch for materialization output: %s", branch_to_use)
    _LOGGER.info("Using config: %s", config)
    camera_names_to_process = config.process_camera_names_csv.split(",")
    _LOGGER.info("Will process cameras %s", camera_names_to_process)
    _LOGGER.info("Judgment queries: %s", config.judgements)

    with Materialization(
        main_branch=branch_to_use,
        input_manifests=[],
        lakefs=lakefs,
        stage_definition=STAGE_DEFINITION,
        incremental=False,  # Incremental processing is handled by the ParquetDatasetWriter.
        delete_bulk_files=config.delete_bulk_files,
    ) as materialization:
        generate_embedded_dataset(
            embedder_type=DatasetVideoEmbedder,
            embedder_output_type=EmbeddedVideo,
            model_ref_or_type=QwenVLJudge,
            materialization=materialization,
            log_slices_dataset=log_slices_dataset,
            config=config,
            code_version=STAGE_DEFINITION.code_version,
            logger=_LOGGER,
            index_columns=["identifiers", "row_id", "sensor_name", "query", "logapps_metadata"],
            row_id_field="row_id",
            frame_stride=1,
            ray_remote_args={
                NUM_CPUS: config.num_cpus_per_judge_actor,
                NUM_GPUS: config.num_gpus_per_judge_actor,
            },
            model_fn_constructor_kwargs={
                "model_path": config.model_path,
                "load_model_from_lakefs": config.load_model_from_lakefs,
                "max_new_tokens": config.max_new_tokens,
            },
            extra_fn_constructor_kwargs={
                "segment_overlapping_secs": config.segment_overlapping_secs,
                "segment_desired_fps": config.segment_desired_fps,
                "camera_names": camera_names_to_process,
                "judgment_queries": config.judgements,
            },
        )


command = build_command(generate_qwenvl_judgments)


if __name__ == "__main__":
    initialize_ray(
        initialize_hpc_filesystem=False,
        # Workaround to load QwenVL in Ray worker.
        runtime_env={
            "env_vars": {
                "HF_HOME": HF_HOME,
                "HF_MODULES_CACHE": HF_CACHE_DIR,
            },
        },
    )
    command()

