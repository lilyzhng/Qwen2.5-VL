"""Main entrypoint to generate the human labels gold dataset locally or on bluemind."""

import logging
from typing import Final

from autonomy.perception.datasets.active_learning.alfa_curate.config import AlfaCurateConfig, load_scenarios_from_yaml
from autonomy.perception.datasets.active_learning.alfa_curate.data_types import ScenarioConfig
from autonomy.perception.datasets.active_learning.alfa_curate.stage import STAGE_DEFINITION
from autonomy.perception.datasets.active_learning.alfa_curate.utils import (
    SearchResult,
    deduplicate_by_base_slice,
    get_slice_ids_to_exclude,
    load_table,
    run_text_query,
    similarity_to_l2_distance,
)
from autonomy.perception.datasets.active_learning.framework.ray_worker_impl import PARQUET_FILE_NAME
from autonomy.perception.datasets.active_learning.selection.data_model import ActiveLearningSelection
from kits.ml.onnx.model_management.av_path import access_av_path
from kits.scalex.dataset.bluemind.command import initialize_ray
from kits.scalex.dataset.config import get_updated_config
from kits.scalex.dataset.file_name import get_slice_id
from kits.scalex.dataset.lakefs_branches import get_branch_for_env_and_user
from kits.scalex.dataset.orchestration import Materialization
from kits.scalex.dataset.stage_str import get_manifest_from_stage_str
from kits.scalex.dataset.stage_support import build_command, update_config_for_user
from kits.scalex.hpc.tiered_file_system import tiered_filesystem
from kits.scalex.parquet import PARQUET
from kits.scalex.writer.writer import write_to_single_file
from platforms.lakefs.client import LakeFS
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.infer import VLMJudge
from autonomy.perception.datasets.active_learning.alfa_curate.vlm_judge.utils import (
    load_frames_for_search_result,
)

_LOGGER: Final = logging.getLogger(__name__)


def select_slices_for_scenario(
    config: AlfaCurateConfig,
    scenario: ScenarioConfig,
) -> list[SearchResult]:
    """Select slices for a scenario using text query and VLM judge.
    
    This function implements a two-stage selection pipeline:
    1. Text-to-video embedding similarity search with deduplication
    2. VLM judge to verify and filter candidates
    
    Args:
        config: Configuration for the selection process
        scenario: Scenario configuration with prompts
        
    Returns:
        List of SearchResult objects that passed both stages
    """
    # Stage 1: Run text queries and deduplicate
    _LOGGER.info("Stage 1: Running text-to-video similarity search...")
    results = []
    table = load_table(config.repo, config.lance_db_branch, config.table_name)
    
    for prompt_config in scenario.prompts:
        results += run_text_query(
            table,
            prompt_config.prompt,
            config.model_size,
            camera_names=prompt_config.camera_names,
            upper_bound_distance=similarity_to_l2_distance(prompt_config.similarity_threshold),
        )
    
    _LOGGER.info("Number of initial results: %d.", len(results))
    results = deduplicate_by_base_slice(results)
    _LOGGER.info("Number of unique slices after deduplication: %d.", len(results))
    
    # Stage 2: VLM judge filtering
    _LOGGER.info("Stage 2: Running VLM judge to verify candidates...")
    filtered_results = apply_vlm_judge(
        results=results,
        scenario=scenario,
        config=config,
    )
    _LOGGER.info("Number of slices after VLM filtering: %d.", len(filtered_results))
    
    return filtered_results


def apply_vlm_judge(
    results: list[SearchResult],
    scenario: ScenarioConfig,
    config: AlfaCurateConfig,
) -> list[SearchResult]:
    """Apply VLM judge to filter and re-score search results.
    
    Args:
        results: Initial search results from embedding similarity
        scenario: Scenario configuration with prompts
        config: Configuration for VLM inference
        
    Returns:
        Filtered and re-scored results based on VLM judgement
    """
    # Skip VLM judge if disabled
    if not config.vlm_judge.enable_vlm_judge:
        _LOGGER.info("VLM judge is disabled, skipping filtering")
        return results
    
    # Limit candidates to top K for efficiency
    max_candidates = config.vlm_judge.max_candidates_for_vlm
    if len(results) > max_candidates:
        _LOGGER.info("Limiting VLM judge to top %d candidates (from %d)", max_candidates, len(results))
        results = results[:max_candidates]
    
    if not results:
        return results
    
    # Initialize VLM judge
    _LOGGER.info("Initializing VLM judge with model: %s", config.vlm_judge.vlm_model_path)
    judge = VLMJudge(
        model_path=config.vlm_judge.vlm_model_path,
        load_model_from_lakefs=config.vlm_judge.load_model_from_lakefs,
        use_flash_attn=config.vlm_judge.use_flash_attn,
        max_new_tokens=config.vlm_judge.max_new_tokens,
    )
    
    # Get manifest for loading video data
    log_slices_silver = get_manifest_from_stage_str(config.log_slices_silver_reference, lakefs=LakeFS(), use_index=True)
    
    # Build judgment query from scenario prompts
    if not scenario.prompts:
        _LOGGER.warning("No prompts in scenario, skipping VLM judge")
        return results
    
    judgment_query = scenario.prompts[0].prompt
    _LOGGER.info("VLM judgment query: %s", judgment_query)
    
    # Filter results using VLM judge
    filtered_results = []
    for i, result in enumerate(results):
        try:
            frames = load_frames_for_search_result(
                result,
                log_slices_silver,
                desired_fps=config.vlm_judge.segment_desired_fps,
                max_frames=config.vlm_judge.max_frames_per_segment,
            )
            
            if not frames:
                _LOGGER.warning("No frames loaded for slice_id: %s, skipping", result.slice_id)
                continue
            
            # Run VLM judgment
            judgment = judge.judge_frames(frames, judgment_query)
            
            # Filter based on judgment and confidence threshold
            if judgment.match and judgment.confidence >= config.vlm_judge.vlm_confidence_threshold:
                filtered_results.append(result)
                _LOGGER.info(
                    "PASS [%d/%d]: slice_id=%s, vlm_confidence=%.2f, embedding_sim=%.2f\n"
                    "  Observation: %s\n"
                    "  Reason: %s",
                    i + 1, len(results), result.slice_id, judgment.confidence, result.similarity,
                    judgment.observation, judgment.reason,
                )
            else:
                _LOGGER.info(
                    "FAIL [%d/%d]: slice_id=%s, vlm_match=%s, vlm_confidence=%.2f, embedding_sim=%.2f\n"
                    "  Observation: %s\n"
                    "  Reason: %s",
                    i + 1, len(results), result.slice_id, judgment.match, judgment.confidence, result.similarity,
                    judgment.observation, judgment.reason,
                )
        
        except Exception as e:
            _LOGGER.error("Error judging slice_id=%s: %s", result.slice_id, e, exc_info=True)
            continue
    
    _LOGGER.info(
        "VLM judge filtered %d -> %d results (%.1f%% pass rate)",
        len(results), len(filtered_results),
        100.0 * len(filtered_results) / len(results) if results else 0.0,
    )
    
    return filtered_results


def _materialize_scenario(
    config: AlfaCurateConfig,
    lakefs: LakeFS,
    slice_id_to_key: dict[str, str],
    scenario: ScenarioConfig,
    branch_to_use: str,
) -> None:
    """Materialize Alfa selections for a specific scenario."""
    with Materialization(
        main_branch=branch_to_use,
        input_manifests=[],
        lakefs=lakefs,
        stage_definition=STAGE_DEFINITION,
        incremental=config.incremental,
        stage_config=config,
    ) as materialization:
        # Run the two-stage selection pipeline
        results = select_slices_for_scenario(config, scenario)

        selected_data = [
            ActiveLearningSelection(
                key=slice_id_to_key[result.slice_id],
                selection_strategy=f"alfa_curate_{scenario.name}",
                selection_strategy_version=str(STAGE_DEFINITION.code_version),
                strategy_score=-result.similarity,
                unlabeled_dataset_description=config.log_slices_silver_reference,
                labeled_dataset_description=config.human_labels_gold_reference,
            )
            for result in results
            if result.slice_id in slice_id_to_key
        ]
        _LOGGER.info("Length after filtering out labeled slices: %d", len(selected_data))

        output_filename = materialization.uris.data_file(PARQUET_FILE_NAME, PARQUET)
        if selected_data:
            write_to_single_file(
                selected_data, output_filename, materialization.uris.common_metadata, materialization.lakefs.s3
            )
        else:
            lakefs.s3.delete(output_filename)


def generate_alfa_curate(config: AlfaCurateConfig) -> None:
    """Generate the human labels gold dataset.

    Args:
        config: an instance of a dataclass with the config parameters.
    """
    lakefs = LakeFS()
    read_filesystem = tiered_filesystem()

    config = update_config_for_user(config)
    config = get_updated_config(config, update_git_fields=True)
    base_branch_name = config.branch

    exclude_slice_ids = get_slice_ids_to_exclude(config.active_learning_filter_reference, lakefs, read_filesystem)
    log_slices_silver = get_manifest_from_stage_str(config.log_slices_silver_reference, lakefs, use_index=True)
    slice_id_to_key = {
        slice_id: reference.key
        for reference in log_slices_silver.key_to_data_file.values()
        if (slice_id := get_slice_id(reference.path)) not in exclude_slice_ids
    }

    scenarios: list[ScenarioConfig] = load_scenarios_from_yaml(access_av_path(config.prompt_yaml_path))
    for i, scenario in enumerate(scenarios):
        branch_name = f"{base_branch_name}_{scenario.name}"
        branch_to_use = get_branch_for_env_and_user(
            STAGE_DEFINITION.stage.repo, branch_name, config.env, config.user, lakefs, _LOGGER
        )
        _LOGGER.info("Scenario %d of %d: %s, writing to branch %s.", i + 1, len(scenarios), scenario, branch_to_use)
        _materialize_scenario(config, lakefs, slice_id_to_key, scenario, branch_to_use)


command = build_command(generate_alfa_curate)


if __name__ == "__main__":
    initialize_ray(initialize_hpc_filesystem=False)
    command()
