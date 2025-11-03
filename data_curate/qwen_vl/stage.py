"""Define the current dataset stage for QwenVL video judgment."""

from typing import Final

from autonomy.perception.datasets.features.dataset import NAME
from autonomy.perception.datasets.features.qwen_vl.config import QwenVLJudgeConfig
from kits.scalex.dataset.dependency.dependency_list import DependencyList
from kits.scalex.dataset.dependency.deps import SchemaDep
from kits.scalex.dataset.stage import Stage
from kits.scalex.dataset.stage_def import StageDefinition
from kits.scalex.dataset.version import SemanticVersion
from pyarrow import schema, field, string, bool_, timestamp


def _get_judgment_schema():
    """Define the output schema for video judgments."""
    return schema(
        [
            field("row_id", string()),
            field("sensor_name", string()),
            field("query", string()),
            field("judgment", bool_()),
            field("timestamp", timestamp("us")),
        ]
    )


def _get_minor_hash() -> str:
    """Compute hash for schema dependencies."""
    dep_list = DependencyList([SchemaDep(_get_judgment_schema())])
    return dep_list.compute_overall_hash()


STAGE_DEFINITION: Final = StageDefinition(
    stage=Stage(dataset_name=NAME, name="qwen_vl_judge"),
    code_version=SemanticVersion(0, 1, 0, minor_hash=_get_minor_hash()),
    bluemind_image_target="autonomy/perception/datasets/features/qwen_vl:image",
    stage_config_type=QwenVLJudgeConfig,
)

