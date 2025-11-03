"""Defines the current dataset stage for ALFA curate."""

from typing import Final

from autonomy.perception.datasets.active_learning.alfa_curate.config import AlfaCurateConfig
from autonomy.perception.datasets.active_learning.dataset import NAME
from kits.scalex.dataset.stage import Stage
from kits.scalex.dataset.stage_def import StageDefinition
from kits.scalex.dataset.version import SemanticVersion

STAGE_DEFINITION: Final = StageDefinition(
    Stage(dataset_name=NAME, name="selection"),
    SemanticVersion(0, 2, 3),
    "autonomy/perception/datasets/active_learning/alfa_curate:image",
    AlfaCurateConfig,
)
