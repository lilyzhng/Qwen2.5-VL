"""Config for ALFA-based data selection strategy using text prompts to select relevant video scenarios."""


from pathlib import Path

from ruamel.yaml import YAML

from autonomy.perception.datasets.active_learning.alfa_curate.data_types import ScenarioConfig
from kits.scalex.dataset.config import BaseStageConfigV2


class AlfaCurateConfig(BaseStageConfigV2):
    """Configuration for the alfa-based active learning selection strategy."""

    #: References to input data.
    log_slices_silver_reference: str = "sensing--log-slices--silver/main"
    human_labels_gold_reference: str = "sensing--human-labels--gold/main"
    active_learning_filter_reference: str = "sensing--active-learning--selection/filter"

    #: Repo.
    repo: str = "sensing--features--cosmos-index"

    #: LanceDB branch to read from.
    lance_db_branch: str = "main"

    # : LanceDB table name.
    table_name: str = "data"

    #: Cosmos model size for text embeddings.
    model_size: str = "Cosmos-Embed1-448p"

    #: Path to YAML file containing prompts.
    prompt_yaml_path: str = "autonomy/perception/datasets/active_learning/alfa_curate/resources/prompts.yaml"

    #: Maximum number of slices to process in a single batch.
    batch_size: int = 1000

    #: Standard configuration options.
    branch: str = "alfa_curate"
    incremental: bool = False


def load_scenarios_from_yaml(path: str | Path) -> list[ScenarioConfig]:
    """Load scenarios from a YAML file."""
    with open(path, "r") as f:
        yaml = YAML(typ="safe")
        data = yaml.load(f)

    return [ScenarioConfig.from_dict(item) for item in data.get("scenarios", [])]  # type: ignore[attr-defined]
