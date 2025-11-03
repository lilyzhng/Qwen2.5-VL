"""Data types for ALFA-based data selection strategy."""

from dataclasses import dataclass
from typing import NamedTuple

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class PromptConfig:
    """Configuration for a specific prompt."""

    prompt: str
    camera_names: list[str]
    similarity_threshold: float

    def __post_init__(self) -> None:
        """Validate the prompt configuration."""
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("Prompt must be a non-empty string")


@dataclass_json
@dataclass
class ScenarioConfig:
    """Configuration for a single prompt used in selection."""

    name: str
    prompts: list[PromptConfig]


class RowIdComponents(NamedTuple):
    """Components of a parsed row ID."""

    base_row_id: str
    start_ns: str
    end_ns: str
    camera_name: str