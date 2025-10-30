from dataclasses import dataclass
from typing import NamedTuple

from dataclass_json import dataclass_json

@dataclass_json
@dataclass
class PromptConfig:
    prompt: str
    threshold: float
    top_k: int

    def __post_init__(self):
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("Prompt cannot be empty")

class RowIdComponents(NamedTuple):
    base_row_id: str
    start_ns: str
    end_ns: str
    camera_name: str
