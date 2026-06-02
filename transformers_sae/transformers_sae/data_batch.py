from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


@dataclass
class ExampleInfo:
    batch_index: Optional[int] = None
    token_range: Optional[tuple[int, int]] = None
    original_example: Optional[Any] = None


@dataclass
class TokenizedExample:
    token_ids: torch.Tensor
    info: ExampleInfo


@dataclass
class DataBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    num_tokens: int
    batch_size: int
    num_dataset_rows: int
    input_lens: List[int]
    token_mask: torch.Tensor
    special_token_indices: torch.Tensor
    skipped: bool = False
    example_info: list[ExampleInfo] = field(default_factory=list)

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.position_ids = self.position_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.token_mask = self.token_mask.to(*args, **kwargs)
        return self
