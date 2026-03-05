from dataclasses import dataclass
from typing import List

import torch


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

    def to(self, *args, **kwargs):
        self.input_ids = self.input_ids.to(*args, **kwargs)
        self.position_ids = self.position_ids.to(*args, **kwargs)
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        self.token_mask = self.token_mask.to(*args, **kwargs)
