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
