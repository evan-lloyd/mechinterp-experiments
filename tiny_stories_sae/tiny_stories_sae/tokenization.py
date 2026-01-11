from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

from .data_batch import DataBatch

# from model card, value used in training
CONTEXT_LENGTH = 512


def _attention_mask_row(dtype):
    return torch.full(
        (1, CONTEXT_LENGTH, CONTEXT_LENGTH),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
    )


def _ensure_tokenized(
    inputs: List[str | torch.Tensor], tokenizer: AutoTokenizer
) -> Dict[int, torch.Tensor]:
    needed_tokenizations = {
        i: _in for i, _in in enumerate(inputs) if isinstance(_in, str)
    }
    if needed_tokenizations:
        new_tokenizations = tokenizer(
            list(needed_tokenizations.values()), return_attention_mask=False
        )["input_ids"]
        for idx, i in enumerate(needed_tokenizations):
            needed_tokenizations[i] = new_tokenizations[idx]
    token_ids = {i: needed_tokenizations.get(i, inputs[i]) for i in range(len(inputs))}
    for i, t in token_ids.items():
        if len(t) > CONTEXT_LENGTH:
            token_ids[i] = t[:CONTEXT_LENGTH]
    return token_ids


def _fill_context(
    token_ids: Dict[int, torch.Tensor],
    max_tokens: Optional[int],
    max_batch_size: Optional[int],
) -> Iterator[Tuple[List[torch.Tensor], int, int]]:
    if max_tokens is None:
        max_tokens = inf
    if max_batch_size is None:
        max_batch_size = inf
    row_len = 0
    total_tokens = 0
    new_tokens = 0
    batch_size = 0
    need_new_row = False
    used_inputs = []
    while token_ids and total_tokens < max_tokens:
        if need_new_row:
            batch_size += 1
            yield used_inputs, row_len, new_tokens
            used_inputs = []
            row_len = 0
            new_tokens = 0

            if batch_size >= max_batch_size:
                return

        # Greedily consume entire inputs until we fill each CONTEXT_LENGTH row of stack
        need_new_row = True
        for i, cur_input in token_ids.items():
            if len(cur_input) + row_len <= CONTEXT_LENGTH:
                total_tokens += len(cur_input)
                new_tokens += len(cur_input)
                used_inputs.append(cur_input)
                row_len += len(cur_input)
                del token_ids[i]
                need_new_row = False
                break


def tokenize_strings(
    inputs: List[str | torch.Tensor],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    max_tokens: Optional[int] = None,
    max_batch_size: Optional[int] = None,
):
    token_ids = _ensure_tokenized(inputs, tokenizer)

    row_lens = []
    input_id_stack = []
    position_id_stack = []
    attention_mask_stack = []
    num_tokens = 0
    batch_size = 0
    num_rows = 0
    for input_batch, row_len, new_tokens in _fill_context(
        token_ids, max_tokens, max_batch_size
    ):
        batch_size += 1
        num_rows += len(input_batch)
        num_tokens += new_tokens
        row_lens.append(row_len)
        input_id_stack.append(
            torch.cat([torch.tensor(in_, dtype=torch.int64) for in_ in input_batch])
        )
        position_id_stack.append(
            torch.cat(
                [
                    torch.arange(len(cur_input), dtype=torch.int64)
                    for cur_input in input_batch
                ]
            )
        )
        if new_tokens < CONTEXT_LENGTH:
            position_id_stack[-1] = torch.cat(
                (
                    position_id_stack[-1],
                    torch.full(
                        (CONTEXT_LENGTH - new_tokens,),
                        -1,
                        dtype=torch.int64,
                    ),
                )
            )
            input_id_stack[-1] = torch.cat(
                (
                    input_id_stack[-1],
                    torch.full(
                        (CONTEXT_LENGTH - new_tokens,),
                        tokenizer.bos_token_id,
                        dtype=torch.int64,
                    ),
                )
            )
        attention_mask_stack.append(_attention_mask_row(model.dtype))
        cur_token = 0
        # TODO: should be vectorizable
        for cur_input in input_batch:
            for j in range(cur_token, cur_token + len(cur_input)):
                attention_mask_stack[-1][0, j, cur_token : j + 1] = 0.0
            cur_token += len(cur_input)

    unused_inputs = list(token_ids.values())
    position_ids = torch.stack(position_id_stack)
    token_mask = (position_ids >= 0).float()

    # These have to be non-negative on CUDA kernels
    position_ids = torch.where(position_ids >= 0, position_ids, 0)
    return DataBatch(
        torch.stack(input_id_stack),
        position_ids,
        torch.stack(attention_mask_stack),
        num_tokens,
        batch_size,
        num_rows,
        row_lens,
        token_mask,
    ), unused_inputs


@dataclass
class _IterState:
    batch_inputs: List[str | torch.Tensor] = field(default_factory=list)
    num_tokens_generated: int = 0
    num_rows_consumed: int = 0
    num_batches: int = 0
    tokens_to_generate: int = 0


def _iter_dataset(
    dataset: IterableDataset,
    max_tokens: Optional[int],
    max_rows: Optional[int],
    max_batches: Optional[int],
    tokenizer_batch_size,
    inference_batch_size,
):
    state = _IterState()
    dataset_iterable = dataset.iter(max(tokenizer_batch_size // 2, 1))

    while state.num_tokens_generated < (
        max_tokens if max_tokens is not None else inf
    ) and state.num_rows_consumed < (max_rows if max_rows is not None else inf):
        if max_batches is not None and state.num_batches >= max_batches:
            break
        while len(state.batch_inputs) < tokenizer_batch_size:
            try:
                state.batch_inputs.extend(next(dataset_iterable)["text"])
            except StopIteration:
                break
        if state.batch_inputs:
            prev_len = len(state.batch_inputs)
            if max_tokens is not None:
                state.tokens_to_generate = max(
                    max_tokens - state.num_tokens_generated,
                    CONTEXT_LENGTH * inference_batch_size,
                )
            else:
                state.tokens_to_generate = None

            # Caller must update number of generated tokens and consume batch_inputs
            yield state

            state.num_rows_consumed += prev_len - len(state.batch_inputs)
            state.num_batches += 1
        else:
            return


def input_generator(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    max_tokens: Optional[int] = None,
    max_rows: Optional[int] = None,
    max_batches: Optional[int] = None,
    tokenizer_batch_size: int = 1,
    inference_batch_size: int = 1,
    offset: int = 0,
):
    for state in _iter_dataset(
        dataset,
        max_tokens,
        max_rows,
        max_batches,
        tokenizer_batch_size,
        inference_batch_size,
    ):
        batch, state.batch_inputs = tokenize_strings(
            state.batch_inputs,
            model,
            tokenizer,
            state.tokens_to_generate,
            inference_batch_size,
        )
        if state.num_tokens_generated + batch.num_tokens > offset:
            # NB: this should maybe technically discard some rows from the start, since we'll effectively
            # be training on the overlapping tokens twice, but I doubt this matters much.
            yield batch
        state.num_tokens_generated += batch.num_tokens
