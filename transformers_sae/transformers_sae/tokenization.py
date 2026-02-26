import multiprocessing
from dataclasses import dataclass, field
from math import inf
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import AutoTokenizer

from .data_batch import DataBatch
from .replacement_model import ReplacementModel

_ONES: Dict[int, torch.Tensor] = {}
_ZEROS: Dict[int, torch.Tensor] = {}


def _ensure_tokenized(
    inputs: List[str | torch.Tensor],
    tokenizer: AutoTokenizer,
    context_length: int,
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
        if len(t) > context_length:
            token_ids[i] = t[:context_length]
    return token_ids


def _fill_context(
    token_ids: Dict[int, torch.Tensor],
    max_tokens: Optional[int | float],
    max_batch_size: Optional[int | float],
    context_length: int,
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
            if len(cur_input) + row_len <= context_length:
                total_tokens += len(cur_input)
                new_tokens += len(cur_input)
                used_inputs.append(cur_input)
                row_len += len(cur_input)
                del token_ids[i]
                need_new_row = False
                break


def tokenize_strings(
    inputs: List[str | torch.Tensor],
    dtype: torch.dtype,
    tokenizer: AutoTokenizer,
    context_length: int,
    max_tokens: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    token_offset: int = 0,
):
    token_ids = _ensure_tokenized(inputs, tokenizer, context_length)

    row_lens = []
    input_id_stack = []
    position_id_stack = []
    attention_mask_stack = []
    num_tokens = 0
    batch_size = 0
    num_rows = 0
    input_batches = []
    row_lens = []
    new_tokenses = []

    # Minimize the amount of work we do in case we need to seek to token_offset. If we haven't generated
    # enough tokens then we will be skipping this batch anyway, so no need to make tensors for it.
    for input_batch, row_len, new_tokens in _fill_context(
        token_ids, max_tokens, max_batch_size, context_length
    ):
        num_tokens += new_tokens
        batch_size += 1
        num_rows += len(input_batch)
        input_batches.append(input_batch)
        row_lens.append(row_len)
        new_tokenses.append(new_tokens)

    if num_tokens <= token_offset:
        return DataBatch(
            torch.empty((0,)),
            torch.empty((0,)),
            torch.empty((0,)),
            num_tokens,
            batch_size,
            num_rows,
            row_lens,
            torch.empty((0,)),
        ), list(token_ids.values())

    if context_length not in _ONES:
        ones = _ONES[context_length] = torch.ones(
            (context_length, context_length), dtype=dtype
        )
    else:
        ones = _ONES[context_length]
    if context_length not in _ZEROS:
        zeros = _ZEROS[context_length] = torch.zeros(
            (context_length, context_length), dtype=dtype
        )
    else:
        zeros = _ZEROS[context_length]

    for input_batch, row_len, new_tokens in zip(input_batches, row_lens, new_tokenses):
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
        if new_tokens < context_length:
            position_id_stack[-1] = torch.cat(
                (
                    position_id_stack[-1],
                    torch.full(
                        (context_length - new_tokens,),
                        -1,
                        dtype=torch.int64,
                    ),
                )
            )
            input_id_stack[-1] = torch.cat(
                (
                    input_id_stack[-1],
                    torch.full(
                        (context_length - new_tokens,),
                        tokenizer.bos_token_id,
                        dtype=torch.int64,
                    ),
                )
            )
        # Set causal mask, with additional constraint to ignore tokens from other inputs.
        # Eg, if we have inputs of length 4, 3, 2, 4, we want our mask to look like,
        # before replacing ones with the minimum value for our dtype:
        # x, y, z, w = torch.ones((4, 4)), torch.ones((3, 3)), torch.ones((2, 2)), torch.ones((4, 4))
        # 1 - torch.block_diag(x.tril(), y.tril(), z.tril(), w.tril())
        # > tensor([[0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #           [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #           [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #           [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1.],
        #           [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.]])
        mask_blocks = [ones[0 : len(i), 0 : len(i)] for i in input_batch]
        # Pad?
        if new_tokens < context_length:
            mask_blocks.append(
                zeros[0 : context_length - new_tokens, 0 : context_length - new_tokens]
            )

        # TODO: how hard would it be to write a custom kernel for this, or find some other way to
        # speed it up? Profiler shows this taking up a significant amount of time, ~twice that for tokenization.
        attention_mask_stack.append(
            ((1.0 - torch.block_diag(*mask_blocks)) * torch.finfo(dtype).min).unsqueeze(
                0
            )
        )

    unused_inputs = list(token_ids.values())
    position_ids = torch.stack(position_id_stack)
    token_mask = (position_ids >= 0).to(dtype=torch.float32)

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
    tokens_to_generate: Optional[int] = 0


def _iter_dataset(
    dataset: IterableDataset,
    max_tokens: Optional[int],
    max_rows: Optional[int],
    max_batches: Optional[int],
    tokenizer_batch_size: int,
    inference_batch_size: int,
    context_length: int,
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
                    context_length * inference_batch_size,
                )
            else:
                state.tokens_to_generate = None

            # Caller must update number of generated tokens and consume batch_inputs
            yield state

            state.num_rows_consumed += prev_len - len(state.batch_inputs)
            state.num_batches += 1
        else:
            return


def _input_generator(
    model: ReplacementModel,
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
        model.context_length,
    ):
        batch, state.batch_inputs = tokenize_strings(
            state.batch_inputs,
            model.dtype,
            tokenizer,
            model.context_length,
            state.tokens_to_generate,
            inference_batch_size,
            offset - state.num_tokens_generated,
        )
        # TODO: we should find a way to not have to tokenize everything up until this point. Unfortunately
        # we can't really just skip to a particular row of the dataset because of the way we pack into
        # batches. Possibly we can live with this being slightly non-deterministic, or maybe there's some
        # simple bit of info we could pass along that would still keep determinism (eg, greatest row i
        # such that we have definitely already used all tokens for rows < i)
        if state.num_tokens_generated + batch.num_tokens > offset:
            # NB: this should maybe technically discard some rows from the start, since we'll effectively
            # be training on the overlapping tokens twice, but I doubt this matters much.
            yield batch
        state.num_tokens_generated += batch.num_tokens


def make_dataloader(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    max_tokens: int | None,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    offset: int = 0,
    max_batches: int | None = None,
):
    class InputGeneratorDataset(TorchIterableDataset):
        def __iter__(self):
            return iter(
                _input_generator(
                    model,
                    tokenizer,
                    dataset,
                    max_tokens=max_tokens,
                    max_batches=max_batches,
                    tokenizer_batch_size=tokenizer_batch_size,
                    inference_batch_size=inference_batch_size,
                    offset=offset,
                )
            )

    # Can't/don't want to serialize process args if we're on MacOS, so just return a regular iterator
    if multiprocessing.get_start_method() == "spawn":
        return iter(
            _input_generator(
                model,
                tokenizer,
                dataset,
                max_tokens=max_tokens,
                max_batches=max_batches,
                tokenizer_batch_size=tokenizer_batch_size,
                inference_batch_size=inference_batch_size,
                offset=offset,
            )
        )

    return DataLoader(
        InputGeneratorDataset(),
        batch_size=None,
        num_workers=1,
        pin_memory=model.device.type == "cuda",
    )
