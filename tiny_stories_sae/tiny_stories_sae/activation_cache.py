from __future__ import annotations

import json
from contextlib import ExitStack
from functools import partial
from typing import Dict

import numpy as np
import torch
import zarr
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data_batch import DataBatch
from .ops import ensure_directory
from .tokenization import DEFAULT_CONTEXT_LENGTH, make_dataloader


@torch.no_grad
def _run_base_model(
    base_model: torch.nn.Module,
    batch: DataBatch,
    hooks: Dict[str, torch.nn.Module],
):
    activation_cache = {}

    def _hook_output(probe_name, _module, _args, out):
        activation_cache[probe_name] = out

    with ExitStack() as hook_stack:
        for hook_name, module in hooks.items():
            hook_stack.enter_context(
                module.register_forward_hook(partial(_hook_output, hook_name))
            )
        output = base_model(
            input_ids=batch.input_ids.to(base_model.device),
            position_ids=batch.position_ids.to(base_model.device),
            attention_mask=batch.attention_mask.to(base_model.device),
            use_cache=False,
        )

    return activation_cache, output.logits


CACHE_METADATA_FILENAME = "metadata.json"


def init_cache(
    model: torch.nn.Module,
    cache_dir: str,
    num_layers: int,
    inference_batch_size: int,
    batches_per_file: int,
    tokens_per_row: int,
    d_model: int,
):
    zarr_kwargs = {
        "overwrite": True,
        "compressors": None,
    }
    metadata = {"num_tokens": 0, "num_rows": 0, "num_layers": num_layers}
    ensure_directory(cache_dir)
    zarr.create_array(
        cache_dir,
        name="activations",
        dtype=np.float32,
        shape=(
            num_layers,
            0,
            tokens_per_row,
            d_model,
        ),
        chunks=(
            1,
            inference_batch_size,
            tokens_per_row,
            model.config.hidden_size,
        ),
        shards=(
            model.config.num_layers,
            batches_per_file * inference_batch_size,
            tokens_per_row,
            model.config.hidden_size,
        ),
        **zarr_kwargs,
    )
    zarr.create_array(
        cache_dir,
        name="position_ids",
        dtype=np.int16,
        shape=(0, tokens_per_row),
        chunks=(inference_batch_size, tokens_per_row),
        shards=(batches_per_file * inference_batch_size, tokens_per_row),
        **zarr_kwargs,
    )
    json.dump(metadata, open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "w"))


def cache_on_disk(cache_dir: str, model: torch.nn.Module, batch: DataBatch):
    metadata = json.load(open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "r"))

    activations, _ = _run_base_model(
        model,
        batch,
        {layer: model.transformer.h[layer] for layer in range(model.config.num_layers)},
    )

    cache = zarr.open_group(cache_dir, mode="r+")
    old_num_rows = metadata["num_rows"]
    new_num_rows = old_num_rows + batch.batch_size
    cache["position_ids"].resize((new_num_rows, cache["position_ids"].shape[1]))
    cache["activations"].resize(
        (cache["activations"].shape[0], new_num_rows, *cache["activations"].shape[2:])
    )
    cache["position_ids"][old_num_rows:new_num_rows, :] = batch.position_ids.to(
        "cpu"
    ).numpy()
    cache["activations"][:, old_num_rows:new_num_rows, :, :] = (
        torch.stack(
            tuple(v[0] if isinstance(v, tuple) else v for v in activations.values())
        )
        .to("cpu")
        .numpy()
    )

    metadata["num_tokens"] += batch.num_tokens
    metadata["num_rows"] = new_num_rows

    json.dump(metadata, open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "w"))


def build_cache(
    cache_dir: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    num_tokens: int,
    tokenizer_batch_size: int,
    inference_batch_size: int,
):
    progress = tqdm(total=num_tokens, desc="Building activation cache")

    init_cache(
        model,
        cache_dir,
        model.config.num_layers,
        inference_batch_size,
        1,
        DEFAULT_CONTEXT_LENGTH,
        model.config.hidden_size,
    )
    num_used_tokens = 0

    model.eval()
    for batch in make_dataloader(
        model,
        tokenizer,
        dataset,
        max_tokens=num_tokens,
        tokenizer_batch_size=tokenizer_batch_size,
        inference_batch_size=inference_batch_size,
    ):
        cache_on_disk(cache_dir, model, batch)
        num_used_tokens += batch.num_tokens
        progress.total = max(num_tokens, num_used_tokens)
        progress.update(batch.num_tokens)
    progress.close()


def load_cache(num_layers: int, cache_dir: str, cache_offset: int, batch: DataBatch):
    # TODO: we should reconstruct the batch from cache, rather than needing the
    # batch object here
    cache = zarr.open_group(cache_dir, mode="r")
    return {
        layer: torch.tensor(
            cache["activations"][
                layer, cache_offset : cache_offset + batch.batch_size, :, :
            ]
        )
        for layer in range(0, num_layers)
    }
