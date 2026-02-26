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
from .replacement_model import ReplacementModel
from .tokenization import make_dataloader


@torch.no_grad()
def _run_base_model(
    base_model: ReplacementModel,
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
    model: ReplacementModel,
    cache_dir: str,
    inference_batch_size: int,
    batches_per_file: int,
):
    zarr_kwargs = {
        "overwrite": True,
        "compressors": None,
    }
    metadata = {"num_tokens": 0, "num_rows": 0, "num_layers": model.num_layers}
    ensure_directory(cache_dir)
    if model.dtype == torch.bfloat16:
        dtype = np.uint16
    elif model.dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    zarr.create_array(
        cache_dir,
        name="activations",
        dtype=dtype,  # Store as uint16, since bfloat16 isn't supported directly
        shape=(
            model.num_layers,
            0,
            model.context_length,
            model.d_model,
        ),
        chunks=(
            1,
            inference_batch_size,
            model.context_length,
            model.d_model,
        ),
        shards=(
            model.num_layers,
            batches_per_file * inference_batch_size,
            model.context_length,
            model.d_model,
        ),
        **zarr_kwargs,
    )
    zarr.create_array(
        cache_dir,
        name="position_ids",
        dtype=np.int16,
        shape=(0, model.context_length),
        chunks=(inference_batch_size, model.context_length),
        shards=(batches_per_file * inference_batch_size, model.context_length),
        **zarr_kwargs,
    )
    json.dump(metadata, open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "w"))


def cache_on_disk(cache_dir: str, model: ReplacementModel, batch: DataBatch):
    metadata = json.load(open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "r"))

    activations, _ = _run_base_model(
        model,
        batch,
        {layer: model.transformer.h[layer] for layer in range(model.num_layers)},
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
    torch_stack = torch.stack(
        tuple(v[0] if isinstance(v, tuple) else v for v in activations.values())
    ).to("cpu")
    # Hack to handle bfloat16
    if cache["activations"].dtype == np.uint16:
        torch_stack = torch_stack.view(torch.uint16)
    cache["activations"][:, old_num_rows:new_num_rows, :, :] = torch_stack.numpy()

    metadata["num_tokens"] += batch.num_tokens
    metadata["num_rows"] = new_num_rows

    json.dump(metadata, open(f"{cache_dir}/{CACHE_METADATA_FILENAME}", "w"))


def build_cache(
    cache_dir: str,
    model: ReplacementModel,
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
        inference_batch_size,
        1,
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

    def _maybe_convert_bfloat16(t: torch.Tensor):
        if cache["activations"].dtype == np.uint16:
            return t.view(torch.bfloat16)
        return t

    return {
        layer: _maybe_convert_bfloat16(
            torch.from_numpy(
                cache["activations"][
                    layer, cache_offset : cache_offset + batch.batch_size, :, :
                ],
            )
        )
        for layer in range(num_layers)
    }
