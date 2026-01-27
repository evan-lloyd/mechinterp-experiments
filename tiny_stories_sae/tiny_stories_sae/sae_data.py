from __future__ import annotations

import json
from collections import defaultdict
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import zarr

from .data_batch import DataBatch
from .ops import ensure_directory
from .sae import SAE


@dataclass(kw_only=True)
class SAEData:
    target_layer: int
    original: torch.Tensor
    features: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_eval: Literal["norm"] | Literal["KL"]


def get_logits(
    model: torch.nn.Module,
    data: SAEData,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    output = data.original
    if data.target_layer < model.config.num_layers:
        for layer in model.transformer.h[data.target_layer + 1 :]:
            output = layer(output, attention_mask=attention_mask, use_cache=False)[0]
        output = model.transformer.ln_f(output)

    # Now we have either ln_f output or logits already
    if output.shape[-1] != model.config.vocab_size:
        output = model.lm_head(output.view(-1, output.shape[-2], output.shape[-1]))

    return output


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
    metadata = {"num_tokens": 0, "num_rows": 0}
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
            model.config.num_layers + 1,
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
        {
            layer: model.transformer.h[layer]
            if layer < model.config.num_layers
            else model.transformer.ln_f
            for layer in range(model.config.num_layers + 1)
        },
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


def _ensure_tensor(
    maybe_tensor: Tuple[torch.Tensor, ...] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(maybe_tensor, tuple):
        return maybe_tensor[0]
    return maybe_tensor


def get_sae_data(
    base_model: torch.nn.Module,
    saes: Dict[int, SAE],
    batch: DataBatch,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    use_downstream_saes: bool = True,
    cache_dir: Optional[str] = None,
    cache_offset: Optional[int] = None,
    for_validation: bool = False,
) -> Tuple[Dict[Tuple[int, ...], Dict[int, SAEData]], Optional[torch.Tensor]]:
    if end_layer is None:
        end_layer = base_model.config.num_layers

    assert end_layer <= base_model.config.num_layers
    assert end_layer > start_layer
    assert start_layer >= 0

    if cache_dir is None:
        # TODO: early stopping
        activation_cache, logits = _run_base_model(
            base_model,
            batch,
            {
                layer: base_model.transformer.h[layer]
                if layer < base_model.config.num_layers
                else base_model.transformer.ln_f
                for layer in range(start_layer, end_layer + 1)
            },
        )
    else:
        # TODO: we should reconstruct the batch from cache, rather than needing the
        # batch object here
        try:
            cache = zarr.open_group(cache_dir, mode="r")
        except Exception:
            init_cache(
                base_model,
                cache_dir,
                base_model.num_layers,
                batch.batch_size,
            )
        activation_cache = {
            layer: torch.tensor(
                cache["activations"][
                    layer, cache_offset : cache_offset + batch.batch_size, :, :
                ]
            ).to(base_model.device)
            for layer in range(start_layer, end_layer + 1)
        }
        logits = None

    # Get data with no activation replacements
    baseline_data = {}

    # Handle model output as the n+1th "layer"
    if end_layer == base_model.config.num_layers:
        if logits is None:
            layer_norm_output = _ensure_tensor(activation_cache[end_layer])
            logits = base_model.lm_head(
                layer_norm_output.view(
                    (-1, layer_norm_output.shape[-2], layer_norm_output.shape[-1])
                )
            )
        final_layer_output = logits
        baseline_data[base_model.config.num_layers] = SAEData(
            target_layer=end_layer,
            original=final_layer_output,
            features=None,
            reconstruction=None,
            reconstruction_eval="KL",
        )

    for layer in range(
        min(end_layer, base_model.config.num_layers - 1), start_layer - 1, -1
    ):
        sae = saes[layer]
        original = _ensure_tensor(activation_cache[layer])
        features = sae.encode(original)
        reconstruction = _ensure_tensor(sae.decode(features))

        baseline_data[layer] = SAEData(
            target_layer=layer,
            original=original,
            features=features,
            reconstruction=reconstruction,
            reconstruction_eval="norm",
        )

    # Get data with activations where start layer replaced by its SAE
    # TODO: if using downstream SAEs, we should use a replacement model to make this cleaner
    replacement_data: Dict[Tuple[int, ...], Dict[int, SAEData]] = defaultdict(dict)
    replacement_data[(start_layer,)][start_layer] = copy(baseline_data[start_layer])

    # Make the "original" be our reconstruction to simplify the below loop regardless of whether
    # we are using downstream SAEs, since we *always* want the result after reconstructing the start layer.
    replacement_data[(start_layer,)][start_layer].original = replacement_data[
        (start_layer,)
    ][start_layer].reconstruction
    if use_downstream_saes:
        prev_layer_attr = "reconstruction"
    else:
        prev_layer_attr = "original"

    for layer in range(start_layer + 1, end_layer + 1):
        if for_validation:
            layer_input = baseline_data[layer - 1].reconstruction
        else:
            layer_input = getattr(
                replacement_data[(start_layer,)][layer - 1], prev_layer_attr
            )
        sae = saes.get(layer)
        if layer == base_model.config.num_layers:
            replacement_original = _ensure_tensor(
                base_model.transformer.ln_f(layer_input)
            )
            replacement_original = base_model.lm_head(
                replacement_original.view(
                    (
                        -1,
                        replacement_original.shape[-2],
                        replacement_original.shape[-1],
                    )
                )
            )
            reconstruction_eval = "KL"
        else:
            replacement_original = _ensure_tensor(
                base_model.transformer.h[layer](
                    layer_input,
                    attention_mask=batch.attention_mask.to(base_model.device),
                    use_cache=False,
                )
            )
            reconstruction_eval = "norm"
        if use_downstream_saes and sae is not None:
            replacement_features = sae.encode(replacement_original)
            replacement_reconstruction = _ensure_tensor(sae.decode(replacement_features))
        else:
            replacement_features = None
            replacement_reconstruction = None
        replacement_data[(layer - 1,) if for_validation else (start_layer,)][layer] = (
            SAEData(
                target_layer=layer,
                original=replacement_original,
                features=replacement_features,
                reconstruction=replacement_reconstruction,
                reconstruction_eval=reconstruction_eval,
            )
        )

    return {(): baseline_data, **replacement_data}, logits
