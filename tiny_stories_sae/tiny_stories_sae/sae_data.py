from __future__ import annotations

import json
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import zarr

from .data_batch import DataBatch
from .ops import ensure_directory
from .replacement_model import ReplacementModel
from .sae import SAE


@dataclass(kw_only=True)
class SAEData:
    target_layer: int
    original: torch.Tensor
    normalized_original: torch.Tensor
    features: torch.Tensor
    reconstruction: torch.Tensor
    denormalized_reconstruction: torch.Tensor
    reconstruction_eval: Literal["norm"] | Literal["KL"]
    norms: Optional[torch.Tensor] = None
    dense_encoding: Optional[torch.Tensor] = None
    dense_decoding: Optional[torch.Tensor] = None


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


@contextmanager
def _truncated_model(model: torch.nn.Module, start_layer: int, end_layer: int):
    class _EarlyStopping(Exception):
        pass

    class _StopLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise _EarlyStopping()

    patched_layers = torch.nn.ModuleList(
        model.transformer.h[max(start_layer, 0) : end_layer]
        + ([_StopLayer()] if end_layer < model.config.num_layers else [])
    )

    class _TruncatedTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, *args, **kwargs):
            residual = x
            for layer in patched_layers:
                residual = layer(
                    residual,
                    attention_mask=kwargs.get("attention_mask"),
                    use_cache=kwargs.get("use_cache"),
                )[0]
            if end_layer > model.config.num_layers:
                residual = model.transformer.ln_f(residual)
                residual = model.lm_head(
                    residual.view(-1, residual.shape[-2], residual.shape[-1])
                )
            return (residual,)

    try:
        orig_layers = model.transformer.h

        # If starting from embedding, we only need to handle early stopping
        if start_layer == -1:
            model.transformer.h = patched_layers
            truncated_model = model
        else:
            truncated_model = _TruncatedTransformer()
        try:
            yield truncated_model
        except _EarlyStopping:
            pass
    finally:
        model.transformer.h = orig_layers


def _run_model(
    model: torch.nn.Module,
    start_layer: int,
    end_layer: int,
    batch: DataBatch,
    starting_input: Optional[torch.Tensor],
    hooks: Dict[str, torch.nn.Module],
):
    if start_layer != -1:
        assert starting_input is not None, (
            "Must provide first layer's input if not starting from embedding"
        )

    activation_cache = {}

    def _hook_output(probe_name, _module, _args, out):
        activation_cache[probe_name] = out

    with ExitStack() as hook_stack:
        for hook_name, module in hooks.items():
            hook_stack.enter_context(
                module.register_forward_hook(partial(_hook_output, hook_name))
            )
        with _truncated_model(model, start_layer, end_layer) as truncated_model:
            if start_layer == -1:
                truncated_model(
                    input_ids=batch.input_ids.to(model.device),
                    position_ids=batch.position_ids.to(model.device),
                    attention_mask=batch.attention_mask.to(model.device),
                    use_cache=False,
                )
            else:
                truncated_model(
                    starting_input,
                    attention_mask=batch.attention_mask.to(model.device),
                    use_cache=False,
                )

    return activation_cache


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


def get_sae_data_with_replacement(
    base_model: torch.nn.Module,
    replacement_model: ReplacementModel,
    saes: Dict[int, SAE],
    batch: DataBatch,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    cache_dir: Optional[str] = None,
    cache_offset: Optional[int] = None,
):
    if end_layer is None:
        end_layer = base_model.config.num_layers + 1

    if cache_dir is None:
        with torch.no_grad():
            original_activations = _run_model(
                base_model,
                -1,
                end_layer,
                batch,
                None,
                {
                    layer: base_model.transformer.h[layer]
                    if layer < base_model.config.num_layers
                    else base_model.lm_head
                    for layer in range(0, end_layer)
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
        original_activations = {
            layer: torch.tensor(
                cache["activations"][
                    layer, cache_offset : cache_offset + batch.batch_size, :, :
                ]
            ).to(base_model.device)
            for layer in range(start_layer, end_layer)
        }
        # We cache the ln_f output instead of logits to save space
        if end_layer > base_model.config.num_layers:
            original_activations[end_layer - 1] = base_model.lm_head(
                original_activations[end_layer - 1].view(
                    -1,
                    original_activations[end_layer - 1].shape[-2],
                    original_activations[end_layer - 1].shape[-1],
                )
            )

    # TODO: this data structure isn't well aligned with the replacement model formulation, we should refactor
    baseline_data = {}
    # no_grad for the baseline
    with torch.no_grad():
        for layer in range(start_layer, end_layer):
            sae = saes.get(layer)
            original = _ensure_tensor(original_activations[layer])

            # TODO: support normalization
            normalized_original = original

            if sae is not None:
                features = sae.encode(normalized_original)
                reconstruction = _ensure_tensor(sae.decode(features))
                denormalized_reconstruction = reconstruction
            else:
                features = None
                reconstruction = None
                denormalized_reconstruction = None

            baseline_data[layer] = SAEData(
                target_layer=layer,
                original=original,
                normalized_original=normalized_original,
                features=features,
                reconstruction=reconstruction,
                denormalized_reconstruction=denormalized_reconstruction,
                reconstruction_eval="KL"
                if layer >= base_model.config.num_layers
                else "norm",
                norms=None,
                # TODO: support mixed sparse/dense
                dense_encoding=None,
                dense_decoding=None,
            )

    replacement_hooks = {}
    for layer in range(start_layer, end_layer):
        if layer < replacement_model.config.num_layers:
            if layer in replacement_model.sae_layers:
                sae = replacement_model.sae_layers[layer]
                replacement_hooks[f"{layer}.original"] = (
                    replacement_model.transformer.h[layer].original_layer
                )
                replacement_hooks[f"{layer}.features"] = sae.encoder_hook
                replacement_hooks[f"{layer}.reconstruction"] = (
                    replacement_model.transformer.h[layer]
                )
            else:
                replacement_hooks[f"{layer}.original"] = (
                    replacement_model.transformer.h[layer]
                )
        else:
            replacement_hooks[f"{layer}.original"] = replacement_model.lm_head

    # yes_grad for replacement model
    replacement_activations = _run_model(
        replacement_model,
        start_layer,
        end_layer,
        batch,
        _ensure_tensor(original_activations[start_layer]),
        replacement_hooks,
    )
    replacement_data = {}
    for layer in range(start_layer, end_layer):
        # This must always exist
        original = _ensure_tensor(replacement_activations[f"{layer}.original"])

        # TODO: support normalization
        normalized_original = original
        features = _ensure_tensor(replacement_activations.get(f"{layer}.features"))
        reconstruction = _ensure_tensor(
            replacement_activations.get(f"{layer}.reconstruction")
        )
        denormalized_reconstruction = reconstruction

        replacement_data[layer] = SAEData(
            target_layer=layer,
            original=original,
            normalized_original=normalized_original,
            features=features,
            reconstruction=reconstruction,
            denormalized_reconstruction=denormalized_reconstruction,
            reconstruction_eval="KL"
            if layer >= base_model.config.num_layers
            else "norm",
            norms=None,
            # TODO: support mixed sparse/dense
            dense_encoding=None,
            dense_decoding=None,
        )
    return baseline_data, replacement_data


def get_sae_data(
    base_model: torch.nn.Module,
    saes: Dict[int, SAE],
    batch: DataBatch,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    use_downstream_saes: bool = True,
    cache_dir: Optional[str] = None,
    cache_offset: Optional[int] = None,
    replace_previous_layer_only: bool = False,
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
            normalized_original=final_layer_output,
            features=None,
            reconstruction=None,
            reconstruction_eval="KL",
            denormalized_reconstruction=None,
            norms=None,
            dense_encoding=None,
            dense_decoding=None,
        )

    for layer in range(
        min(end_layer, base_model.config.num_layers - 1), start_layer - 1, -1
    ):
        sae = saes[layer]
        original = _ensure_tensor(activation_cache[layer])
        # with torch.no_grad():
        true_norms = torch.linalg.vector_norm(original, dim=-1)
        if sae.normalize_activations:
            est_norms = sae.normalizer(batch.position_ids.to(sae.device)).unsqueeze(-1)
            scaled_original = original / est_norms
        else:
            scaled_original = original
        if getattr(sae, "d_dense", None) is not None:
            features, dense_encoding = sae.encode(scaled_original)
            reconstruction, dense_decoding = sae.decode(features, dense_encoding)
        else:
            features = sae.encode(scaled_original)
            reconstruction = _ensure_tensor(sae.decode(features))
            dense_encoding = dense_decoding = None
        if sae.normalize_activations:
            denormalized_reconstruction = reconstruction * est_norms
        else:
            denormalized_reconstruction = reconstruction
        baseline_data[layer] = SAEData(
            target_layer=layer,
            original=original,
            normalized_original=scaled_original,
            features=features,
            reconstruction=reconstruction,
            denormalized_reconstruction=denormalized_reconstruction,
            reconstruction_eval="norm",
            norms=true_norms,
            dense_encoding=dense_encoding,
            dense_decoding=dense_decoding,
        )

    # Get data with activations where start layer replaced by its SAE
    # TODO: if using downstream SAEs, we should use a replacement model to make this cleaner
    replacement_data: Dict[Tuple[int, ...], Dict[int, SAEData]] = defaultdict(dict)
    replacement_data[(start_layer,)][start_layer] = copy(baseline_data[start_layer])

    # Make the "original" be our reconstruction to simplify the below loop regardless of whether
    # we are using downstream SAEs, since we *always* want the result after reconstructing the start layer.
    replacement_data[(start_layer,)][start_layer].original = replacement_data[
        (start_layer,)
    ][start_layer].denormalized_reconstruction
    if use_downstream_saes:
        prev_layer_attr = "denormalized_reconstruction"
    else:
        prev_layer_attr = "original"

    for layer in range(start_layer + 1, end_layer + 1):
        if replace_previous_layer_only:
            layer_input = baseline_data[layer - 1].denormalized_reconstruction
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
            # with torch.no_grad():
            replacement_true_norms = torch.linalg.vector_norm(
                replacement_original, dim=-1
            )
            if sae.normalize_activations:
                replacement_est_norms = sae.normalizer(
                    batch.position_ids.to(sae.device)
                ).unsqueeze(-1)
                replacement_scaled_original = (
                    replacement_original / replacement_est_norms
                )
            else:
                replacement_scaled_original = replacement_original
            if getattr(sae, "d_dense", None) is not None:
                replacement_features, replacement_dense_encoding = sae.encode(
                    replacement_scaled_original
                )
                replacement_reconstruction, replacement_dense_decoding = sae.decode(
                    features, replacement_dense_encoding
                )
            else:
                replacement_features = sae.encode(replacement_scaled_original)
                replacement_reconstruction = _ensure_tensor(sae.decode(features))
                replacement_dense_encoding = None
                replacement_dense_decoding = None
            if sae.normalize_activations:
                replacement_denormalized_reconstruction = (
                    replacement_reconstruction * replacement_est_norms
                )
            else:
                replacement_denormalized_reconstruction = replacement_reconstruction
        else:
            if sae is not None and sae.normalize_activations:
                replacement_scaled_original = replacement_original / sae.normalizer(
                    batch.position_ids.to(sae.device)
                ).unsqueeze(-1)
            else:
                replacement_scaled_original = replacement_original
            replacement_features = None
            replacement_reconstruction = None
            replacement_dense_encoding = None
            replacement_dense_decoding = None
            replacement_true_norms = None
            replacement_denormalized_reconstruction = None
            replacement_true_norms = None
        replacement_data[
            (layer - 1,) if replace_previous_layer_only else (start_layer,)
        ][layer] = SAEData(
            target_layer=layer,
            original=replacement_original,
            normalized_original=replacement_scaled_original,
            features=replacement_features,
            reconstruction=replacement_reconstruction,
            denormalized_reconstruction=replacement_denormalized_reconstruction,
            reconstruction_eval=reconstruction_eval,
            norms=replacement_true_norms,
            dense_encoding=replacement_dense_encoding,
            dense_decoding=replacement_dense_decoding,
        )

    return {(): baseline_data, **replacement_data}, logits
