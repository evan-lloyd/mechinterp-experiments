from collections import defaultdict
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import torch

from .data_batch import DataBatch
from .ops import ensure_tensor
from .replacement_model import ReplacementModel
from .truncated_model import truncated_model


@dataclass(kw_only=True)
class ActivationBatch:
    layer_output: torch.Tensor | None = None
    sae_features: torch.Tensor | None = None
    sae_output: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None


@dataclass
class TrainingBatch:
    input_data: DataBatch
    replacement_activations: Dict[int, ActivationBatch]
    baseline_activations: Dict[int, ActivationBatch]
    replacement_layers: List[int]


def _run_replacement_model(
    model: ReplacementModel,
    hooks: Dict[str, torch.nn.Module],
    batch: DataBatch,
    start_input: Optional[torch.Tensor] = None,
    start_layer: int = -1,
    end_layer: Optional[int] = None,
    start_at_sae: bool = False,
):
    input_args, input_kwargs = model.get_base_model_args(
        batch, start_input, start_layer == -1
    )

    activation_cache = {}

    def _hook_output(probe_key, _module, _args, out):
        activation_cache[probe_key] = out

    if end_layer is None:
        end_layer = model.num_layers + 1
    with (
        ExitStack() as hook_stack,
        truncated_model(
            model,
            start_layer,
            end_layer,
            start_at_sae,
            model.get_sae_kwargs(batch),
        ) as model_to_run,
    ):
        for hook_name, module in hooks.items():
            # Special case, since we make a new model on the fly
            if module is model:
                module = model_to_run
            hook_stack.enter_context(
                module.register_forward_hook(partial(_hook_output, hook_name))
            )
        model_to_run(
            *input_args,
            **input_kwargs,
            use_cache=False,
        )

    return {k: ensure_tensor(v) for k, v in activation_cache.items()}


def make_activation_batch(
    replacement_model: ReplacementModel,
    request_spec: List[Tuple[int, Literal["layer", "sae"]]],
    batch: DataBatch,
    start_input: Optional[torch.Tensor] = None,
    start_layer: int = -1,
    end_layer: Optional[int] = None,
    start_at_sae: bool = False,
) -> Dict[int, ActivationBatch]:
    hooks = {}
    request_attrs_by_layer = defaultdict(list)
    for r in request_spec:
        if r[0] == replacement_model.num_layers:
            attrs = ("log_probs",)
            submodule_paths = ("",)
        else:
            if r[1] == "sae":
                assert r[0] in replacement_model.sae_layers, f"No SAE at layer {r[0]}"
                attrs = ("sae_output", "sae_features")
                submodule_paths = (
                    f"{replacement_model.layer_path}.{r[0]}.sae",
                    f"{replacement_model.layer_path}.{r[0]}.sae.encoder",
                )
            elif r[1] == "layer":
                attrs = ("layer_output",)
                if (
                    hasattr(replacement_model, "sae_layers")
                    and r[0] in replacement_model.sae_layers
                ):
                    submodule_paths = (
                        f"{replacement_model.layer_path}.{r[0]}.original_layer",
                    )
                else:
                    submodule_paths = (f"{replacement_model.layer_path}.{r[0]}",)
        for attr, submodule in zip(attrs, submodule_paths):
            hooks[(r[0], attr)] = replacement_model.get_submodule(submodule)
        request_attrs_by_layer[r[0]].extend(attrs)

    model_run = _run_replacement_model(
        replacement_model,
        hooks,
        batch,
        start_input,
        start_layer,
        end_layer,
        start_at_sae,
    )
    if (replacement_model.num_layers, "log_probs") in model_run:
        model_run[(replacement_model.num_layers, "log_probs")] = ensure_tensor(
            model_run[(replacement_model.num_layers, "log_probs")]
        ).log_softmax(-1)
    result = {}
    for layer in sorted(list(request_attrs_by_layer.keys())):
        result[layer] = ActivationBatch(
            **{
                k: ensure_tensor(model_run[(layer, k)])
                for k in request_attrs_by_layer[layer]
            }
        )
    return result


def make_batch_for_evals(
    base_model: ReplacementModel,
    replacement_model: ReplacementModel,
    training_batch: TrainingBatch,
    wanted_layers: List[int],
) -> TrainingBatch:
    start_layer = min(wanted_layers)
    end_layer = max(wanted_layers) + 1
    baseline_activations = copy(training_batch.baseline_activations)
    replacement_activations = copy(training_batch.replacement_activations)

    needs_baseline_layers = [
        layer for layer in wanted_layers if layer not in baseline_activations
    ]

    # Get activation data for any layers we don't already have it for
    if needs_baseline_layers:
        if needs_baseline_layers[0] - 1 in training_batch.baseline_activations:
            prev_layer_data = training_batch.baseline_activations[
                needs_baseline_layers[0] - 1
            ].layer_output
        else:
            prev_layer_data = None
        new_baseline_run = _run_replacement_model(
            base_model,
            {layer: base_model.get_layer(layer) for layer in needs_baseline_layers},
            training_batch.input_data,
            start_input=prev_layer_data,
            start_layer=needs_baseline_layers[0] if prev_layer_data is not None else -1,
            end_layer=end_layer,
        )
        for layer in needs_baseline_layers:
            if layer == base_model.num_layers:
                baseline_activations[layer] = ActivationBatch(
                    log_probs=ensure_tensor(new_baseline_run[layer]).log_softmax(-1)
                )
            else:
                baseline_activations[layer] = ActivationBatch(
                    layer_output=new_baseline_run[layer]
                )

    # The existing batch may have not used a full replacement model. We can still shave off time
    # by starting at the first layer that wasn't replaced.
    missing_replacement_model_layers = sorted(
        list(
            # We never replace the logits layer, so don't count it as missing
            set(range(start_layer, min(end_layer, replacement_model.num_layers)))
            - set(training_batch.replacement_layers)
        )
    )
    needs_replacement_layers = [
        layer
        for layer in range(start_layer, end_layer)
        # If we never collected it...
        if layer not in replacement_activations
        # Or, we *did*, but we skipped some SAEs, so we need to re-run for evals
        or (
            missing_replacement_model_layers
            and layer >= missing_replacement_model_layers[0]
            and layer in replacement_model.sae_layers
        )
    ]

    if needs_replacement_layers:
        hooks = {}
        for layer in needs_replacement_layers:
            if layer not in wanted_layers:
                continue
            if layer >= replacement_model.num_layers:
                hooks[layer] = replacement_model
            elif layer in replacement_model.sae_layers:
                hooks[(layer, "layer_output")] = replacement_model.get_layer(
                    layer
                ).original_layer
                hooks[(layer, "sae_features")] = replacement_model.get_layer(
                    layer
                ).sae.encoder
                hooks[(layer, "sae_output")] = replacement_model.get_layer(layer).sae
            else:
                hooks[(layer, "layer_output")] = replacement_model.get_layer(layer)
        # The previously used replacement model was good up until this layer, so we can start here
        # if we have the data. Otherwise, we need to start at the start_layer from baseline data.
        if needs_replacement_layers[0] - 1 in replacement_activations:
            replacement_start_at_layer = needs_replacement_layers[0]
            if (
                replacement_activations[needs_replacement_layers[0] - 1].sae_output
                is not None
            ):
                replacement_start_input = replacement_activations[
                    needs_replacement_layers[0] - 1
                ].sae_output
                replacement_start_at_sae = False
            else:
                replacement_start_input = replacement_activations[
                    needs_replacement_layers[0] - 1
                ].layer_output
                replacement_start_at_sae = True
        else:
            replacement_start_at_layer = start_layer
            replacement_start_input = baseline_activations[
                needs_replacement_layers[0]
            ].layer_output
            replacement_start_at_sae = True
        new_replacement_run = _run_replacement_model(
            replacement_model,
            hooks,
            training_batch.input_data,
            start_input=replacement_start_input,
            start_layer=replacement_start_at_layer,
            end_layer=end_layer,
            start_at_sae=replacement_start_at_sae,
        )
        # Handle edge case. We didn't actually run that layer, so we need to populate it
        # from the baseline data we started with.
        if replacement_start_at_sae:
            new_replacement_run[(start_layer, "layer_output")] = replacement_start_input

        for layer in needs_replacement_layers:
            if layer not in wanted_layers:
                continue
            if layer == base_model.num_layers:
                replacement_activations[layer] = ActivationBatch(
                    log_probs=ensure_tensor(new_replacement_run[layer]).log_softmax(-1),
                )
            else:
                replacement_activations[layer] = ActivationBatch(
                    layer_output=new_replacement_run[(layer, "layer_output")],
                    sae_features=new_replacement_run.get((layer, "sae_features")),
                    sae_output=new_replacement_run.get((layer, "sae_output")),
                )

    return TrainingBatch(
        training_batch.input_data,
        replacement_activations,
        baseline_activations,
        replacement_layers=list(range(start_layer, end_layer)),
    )
