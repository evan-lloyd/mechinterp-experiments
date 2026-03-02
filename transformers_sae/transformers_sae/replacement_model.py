from __future__ import annotations

import copy
from typing import Dict, Optional, Type, overload

import torch

from .data_batch import DataBatch
from .sae import SAE


class SAEReplacementLayer(torch.nn.Module):
    def __init__(self, original_layer: torch.nn.Module, sae: torch.nn.Module):
        super().__init__()
        self.original_layer = original_layer
        self.sae = sae

    def forward(self, *args, **kwargs):
        original_output = self.original_layer(*args, **kwargs)
        if isinstance(original_output, tuple):
            original_output, *rest = original_output
        else:
            rest = []
        reconstruction = self.sae(original_output, *args, **kwargs)
        if not isinstance(reconstruction, tuple):
            reconstruction = (reconstruction,)
        return reconstruction + tuple(rest)


class ReplacementModel:
    sae_layers: Dict[int, SAE]
    num_layers: int
    context_length: int
    d_model: int
    layer_path: str
    norm_path: str

    def __init__(self):
        raise NotImplementedError(
            "ReplacementModel should not be instantiated directly; use make_replacement_model instead"
        )

    def get_layer(self, i: int):
        if i < self.num_layers:
            return self.get_submodule(self.layer_path)[i]
        else:
            return self.lm_head

    def get_layer_args(self, *args, **kwargs):
        return args, dict(
            attention_mask=kwargs.get("attention_mask"),
            use_cache=kwargs.get("use_cache"),
        )

    def get_model_args(
        self,
        batch: DataBatch,
        model_input: torch.Tensor | None,
        start_at_embedding: bool,
    ):
        if start_at_embedding:
            input_args = []
            input_kwargs = batch.model_kwargs()
        else:
            assert model_input is not None, (
                "Must provide first layer's input if not starting from embedding"
            )
            input_args = [model_input]
            input_kwargs = {"attention_mask": batch.attention_mask}
        return input_args, input_kwargs


def _shallow_copy_model(source: torch.nn.Module):
    copied = copy.copy(source)
    copied._modules = {}
    copied._buffers = dict(**source._buffers)
    copied._parameters = dict(**source._parameters)
    copied._non_persistent_buffers_set = copy.copy(source._non_persistent_buffers_set)
    copied.training = source.training

    # Recursively copy all submodules
    for name, module in source._modules.items():
        if module is not None:
            copied._modules[name] = _shallow_copy_model(module)

    return copied


@overload
def make_replacement_model(
    original: ReplacementModel,
    sae_layers: Dict[int, SAE],
    layer_path: str = "transformer.h",
    norm_path: str = "transformer.ln_f",
) -> ReplacementModel: ...


@overload
def make_replacement_model(
    original: torch.nn.Module,
    sae_layers: Dict[int, SAE],
    num_layers: int,
    context_length: int,
    d_model: int,
    layer_path: str = "transformer.h",
    norm_path: str = "transformer.ln_f",
) -> ReplacementModel: ...


def make_replacement_model(
    original: torch.nn.Module | ReplacementModel,
    sae_layers: Dict[int, SAE],
    num_layers: Optional[int] = None,
    context_length: Optional[int] = None,
    d_model: Optional[int] = None,
    layer_path: str = "transformer.h",
    norm_path: str = "transformer.ln_f",
    replacement_class: Type[ReplacementModel] = ReplacementModel,
) -> ReplacementModel:
    # Shallow copy into a new module instance, adding ReplacementModel as a mixin
    new_instance = _shallow_copy_model(original)
    replacement_layers = {}

    if isinstance(original, ReplacementModel):
        layer_path = original.layer_path
        norm_path = original.norm_path

    for target_layer, sae in sae_layers.items():
        module_path = f"{layer_path}.{target_layer}"
        original_layer = original.get_submodule(module_path)
        replacement_layer = SAEReplacementLayer(original_layer, sae)
        new_instance.set_submodule(module_path, replacement_layer)
        replacement_layers[target_layer] = replacement_layer

    if not isinstance(original, ReplacementModel):
        new_instance.__class__ = type(
            replacement_class.__name__, (replacement_class, original.__class__), {}
        )
        object.__setattr__(new_instance, "num_layers", num_layers)
        object.__setattr__(new_instance, "context_length", context_length)
        object.__setattr__(new_instance, "d_model", d_model)
    else:
        object.__setattr__(new_instance, "num_layers", original.num_layers)
        object.__setattr__(new_instance, "context_length", original.context_length)
        object.__setattr__(new_instance, "d_model", original.d_model)
        for layer, sae in original.sae_layers.items():
            if layer not in replacement_layers:
                replacement_layers[layer] = sae

    object.__setattr__(new_instance, "sae_layers", replacement_layers)
    object.__setattr__(new_instance, "layer_path", layer_path)
    object.__setattr__(new_instance, "norm_path", norm_path)

    assert isinstance(new_instance, ReplacementModel)
    return new_instance
