from __future__ import annotations

import copy
from typing import Dict

import torch

from .sae import SAE


class SAEReplacementLayer(torch.nn.Module):
    def __init__(self, original_layer: torch.nn.Module, sae: torch.nn.Module):
        super().__init__()
        self.original_layer = original_layer
        self.sae = sae

    def forward(self, *args, **kwargs):
        original_output, *rest = self.original_layer(*args, **kwargs)
        reconstruction = self.sae(original_output, *args, **kwargs)
        if not isinstance(reconstruction, tuple):
            reconstruction = (reconstruction,)
        return reconstruction + tuple(rest)


class ReplacementModel:
    sae_layers: Dict[int, SAE]

    def __init__(self):
        raise NotImplementedError(
            "ReplacementModel should not be instantiated directly; use make_replacement_model instead"
        )


def _shallow_copy_model(source: torch.nn.Module):
    copied = copy.copy(source)
    copied._modules = {}
    copied._buffers = dict(**source._buffers)

    # Recursively copy all submodules
    for name, module in source._modules.items():
        if module is not None:
            copied._modules[name] = _shallow_copy_model(module)

    return copied


def make_replacement_model(
    original: torch.nn.Module,
    sae_layers: Dict[int, SAE],
    module_path_prefix: str = "transformer.h.",
) -> ReplacementModel:
    # Shallow copy into a new module instance, adding ReplacementModel as a mixin
    new_instance = _shallow_copy_model(original)
    new_instance.__class__ = type(
        "ReplacementModel", (ReplacementModel, original.__class__), {}
    )
    replacement_layers = {}
    for target_layer, sae in sae_layers.items():
        module_path = f"{module_path_prefix}{target_layer}"
        original_layer = original.get_submodule(module_path)
        replacement_layer = SAEReplacementLayer(original_layer, sae)
        new_instance.set_submodule(module_path, replacement_layer)
        replacement_layers[target_layer] = replacement_layer

    object.__setattr__(new_instance, "sae_layers", replacement_layers)

    assert isinstance(new_instance, ReplacementModel)
    return new_instance
