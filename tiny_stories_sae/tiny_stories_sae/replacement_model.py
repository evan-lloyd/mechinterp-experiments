from __future__ import annotations

import copy
from typing import Dict

import torch

from .sae import SAE


class GenericReplacementLayer(torch.nn.Module):
    def __init__(self, original_layer: torch.nn.Module, sae: torch.nn.Module):
        super().__init__()
        self.original_layer = original_layer
        self.sae = sae

    def forward(self, *args, **kwargs):
        original_output, *rest = self.original_layer(*args, **kwargs)
        if "position_ids" not in kwargs:
            if "cache_position" in kwargs:
                kwargs["position_ids"] = kwargs["cache_position"].unsqueeze(0)
            else:
                kwargs["position_ids"] = torch.arange(
                    args[0].shape[-2], device=args[0].device, dtype=torch.int64
                ).unsqueeze(0)
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
) -> ReplacementModel:
    # Shallow copy into a new module instance, adding ReplacementModel as a mixin
    new_instance = _shallow_copy_model(original)
    new_instance.__class__ = type(
        "ReplacementModel", (ReplacementModel, original.__class__), {}
    )
    for layer, sae in sae_layers.items():
        module_path = f"transformer.h.{layer}"
        original_layer = original.get_submodule(module_path)
        replacement_layer = GenericReplacementLayer(original_layer, sae)
        new_instance.set_submodule(module_path, replacement_layer)

    object.__setattr__(new_instance, "sae_layers", sae_layers)

    assert isinstance(new_instance, ReplacementModel)
    return new_instance
