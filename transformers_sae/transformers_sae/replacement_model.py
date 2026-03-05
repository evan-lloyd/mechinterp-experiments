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

    def __getattr__(self, name: str):
        if name == "forward":
            return object.__getattribute__(self, "forward")
        elif name in ("sae", "original_layer"):
            return object.__getattribute__(self, "_modules")[name]
        return getattr(self.original_layer, name)

    def forward(self, *args, pass_through_positions: torch.Tensor, **kwargs):
        original_output = self.original_layer(*args, **kwargs)
        tuple_expected = isinstance(original_output, tuple)
        if tuple_expected:
            original_output, *rest = original_output
        else:
            rest = []
        reconstruction = self.sae(
            original_output,
            *args,
            pass_through_positions=pass_through_positions,
            **kwargs,
        )
        if tuple_expected:
            return (reconstruction,) + tuple(rest)
        return reconstruction


class ReplacementModel:
    sae_layers: Dict[int, SAE]
    num_layers: int
    context_length: int
    d_model: int
    layer_path: str
    transformers_class: type

    def __init__(self):
        raise NotImplementedError(
            "ReplacementModel should not be instantiated directly; use make_replacement_model instead"
        )

    def get_logits(self, residual, **kwargs):
        residual = self.transformer.ln_f(residual)
        residual = self.lm_head(
            residual.view(-1, residual.shape[-2], residual.shape[-1])
        )
        return residual

    def get_layer(self, i: int):
        if i < self.num_layers:
            return self.get_submodule(self.layer_path)[i]
        else:
            return self.lm_head

    def get_layer_args(self, layer_idx, layer, *args, **kwargs):
        layer_kwargs = dict(
            attention_mask=kwargs.get("attention_mask"),
            use_cache=kwargs.get("use_cache"),
        )
        if layer_idx in self.sae_layers:
            layer_kwargs["pass_through_positions"] = kwargs.get("pass_through_positions")
            layer_kwargs["token_mask"] = kwargs.get("token_mask")
        return (args, layer_kwargs)

    def get_model_args(
        self,
        batch: DataBatch,
        model_input: torch.Tensor | None,
        start_at_embedding: bool,
    ):
        if start_at_embedding:
            input_args = []
            input_kwargs = {
                "input_ids": batch.input_ids,
                "position_ids": batch.position_ids,
                "attention_mask": batch.attention_mask,
            }
        else:
            assert model_input is not None, (
                "Must provide first layer's input if not starting from embedding"
            )
            input_args = [model_input]
            input_kwargs = {"attention_mask": batch.attention_mask}

        input_kwargs["pass_through_positions"] = batch.special_token_indices
        input_kwargs["token_mask"] = batch.token_mask
        return input_args, input_kwargs


class GemmaReplacement(ReplacementModel):
    def get_model_args(self, batch, model_input, start_at_embedding):
        input_args, input_kwargs = super().get_model_args(
            batch, model_input, start_at_embedding
        )
        if not start_at_embedding:
            input_kwargs["position_embeddings"] = self.model.rotary_emb(
                model_input, batch.position_ids
            )
        input_kwargs["attention_mask"] = {
            "full_attention": input_kwargs["attention_mask"],
            # TODO: this is only correct while we are using a context that's <= sliding attention
            "sliding_attention": input_kwargs["attention_mask"],
        }

        return input_args, input_kwargs

    def get_logits(self, residual, **kwargs):
        logits = self.lm_head(self.model.norm(residual))
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits

    def get_layer_args(self, layer_idx, layer, *args, **kwargs):
        layer_args, layer_kwargs = super().get_layer_args(
            layer_idx, layer, *args, **kwargs
        )
        layer_kwargs["position_embeddings"] = kwargs.get("position_embeddings")
        layer_kwargs["attention_mask"] = kwargs.get("attention_mask")[
            layer.attention_type
        ]
        return layer_args, layer_kwargs


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
) -> ReplacementModel: ...


@overload
def make_replacement_model(
    original: torch.nn.Module,
    sae_layers: Dict[int, SAE],
    num_layers: int,
    context_length: int,
    d_model: int,
    layer_path: str = "transformer.h",
) -> ReplacementModel: ...


def make_replacement_model(
    original: torch.nn.Module | ReplacementModel,
    sae_layers: Dict[int, SAE],
    num_layers: Optional[int] = None,
    context_length: Optional[int] = None,
    d_model: Optional[int] = None,
    layer_path: str = "transformer.h",
    replacement_class: Type[ReplacementModel] = ReplacementModel,
) -> ReplacementModel:
    # Shallow copy into a new module instance, adding ReplacementModel as a mixin
    new_instance = _shallow_copy_model(original)
    replacement_layers = {}

    if isinstance(original, ReplacementModel):
        layer_path = original.layer_path

    for target_layer, sae in sae_layers.items():
        module_path = f"{layer_path}.{target_layer}"
        original_layer = original.get_submodule(module_path)
        replacement_layer = SAEReplacementLayer(original_layer, sae)
        new_instance.set_submodule(module_path, replacement_layer)
        replacement_layers[target_layer] = replacement_layer

    if not isinstance(original, ReplacementModel):
        new_instance.__class__ = type(
            f"{replacement_class.__name__}Instance",
            (replacement_class, original.__class__),
            {},
        )
        object.__setattr__(new_instance, "num_layers", num_layers)
        object.__setattr__(new_instance, "context_length", context_length)
        object.__setattr__(new_instance, "d_model", d_model)
        object.__setattr__(new_instance, "transformers_class", original.__class__)
    else:
        object.__setattr__(new_instance, "num_layers", original.num_layers)
        object.__setattr__(new_instance, "context_length", original.context_length)
        object.__setattr__(new_instance, "d_model", original.d_model)
        object.__setattr__(
            new_instance, "transformers_class", original.transformers_class
        )
        for layer, sae in original.sae_layers.items():
            if layer not in replacement_layers:
                replacement_layers[layer] = sae

    object.__setattr__(new_instance, "sae_layers", replacement_layers)
    object.__setattr__(new_instance, "layer_path", layer_path)

    assert isinstance(new_instance, ReplacementModel)
    return new_instance
