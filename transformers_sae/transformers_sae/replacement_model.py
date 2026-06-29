from __future__ import annotations

from typing import Dict, Optional, Type, overload

import torch

from .data_batch import DataBatch
from .ops import _shallow_copy_model
from .sae import SAE


class SAEReplacementLayer(torch.nn.Module):
    sae: SAE
    original_layer: torch.nn.Module

    def __init__(self, original_layer: torch.nn.Module, sae: SAE):
        super().__init__()
        self.original_layer = original_layer
        self.sae = sae

    def __getattr__(self, name: str):
        if name == "forward":
            return object.__getattribute__(self, "forward")
        elif name in ("sae", "original_layer"):
            return object.__getattribute__(self, "_modules")[name]
        return getattr(self.original_layer, name)

    def forward(self, *args, **kwargs):
        additional_sae_kwargs = self.sae.pop_sae_kwargs(kwargs)
        bypass_sae = kwargs.pop("bypass_sae", False)

        original_output = self.original_layer(*args, **kwargs)

        if bypass_sae:
            return original_output

        tuple_expected = isinstance(original_output, tuple)
        if tuple_expected:
            original_output, *rest = original_output
        else:
            rest = []
        reconstruction = self.sae(
            original_output,
            *args,
            **kwargs,
            **additional_sae_kwargs,
        )
        if tuple_expected:
            return (reconstruction,) + tuple(rest)
        return reconstruction


class ReplacementModel:
    sae_layers: Dict[int, SAEReplacementLayer]
    num_layers: int
    context_length: int
    d_model: int
    layer_path: str
    transformers_class: type
    layer_class: type

    def post_init(self):
        pass

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
            attention_mask=kwargs["attention_mask"],
            use_cache=kwargs["use_cache"],
        )
        return (args, layer_kwargs)

    def get_sae_kwargs(
        self,
        batch: DataBatch,
    ):
        return {
            "pass_through_positions": batch.special_token_indices,
            "token_mask": batch.token_mask,
        }

    def get_base_model_args(
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

        return input_args, input_kwargs


class GemmaReplacement(ReplacementModel):
    def post_init(self):
        # So we don't crash on virtual logits layer
        if hasattr(self.config, "layer_types"):
            self.config.layer_types.append(self.config.layer_types[-1])

    def get_base_model_args(self, batch, model_input, start_at_embedding):
        input_args, input_kwargs = super().get_base_model_args(
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

    def get_logits(self, residual, for_token_ids=slice(None), **kwargs):
        residual = self.model.norm(residual)
        # logits = self.lm_head(residual)
        logits = residual @ self.lm_head.weight[for_token_ids, :].T
        # logits = torch.utils.checkpoint.checkpoint(
        #     self.lm_head, residual, use_reentrant=False
        # )
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits

    def get_layer_args(self, layer_idx, layer, *args, **kwargs):
        layer_args, layer_kwargs = super().get_layer_args(
            layer_idx, layer, *args, **kwargs
        )
        layer_kwargs["position_embeddings"] = kwargs["position_embeddings"]
        if hasattr(layer, "attention_type"):
            layer_kwargs["attention_mask"] = kwargs["attention_mask"][
                layer.attention_type
            ]
        else:
            layer_kwargs["attention_mask"] = kwargs["attention_mask"][
                layer.self_attn.layer_type
            ]
        return layer_args, layer_kwargs


@overload
def make_replacement_model(
    original: ReplacementModel,
    sae_layers: Dict[int, SAE],
    *,
    layer_path: str = "transformer.h",
    override_replacement_layers: bool = False,
) -> ReplacementModel: ...


@overload
def make_replacement_model(
    original: torch.nn.Module,
    sae_layers: Dict[int, SAE],
    *,
    num_layers: int,
    context_length: int,
    d_model: int,
    layer_path: str = "transformer.h",
    replacement_class: Type[ReplacementModel] = ReplacementModel,
    layer_class: Type[SAEReplacementLayer] = SAEReplacementLayer,
) -> ReplacementModel: ...


# TODO: all the customization (d_model, layer_path, etc) should be handled by the subclass
def make_replacement_model(
    original: torch.nn.Module | ReplacementModel,
    sae_layers: Dict[int, SAE],
    *,
    num_layers: Optional[int] = None,
    context_length: Optional[int] = None,
    d_model: Optional[int] = None,
    layer_path: str = "transformer.h",
    replacement_class: Type[ReplacementModel] = ReplacementModel,
    layer_class: Type[SAEReplacementLayer] = SAEReplacementLayer,
    override_replacement_layers: bool = False,
) -> ReplacementModel:
    # Shallow copy into a new module instance, adding ReplacementModel as a mixin
    new_instance = _shallow_copy_model(original)
    replacement_layers = {}

    if isinstance(original, ReplacementModel):
        layer_path = original.layer_path
        layer_class = original.layer_class
        layer_range = range(original.num_layers)
    else:
        layer_range = sae_layers.keys()

    for target_layer in layer_range:
        module_path = f"{layer_path}.{target_layer}"
        original_submodule = original.get_submodule(module_path)

        if isinstance(original_submodule, SAEReplacementLayer):
            original_layer = original_submodule.original_layer
        else:
            original_layer = original_submodule

        if target_layer in sae_layers:
            # Always make a fresh SAEReplacementLayer for SAEs we explicitly wanted
            replacement_layer = layer_class(original_layer, sae_layers[target_layer])
        elif override_replacement_layers:
            replacement_layer = original_layer
        else:
            # Otherwise, we may or may not be inheriting an existing replacement layer from original
            replacement_layer = original_submodule

        if isinstance(replacement_layer, SAEReplacementLayer):
            replacement_layers[target_layer] = replacement_layer

        new_instance.set_submodule(module_path, replacement_layer)

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

    object.__setattr__(new_instance, "sae_layers", replacement_layers)
    object.__setattr__(new_instance, "layer_path", layer_path)
    object.__setattr__(new_instance, "layer_class", layer_class)
    new_instance.post_init()

    assert isinstance(new_instance, ReplacementModel)
    return new_instance
