from sae_lens.saes.sae import SAEMetadata
from types import MappingProxyType
from typing import Dict, Type

import torch
from sae_lens import SAE as SAELens
from sae_lens import (
    JumpReLUSAE,
    JumpReLUSAEConfig,
    StandardSAE,
    StandardSAEConfig,
    TopKSAE,
    TopKSAEConfig,
)
from sae_lens import SAEConfig as SAELensConfig

from transformers_sae.encoder import EncoderKind

from .sae import SAE as MySAE


class SAELensSAEWrapper(torch.nn.Module):
    def __init__(
        self,
        sae_lens_config: dict,
        sae_lens_sae: SAELens,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.sae_lens_config = sae_lens_config
        self.sae_lens_sae = sae_lens_sae
        self.device = device
        self.dtype = dtype

    @property
    def encoder(self):
        return self.sae_lens_sae.hook_sae_acts_post

    def pop_sae_kwargs(self, kwargs):
        return {
            "token_mask": kwargs.pop("token_mask"),
            "pass_through_positions": kwargs.pop("pass_through_positions"),
        }

    def forward(
        self,
        x: torch.Tensor,
        *args,
        pass_through_positions: torch.Tensor,
        token_mask: torch.Tensor,
        **kwargs,
    ):
        orig_dtype = x.dtype
        result = self.sae_lens_sae(x.to(self.dtype)).to(orig_dtype)
        # We want special tokens to "pass through" the SAE, since we don't train on them.
        result.view(x.shape[0] * x.shape[1], x.shape[2])[pass_through_positions, :] = (
            x.view(x.shape[0] * x.shape[1], x.shape[2])[pass_through_positions, :]
        )
        return result

    def offload(self):
        self.to(torch.device("cpu"))

    def onload(self):
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.requires_grad_(mode)

    def decode(self, x: torch.Tensor, should_cast: bool = True):
        orig_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self.sae_lens_sae.decode(x)
        if should_cast:
            result = result.to(orig_dtype)
        return result

    def encode(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        should_cast: bool = True,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self.sae_lens_sae.decode(x)
        if should_cast:
            result = result.to(out_dtype)
        return result


def wrap_sae_lens_pretrained(**kwargs) -> SAELensSAEWrapper:
    saelens, saelens_config, _ = SAELens.from_pretrained_with_cfg_and_sparsity(**kwargs)
    return SAELensSAEWrapper(
        saelens_config, saelens, kwargs.get("device"), kwargs.get("dtype")
    )


SAE_KIND_TO_SAE_LENS: MappingProxyType[EncoderKind, Type[SAELens]] = MappingProxyType(
    {
        "relu": StandardSAE,
        "topk": TopKSAE,
        "batch_topk": JumpReLUSAE,
    }
)
SAE_KIND_TO_SAE_CONFIG: MappingProxyType[EncoderKind, Type[SAELensConfig]] = (
    MappingProxyType(
        {
            "relu": StandardSAEConfig,
            "topk": TopKSAEConfig,
            "batch_topk": JumpReLUSAEConfig,
        }
    )
)


def to_sae_lens(my_sae: MySAE, layer: int) -> SAELens:
    encoder_kind = my_sae.config.encoder.activation_function.kind
    sae_lens_class = SAE_KIND_TO_SAE_LENS[encoder_kind]
    sae_lens_config = SAE_KIND_TO_SAE_CONFIG[encoder_kind](
        my_sae.config.d_model,
        my_sae.config.d_sae,
        str(my_sae.config.inference_dtype).replace("torch.", ""),
        "meta",
        apply_b_dec_to_input=False,
        normalize_activations="none",
        reshape_activations="none",
        metadata=SAEMetadata(hook_name=f"blocks.{layer}.hook_resid_post"),
    )

    state_dict = {
        "W_enc": my_sae.encoder.linear.weight.T,
        "b_enc": my_sae.encoder.linear.bias,
        "W_dec": my_sae.decoder.linear.weight.T,
        "b_dec": my_sae.decoder.linear.bias,
    }

    if encoder_kind == "batch_topk":
        state_dict["threshold"] = torch.full_like(
            my_sae.encoder.linear.bias, my_sae.encoder.activation[0].threshold.item()
        )

    sae_lens_sae = sae_lens_class(sae_lens_config)
    sae_lens_sae.cfg.device = my_sae.config.device
    sae_lens_sae.load_state_dict(state_dict, assign=True)
    sae_lens_sae.to(dtype=my_sae.config.inference_dtype, device=my_sae.config.device)
    return sae_lens_sae
