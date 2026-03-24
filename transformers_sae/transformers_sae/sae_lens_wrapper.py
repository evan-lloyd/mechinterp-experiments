import re
from importlib.resources import files
from types import MappingProxyType
from typing import Type

import torch
import yaml
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
from sae_lens.saes.sae import SAEMetadata

from transformers_sae.encoder import EncoderKind

from .sae import SAE as MySAE


class SAELensSAEWrapper(torch.nn.Module):
    def __init__(
        self,
        sae_lens_config: dict,
        sae_lens_sae: SAELens,
        device: torch.device,
        dtype: torch.dtype,
        target_l0: int,
    ):
        super().__init__()
        self.sae_lens_config = sae_lens_config
        self.sae_lens_sae = sae_lens_sae
        self.device = device
        self.dtype = dtype
        self.threshold_lr = 1e-4
        self.target_l0 = target_l0
        self.register_buffer(
            "threshold_offset",
            torch.tensor(
                0.0,
                dtype=torch.double,
                device=device,
                requires_grad=False,
            ),
            persistent=True,
        )

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
        decoder_result = self.decode(
            self.encode(x.to(self.dtype), token_mask=token_mask, should_cast=False),
            should_cast=False,
        ).to(x.dtype)
        # We want special tokens to "pass through" the SAE, since we don't train on them.
        decoder_result.view(x.shape[0] * x.shape[1], x.shape[2])[
            pass_through_positions, :
        ] = x.view(x.shape[0] * x.shape[1], x.shape[2])[pass_through_positions, :]
        return decoder_result

    def offload(self):
        self.to(torch.device("cpu"))

    def onload(self):
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.requires_grad_(mode)

    def train_encoder(self):
        self.train()

    def activation_thresholds(self):
        return (self.threshold_offset.item(),)

    def set_activation_threshold_lr(self, lr: float):
        self.threshold_lr = lr

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

        linear_out = None

        def _capture_pre_act(_module, _args, out):
            nonlocal linear_out
            linear_out = out

        orig_threshold = self.sae_lens_sae.threshold
        try:
            self.sae_lens_sae.threshold = torch.nn.Parameter(
                self.sae_lens_sae.threshold + self.threshold_offset
            )
            with self.sae_lens_sae.hook_sae_acts_pre.register_forward_hook(
                _capture_pre_act
            ):
                result = self.sae_lens_sae.encode(x)
        finally:
            self.sae_lens_sae.threshold = orig_threshold

        # Update threshold_multiplier?
        if self.training:
            with torch.no_grad():
                linear_out[~token_mask.bool()] = torch.finfo(linear_out.dtype).min

            num_tokens = linear_out.shape[0] * linear_out.shape[1]
            topk = torch.topk(
                # Offset by the (treated as constant) existing JumpReLU threshold, so that
                # we target our offset to get us the expected final L0.
                (linear_out - self.sae_lens_sae.threshold).view(-1),
                k=self.target_l0 * num_tokens,
                dim=-1,
                sorted=False,
            )

            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                self.threshold_offset = (
                    1 - self.threshold_lr
                ) * self.threshold_offset + self.threshold_lr * topk.values.min().to(
                    torch.double
                )

        if should_cast:
            result = result.to(out_dtype)
        return result


def wrap_sae_lens_pretrained(target_l0: int, **sae_lens_kwargs) -> SAELensSAEWrapper:
    saelens, saelens_config, _ = SAELens.from_pretrained_with_cfg_and_sparsity(
        **sae_lens_kwargs
    )
    return SAELensSAEWrapper(
        saelens_config,
        saelens,
        sae_lens_kwargs.get("device"),
        sae_lens_kwargs.get("dtype"),
        target_l0,
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
