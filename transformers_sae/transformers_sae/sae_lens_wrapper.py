from transformers_sae.decoder import DecoderConfig
import re
from importlib.resources import files
from types import MappingProxyType
from typing import Type, Any, Dict

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

from transformers_sae.encoder import (
    ActivationKind,
    EncoderConfig,
    JumpReluActivationFunctionConfig,
)

from .sae import SAE as MySAE, SAEConfig


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
                dtype=torch.float64 if device.type != "mps" else torch.float32,
                device=device,
                requires_grad=False,
            ),
            persistent=True,
        )
        self.encoder.scale = torch.nn.Parameter(torch.ones((1,), device=device))
        self.training_activations = True

        def interaction_params():
            yield ("scale", self.encoder.scale)

        def train_activations():
            self.training_activations = True
            self.encoder.scale.requires_grad_(True)

        self.encoder.interaction_params = interaction_params
        self.encoder.train_activations = train_activations

    @property
    def encoder(self):
        return self.sae_lens_sae.hook_sae_acts_post

    def pop_sae_kwargs(self, kwargs):
        return {
            "token_mask": kwargs.pop("token_mask"),
            "pass_through_positions": kwargs.pop("pass_through_positions"),
            "feature_soft_cap": kwargs.pop("feature_soft_cap", None),
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
        self.training_activations = True
        return decoder_result

    def offload(self):
        self.to(torch.device("cpu"))

    def onload(self):
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.training_activtions = mode
        self.encoder.scale.requires_grad_(mode)
        self.requires_grad_(mode)

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
            linear_out = out * self.encoder.scale.to(out.dtype)
            return linear_out

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
        if self.training_activations:
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
                    self.threshold_offset.dtype
                )

        # Apply softcap to features to avoid exploding reconstructions
        result = 1000.0 * torch.tanh(result / 1000.0)

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
        torch.device(sae_lens_kwargs.get("device")),
        sae_lens_kwargs.get("dtype"),
        target_l0,
    )


@torch.no_grad()
def convert_sae_lens(
    target_k: int, saelens: SAELens, saelens_config: Dict[str, Any], device=None
):
    device = torch.device(device or saelens_config["device"])

    if saelens_config["architecture"] == "jumprelu":
        assert saelens_config["apply_b_dec_to_input"] == False, (
            "Not handled: apply_b_dec_to_input"
        )
        assert saelens_config["normalize_activations"] == "none", (
            "Not handled: normalize_activations"
        )

        encoder_config = EncoderConfig(
            d_model=saelens_config["d_in"],
            d_sae=saelens_config["d_sae"],
            device=device,
            train_dtype=getattr(torch, saelens_config["dtype"]),
            inference_dtype=torch.bfloat16,
            activation_function=JumpReluActivationFunctionConfig(
                d_sae=saelens_config["d_sae"],
                k=target_k,
            ),
        )
    else:
        raise NotImplementedError(
            f"Conversion from sae_lens architecture {saelens_config['architecture']} not implemented"
        )

    decoder_config = DecoderConfig(
        d_model=saelens_config["d_in"],
        d_sae=saelens_config["d_sae"],
        device=device,
        train_dtype=getattr(torch, saelens_config["dtype"]),
        inference_dtype=torch.bfloat16,
    )

    config = SAEConfig(
        d_model=saelens_config["d_in"],
        d_sae=saelens_config["d_sae"],
        device=device,
        train_dtype=getattr(torch, saelens_config["dtype"]),
        inference_dtype=torch.bfloat16,
        encoder=encoder_config,
        decoder=decoder_config,
    )
    sae = MySAE(config)
    sae._device_tracker = torch.nn.Buffer(torch.empty((0,), device=config.device))
    sae.encoder.linear.weight = torch.nn.Parameter(
        saelens.W_enc.T.to(device, copy=True).detach().contiguous()
    )
    sae.encoder.linear.bias = torch.nn.Parameter(
        saelens.b_enc.to(device, copy=True).detach().contiguous()
    )
    sae.decoder.linear.weight = torch.nn.Parameter(
        saelens.W_dec.T.to(device, copy=True).detach().contiguous()
    )
    sae.decoder.linear.bias = torch.nn.Parameter(
        saelens.b_dec.to(device, copy=True).detach().contiguous()
    )
    sae.encoder.activation[0].init_weights()
    sae.encoder.activation[0].threshold = (
        saelens.threshold.to(device, copy=True).detach().contiguous()
    )

    return sae


def convert_sae_lens_pretrained(target_k: int, **sae_lens_kwargs) -> MySAE:
    saelens, saelens_config, _ = SAELens.from_pretrained_with_cfg_and_sparsity(
        **sae_lens_kwargs
    )
    return convert_sae_lens(target_k, saelens, saelens_config)


SAE_KIND_TO_SAE_LENS: MappingProxyType[ActivationKind, Type[SAELens]] = (
    MappingProxyType(
        {
            "relu": StandardSAE,
            "topk": TopKSAE,
            "batch_topk": JumpReLUSAE,
        }
    )
)
SAE_KIND_TO_SAE_CONFIG: MappingProxyType[ActivationKind, Type[SAELensConfig]] = (
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
