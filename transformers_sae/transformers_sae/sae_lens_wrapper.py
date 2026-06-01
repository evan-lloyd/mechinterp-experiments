import re
from importlib.resources import files
from types import MappingProxyType
from typing import Any, Dict, Type

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

from transformers_sae.decoder import DecoderConfig
from transformers_sae.encoder import (
    ActivationKind,
    EncoderConfig,
    JumpReluActivationFunctionConfig,
)

from .sae import SAE as MySAE
from .sae import SAEConfig


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
