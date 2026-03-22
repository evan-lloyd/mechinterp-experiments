from dataclasses import dataclass
from typing import Optional

import torch

from .decoder import Decoder, DecoderConfig
from .encoder import (  # noqa: F401
    BatchTopKActivationFunctionConfig,
    Encoder,
    EncoderConfig,
    EncoderKind,
    InteractionEncoder,
    InteractionEncoderConfig,
    ReluActivationFunctionConfig,
    TopKActivationFunctionConfig,
)


@dataclass
class SAEConfig:
    d_model: int
    d_sae: int
    device: torch.device
    train_dtype: torch.dtype
    inference_dtype: torch.dtype
    encoder: EncoderConfig
    decoder: DecoderConfig


def make_sae_config(
    *,
    d_model: int,
    d_sae: int,
    device: str | torch.device,
    train_dtype: torch.dtype,
    inference_dtype: torch.dtype,
    encoder_kind: EncoderKind,
    top_k: int | None = None,
    with_interaction: bool = False,
) -> SAEConfig:
    if encoder_kind == "relu":
        activation_config = ReluActivationFunctionConfig()
    elif encoder_kind == "topk":
        assert top_k is not None, "Must specify top_k for TopK SAE"
        activation_config = TopKActivationFunctionConfig(top_k)
    elif encoder_kind == "batch_topk":
        assert top_k is not None, "Must specify top_k for BatchTopK SAE"
        activation_config = BatchTopKActivationFunctionConfig(top_k)
    else:
        raise ValueError(f"Unknown encoder_kind {encoder_kind}")

    device = torch.device(device)

    if with_interaction:
        encoder_cfg_cls = InteractionEncoderConfig
    else:
        encoder_cfg_cls = EncoderConfig
    encoder_config = encoder_cfg_cls(
        d_model, d_sae, device, train_dtype, inference_dtype, activation_config
    )
    decoder_config = DecoderConfig(d_model, d_sae, device, train_dtype, inference_dtype)
    return SAEConfig(
        d_model,
        d_sae,
        device,
        train_dtype,
        inference_dtype,
        encoder_config,
        decoder_config,
    )


def _check_device(method):
    def wrapper(self, *args, **kwargs):
        if self._device_tracker.device != self.config.device:
            raise RuntimeError(
                f"SAE weights are on {self._device_tracker.device} but expected to be on {self.config.device}"
            )
        return method(self, *args, **kwargs)

    return wrapper


class SAE(torch.nn.Module):
    encoder: Encoder
    decoder: Decoder
    config: SAEConfig
    _device_tracker: torch.nn.Buffer

    def __init__(
        self,
        config: SAEConfig,
    ):
        super().__init__()
        self.config = config
        if isinstance(config.encoder, InteractionEncoderConfig):
            self.encoder = InteractionEncoder(config.encoder)
        else:
            self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)
        self._device_tracker = torch.nn.Buffer(torch.empty((0,), device="meta"))

    @torch.no_grad()
    def init_weights(
        self,
        init_from: Optional["SAE"] = None,
        to_device: Optional[str] = None,
    ):
        if init_from is None:
            self.decoder.init_weights(None, to_device)
            self.encoder.init_weights(self.decoder, to_device)
            self._device_tracker = torch.nn.Buffer(
                torch.empty((0,), device=to_device or self.config.device)
            )
        else:
            self.decoder.init_weights(init_from.decoder, to_device)
            self.encoder.init_weights(init_from.encoder, to_device)
            self._device_tracker = torch.nn.Buffer(
                torch.empty((0,), device=to_device or self.config.device)
            )

    def change_configured_device(self, device: torch.device | str):
        """Change our configured device, and move to it."""
        if isinstance(device, str):
            device = torch.device(device)
        self.config.device = device
        self.config.encoder.device = device
        self.config.decoder.device = device
        self.encoder.config.device = device
        self.decoder.config.device = device
        self.to(device)
        return self

    def offload(self):
        if self._device_tracker.device != torch.device("meta"):
            self.to(torch.device("cpu"))

    def onload(self):
        if self._device_tracker.device != torch.device("meta"):
            self.to(self.config.device)

    @_check_device
    def decode(self, x: torch.Tensor, should_cast: bool = True):
        return self.decoder(x, should_cast=should_cast)

    @_check_device
    def encode(
        self, x: torch.Tensor, token_mask: torch.Tensor, should_cast: bool = True
    ):
        return self.encoder(x, token_mask=token_mask, should_cast=should_cast)

    def pop_sae_kwargs(self, kwargs):
        return {
            "token_mask": kwargs.pop("token_mask"),
            "pass_through_positions": kwargs.pop("pass_through_positions"),
        }

    @_check_device
    def forward(
        self,
        x: torch.Tensor,
        *args,
        pass_through_positions: torch.Tensor,
        token_mask: torch.Tensor,
        **kwargs,
    ):
        decoder_result = self.decode(
            self.encode(
                x.to(self.encoder.dtype), token_mask=token_mask, should_cast=False
            ),
            should_cast=False,
        ).to(x.dtype)
        # We want special tokens to "pass through" the SAE, since we don't train on them.
        decoder_result.view(x.shape[0] * x.shape[1], x.shape[2])[
            pass_through_positions, :
        ] = x.view(x.shape[0] * x.shape[1], x.shape[2])[pass_through_positions, :]
        return decoder_result
