from dataclasses import dataclass
from typing import Optional

import torch

from .decoder import Decoder, DecoderConfig
from .encoder import (  # noqa: F401
    Encoder,
    EncoderConfig,
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
    encoder: EncoderConfig
    decoder: DecoderConfig

    def __post_init__(self):
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)


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

    @torch.no_grad
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
        self.to(device)
        return self

    @_check_device
    def decode(self, x: torch.Tensor):
        return self.decoder(x)

    @_check_device
    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    @_check_device
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return (self.decode(self.encode(x)),)
