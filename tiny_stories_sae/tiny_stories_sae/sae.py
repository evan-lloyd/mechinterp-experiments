from dataclasses import dataclass
from typing import Optional

import torch

from .decoder import Decoder, DecoderConfig
from .encoder import (
    Encoder,
    EncoderConfig,
    InteractionEncoder,
    InteractionEncoderConfig,
    ReluActivationFunctionConfig,
    TopKActivationFunctionConfig,
)  # noqa: F401


@dataclass
class SAEConfig:
    d_model: int
    d_sae: int
    device: torch.device
    encoder: EncoderConfig
    decoder: DecoderConfig


class SAE(torch.nn.Module):
    encoder: Encoder
    decoder: Decoder
    config: SAEConfig

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

    @torch.no_grad
    def init_weights(
        self,
        init_from: Optional["SAE"] = None,
    ):
        if init_from is None:
            self.decoder.init_weights()
            self.encoder.init_weights(self.decoder)
        else:
            self.decoder.init_weights(init_from.decoder)
            self.encoder.init_weights(init_from.encoder)

    def decode(self, x: torch.Tensor):
        return self.decoder(x)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return (self.decode(self.encode(x)),)
