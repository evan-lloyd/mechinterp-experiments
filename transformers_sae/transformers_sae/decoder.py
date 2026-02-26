import torch
from dataclasses import dataclass
from typing import Union


@dataclass
class DecoderConfig:
    d_model: int
    d_sae: int
    device: torch.device
    dtype: torch.dtype


class Decoder(torch.nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(
            config.d_sae, config.d_model, device="meta", dtype=self.config.dtype
        )

    @torch.no_grad()
    def init_weights(
        self, init_from: Union["Decoder", None] = None, to_device: str | None = None
    ):
        if init_from is None:
            self.linear = torch.nn.Linear(
                self.config.d_sae,
                self.config.d_model,
                device=to_device or self.config.device,
                dtype=self.config.dtype,
            )
        elif isinstance(init_from, Decoder):
            self.linear.weight = torch.nn.Parameter(
                init_from.linear.weight.to(to_device or self.config.device, copy=True)
                .detach()
                .contiguous()
            )
            self.linear.bias = torch.nn.Parameter(
                init_from.linear.bias.to(to_device or self.config.device, copy=True)
                .detach()
                .contiguous()
            )

        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def forward(self, x: torch.Tensor, should_cast: bool = True):
        orig_dtype = x.dtype
        if should_cast:
            x = x.to(self.config.dtype)
        result = self.linear(x)
        if should_cast:
            result = result.to(orig_dtype)
        return result
