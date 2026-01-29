from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from typing import Literal, Union

import torch


if TYPE_CHECKING:
    from .decoder import Decoder


@dataclass
class ActivationFunctionConfig:
    kind: Literal["topk", "relu"] = field(init=False)


@dataclass
class TopKActivationFunctionConfig(ActivationFunctionConfig):
    k: int

    def __post_init__(self):
        self.kind = "topk"


@dataclass
class ReluActivationFunctionConfig(ActivationFunctionConfig):
    def __post_init__(self):
        self.kind = "relu"


@dataclass
class EncoderConfig:
    d_model: int
    d_sae: int
    device: torch.device
    activation_function: ActivationFunctionConfig


@dataclass
class InteractionEncoderConfig(EncoderConfig):
    n_interaction_iterations: int = 1


class Encoder(torch.nn.Module):
    def __init__(
        self,
        config: EncoderConfig,
    ):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(config.d_model, config.d_sae, device="meta")

    def init_weights(self, init_from: Union["Encoder", "Decoder", None] = None):
        from .decoder import Decoder

        if init_from is None:
            raise ValueError(
                "Encoder weights must be initialized from existing encoder or decoder"
            )

        if isinstance(init_from, Encoder):
            self.linear.weight = torch.nn.Parameter(
                init_from.linear.weight.clone().detach().contiguous()
            )
            self.linear.bias = torch.nn.Parameter(
                init_from.linear.bias.clone().detach().contiguous()
            )
        elif isinstance(init_from, Decoder):
            self.linear.weight = torch.nn.Parameter(
                init_from.linear.weight.T.clone().detach().contiguous()
            )
            self.linear.bias = torch.nn.Parameter(
                torch.zeros(self.config.d_sae, device=self.config.device)
            )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def _activation_fn(self, x: torch.Tensor):
        if self.config.activation_function.kind == "relu":
            return x.relu()
        elif self.config.activation_function.kind == "topk":
            topk = torch.topk(x, k=self.config.activation_function.k, dim=-1)
            result = torch.zeros_like(x)
            result.scatter_(-1, topk.indices, topk.values.relu())
            return result
        else:
            raise NotImplementedError(
                f'"{self.config.activation_function.kind}" not implemented'
            )

    def forward(self, x: torch.Tensor):
        return self._activation_fn(self.linear(x))


class InteractionEncoder(Encoder):
    def __init__(
        self,
        config: InteractionEncoderConfig,
    ):
        super().__init__(config)

    @torch.no_grad
    def init_weights(self, init_from: Union["Encoder", "Decoder", None] = None):
        from .decoder import Decoder

        super().init_weights(init_from)

        if isinstance(init_from, InteractionEncoder):
            self.interaction = torch.nn.Parameter(
                init_from.interaction.clone().detach().contiguous()
            )
        elif isinstance(init_from, Decoder):
            self.interaction = torch.nn.Parameter(
                torch.eye(self.config.d_sae, device=self.config.device)
            )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def forward(self, x: torch.Tensor):
        encoder_output = self.linear(x)
        features = self._activation_fn(encoder_output)
        for _ in range(self.config.n_interaction_iterations):
            features = self._activation_fn(encoder_output + features @ self.interaction)

        return features
