from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import torch

if TYPE_CHECKING:
    from .decoder import Decoder

EncoderKind: TypeAlias = Literal["relu", "topk", "batch_topk"]


@dataclass
class ActivationFunctionConfig:
    kind: EncoderKind = field(init=False)


@dataclass
class TopKActivationFunctionConfig(ActivationFunctionConfig):
    k: int

    def __post_init__(self):
        self.kind = "topk"


@dataclass
class BatchTopKActivationFunctionConfig(ActivationFunctionConfig):
    k: int
    # From SAELens
    threshold_lr: float = 0.01

    def __post_init__(self):
        self.kind = "batch_topk"


@dataclass
class ReluActivationFunctionConfig(ActivationFunctionConfig):
    def __post_init__(self):
        self.kind = "relu"


@dataclass
class EncoderConfig:
    d_model: int
    d_sae: int
    device: torch.device
    train_dtype: torch.dtype
    inference_dtype: torch.dtype
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
        self.linear = torch.nn.Linear(
            config.d_model, config.d_sae, device="meta", dtype=self.config.train_dtype
        )
        if self.config.activation_function.kind == "batch_topk":
            # We don't find this by standard optimization, rather we estimate it manually during training
            self.register_buffer(
                "batch_topk_threshold",
                torch.tensor(
                    0.0,
                    dtype=torch.double,
                    device=self.config.device,
                    requires_grad=False,
                ),
                persistent=True,
            )

    def init_weights(
        self,
        init_from: Union["Encoder", "Decoder", None] = None,
        to_device: str | None = None,
    ):
        from .decoder import Decoder

        if init_from is None:
            raise ValueError(
                "Encoder weights must be initialized from existing encoder or decoder"
            )

        if isinstance(init_from, Encoder):
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
        elif isinstance(init_from, Decoder):
            self.linear.weight = torch.nn.Parameter(
                init_from.linear.weight.T.to(to_device or self.config.device, copy=True)
                .detach()
                .contiguous()
            )
            self.linear.bias = torch.nn.Parameter(
                torch.zeros(
                    self.config.d_sae,
                    device=to_device or self.config.device,
                    dtype=self.config.train_dtype,
                )
            )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def _activation_fn(self, x: torch.Tensor, token_mask: torch.Tensor):
        if self.config.activation_function.kind == "relu":
            return x.relu()
        elif self.config.activation_function.kind == "topk" or (
            not self.training and self.config.activation_function.kind == "batch_topk"
        ):
            topk = torch.topk(
                x, k=self.config.activation_function.k, dim=-1, sorted=False
            )
            result = torch.zeros_like(x)
            result.scatter_(-1, topk.indices, topk.values.relu())
            return result
        elif self.config.activation_function.kind == "batch_topk":
            # Adapted from https://github.com/decoderesearch/SAELens/blob/69c4c62b0dc24e5ba23fc773a0286149514b4a23/sae_lens/saes/batchtopk_sae.py
            # BatchTopK during training
            if self.training:
                # This is crucial; otherwise we are wasting our non-zero activations on tokens that aren't even
                # being evaluated or trained on.
                with torch.no_grad():
                    x[~token_mask.bool()] = torch.finfo(x.dtype).min
                num_tokens = x.shape[0] * x.shape[1]
                topk = torch.topk(
                    x.view(-1),
                    k=self.config.activation_function.k * num_tokens,
                    dim=-1,
                    sorted=False,
                )
                result = torch.zeros_like(x).view(-1)
                result.scatter_(-1, topk.indices, topk.values.relu())
                result = result.reshape(*x.shape)
                lr = self.config.activation_function.threshold_lr

                with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                    pos_values = topk.values > 0
                    if pos_values.any():
                        self.batch_topk_threshold = (
                            1 - lr
                        ) * self.batch_topk_threshold + lr * topk.values[
                            pos_values
                        ].min().to(torch.double)
                return result
            # JumpReLU during inference
            # TODO: what if we made it just regular TopK?
            else:
                return x * (x > self.batch_topk_threshold)
        else:
            raise NotImplementedError(
                f'"{self.config.activation_function.kind}" not implemented'
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.to(dtype=self.config.train_dtype if mode else self.config.inference_dtype)
        self.requires_grad_(mode)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.train_dtype if self.training else self.config.inference_dtype

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        should_cast: bool = True,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self._activation_fn(self.linear(x), token_mask)
        if should_cast:
            result = result.to(out_dtype)
        return result


class InteractionEncoder(Encoder):
    config: InteractionEncoderConfig

    def __init__(
        self,
        config: InteractionEncoderConfig,
    ):
        super().__init__(config)
        self.interaction = torch.nn.Parameter(
            torch.empty(
                (config.d_sae, config.d_sae),
                device="meta",
                dtype=self.config.train_dtype,
            )
        )

    @torch.no_grad()
    def init_weights(
        self,
        init_from: Union["Encoder", "Decoder", None] = None,
        to_device: str | None = None,
    ):
        from .decoder import Decoder

        super().init_weights(init_from, to_device)

        if isinstance(init_from, InteractionEncoder):
            self.interaction = torch.nn.Parameter(
                init_from.interaction.to(to_device or self.config.device, copy=True)
                .detach()
                .contiguous()
            )
        elif isinstance(init_from, Decoder):
            self.interaction = torch.nn.Parameter(
                torch.eye(
                    self.config.d_sae,
                    device=to_device or self.config.device,
                    dtype=self.config.train_dtype,
                )
            )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor, should_cast: bool = True
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        encoder_output = self.linear(x)
        features = self._activation_fn(encoder_output, token_mask)
        for _ in range(self.config.n_interaction_iterations):
            features = self._activation_fn(
                encoder_output + features @ self.interaction, token_mask
            )

        if should_cast:
            features = features.to(out_dtype)
        return features
