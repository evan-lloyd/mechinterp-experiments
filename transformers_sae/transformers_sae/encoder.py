from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, TypeAlias, Union

import torch

if TYPE_CHECKING:
    from .decoder import Decoder

EncoderKind: TypeAlias = Literal["relu", "topk", "batch_topk", "firm_topk"]


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
class FirmTopKActivationFunctionConfig(ActivationFunctionConfig):
    k_soft: int
    k_hard: int
    # From SAELens
    threshold_lr: float = 0.01

    def __post_init__(self):
        assert self.k_hard < self.k_soft, (
            "k for hard threshold must be strictly smaller than k for soft threshold"
        )
        self.kind = "firm_topk"


@dataclass
class ReluActivationFunctionConfig(ActivationFunctionConfig):
    def __post_init__(self):
        self.kind = "relu"


class ActivationFunction(torch.nn.Module):
    config: ActivationFunctionConfig

    def __init__(self, config: ActivationFunctionConfig, device: torch.device):
        super().__init__()
        self.config = config

    def init_weights(self):
        return


class ReluActivationFunction(ActivationFunction):
    config: ReluActivationFunctionConfig

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        return x.relu()


class TopKActivationFunction(ActivationFunction):
    config: TopKActivationFunctionConfig

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        topk = torch.topk(x, k=self.config.k, dim=-1, sorted=False)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, topk.values.relu())
        return result


class BatchTopKActivationFunction(ActivationFunction):
    config: BatchTopKActivationFunctionConfig

    def __init__(
        self,
        config: BatchTopKActivationFunctionConfig,
        device: torch.device,
    ):
        super().__init__(config, device)
        self.config = config
        # We don't find this by standard optimization, rather we estimate it manually during training
        self.register_buffer(
            "threshold",
            torch.tensor(
                0.0,
                dtype=torch.double if device.type != "mps" else torch.float32,
                device=device,
                requires_grad=False,
            ),
            persistent=True,
        )

    def init_weights(self):
        # Doesn't transfer well across layers, so start from scratch
        self.threshold.fill_(0.0)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
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
                k=self.config.k * num_tokens,
                dim=-1,
                sorted=False,
            )
            result = torch.zeros_like(x).view(-1)
            result.scatter_(-1, topk.indices, topk.values.relu())
            result = result.reshape(*x.shape)
            lr = self.config.threshold_lr

            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                pos_values = topk.values > 0
                # TODO: handle mocking more cleanly
                if (
                    not isinstance(pos_values, torch._subclasses.FakeTensor)
                    and pos_values.any()
                ):
                    self.threshold = (1 - lr) * self.threshold + lr * topk.values[
                        pos_values
                    ].min().to(self.threshold.dtype)
            return result
        # JumpReLU during inference
        else:
            return x * (x > self.threshold)


class FirmTopKActivationFunction(ActivationFunction):
    """FirmTopK is a merging of the "thresholding operator with support selection" from Chen et al 2018, which
    is essentially the firm threshold function with upper threshold set via top k, and the BatchTopK activation
    function used in SAEs. During training it selects its thresholds by (batch) top k, and applies the appropriate
    thresholding function to them; values in the top k_hard range are passed through unchanged (hard threshold)
    while those in the range [k_hard, k_soft] are soft-thresholded by subtracting the value of the k_soft^th entry.

    At inference time, the activation function becomes the positive part of the firm threshold function, using
    thresholds that were learned by exponential moving average during training.
    """

    config: FirmTopKActivationFunctionConfig
    threshold: torch.Tensor

    def __init__(
        self,
        config: FirmTopKActivationFunctionConfig,
        device: torch.device,
    ):
        super().__init__(config, device)
        self.config = config
        # We don't find this by standard optimization, rather we estimate it manually during training
        self.register_buffer(
            "threshold",
            torch.tensor(
                [0.0, 0.0],
                dtype=torch.double if device.type != "mps" else torch.float32,
                device=device,
                requires_grad=False,
            ),
            persistent=True,
        )

    def init_weights(self):
        # Doesn't transfer well across layers, so start from scratch
        self.threshold.fill_(0.0)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            # This is crucial; otherwise we are wasting our non-zero activations on tokens that aren't even
            # being evaluated or trained on.
            # TODO: probably this should happen automatically somewhere upstream?
            with torch.no_grad():
                x[~token_mask.bool()] = torch.finfo(x.dtype).min
            num_tokens = x.shape[0] * x.shape[1]
            topk = torch.topk(
                x.view(-1),
                k=self.config.k_soft * num_tokens,
                dim=-1,
                sorted=True,
            )
            threshold = torch.stack(
                (
                    topk.values[self.config.k_soft * num_tokens - 1],
                    topk.values[self.config.k_hard * num_tokens - 1],
                )
            )
            lr = self.config.threshold_lr

            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                if threshold[0] > 0.0:
                    self.threshold = (1 - lr) * self.threshold + lr * threshold
        else:
            threshold = self.threshold

        # Return non-negative part of firm threshold function:
        # if x > hard_threshold:
        #   y = x
        # elif x > soft_threshold:
        #   y = x - soft_threshold
        # else:
        #   y = 0
        # TODO: performance check different variants, seems like there's a lot of potential ways to vectorize
        soft_x = (x - threshold[0]) * (x > threshold[0])
        hard_x = (soft_x + threshold[0]) * (x > threshold[1])
        return soft_x + hard_x


@dataclass
class EncoderConfig:
    d_model: int
    d_sae: int
    device: torch.device
    train_dtype: torch.dtype
    inference_dtype: torch.dtype
    activation_function: ActivationFunctionConfig


@dataclass
class LISTAConfig(EncoderConfig):
    n_iterations: int = 1


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
        self.activation = torch.nn.ModuleList(
            [
                self.activation_module_from_config(
                    self.config.activation_function,
                    device=self.config.device,
                )
            ]
        )

    @classmethod
    def activation_module_from_config(
        cls,
        activation_config: ActivationFunctionConfig,
        device: torch.device,
    ) -> torch.nn.Module:
        if isinstance(activation_config, ReluActivationFunctionConfig):
            return ReluActivationFunction(activation_config, device)
        if isinstance(activation_config, TopKActivationFunctionConfig):
            return TopKActivationFunction(activation_config, device)
        if isinstance(activation_config, BatchTopKActivationFunctionConfig):
            return BatchTopKActivationFunction(activation_config, device)
        raise NotImplementedError(f'"{activation_config.kind}" not implemented')

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

        for submodule in self.activation:
            assert isinstance(submodule, ActivationFunction)
            submodule.init_weights()

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

    def train(self, mode: bool = True):
        super().train(mode)

        to_dtype = self.config.train_dtype if mode else self.config.inference_dtype
        # Unsure why, but have to do it this way for compatibility with FakeTensorMode, which
        # is useful to support for memory profiling purposes.
        self.linear.weight = torch.nn.Parameter(self.linear.weight.to(to_dtype))
        self.linear.bias = torch.nn.Parameter(self.linear.bias.to(to_dtype))
        self.requires_grad_(mode)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.train_dtype if self.training else self.config.inference_dtype

    def encoder_params(self) -> Iterator[torch.nn.Parameter]:
        yield from self.linear.parameters()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        should_cast: bool = True,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self.activation[0](self.linear(x), token_mask)
        if should_cast:
            result = result.to(out_dtype)
        return result


class LISTA(Encoder):
    config: LISTAConfig
    decoder: "Decoder"
    weight: torch.nn.ParameterList
    activation: torch.nn.ModuleList

    def __init__(
        self,
        config: LISTAConfig,
    ):
        torch.nn.Module.__init__(self)
        self.config = config

        self.weight = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.empty(
                        (config.d_model, config.d_sae),
                        device="meta",
                        dtype=config.train_dtype,
                    )
                )
                for _ in range(config.n_iterations)
            ]
        )
        # TODO: make this configurable, instead of being the same at each step?
        self.activation = torch.nn.ModuleList(
            [
                self.activation_module_from_config(
                    config.activation_function, config.device
                )
                for _ in range(config.n_iterations)
            ]
        )

    @torch.no_grad()
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

        for submodule in self.activation:
            assert isinstance(submodule, ActivationFunction)
            submodule.init_weights()

        if isinstance(init_from, LISTA):
            self.weight = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        weight.to(to_device or self.config.device, copy=True)
                        .detach()
                        .contiguous()
                    )
                    for weight in init_from.weight
                ]
            )
        elif isinstance(init_from, Decoder):
            # Avoid adding to module hierarchy; we want a simple reference to it
            object.__setattr__(self, "decoder", init_from)
            for i, weight in enumerate(self.weight):
                if i == 0:
                    self.weight[i] = torch.nn.Parameter(
                        init_from.linear.weight.to(
                            to_device or self.config.device, copy=True
                        )
                        .detach()
                        .contiguous()
                    )
                else:
                    self.weight[i] = torch.nn.Parameter(
                        torch.nn.init.kaiming_normal_(
                            torch.empty_like(
                                weight, device=to_device or self.config.device
                            )
                        )
                    )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def train(self, mode: bool = True):
        torch.nn.Module.train(self, mode)

        to_dtype = self.config.train_dtype if mode else self.config.inference_dtype
        self.weight = torch.nn.ParameterList(
            torch.nn.Parameter(weight.to(to_dtype)) for weight in self.weight
        )
        self.requires_grad_(mode)

    def encoder_params(self) -> Iterator[torch.nn.Parameter]:
        yield from ()

    def interaction_params(self) -> Iterator[torch.nn.Parameter]:
        yield from self.weight.parameters()

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor, should_cast: bool = True
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)

        # Chen et al 2018, though the initialization here is a bit of an ad-lib on my part (paper
        # uses 0)
        # features = self.activation[0](x @ self.decoder.linear.weight, token_mask)
        features = torch.zeros(
            (x.shape[0], x.shape[1], self.config.d_sae), device=x.device, dtype=x.dtype
        )
        for i in range(self.config.n_iterations):
            features = self.activation[i](
                features + (x - self.decoder(features)) @ self.weight[i],
                token_mask,
            )

        if should_cast:
            features = features.to(out_dtype)
        return features
