from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import torch

if TYPE_CHECKING:
    from .decoder import Decoder

ActivationKind: TypeAlias = Literal["relu", "topk", "batch_topk"]
EncoderKind: TypeAlias = Literal["encoder", "interaction", "lista"]


@dataclass
class ActivationFunctionConfig:
    kind: ActivationKind = field(init=False)


@dataclass
class TopKActivationFunctionConfig(ActivationFunctionConfig):
    k: int

    def __post_init__(self):
        self.kind = "topk"


@dataclass
class BatchTopKActivationFunctionConfig(ActivationFunctionConfig):
    k: int
    # TODO: this should live in TrainingConfig
    # From SAELens
    threshold_lr: float = 0.01

    def __post_init__(self):
        self.kind = "batch_topk"


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
                sorted=True,
            )
            threshold = torch.maximum(
                topk.values[-1], torch.zeros_like(topk.values[-1])
            )
            lr = self.config.threshold_lr

            # Adapted from https://github.com/decoderesearch/SAELens/blob/69c4c62b0dc24e5ba23fc773a0286149514b4a23/sae_lens/saes/batchtopk_sae.py
            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                self.threshold = (1 - lr) * self.threshold + lr * threshold.to(
                    self.threshold.dtype
                )
        # JumpReLU during inference
        else:
            threshold = self.threshold

        return x * (x >= threshold)


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
    # Per-layer activation configs. When set, must have length == n_iterations.
    # When None, `activation_function` is replicated across all iterations.
    per_layer_activation_functions: Optional[List[ActivationFunctionConfig]] = None

    def __post_init__(self):
        if self.per_layer_activation_functions is not None:
            if len(self.per_layer_activation_functions) != self.n_iterations:
                raise ValueError(
                    f"per_layer_activation_functions has {len(self.per_layer_activation_functions)} entries "
                    f"but n_iterations={self.n_iterations}"
                )

    @classmethod
    def with_batch_topk_schedule(
        cls,
        k_values: List[int],
        *,
        d_model: int,
        d_sae: int,
        device: torch.device,
        train_dtype: torch.dtype,
        inference_dtype: torch.dtype,
        threshold_lr: float = 0.01,
    ) -> "LISTAConfig":
        """Convenience constructor for LISTA with a different BatchTopK k at each iteration.

        Args:
            k_values: k for each LISTA iteration, e.g. [128, 64, 32].
                      The length determines n_iterations.
        """
        per_layer = [
            BatchTopKActivationFunctionConfig(k=k, threshold_lr=threshold_lr)
            for k in k_values
        ]
        # Use the first layer's config as the nominal activation_function so that
        # code that inspects config.activation_function still gets a sensible value.
        return cls(
            d_model=d_model,
            d_sae=d_sae,
            device=device,
            train_dtype=train_dtype,
            inference_dtype=inference_dtype,
            activation_function=per_layer[0],
            n_iterations=len(k_values),
            per_layer_activation_functions=per_layer,
        )


class InteractionLISTAConfig(LISTAConfig):
    def __post_init__(self):
        if self.per_layer_activation_functions is not None:
            if len(self.per_layer_activation_functions) != self.n_iterations + 1:
                raise ValueError(
                    f"per_layer_activation_functions has {len(self.per_layer_activation_functions)} entries "
                    f"but n_iterations={self.n_iterations}"
                )


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
        elif isinstance(activation_config, TopKActivationFunctionConfig):
            return TopKActivationFunction(activation_config, device)
        elif isinstance(activation_config, BatchTopKActivationFunctionConfig):
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
    layers: torch.nn.ModuleList
    activation: torch.nn.ModuleList

    def __init__(
        self,
        config: LISTAConfig,
    ):
        torch.nn.Module.__init__(self)
        self.config = config

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    config.d_model,
                    config.d_sae,
                    device="meta",
                    dtype=self.config.train_dtype,
                )
                for _ in range(config.n_iterations)
            ]
        )
        activation_configs = (
            config.per_layer_activation_functions
            if config.per_layer_activation_functions is not None
            else [config.activation_function] * config.n_iterations
        )
        self.activation = torch.nn.ModuleList(
            [
                self.activation_module_from_config(act_cfg, config.device)
                for act_cfg in activation_configs
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
            for layer, other in zip(self.layers, init_from.layers):
                layer.weight = torch.nn.Parameter(
                    other.weight.to(to_device or self.config.device, copy=True)
                    .detach()
                    .contiguous()
                )
                layer.bias = torch.nn.Parameter(
                    other.bias.to(to_device or self.config.device, copy=True)
                    .detach()
                    .contiguous()
                )
        elif isinstance(init_from, Decoder):
            # Avoid adding to module hierarchy; we want a simple reference to it
            object.__setattr__(self, "decoder", init_from)
            self.layers[0].weight = torch.nn.Parameter(
                init_from.linear.weight.T.to(to_device or self.config.device, copy=True)
                .detach()
                .contiguous()
            )
            self.layers[0].bias = torch.nn.Parameter(
                torch.zeros(
                    self.config.d_sae,
                    device=to_device or self.config.device,
                    dtype=self.config.train_dtype,
                )
            )
            for layer in self.layers[1:]:
                layer.weight = torch.nn.Parameter(
                    torch.nn.init.kaiming_normal_(
                        torch.empty_like(
                            layer.weight, device=to_device or self.config.device
                        )
                    )
                )
                layer.bias = torch.nn.Parameter(
                    torch.zeros(
                        self.config.d_sae,
                        device=to_device or self.config.device,
                        dtype=self.config.train_dtype,
                    )
                )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def train(self, mode: bool = True):
        torch.nn.Module.train(self, mode)

        to_dtype = self.config.train_dtype if mode else self.config.inference_dtype
        self.layers.to(to_dtype)
        self.requires_grad_(mode)

    def encoder_params(self) -> Iterator[torch.nn.Parameter]:
        yield from ()

    def interaction_params(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self.layers.named_parameters()

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor, should_cast: bool = True
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)

        # Chen et al 2018
        features = torch.zeros(
            (x.shape[0], x.shape[1], self.config.d_sae), device=x.device, dtype=x.dtype
        )
        for i in range(self.config.n_iterations):
            features = self.activation[i](
                features + self.layers[i]((x - self.decoder(features))),
                token_mask,
            )

        if should_cast:
            features = features.to(out_dtype)
        return features


class InteractionLISTA(Encoder):
    config: LISTAConfig
    decoder: "Decoder"
    layers: torch.nn.ModuleList
    activation: torch.nn.ModuleList

    def __init__(
        self,
        config: LISTAConfig,
    ):
        super().__init__(config)

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    config.d_sae,
                    config.d_sae,
                    device="meta",
                    dtype=self.config.train_dtype,
                    bias=False,
                )
                for _ in range(config.n_iterations)
            ]
        )
        activation_configs = (
            config.per_layer_activation_functions
            if config.per_layer_activation_functions is not None
            else [config.activation_function] * (config.n_iterations + 1)
        )
        self.activation = torch.nn.ModuleList(
            [
                self.activation_module_from_config(act_cfg, config.device)
                for act_cfg in activation_configs
            ]
        )

    @torch.no_grad()
    def init_weights(
        self,
        init_from: Union["Encoder", "Decoder", None] = None,
        to_device: str | None = None,
    ):
        from .decoder import Decoder

        super().init_weights(init_from, to_device)

        if init_from is None:
            raise ValueError(
                "Encoder weights must be initialized from existing encoder or decoder"
            )

        for submodule in self.activation:
            assert isinstance(submodule, ActivationFunction)
            submodule.init_weights()

        if isinstance(init_from, InteractionLISTA):
            for layer, other in zip(self.layers, init_from.layers):
                layer.weight = torch.nn.Parameter(
                    other.weight.to(to_device or self.config.device, copy=True)
                    .detach()
                    .contiguous()
                )
                # layer.bias = torch.nn.Parameter(
                #     other.bias.to(to_device or self.config.device, copy=True)
                #     .detach()
                #     .contiguous()
                # )
        elif isinstance(init_from, Decoder):
            # Avoid adding to module hierarchy; we want a simple reference to it
            object.__setattr__(self, "decoder", init_from)
            for layer in self.layers:
                layer.weight = torch.nn.Parameter(
                    torch.eye(
                        self.config.d_sae,
                        device=to_device or self.config.device,
                        dtype=self.config.train_dtype,
                    )
                )
                # layer.bias = torch.nn.Parameter(
                #     torch.zeros(
                #         self.config.d_sae,
                #         device=to_device or self.config.device,
                #         dtype=self.config.train_dtype,
                #     )
                # )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def train(self, mode: bool = True):
        super().train(mode)

        to_dtype = self.config.train_dtype if mode else self.config.inference_dtype
        self.layers.to(to_dtype)

    def interaction_params(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self.layers.named_parameters()

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor, should_cast: bool = True
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)

        # Gregor and LeCun 2010
        encoder_output = self.linear(x)
        features = self.activation[0](encoder_output, token_mask)
        for i in range(self.config.n_iterations):
            features = self.activation[i + 1](
                encoder_output + self.layers[i](features), token_mask
            )

        if should_cast:
            features = features.to(out_dtype)
        return features
