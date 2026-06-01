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

ActivationKind: TypeAlias = Literal["relu", "topk", "batch_topk", "jump_relu"]
EncoderKind: TypeAlias = Literal["encoder", "interaction", "lista"]


@dataclass
class ActivationFunctionConfig:
    kind: ActivationKind = field(init=False)


@dataclass
class JumpReluActivationFunctionConfig(ActivationFunctionConfig):
    d_sae: int
    k: int
    threshold_lr: float = 0.01

    def __post_init__(self):
        self.kind = "jump_relu"


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

    def onload(self):
        pass

    def offload(self):
        pass


class ReluActivationFunction(ActivationFunction):
    config: ReluActivationFunctionConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.relu()


class JumpReluActivationFunction(ActivationFunction):
    """Stub for a JumpReLU activation function. Note that training is not implemented; this class is currently
    only used to load existing JumpReLU SAEs from SAELens. It can also tune a global threshold offset
    BatchTopK style."""

    config: JumpReluActivationFunctionConfig
    threshold: torch.Tensor
    threshold_offset: torch.Tensor
    device: torch.device

    def __init__(self, config: JumpReluActivationFunctionConfig, device: torch.device):
        super().__init__(config, device)
        self.register_buffer(
            "threshold",
            torch.empty((config.d_sae,), device="meta", requires_grad=False),
            persistent=True,
        )
        # For encoder tuning
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
        self.device = device

    def onload(self):
        self.threshold = self.threshold.to(self.device)

    def offload(self):
        self.threshold = self.threshold.to("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchTopK during training
        if self.training:
            num_tokens = x.shape[0] * x.shape[1]
            topk = torch.topk(
                (x - self.threshold).view(-1),
                k=self.config.k * num_tokens,
                dim=-1,
                sorted=False,
            )
            threshold_offset = topk.values.min()
            lr = self.config.threshold_lr

            # Adapted from https://github.com/decoderesearch/SAELens/blob/69c4c62b0dc24e5ba23fc773a0286149514b4a23/sae_lens/saes/batchtopk_sae.py
            with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
                self.threshold_offset = (
                    1 - lr
                ) * self.threshold_offset + lr * threshold_offset.to(
                    self.threshold_offset.dtype
                )

        # JumpReLU during inference
        else:
            threshold_offset = self.threshold_offset
        return (x * (x >= (self.threshold + threshold_offset))).relu()

    def init_weights(self):
        self.threshold = torch.zeros(
            (self.config.d_sae,), device=self.device, requires_grad=False
        )
        self.threshold_offset = torch.tensor(
            0.0,
            dtype=torch.float64 if self.device.type != "mps" else torch.float32,
            device=self.device,
            requires_grad=False,
        )


class TopKActivationFunction(ActivationFunction):
    config: TopKActivationFunctionConfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchTopK during training
        if self.training:
            num_tokens = x.shape[0] * x.shape[1]
            topk = torch.topk(
                # This is crucial; otherwise we are wasting our non-zero activations on tokens that aren't even
                # being evaluated or trained on.
                x.view(-1),
                k=self.config.k * num_tokens,
                dim=-1,
                sorted=False,
            )
            threshold = torch.maximum(
                topk.values.min(),
                torch.zeros((1,), dtype=x.dtype, device=x.device),
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
class InteractionEncoderConfig(EncoderConfig):
    n_interaction_iterations: int = 1


@dataclass
class LISTAConfig(EncoderConfig):
    n_iterations: int = 1
    # Per-layer activation configs. When set, must have length == n_iterations.
    # When None, `activation_function` is replicated across all iterations.
    per_layer_activation_functions: Optional[List[ActivationFunctionConfig]] = None
    use_onsager_correction: bool = False
    force_unit_scale: bool = False

    def __post_init__(self):
        if self.per_layer_activation_functions is not None:
            if len(self.per_layer_activation_functions) != self.n_iterations:
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

    def train_activations(self):
        for a in self.activation:
            a.train()

    def interaction_params(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from ()

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
        elif isinstance(activation_config, JumpReluActivationFunctionConfig):
            return JumpReluActivationFunction(activation_config, device)

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
            if getattr(init_from.linear, "bias", None) is not None:
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
        if isinstance(self.linear.weight, torch.nn.Parameter):
            self.linear.weight = torch.nn.Parameter(self.linear.weight.to(to_dtype))
        if getattr(self.linear, "bias", None) is not None:
            self.linear.bias = torch.nn.Parameter(self.linear.bias.to(to_dtype))
        self.requires_grad_(mode)

    @property
    def dtype(self) -> torch.dtype:
        return self.config.train_dtype if self.training else self.config.inference_dtype

    def encoder_params(self) -> Iterator[torch.nn.Parameter]:
        yield from self.linear.parameters()

    def onload(self):
        for a in self.activation:
            a.onload()

    def offload(self):
        for a in self.activation:
            a.offload()

    def forward(
        self,
        x: torch.Tensor,
        should_cast: bool = True,
        feature_soft_cap: Optional[torch.Tensor] = None,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        features = self.activation[0](self.linear(x))

        if feature_soft_cap is not None:
            features = feature_soft_cap * (features / feature_soft_cap).tanh()

        if should_cast:
            features = features.to(out_dtype)
        return features


class LISTA(Encoder):
    config: LISTAConfig
    decoder: "Decoder"
    scale: torch.nn.Parameter
    feature_scale: torch.nn.Parameter
    activation: torch.nn.ModuleList

    def __init__(
        self,
        config: LISTAConfig,
    ):
        super().__init__(config)

        self.scale = torch.nn.Parameter(
            torch.empty(
                (config.n_iterations,), device="meta", dtype=self.config.train_dtype
            )
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
        self.feature_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=self.config.train_dtype, device="meta"),
            requires_grad=True,
        )

    def _set_parametrization(self):
        if self.training and not torch.nn.utils.parametrize.is_parametrized(
            self.linear
        ):
            self.linear = torch.nn.utils.parametrizations.spectral_norm(self.linear)
        elif not self.training and torch.nn.utils.parametrize.is_parametrized(
            self.linear
        ):
            self.linear = torch.nn.utils.parametrize.remove_parametrizations(
                self.linear, "weight"
            )

    @torch.no_grad()
    def init_weights(
        self,
        init_from: Union["Encoder", "Decoder", None] = None,
        to_device: str | None = None,
    ):
        from .decoder import Decoder

        # Need to remove existing parametrization if we're re-initializing weights, otherwise
        # torch complains.
        if torch.nn.utils.parametrize.is_parametrized(self.linear):
            self.linear = torch.nn.utils.parametrize.remove_parametrizations(
                self.linear, "weight"
            )

        super().init_weights(init_from, to_device)
        self.linear.bias = None
        self._set_parametrization()

        if init_from is None:
            raise ValueError(
                "Encoder weights must be initialized from existing encoder or decoder"
            )

        for submodule in self.activation:
            assert isinstance(submodule, ActivationFunction)
            submodule.init_weights()

        if isinstance(init_from, LISTA):
            # If using spectral norm, uv vectors are randomized when we _set_parametrization, so
            # we should make sure we match the source.
            if torch.nn.utils.parametrize.is_parametrized(self.linear):
                self.linear.load_state_dict(init_from.linear.state_dict())

            self.scale = torch.nn.Parameter(
                init_from.scale.to(to_device or self.config.device, copy=True)
            )
            if self.config.n_iterations != init_from.config.n_iterations:
                # Maybe truncate
                self.scale.data = self.scale.data[0 : self.config.n_iterations]
                # Maybe expand
                self.scale.data = torch.cat(
                    (
                        self.scale.data,
                        torch.full(
                            (self.config.n_iterations - init_from.config.n_iterations,),
                            1.0 / self.config.n_iterations,
                            device=self.scale.device,
                            dtype=self.scale.dtype,
                        ),
                    )
                )
            self.feature_scale = torch.nn.Parameter(
                init_from.feature_scale.to(to_device or self.config.device, copy=True)
            )
        elif isinstance(init_from, Decoder):
            # Avoid adding to module hierarchy; we want a simple reference to it
            object.__setattr__(self, "decoder", init_from)

            self.scale = torch.nn.Parameter(
                torch.full_like(
                    self.scale,
                    1.0 / self.config.n_iterations,
                    device=to_device or self.config.device,
                )
            )
            self.feature_scale = torch.nn.Parameter(
                torch.ones_like(
                    self.feature_scale,
                    device=to_device or self.config.device,
                ),
                requires_grad=True,
            )
        else:
            raise ValueError(f"Invalid initialization source: {type(init_from)}")

    def train(self, mode: bool = True):
        self.training = mode
        self._set_parametrization()
        super().train(mode)

        # NB: deliberately *not* casting dtype of scale or activation thresholds
        self.scale.requires_grad_(mode)
        self.feature_scale.requires_grad_(mode)

    def train_activations(self):
        super().train_activations()
        self.scale.requires_grad_(True)
        self.feature_scale.requires_grad_(True)

    def interaction_params(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from (("scale", self.scale), ("feature_scale", self.feature_scale))

    def forward(
        self,
        x: torch.Tensor,
        should_cast: bool = True,
        feature_soft_cap: Optional[torch.Tensor] = None,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)

        # Liu et al 2019 + Chen et al 2018
        features = torch.zeros(
            (x.shape[0], x.shape[1], self.config.d_sae), device=x.device, dtype=x.dtype
        )
        for i in range(self.config.n_iterations):
            if i == 0 or not self.config.use_onsager_correction:
                residual = x - self.decoder(features)
            else:
                # "Onsager correction" term (Borgerding and Schniter 2016)
                residual = (
                    x
                    - self.decoder(features)
                    + residual * self.activation[i - 1].config.k / self.config.d_model
                )

            # Rescale features to current iteration. This helps to prevent their magnitude
            # from sometimes blowing up.
            if (
                i >= 1
                and not self.config.use_onsager_correction
                and not self.config.force_unit_scale
            ):
                features = features * self.scale[i] / self.scale[i - 1]

            if self.config.force_unit_scale:
                scale = self.scale.tanh()[i]
            else:
                scale = self.scale[i]
            features = self.activation[i](features + scale * self.linear(residual))

        if self.config.force_unit_scale:
            features = features * self.feature_scale

        if feature_soft_cap is not None:
            features = feature_soft_cap * (features / feature_soft_cap).tanh()

        if should_cast:
            features = features.to(out_dtype)
        return features


class InteractionEncoder(Encoder):
    config: InteractionEncoderConfig
    interaction: torch.nn.Parameter

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
        # TODO: make this configurable, instead of being the same at each step?
        self.activation += [
            self.activation_module_from_config(
                config.activation_function, device=config.device
            )
            for _ in range(self.config.n_interaction_iterations)
        ]

    def interaction_params(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from (("interaction", self.interaction),)

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

    def train(self, mode: bool = True):
        super().train(mode)
        self.interaction = torch.nn.Parameter(
            self.interaction.to(
                dtype=self.config.train_dtype if mode else self.config.inference_dtype
            ),
            requires_grad=mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        should_cast: bool = True,
        feature_soft_cap: Optional[torch.Tensor] = None,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        encoder_output = self.linear(x)
        features = self.activation[0](encoder_output)
        for i in range(self.config.n_interaction_iterations):
            features = self.activation[i + 1](
                encoder_output + features @ self.interaction
            )

        if feature_soft_cap is not None:
            features = feature_soft_cap * (features / feature_soft_cap).tanh()

        if should_cast:
            features = features.to(out_dtype)
        return features
