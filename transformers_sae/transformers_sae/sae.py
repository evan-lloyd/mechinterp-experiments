from dataclasses import dataclass
from itertools import repeat
from typing import Any, List, Mapping, Optional

import torch

from .decoder import Decoder, DecoderConfig
from .encoder import (  # noqa: F401
    LISTA,
    ActivationKind,
    BatchTopKActivationFunctionConfig,
    Encoder,
    EncoderConfig,
    EncoderKind,
    InteractionLISTA,
    InteractionLISTAConfig,
    LISTAConfig,
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
    activation_kind: ActivationKind | List[ActivationKind],
    top_k: int | List[int] | None = None,
    encoder_kind: EncoderKind = "encoder",
    n_iterations: int | None = None,
) -> SAEConfig:
    if isinstance(activation_kind, list):
        assert activation_kind, (
            "Must use with_interaction if specifying multiple encoder configurations"
        )
    else:
        if isinstance(top_k, list):
            activation_kind = [activation_kind] * len(top_k)
        else:
            activation_kind = [activation_kind]

    if isinstance(top_k, list):
        assert activation_kind and len(activation_kind) == len(top_k), (
            "Must use with_interaction if specifying multiple encoder configurations"
        )
        top_k_value = top_k.__iter__()
    else:
        top_k_value = repeat(top_k).__iter__()
    activation_config = []
    for ek in activation_kind:
        if ek == "relu":
            activation_config.append(ReluActivationFunctionConfig())
        elif ek == "topk":
            assert top_k is not None, "Must specify top_k for TopK SAE"
            activation_config.append(TopKActivationFunctionConfig(next(top_k_value)))
        elif ek == "batch_topk":
            assert top_k is not None, "Must specify top_k for BatchTopK SAE"
            activation_config.append(
                BatchTopKActivationFunctionConfig(next(top_k_value))
            )
        else:
            raise ValueError(f"Unknown encoder_kind {ek}")

    device = torch.device(device)

    encoder_class_kwargs = dict(
        d_model=d_model,
        d_sae=d_sae,
        device=device,
        train_dtype=train_dtype,
        inference_dtype=inference_dtype,
    )
    if encoder_kind in ("lista", "interaction"):
        if encoder_kind == "lista":
            encoder_cfg_class = LISTAConfig
            n_iterations = len(top_k) if isinstance(top_k, list) else n_iterations
        elif encoder_kind == "interaction":
            encoder_cfg_class = InteractionLISTAConfig
            n_iterations = len(top_k) - 1 if isinstance(top_k, list) else n_iterations

        encoder_class_kwargs["activation_function"] = activation_config[0]
        if len(activation_config) > 1:
            encoder_class_kwargs["per_layer_activation_functions"] = activation_config
        encoder_class_kwargs["n_iterations"] = n_iterations
    else:
        encoder_cfg_class = EncoderConfig
        encoder_class_kwargs["activation_function"] = activation_config[0]
    encoder_config = encoder_cfg_class(**encoder_class_kwargs)

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
        if isinstance(config.encoder, InteractionLISTAConfig):
            self.encoder = InteractionLISTA(config.encoder)
        elif isinstance(config.encoder, LISTAConfig):
            self.encoder = LISTA(config.encoder)
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
            if isinstance(self.encoder, LISTA):
                object.__setattr__(self.encoder, "decoder", self.decoder)
            self._device_tracker = torch.nn.Buffer(
                torch.empty((0,), device=to_device or self.config.device)
            )

    def change_inference_dtype(self, dtype: torch.dtype | str):
        """Change our configured inference_dtype."""
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.config.inference_dtype = dtype
        self.config.encoder.inference_dtype = dtype
        self.config.decoder.inference_dtype = dtype
        self.encoder.config.inference_dtype = dtype
        self.decoder.config.inference_dtype = dtype
        return self

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

    def train_activations(self):
        for a in self.encoder.activation:
            a.train()

    def activation_thresholds(self):
        return tuple(a.threshold.item() for a in self.encoder.activation if hasattr(a, "threshold"))

    def set_activation_threshold_lr(self, lr: float):
        for a in self.encoder.activation:
            if hasattr(a.config, "threshold_lr"):
                a.config.threshold_lr = lr

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

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        super().load_state_dict(state_dict, strict, assign)
        # TODO: Refactor. Maybe every encoder should have a reference to the decoder?
        if isinstance(self.encoder, LISTA):
            object.__setattr__(self.encoder, "decoder", self.decoder)

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
