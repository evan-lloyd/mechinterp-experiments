from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .data_batch import DataBatch
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE


def _batch_loss(fn):
    def _inner(x: torch.Tensor, y: torch.Tensor, batch: DataBatch):
        return (fn(x, y) * batch.token_mask).sum() / batch.num_tokens

    return _inner


def _cos_dist(x: torch.Tensor, y: torch.Tensor):
    return 1 - torch.nn.functional.cosine_similarity(x, y, dim=-1)


def _mse(x: torch.Tensor, y: torch.Tensor):
    return ((x - y) ** 2).mean(dim=-1)


def _kl(x: torch.Tensor, y: torch.Tensor):
    return torch.nn.KLDivLoss(reduction="none", log_target=True)(
        x.log_softmax(-1),
        y.log_softmax(-1),
    ).mean(dim=-1)


cos_dist_loss = _batch_loss(_cos_dist)
mse_loss = _batch_loss(_mse)
kl_loss = _batch_loss(_kl)


@dataclass(kw_only=True)
class ActivationBatch:
    layer_output: torch.Tensor | None = None
    sae_features: torch.Tensor | None = None
    sae_output: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass
class TrainingBatch:
    input_data: DataBatch
    baseline_activations: Dict[int, ActivationBatch]
    replacement_activations: Dict[int, ActivationBatch]


def downstream_loss(
    batch: DataBatch,
    baseline: ActivationBatch,
    replacement: ActivationBatch,
    downstream_attr: str,
    downstream_fn: Callable[[torch.Tensor, torch.Tensor, DataBatch], torch.Tensor],
) -> torch.Tensor:
    return downstream_fn(
        getattr(replacement, downstream_attr),
        getattr(baseline, downstream_attr),
        batch,
    )


def next_layer_losses(batch: TrainingBatch, target_layer: int):
    if batch.baseline_activations[target_layer + 1].logits is not None:
        downstream_attr = "logits"
        downstream_fn = kl_loss
    else:
        downstream_attr = "sae_features"
        downstream_fn = cos_dist_loss

    downstream_reconstruction_loss = downstream_loss(
        batch.input_data,
        batch.baseline_activations[target_layer + 1],
        batch.replacement_activations[target_layer + 1],
        downstream_attr,
        downstream_fn,
    )

    reconstruction_loss = mse_loss(
        batch.baseline_activations[target_layer],
        batch.replacement_activations[target_layer],
        batch.input_data,
    )
    return {
        "reconstruction": reconstruction_loss,
        "downstream_reconstruction": downstream_reconstruction_loss,
    }


def e2e_losses(batch: TrainingBatch, start_layer: int, end_layer: int):
    downstream_reconstruction_loss = []
    for layer in range(start_layer, end_layer):
        if batch.baseline_activations[layer].logits is not None:
            downstream_attr = "logits"
            downstream_fn = kl_loss
        else:
            downstream_attr = "layer_output"
            downstream_fn = mse_loss
        downstream_reconstruction_loss.append(
            downstream_loss(
                batch.input_data,
                batch.baseline_activations[layer + 1],
                batch.replacement_activations[layer + 1],
                downstream_attr,
                downstream_fn,
            )
        )

    return {"downstream_reconstruction": downstream_reconstruction_loss}


def kl_finetune_losses(batch: TrainingBatch, target_layer: int, logits_layer: int):
    downstream_reconstruction_loss = downstream_loss(
        batch.baseline_activations[logits_layer],
        batch.replacement_activations[logits_layer],
        "logits",
        kl_loss,
    )

    reconstruction_loss = mse_loss(
        batch.baseline_activations[target_layer],
        batch.replacement_activations[target_layer],
        batch.input_data,
    )
    return {
        "reconstruction": reconstruction_loss,
        "downstream_reconstruction": downstream_reconstruction_loss,
    }


class Stepper(ABC):
    base_model: torch.nn.Module
    replacement_model: ReplacementModel
    full_replacement_model: ReplacementModel

    def __init__(
        self,
        base_model: torch.nn.Module,
        replacement_model: torch.nn.Module,
        saes: Dict[int, SAE],
    ):
        self.base_model = base_model
        self.replacement_model = replacement_model
        self.full_replacement_model = make_replacement_model(
            base_model,
            {f"transformer.h.{layer}": sae for layer, sae in saes.items()},
        )

    @abstractmethod
    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch: ...

    @abstractmethod
    def step(self, training_batch: TrainingBatch) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def augment_batch_for_evals(
        self, training_batch: TrainingBatch, losses: Dict[str, torch.Tensor]
    ) -> TrainingBatch: ...
