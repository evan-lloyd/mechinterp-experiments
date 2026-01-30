from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from .data_batch import DataBatch
from .metrics import cos_dist_loss, kl_loss, mse_loss
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE


@dataclass(kw_only=True)
class ActivationBatch:
    layer_output: torch.Tensor | None = None
    sae_features: torch.Tensor | None = None
    sae_output: torch.Tensor | None = None
    logits: torch.Tensor | None = None


@dataclass
class TrainingBatch:
    input_data: DataBatch
    replacement_activations: Dict[int, ActivationBatch]
    baseline_activations: Dict[int, ActivationBatch]


def downstream_loss(
    replacement: ActivationBatch,
    baseline: ActivationBatch,
    batch: DataBatch,
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
        batch.replacement_activations[target_layer + 1],
        batch.baseline_activations[target_layer + 1],
        batch.input_data,
        downstream_attr,
        downstream_fn,
    )

    reconstruction_loss = mse_loss(
        batch.replacement_activations[target_layer],
        batch.baseline_activations[target_layer],
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
                batch.replacement_activations[layer + 1],
                batch.baseline_activations[layer + 1],
                batch.input_data,
                downstream_attr,
                downstream_fn,
            )
        )

    return {"downstream_reconstruction": downstream_reconstruction_loss}


def kl_finetune_losses(batch: TrainingBatch, target_layer: int, logits_layer: int):
    downstream_reconstruction_loss = downstream_loss(
        batch.replacement_activations[logits_layer],
        batch.baseline_activations[logits_layer],
        batch.input_data,
        "logits",
        kl_loss,
    )

    reconstruction_loss = mse_loss(
        batch.replacement_activations[target_layer],
        batch.baseline_activations[target_layer],
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
