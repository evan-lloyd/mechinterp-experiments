from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

from .activation_data import ActivationBatch, TrainingBatch, run_replacement_model
from .data_batch import DataBatch
from .metrics import cos_dist_loss, kl_loss, mse_loss
from .replacement_model import ReplacementModel

if TYPE_CHECKING:
    from .training import TrainingConfig


# def e2e_losses(batch: TrainingBatch, start_layer: int, end_layer: int):
#     downstream_reconstruction_loss = []
#     for layer in range(start_layer, end_layer):
#         if batch.baseline_activations[layer].logits is not None:
#             downstream_attr = "logits"
#             downstream_fn = kl_loss
#         else:
#             downstream_attr = "layer_output"
#             downstream_fn = mse_loss
#         downstream_reconstruction_loss.append(
#             downstream_loss(
#                 batch.replacement_activations[layer + 1],
#                 batch.baseline_activations[layer + 1],
#                 batch.input_data,
#                 downstream_attr,
#                 downstream_fn,
#             )
#         )

#     return {"downstream_reconstruction": downstream_reconstruction_loss}


# def kl_finetune_losses(batch: TrainingBatch, target_layer: int, logits_layer: int):
#     downstream_reconstruction_loss = downstream_loss(
#         batch.replacement_activations[logits_layer],
#         batch.baseline_activations[logits_layer],
#         batch.input_data,
#         "logits",
#         kl_loss,
#     )

#     reconstruction_loss = mse_loss(
#         batch.replacement_activations[target_layer],
#         batch.baseline_activations[target_layer],
#         batch.input_data,
#     )
#     return {
#         "reconstruction": reconstruction_loss,
#         "downstream_reconstruction": downstream_reconstruction_loss,
#     }


class Stepper(ABC):
    base_model: torch.nn.Module
    train_model: ReplacementModel

    def __init__(
        self,
        base_model: torch.nn.Module,
        train_model: torch.nn.Module,
    ):
        self.base_model = base_model
        self.train_model = train_model

    @abstractmethod
    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch: ...

    @abstractmethod
    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]: ...

    def run_baseline(
        self, wanted_layers: List[int], batch: DataBatch, cache: torch.Tensor | None
    ):
        if cache is not None:
            base_run = cache
            for k, v in base_run.items():
                base_run[k] = v.to(self.base_model.device)
        else:
            base_run = run_replacement_model(
                self.base_model,
                {
                    layer: self.base_model.transformer.h[layer]
                    if layer < self.base_model.config.num_layers
                    else self.base_model.lm_head
                    for layer in wanted_layers
                },
                batch,
                start_layer=-1,  # not cached, so we start from raw input
                end_layer=max(wanted_layers) + 1,  # non-inclusive range
            )

        # We don't cache logits, so we may need to reconstruct them if loading from cache.
        if (
            self.base_model.config.num_layers in wanted_layers
            and self.base_model.config.num_layers not in base_run
        ):
            base_run[self.base_model.config.num_layers] = run_replacement_model(
                self.base_model,
                {self.base_model.config.num_layers: self.base_model.lm_head},
                batch,
                start_layer=-1,  # not cached, so we start from raw input
                end_layer=max(wanted_layers) + 1,  # non-inclusive range
            )[self.base_model.config.num_layers]
        return base_run
