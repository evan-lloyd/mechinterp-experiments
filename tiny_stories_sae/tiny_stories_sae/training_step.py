from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from .activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from .data_batch import DataBatch
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
    replacement_model: ReplacementModel

    def __init__(
        self,
        base_model: torch.nn.Module,
        train_model: torch.nn.Module,
    ):
        self.base_model = base_model
        self.replacement_model = train_model

    @property
    def run_layers(self) -> List[int]:
        return sorted(list(self.replacement_model.sae_layers.keys()))

    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch:
        baseline_activations = self.run_baseline(batch, cache)
        replacement_activations = self.run_replacement(batch, baseline_activations)
        return TrainingBatch(
            batch,
            replacement_activations=replacement_activations,
            baseline_activations=baseline_activations,
            replacement_layers=self.run_layers,
        )

    @abstractmethod
    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]: ...

    def run_baseline(
        self, batch: DataBatch, cache: torch.Tensor | None
    ) -> Dict[int, ActivationBatch]:
        if cache is not None:
            base_run = cache
            for k, v in list(base_run.items()):
                base_run[k] = ActivationBatch(layer_output=v.to(self.base_model.device))
        else:
            with torch.no_grad():
                base_run = make_activation_batch(
                    self.base_model,
                    [(layer, "layer") for layer in self.run_layers],
                    batch,
                    start_layer=-1,  # not cached, so we start from raw input
                    end_layer=max(self.run_layers) + 1,  # non-inclusive range
                )

        # We don't cache logits, so we may need to reconstruct them if loading from cache.
        if (
            self.base_model.config.num_layers in self.run_layers
            and self.base_model.config.num_layers not in base_run
        ):
            with torch.no_grad():
                start_layer = max(k for k in base_run)
                base_run[self.base_model.config.num_layers] = make_activation_batch(
                    self.base_model,
                    [(self.base_model.config.num_layers, "layer")],
                    batch,
                    start_layer=start_layer + 1,
                    start_input=base_run[start_layer].layer_output,
                    end_layer=self.base_model.config.num_layers
                    + 1,  # non-inclusive range
                )[self.base_model.config.num_layers]

        return {k: v for k, v in base_run.items() if k in self.run_layers}
