from typing import TYPE_CHECKING, Dict, Optional

import torch

from .activation_data import TrainingBatch, make_activation_batch
from .data_batch import DataBatch
from .metrics import mse_loss
from .replacement_model import (
    make_replacement_model,
)
from .sae import SAE
from .training_step import Stepper

if TYPE_CHECKING:
    from .training import TrainingConfig


class StandardTrainingStepper(Stepper):
    sae: SAE
    target_layer: int

    def __init__(
        self, base_model: torch.nn.Module, target_layer: int, saes: Dict[int, SAE]
    ):
        super().__init__(
            base_model,
            make_replacement_model(base_model, {target_layer: saes[target_layer]}),
        )
        self.target_layer = target_layer
        self.sae = saes[target_layer]

    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch:
        baseline_activations = self.run_baseline([self.target_layer], batch, cache)
        replacement_activations = make_activation_batch(
            self.train_model,
            [(self.target_layer, "sae")],
            batch,
            start_input=baseline_activations[self.target_layer].layer_output,
            start_layer=self.target_layer,
            end_layer=self.target_layer + 1,
            start_at_sae=True,
        )
        return TrainingBatch(
            batch,
            replacement_activations=replacement_activations,
            baseline_activations=baseline_activations,
            replacement_layers=[self.target_layer],
        )

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]:
        reconstruction_loss = mse_loss(
            training_batch.replacement_activations[self.target_layer].sae_output,
            training_batch.baseline_activations[self.target_layer].layer_output.to(
                self.base_model.device
            ),
            training_batch.input_data,
        )

        reconstruction_loss.backward()

        return {
            "raw_loss.reconstruction": reconstruction_loss.item(),
        }
