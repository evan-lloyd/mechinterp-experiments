from typing import TYPE_CHECKING, Dict, Optional

import torch

from .activation_data import (
    TrainingBatch,
    make_activation_batch,
)
from .data_batch import DataBatch
from .metrics import cos_dist_loss, downstream_loss, kl_loss, mse_loss
from .replacement_model import (
    make_replacement_model,
)
from .sae import SAE
from .training_step import Stepper

if TYPE_CHECKING:
    from .training import TrainingConfig


class NextLayerTrainingStepper(Stepper):
    sae: SAE
    target_layer: int

    def __init__(
        self, base_model: torch.nn.Module, target_layer: int, saes: Dict[int, SAE]
    ):
        replacement_model_dict = {target_layer: saes[target_layer]}
        if target_layer + 1 in saes:
            replacement_model_dict[target_layer + 1] = saes[target_layer + 1]

        super().__init__(
            base_model,
            make_replacement_model(base_model, replacement_model_dict),
        )
        self.target_layer = target_layer
        self.sae = saes[target_layer]

    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch:
        relevant_layers = [self.target_layer, self.target_layer + 1]
        baseline_activations = self.run_baseline(relevant_layers, batch, cache)

        # Need SAE features for next layer?
        if self.target_layer + 1 < self.train_model.config.num_layers:
            with torch.no_grad():
                baseline_activations[
                    self.target_layer + 1
                ].sae_features = self.train_model.transformer.h[
                    self.target_layer + 1
                ].sae.encode(baseline_activations[self.target_layer + 1].layer_output)

        replacement_activations = make_activation_batch(
            self.train_model,
            [
                (self.target_layer, "sae"),
                (self.target_layer + 1, "layer")
                if self.target_layer + 1 >= self.train_model.config.num_layers
                else (self.target_layer + 1, "sae"),
            ],
            batch,
            start_input=baseline_activations[self.target_layer].layer_output,
            start_layer=self.target_layer,
            end_layer=self.target_layer + 2,
            start_at_sae=True,
        )
        return TrainingBatch(
            batch,
            replacement_activations=replacement_activations,
            baseline_activations=baseline_activations,
            replacement_layers=[self.target_layer, self.target_layer + 1],
        )

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]:
        if (
            training_batch.baseline_activations[self.target_layer + 1].logits
            is not None
        ):
            downstream_attr = "logits"
            downstream_fn = kl_loss
        else:
            downstream_attr = "sae_features"
            downstream_fn = cos_dist_loss

        downstream_reconstruction_loss = downstream_loss(
            training_batch.replacement_activations[self.target_layer + 1],
            training_batch.baseline_activations[self.target_layer + 1],
            training_batch.input_data,
            downstream_attr,
            downstream_fn,
        )

        reconstruction_loss = mse_loss(
            training_batch.replacement_activations[self.target_layer].sae_output,
            training_batch.baseline_activations[self.target_layer].layer_output.to(
                self.base_model.device
            ),
            training_batch.input_data,
        )

        reconstruction_scale = config.reconstruction_weight[self.target_layer]
        downstream_scale = config.downstream_reconstruction_weight[self.target_layer]
        if config.balance_reconstruction_losses[self.target_layer]:
            downstream_scale *= reconstruction_loss.item() / (
                downstream_reconstruction_loss.item() + 1e-8
            )

        weighted_reconstruction_loss = reconstruction_scale * reconstruction_loss
        weighted_downstream_reconstruction_loss = (
            downstream_scale * downstream_reconstruction_loss
        )

        loss = weighted_reconstruction_loss + weighted_downstream_reconstruction_loss
        loss.backward()

        return {
            "total_loss": loss.item(),
            "raw_loss.reconstruction": reconstruction_loss.item(),
            "raw_loss.downstream_reconstruction": downstream_reconstruction_loss.item(),
            "weighted_loss.reconstruction": weighted_reconstruction_loss.item(),
            "weighted_loss.downstream_reconstruction": weighted_downstream_reconstruction_loss.item(),
        }
