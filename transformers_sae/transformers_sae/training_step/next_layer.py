from typing import TYPE_CHECKING, Dict, List

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..metrics import cos_dist_loss, kl_loss, mse_loss
from ..replacement_model import make_replacement_model, ReplacementModel
from ..sae import SAE
from .training_step import Stepper

if TYPE_CHECKING:
    from ..training import TrainingConfig


class NextLayerTrainingStepper(Stepper):
    sae: SAE
    target_layer: int

    def __init__(
        self, base_model: ReplacementModel, target_layer: int, saes: Dict[int, SAE]
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

    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]:
        return make_activation_batch(
            self.replacement_model,
            [
                (self.target_layer, "sae"),
                (self.target_layer + 1, "layer")
                if self.target_layer + 1 >= self.replacement_model.num_layers
                else (self.target_layer + 1, "sae"),
            ],
            batch,
            start_input=baseline_activations[self.target_layer].layer_output,
            start_layer=self.target_layer,
            end_layer=self.target_layer + 2,
            start_at_sae=True,
        )

    @property
    def run_layers(self) -> List[int]:
        return [self.target_layer, self.target_layer + 1]

    def run_baseline(
        self, batch: DataBatch, cache: torch.Tensor | None
    ) -> Dict[int, ActivationBatch]:
        baseline_activations = super().run_baseline(batch, cache)

        # Need SAE features for next layer?
        if self.target_layer + 1 < self.replacement_model.num_layers:
            with torch.no_grad():
                baseline_activations[
                    self.target_layer + 1
                ].sae_features = self.replacement_model.get_layer(
                    self.target_layer + 1
                ).sae.encode(baseline_activations[self.target_layer + 1].layer_output)

        return baseline_activations

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]:
        if (
            training_batch.baseline_activations[self.target_layer + 1].logits
            is not None
        ):
            downstream_reconstruction_loss = kl_loss(
                training_batch.replacement_activations[self.target_layer + 1].logits,
                training_batch.baseline_activations[self.target_layer + 1].logits,
                training_batch.input_data,
            )
        else:
            downstream_reconstruction_loss = cos_dist_loss(
                training_batch.replacement_activations[
                    self.target_layer + 1
                ].sae_features,
                training_batch.baseline_activations[self.target_layer + 1].sae_features,
                training_batch.input_data,
            )

        reconstruction_loss = mse_loss(
            training_batch.replacement_activations[self.target_layer].sae_output,
            training_batch.baseline_activations[self.target_layer].layer_output,
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

        loss = (
            weighted_reconstruction_loss + weighted_downstream_reconstruction_loss
        ) / 2

        return loss, {
            "total_loss": loss.item(),
            "raw_loss.reconstruction": reconstruction_loss.item(),
            "raw_loss.downstream_reconstruction": downstream_reconstruction_loss.item(),
            "weighted_loss.reconstruction": weighted_reconstruction_loss.item(),
            "weighted_loss.downstream_reconstruction": weighted_downstream_reconstruction_loss.item(),
        }
