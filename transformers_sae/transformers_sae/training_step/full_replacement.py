from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..metrics import cos_dist_loss, kl_loss, mse_loss
from ..ops import clone_sae
from ..replacement_model import ReplacementModel, make_replacement_model
from ..sae import SAE
from .training_step import Stepper

if TYPE_CHECKING:
    from ..training import TrainingConfig


class FullReplacementTrainingStepper(Stepper):
    saes: Dict[int, SAE]

    def __init__(self, base_model: ReplacementModel, saes: Dict[int, SAE]):
        super().__init__(
            base_model,
            make_replacement_model(base_model, saes),
        )
        self.saes = {**saes}

    def make_checkpoint(self, layer: int, offload_to_cpu: bool = True) -> SAE:
        return clone_sae(self.saes[layer], to_device="cpu" if offload_to_cpu else None)

    def run_baseline(
        self, batch: DataBatch, cache: torch.Tensor | None
    ) -> Dict[int, ActivationBatch]:
        """Need to calculate "baseline" features for every layer after the first, to use in
        cos_dist loss. These are the counterfactual features that would have been computed on
        the baseline activations.
        """
        baseline_activations = super().run_baseline(batch, cache)

        with torch.no_grad(), self.autocast():
            for layer in range(1, self.base_model.num_layers):
                baseline_activations[
                    layer
                ].sae_features = self.replacement_model.get_layer(layer).sae.encode(
                    baseline_activations[layer].layer_output,
                    token_mask=batch.token_mask,
                )

        return baseline_activations

    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]:
        return make_activation_batch(
            self.replacement_model,
            [(layer, "sae") for layer in self.saes.keys()]
            + [(self.replacement_model.num_layers, "layer")],
            batch,
            start_input=baseline_activations[0].layer_output,
            start_layer=0,
            end_layer=self.replacement_model.num_layers + 1,
            start_at_sae=True,
        )

    @property
    def run_layers(self) -> List[int]:
        return list(range(self.base_model.num_layers + 1))

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Tuple[torch.Tensor, Dict[int, Dict[str, float]]]:
        # Reconstruction loss for first layer
        reconstruction_loss = mse_loss(
            training_batch.replacement_activations[0].sae_output,
            training_batch.baseline_activations[0].layer_output,
            training_batch.input_data,
        )

        # Feature cosdist loss for other layers
        feature_loss = torch.zeros((1,), device=self.base_model.device)
        for layer in range(1, self.base_model.num_layers):
            feature_loss += cos_dist_loss(
                training_batch.replacement_activations[layer].sae_features,
                training_batch.baseline_activations[layer].sae_features,
                training_batch.input_data,
            )

        downstream_kl_loss = kl_loss(
            training_batch.replacement_activations[
                self.base_model.num_layers
            ].log_probs,
            training_batch.baseline_activations[self.base_model.num_layers].log_probs,
            training_batch.input_data,
        )

        # Balance KL loss to MSE loss so that its scale is on equal footing with our other methods.
        kl_scale = 1.0
        feature_scale = config.downstream_reconstruction_weight[0]
        reconstruction_scale = config.reconstruction_weight[0]
        if config.balance_reconstruction_losses[0]:
            kl_scale *= reconstruction_loss.item() / (downstream_kl_loss.item() + 1e-8)
            feature_scale *= reconstruction_loss.item() / (feature_loss.item() + 1e-8)

        weighted_kl_loss = kl_scale * downstream_kl_loss
        weighted_reconstruction_loss = reconstruction_scale * reconstruction_loss
        weighted_feature_loss = feature_scale * feature_loss

        loss = (
            weighted_kl_loss + weighted_reconstruction_loss + weighted_feature_loss
        ) / (self.base_model.num_layers + 1)

        return loss, {
            # TODO: refactor to save information per-layer, without redundancy for stuff like total loss
            # and KL
            layer: {
                "total_loss": loss.item(),
                "raw_loss.reconstruction": reconstruction_loss.item(),
                "raw_loss.kl": downstream_kl_loss.item(),
                "raw_loss.features": feature_loss.item(),
                "weighted_loss.reconstruction": weighted_reconstruction_loss.item(),
                "weighted_loss.kl": weighted_kl_loss.item(),
                "weighted_loss.features": weighted_feature_loss.item(),
            }
            for layer in range(self.base_model.num_layers)
        }
