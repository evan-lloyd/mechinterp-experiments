from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..metrics import cos_dist_loss, kl_loss, mse_loss
from ..replacement_model import ReplacementModel, make_replacement_model
from ..sae import SAE
from .training_step import MultiSAEStepper

if TYPE_CHECKING:
    from ..training import TrainingConfig


class FullReplacementTrainingStepper(MultiSAEStepper):
    def _make_replacement_model(
        self, base_model: ReplacementModel, saes: dict[int, SAE]
    ) -> ReplacementModel:
        return make_replacement_model(base_model, saes)

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
                sae = self.replacement_model.get_layer(layer).sae
                orig_train_state = [a.training for a in sae.encoder.activation]
                # Turn off train mode for activation functions, so that for eg BatchTopK,
                # we don't tune the thresholds on the baseline model.
                for a in sae.encoder.activation:
                    a.train(False)
                try:
                    baseline_activations[layer].sae_features = sae.encode(
                        baseline_activations[layer].layer_output,
                        token_mask=batch.token_mask,
                    )
                finally:
                    for t, a in zip(orig_train_state, sae.encoder.activation):
                        a.train(t)

        return baseline_activations

    def run_replacement(
        self, batch: DataBatch, baseline_activations: dict[int, ActivationBatch]
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
        # Mean MSE, which will serve as a reference for loss scaling
        # reconstruction_loss = torch.zeros((1,), device=self.base_model.device)
        reconstruction_losses = []
        mean_reconstruction = 0.0
        for layer in range(0, self.base_model.num_layers):
            layer_reconstruction = mse_loss(
                training_batch.replacement_activations[layer].sae_output,
                training_batch.baseline_activations[layer].layer_output,
                training_batch.input_data,
            )
            reconstruction_losses.append(layer_reconstruction)
            mean_reconstruction += (
                layer_reconstruction.item() / self.base_model.num_layers
            )

        # Rescale each layer's loss to match the mean MSE
        for layer in range(0, self.base_model.num_layers):
            reconstruction_losses[layer] *= mean_reconstruction / (
                reconstruction_losses[layer].item() + 1e-8
            )

        reconstruction_loss = (
            torch.stack(reconstruction_losses).sum() / self.base_model.num_layers
        )

        # Feature cosdist loss for later layers. Note we don't additionally rescale to their mean
        # because cosdist is naturally on the same scale across layers.
        feature_loss = torch.zeros((1,), device=self.base_model.device)
        for layer in range(1, self.base_model.num_layers):
            feature_loss += cos_dist_loss(
                training_batch.replacement_activations[layer].sae_features,
                training_batch.baseline_activations[layer].sae_features,
            ) / (self.base_model.num_layers - 1)

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
            kl_scale *= mean_reconstruction / (downstream_kl_loss.item() + 1e-8)
            feature_scale *= mean_reconstruction / (feature_loss.item() + 1e-8)

        weighted_kl_loss = kl_scale * downstream_kl_loss
        weighted_reconstruction_loss = reconstruction_scale * reconstruction_loss
        weighted_feature_loss = feature_scale * feature_loss

        loss = (
            weighted_kl_loss + weighted_reconstruction_loss + weighted_feature_loss
        ) / 3.0

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
