from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..metrics import cos_dist_loss, kl_loss, mse_loss
from ..replacement_model import ReplacementModel, make_replacement_model
from ..sae import SAE
from .training_step import SingleSAEStepper

if TYPE_CHECKING:
    from ..training import TrainingConfig


class NextLayerFinetunedTrainingStepper(SingleSAEStepper):
    def __init__(
        self, base_model: ReplacementModel, target_layer: int, saes: Dict[int, SAE]
    ):
        super().__init__(
            base_model,
            make_replacement_model(
                base_model,
                {
                    layer: saes[layer]
                    for layer in range(target_layer, base_model.num_layers)
                },
            ),
            target_layer,
            saes[target_layer],
        )

    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]:
        activation_requests = [
            (self.target_layer, "sae"),
            # Always want logits
            (self.replacement_model.num_layers, "layer"),
        ]
        if self.target_layer + 1 < self.replacement_model.num_layers:
            activation_requests.append((self.target_layer + 1, "sae"))
        return make_activation_batch(
            self.replacement_model,
            activation_requests,
            batch,
            start_input=baseline_activations[self.target_layer].layer_output,
            start_layer=self.target_layer,
            end_layer=self.replacement_model.num_layers + 1,
            start_at_sae=True,
        )

    @property
    def run_layers(self) -> List[int]:
        return list(
            # Logits could be the same as next_layer + 1, hence list(set(...))
            {
                self.target_layer,
                self.target_layer + 1,
                self.base_model.num_layers,
            }
        )

    def run_baseline(
        self, batch: DataBatch, cache: torch.Tensor | None
    ) -> Dict[int, ActivationBatch]:
        baseline_activations = super().run_baseline(batch, cache)

        # Need SAE features for next layer?
        if self.target_layer + 1 < self.replacement_model.num_layers:
            with torch.no_grad(), self.autocast():
                baseline_activations[
                    self.target_layer + 1
                ].sae_features = self.replacement_model.get_layer(
                    self.target_layer + 1
                ).sae.encode(
                    baseline_activations[self.target_layer + 1].layer_output,
                    token_mask=batch.token_mask,
                )

        return baseline_activations

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Tuple[torch.Tensor, Dict[int, Dict[str, float]]]:
        downstream_kl_loss = kl_loss(
            training_batch.replacement_activations[
                self.base_model.num_layers
            ].log_probs,
            training_batch.baseline_activations[self.base_model.num_layers].log_probs,
            training_batch.input_data,
        )
        if (
            training_batch.baseline_activations[self.target_layer + 1].log_probs
            is not None
        ):
            # Handled explicitly above
            downstream_reconstruction_loss = torch.zeros(
                (1,), device=self.replacement_model.device
            )
            effective_loss_terms = 2
        else:
            downstream_reconstruction_loss = cos_dist_loss(
                training_batch.replacement_activations[
                    self.target_layer + 1
                ].sae_features,
                training_batch.baseline_activations[self.target_layer + 1].sae_features,
            )
            effective_loss_terms = 3

        reconstruction_loss = mse_loss(
            training_batch.replacement_activations[self.target_layer].sae_output,
            training_batch.baseline_activations[self.target_layer].layer_output,
            training_batch.input_data,
        )

        reconstruction_scale = config.reconstruction_weight[self.target_layer]
        downstream_scale = config.downstream_reconstruction_weight[self.target_layer]
        kl_scale = 1.0
        if config.balance_reconstruction_losses[self.target_layer]:
            downstream_scale *= reconstruction_loss.item() / (
                downstream_reconstruction_loss.item() + 1e-8
            )
            kl_scale *= reconstruction_loss.item() / (downstream_kl_loss.item() + 1e-8)

        weighted_reconstruction_loss = reconstruction_scale * reconstruction_loss
        weighted_downstream_reconstruction_loss = (
            downstream_scale * downstream_reconstruction_loss
        )
        weighted_downstream_kl_loss = kl_scale * downstream_kl_loss
        loss = (
            weighted_reconstruction_loss
            + weighted_downstream_reconstruction_loss
            + weighted_downstream_kl_loss
        ) / effective_loss_terms

        return loss, {
            self.target_layer: {
                "total_loss": loss.item(),
                "raw_loss.reconstruction": reconstruction_loss.item(),
                "raw_loss.downstream_reconstruction": downstream_reconstruction_loss.item(),
                "raw_loss.kl": downstream_kl_loss.item(),
                "weighted_loss.reconstruction": weighted_reconstruction_loss.item(),
                "weighted_loss.downstream_reconstruction": weighted_downstream_reconstruction_loss.item(),
                "weighted_loss.kl": weighted_downstream_kl_loss.item(),
            }
        }
