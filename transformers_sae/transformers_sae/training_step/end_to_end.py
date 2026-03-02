from typing import TYPE_CHECKING, Dict, List

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..metrics import kl_loss, mse_loss
from ..replacement_model import make_replacement_model, ReplacementModel
from ..sae import SAE
from .training_step import Stepper

if TYPE_CHECKING:
    from ..training import TrainingConfig


class EndToEndTrainingStepper(Stepper):
    sae: SAE
    target_layer: int

    def __init__(
        self, base_model: ReplacementModel, target_layer: int, saes: Dict[int, SAE]
    ):
        super().__init__(
            base_model,
            make_replacement_model(base_model, {target_layer: saes[target_layer]}),
        )
        self.target_layer = target_layer
        self.sae = saes[target_layer]

    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]:
        return make_activation_batch(
            self.replacement_model,
            [(layer, "layer") for layer in self.run_layers[1:]]
            + [(self.target_layer, "sae")],
            batch,
            start_input=baseline_activations[self.target_layer].layer_output,
            start_layer=self.target_layer,
            end_layer=self.replacement_model.num_layers + 1,
            start_at_sae=True,
        )

    @property
    def run_layers(self) -> List[int]:
        return list(range(self.target_layer, self.base_model.num_layers + 1))

    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]:
        downstream_reconstruction_loss = torch.zeros(
            (1,), device=self.base_model.device
        )
        for layer in range(self.target_layer + 1, self.base_model.num_layers):
            downstream_reconstruction_loss += mse_loss(
                training_batch.replacement_activations[layer].layer_output,
                training_batch.baseline_activations[layer].layer_output,
                training_batch.input_data,
            )

        downstream_kl_loss = kl_loss(
            training_batch.replacement_activations[self.base_model.num_layers].log_probs,
            training_batch.baseline_activations[self.base_model.num_layers].log_probs,
            training_batch.input_data,
        )

        num_downstream_layers = self.base_model.num_layers - self.target_layer - 1

        # Balance KL loss to the downstream MSE terms (or current layer if none) so that its scale
        # is on equal footing with our other methods.
        kl_scale = 1.0
        downstream_scale = config.downstream_reconstruction_weight[self.target_layer]
        if config.balance_reconstruction_losses[self.target_layer]:
            if num_downstream_layers == 0:
                with torch.no_grad():
                    mse_scale = mse_loss(
                        training_batch.replacement_activations[
                            self.target_layer
                        ].sae_output,
                        training_batch.baseline_activations[
                            self.target_layer
                        ].layer_output,
                        training_batch.input_data,
                    ).item()
            else:
                mse_scale = downstream_reconstruction_loss.item()
            kl_scale *= mse_scale / (downstream_kl_loss.item() + 1e-8)

        weighted_kl_loss = kl_scale * downstream_kl_loss
        weighted_downstream_reconstruction_loss = (
            downstream_scale * downstream_reconstruction_loss
        )

        loss = (weighted_kl_loss + weighted_downstream_reconstruction_loss) / (
            min(num_downstream_layers + 1, 2)
        )
        loss.backward()

        return {
            "total_loss": loss.item(),
            "raw_loss.downstream_reconstruction": downstream_reconstruction_loss.item(),
            "raw_loss.kl": downstream_kl_loss.item(),
            "weighted_loss.downstream_reconstruction": weighted_downstream_reconstruction_loss.item(),
            "weighted_loss.kl": weighted_kl_loss.item(),
        }
