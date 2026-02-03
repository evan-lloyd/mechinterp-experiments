from itertools import product
from typing import TYPE_CHECKING, Dict, Optional

import torch

from .activation_data import ActivationBatch, TrainingBatch, run_replacement_model
from .data_batch import DataBatch
from .metrics import cos_dist_loss, downstream_loss, kl_loss, mse_loss
from .ops import ensure_tensor
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
        baseline_run = self.run_baseline(relevant_layers, batch, cache)

        hooks = {
            (self.target_layer, "sae_output"): self.train_model.transformer.h[
                self.target_layer
            ].sae,
            (self.target_layer, "sae_features"): self.train_model.transformer.h[
                self.target_layer
            ].sae.encoder,
        }
        if self.target_layer + 1 >= self.train_model.config.num_layers:
            hooks[self.target_layer + 1] = self.train_model.lm_head
        else:
            hooks[(self.target_layer + 1, "sae_output")] = (
                self.train_model.transformer.h[self.target_layer + 1].sae
            )

            hooks[(self.target_layer + 1, "sae_features")] = (
                self.train_model.transformer.h[self.target_layer + 1].sae.encoder
            )
        replacement_run = run_replacement_model(
            self.train_model,
            hooks,
            batch,
            start_input=ensure_tensor(
                baseline_run[self.target_layer].to(self.base_model.device)
            ),
            start_layer=self.target_layer,
            end_layer=self.target_layer + 2,
            start_at_sae=True,
        )
        result = TrainingBatch(
            batch,
            replacement_activations={
                layer: ActivationBatch(
                    sae_features=ensure_tensor(
                        replacement_run[(layer, "sae_features")]
                    ),
                    sae_output=ensure_tensor(replacement_run[(layer, "sae_output")]),
                )
                if layer < self.train_model.config.num_layers
                else ActivationBatch(logits=ensure_tensor(replacement_run[layer]))
                for layer in relevant_layers
            },
            baseline_activations={
                layer: ActivationBatch(layer_output=ensure_tensor(baseline_run[layer]))
                if layer < self.base_model.config.num_layers
                else ActivationBatch(logits=ensure_tensor(baseline_run[layer]))
                for layer in relevant_layers
            },
            replacement_layers=[self.target_layer],
        )
        # Need SAE features for next layer?
        if self.target_layer + 1 < self.train_model.config.num_layers:
            result.baseline_activations[
                self.target_layer + 1
            ].sae_features = self.train_model.transformer.h[
                self.target_layer + 1
            ].sae.encode(baseline_run[self.target_layer + 1])
        return result

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
