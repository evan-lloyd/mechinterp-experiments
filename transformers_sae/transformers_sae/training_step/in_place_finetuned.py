from typing import Dict

import torch

from ..activation_data import ActivationBatch, make_activation_batch
from ..data_batch import DataBatch
from ..replacement_model import ReplacementModel, make_replacement_model
from ..sae import SAE
from .full_replacement_finetuned import FullReplacementFinetunedTrainingStepper


class InPlaceFinetunedTrainingStepper(FullReplacementFinetunedTrainingStepper):
    def _make_replacement_model(
        self, base_model: ReplacementModel, saes: dict[int, SAE]
    ) -> ReplacementModel:
        return make_replacement_model(
            base_model,
            saes,
        )

    def run_replacement(
        self, batch: DataBatch, baseline_activations: dict[int, ActivationBatch]
    ) -> Dict[int, ActivationBatch]:
        activation_requests = [
            (self.target_layer, "sae"),
            # Always want logits
            (self.replacement_model.num_layers, "layer"),
        ]
        if self.target_layer + 1 < self.replacement_model.num_layers:
            activation_requests.append((self.target_layer + 1, "sae"))

        with torch.no_grad():
            replacement_input = make_activation_batch(
                self.replacement_model,
                [(self.target_layer, "layer")],
                batch,
                end_layer=self.target_layer + 1,
                stop_before_sae=True,
                # Bit of a hack to prevent loss from exploding on rare, extreme outliers. Unlike in
                # tune_encoders, we don't have a "sensible" baseline based on what our SAE would output
                # on base model activations, because our starting point was already encoder tuned, so
                # instead put an arbitrary constant.
                # TODO: we probably should just bake this in for LISTA.
                additional_sae_kwargs={"feature_soft_cap": 1000.0},
            )[self.target_layer].layer_output
        return make_activation_batch(
            self.replacement_model,
            activation_requests,
            batch,
            start_input=replacement_input,
            start_layer=self.target_layer,
            end_layer=self.replacement_model.num_layers + 1,
            start_at_sae=True,
            additional_sae_kwargs={"feature_soft_cap": 1000.0},
        )
