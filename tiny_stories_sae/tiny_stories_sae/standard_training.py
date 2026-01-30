from copy import copy
from typing import Dict, Optional

import torch

from .data_batch import DataBatch
from .replacement_model import (
    make_replacement_model,
)
from .sae import SAE
from .sae_data import _ensure_tensor, run_replacement_model
from .training_step import ActivationBatch, Stepper, TrainingBatch, mse_loss


class StandardTrainingStepper(Stepper):
    sae: SAE

    def __init__(
        self, base_model: torch.nn.Module, target_layer: int, saes: Dict[int, SAE]
    ):
        super().__init__(
            base_model,
            make_replacement_model(
                base_model, {f"transformer.h.{target_layer}": saes[target_layer]}
            ),
            saes,
        )
        self.target_layer = target_layer
        self.sae = saes[target_layer]

    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch:
        if cache is not None:
            baseline_activations = cache
        else:
            baseline_activations = run_replacement_model(
                self.base_model,
                {self.target_layer: self.base_model.transformer.h[self.target_layer]},
                batch,
                start_layer=-1,  # not cached, so we start from raw input
                end_layer=self.target_layer + 1,  # non-inclusive range
            )
        replacement_activations = run_replacement_model(
            self.replacement_model,
            {
                "sae_output": self.replacement_model.transformer.h[
                    self.target_layer
                ].sae,
            },
            batch,
            start_input=_ensure_tensor(
                baseline_activations[self.target_layer].to(self.base_model.device)
            ),
            start_layer=self.target_layer,
            end_layer=self.target_layer + 1,
            start_at_sae=True,
        )
        return TrainingBatch(
            batch,
            baseline_activations={
                k: ActivationBatch(layer_output=_ensure_tensor(v))
                for k, v in baseline_activations.items()
            },
            replacement_activations={
                self.target_layer: ActivationBatch(
                    sae_output=_ensure_tensor(replacement_activations["sae_output"])
                )
            },
        )

    def step(self, training_batch: TrainingBatch) -> Dict[str, torch.Tensor]:
        reconstruction_loss = mse_loss(
            training_batch.baseline_activations[self.target_layer].layer_output.to(
                self.base_model.device
            ),
            training_batch.replacement_activations[self.target_layer].sae_output,
            training_batch.input_data,
        )

        reconstruction_loss.backward()

        return {
            "raw_loss.reconstruction": reconstruction_loss,
        }

    def augment_batch_for_evals(
        self, training_batch: TrainingBatch, losses: Dict[str, torch.Tensor]
    ) -> TrainingBatch:
        baseline_activations = copy(training_batch.baseline_activations)
        has_next_layer = self.target_layer < self.base_model.config.num_layers - 1
        needs_logits = self.base_model.config.num_layers not in baseline_activations
        needs_next_layer = (
            has_next_layer and self.target_layer + 1 not in baseline_activations
        )

        # Need logits for evals, and next layer (if it exists)
        if needs_logits or needs_next_layer:
            hooks = {}
            if needs_logits:
                hooks["logits"] = self.base_model.lm_head
            if needs_next_layer:
                hooks["next_layer_output"] = self.base_model.transformer.h[
                    self.target_layer + 1
                ]
            new_baseline_run = run_replacement_model(
                self.base_model,
                hooks,
                training_batch.input_data,
                start_input=training_batch.baseline_activations[
                    self.target_layer
                ].layer_output,
                start_layer=self.target_layer + 1,
                end_layer=self.model.config.num_layers + 1,
            )
            if needs_logits:
                baseline_activations[self.base_model.config.num_layers] = (
                    ActivationBatch(logits=_ensure_tensor(new_baseline_run["logits"]))
                )
            if needs_next_layer:
                baseline_activations[self.target_layer + 1] = ActivationBatch(
                    layer_output=_ensure_tensor(new_baseline_run["next_layer_output"])
                )
            del hooks

        hooks = {"logits": self.full_replacement_model.lm_head}
        # If there *is* a next layer
        if has_next_layer:
            hooks["next_layer_sae_output"] = self.full_replacement_model.transformer.h[
                self.target_layer + 1
            ]
        full_replacement_run = run_replacement_model(
            self.full_replacement_model,
            hooks,
            training_batch.input_data,
            start_input=training_batch.replacement_activations[
                self.target_layer
            ].sae_output,
            start_layer=self.target_layer + 1,
            end_layer=self.model.config.num_layers + 1,
        )
        full_replacement_activations = {
            self.base_model.config.num_layers: ActivationBatch(
                logits=_ensure_tensor(full_replacement_run["logits"])
            )
        }
        if has_next_layer:
            full_replacement_activations[self.target_layer + 1] = ActivationBatch(
                sae_output=_ensure_tensor(full_replacement_run["next_layer_sae_output"])
            )
        return TrainingBatch(
            training_batch.input_data,
            baseline_activations,
            full_replacement_activations,
        )
