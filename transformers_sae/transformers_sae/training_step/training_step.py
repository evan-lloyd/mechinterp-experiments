from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from ..activation_data import ActivationBatch, TrainingBatch, make_activation_batch
from ..data_batch import DataBatch
from ..ops import clone_sae
from ..replacement_model import ReplacementModel
from ..sae import SAE

if TYPE_CHECKING:
    from ..training import TrainingConfig


class Stepper(ABC):
    base_model: ReplacementModel
    replacement_model: ReplacementModel

    def __init__(
        self,
        base_model: ReplacementModel,
        replacement_model: ReplacementModel,
    ):
        self.base_model = base_model
        self.replacement_model = replacement_model

    @property
    def replacement_layers(self) -> List[int]:
        return sorted(list(self.replacement_model.sae_layers.keys()))

    @property
    def run_layers(self) -> List[int]:
        return sorted(list(self.replacement_model.sae_layers.keys()))

    def make_batch(
        self, batch: DataBatch, cache: Optional[Dict[str, torch.Tensor]]
    ) -> TrainingBatch:
        baseline_activations = self.run_baseline(batch, cache)
        replacement_activations = self.run_replacement(batch, baseline_activations)
        return TrainingBatch(
            batch,
            replacement_activations=replacement_activations,
            baseline_activations=baseline_activations,
            replacement_layers=self.replacement_layers,
        )

    def make_checkpoint(self, offload_to_cpu: bool = True) -> SAE:
        assert hasattr(self, "sae"), (
            f"{self.__class__} needs to override make_checkpoint"
        )
        return clone_sae(self.sae, to_device="cpu" if offload_to_cpu else None)

    @abstractmethod
    def step(
        self, training_batch: TrainingBatch, config: "TrainingConfig"
    ) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def run_replacement(
        self, batch: DataBatch, baseline_activations: ActivationBatch
    ) -> Dict[int, ActivationBatch]: ...

    def autocast(self):
        return torch.autocast(
            device_type="cuda"
            if self.replacement_model.device.type == "cuda"
            else "cpu",
            dtype=torch.bfloat16,
        )

    def run_baseline(
        self, batch: DataBatch, cache: torch.Tensor | None
    ) -> Dict[int, ActivationBatch]:
        if cache is not None:
            base_run = cache
            for k, v in list(base_run.items()):
                base_run[k] = ActivationBatch(layer_output=v.to(self.base_model.device))
        else:
            with torch.no_grad():
                base_run = make_activation_batch(
                    self.base_model,
                    [(layer, "layer") for layer in self.run_layers],
                    batch,
                    start_layer=-1,  # not cached, so we start from raw input
                    end_layer=max(self.run_layers) + 1,  # non-inclusive range
                )

        # We don't cache logits, so we may need to reconstruct them if loading from cache.
        if (
            self.base_model.num_layers in self.run_layers
            and self.base_model.num_layers not in base_run
        ):
            with torch.no_grad():
                start_layer = max(k for k in base_run)
                base_run[self.base_model.num_layers] = make_activation_batch(
                    self.base_model,
                    [(self.base_model.num_layers, "layer")],
                    batch,
                    start_layer=start_layer + 1,
                    start_input=base_run[start_layer].layer_output,
                    end_layer=self.base_model.num_layers + 1,  # non-inclusive range
                )[self.base_model.num_layers]

        return {k: v for k, v in base_run.items() if k in self.run_layers}
