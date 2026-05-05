from typing import Dict

from ..activation_data import ActivationBatch, make_activation_batch
from ..data_batch import DataBatch
from ..replacement_model import ReplacementModel, make_replacement_model
from ..sae import SAE
from .next_layer_finetuned import NextLayerFinetunedTrainingStepper


class InPlaceFinetunedTrainingStepper(NextLayerFinetunedTrainingStepper):
    def __init__(
        self, base_model: ReplacementModel, target_layer: int, saes: Dict[int, SAE]
    ):
        super(NextLayerFinetunedTrainingStepper, self).__init__(
            base_model,
            make_replacement_model(
                base_model,
                saes,
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
            start_layer=-1,
            end_layer=self.replacement_model.num_layers + 1,
        )
