from __future__ import annotations

import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    ValuesView,
)

import cloudpickle
import numpy as np
import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

from transformers_sae.data_batch import DataBatch
from transformers_sae.metrics import feature_mse_loss, kl_loss, l0_eval

from .activation_cache import load_cache
from .activation_data import (
    TrainingBatch,
    make_activation_batch,
    make_batch_for_evals,
)
from .multiline_progress import MultilineProgress
from .ops import (
    find_checkpoint_after,
    find_latest_checkpoint,
    load_checkpoint,
    save_training_result,
)
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE
from .tokenization import make_dataloader
from .training_step import (
    EndToEndFullTrainingStepper,
    EndToEndTrainingStepper,
    FullReplacementFinetunedTrainingStepper,
    FullReplacementTrainingStepper,
    InPlaceFinetunedTrainingStepper,
    KLFinetuneTrainingStepper,
    MultiSAEStepper,
    NextLayerFinetunedTrainingStepper,
    NextLayerTrainingStepper,
    SingleSAEStepper,
    StandardTrainingStepper,
    Stepper,
)
from .validation import LayerEval, run_evals


class TrainingMethod(Enum):
    standard = "Standard"
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "KL Fine-tuning"
    next_layer_finetuned = "Next Layer + Fine-Tuning"
    e2e_full = "End-to-end Full Replacement"
    full_replacement = "Full Replacement"
    full_replacement_finetuned = "Full Replacement Fine-Tuning"
    in_place_finetuned = "Next Layer + In-Place Fine-Tuning"

    @classmethod
    def _stepper_class_for(cls, method: TrainingMethod) -> type[Stepper]:
        return {
            cls.standard: StandardTrainingStepper,
            cls.next_layer: NextLayerTrainingStepper,
            cls.e2e: EndToEndTrainingStepper,
            cls.e2e_full: EndToEndFullTrainingStepper,
            cls.finetuned: KLFinetuneTrainingStepper,
            cls.next_layer_finetuned: NextLayerFinetunedTrainingStepper,
            cls.full_replacement: FullReplacementTrainingStepper,
            cls.full_replacement_finetuned: FullReplacementFinetunedTrainingStepper,
            cls.in_place_finetuned: InPlaceFinetunedTrainingStepper,
        }[method]

    @classmethod
    def to_stepper(
        cls,
        method: TrainingMethod,
        model: ReplacementModel,
        layer: int,
        saes: dict[int, SAE],
    ):
        stepper_class = cls._stepper_class_for(method)
        if issubclass(stepper_class, SingleSAEStepper):
            return stepper_class(model, layer, saes)
        elif issubclass(stepper_class, MultiSAEStepper):
            return stepper_class(model, saes)
        else:
            raise ValueError(f"Unknown stepper type: {stepper_class}")


class LRSchedule(Protocol):
    def __call__(self, frac_trained: float, **kwargs) -> float: ...


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
    eval_interval: int
    train_layers: List[int]
    betas: Tuple[float, float] = (0.9, 0.999)
    lr: float = 1e-3
    reconstruction_weight: Mapping[int, float] = 1.0
    downstream_reconstruction_weight: Mapping[int, float] = 1.0
    decoder_lr: Mapping[int, float | None] = None
    encoder_lr: Mapping[int, float | None] = None
    interaction_lr: Mapping[int, float | None] = None
    threshold_lr: Mapping[int, float | None] = None
    # Default schedule is constant
    lr_schedule: LRSchedule = lambda frac_trained, **kwargs: 1.0
    balance_reconstruction_losses: bool | Mapping[int, bool] = True
    finetune_fraction: Optional[float] = None
    normalize_decoder: bool | Mapping[int, bool] = True
    method: TrainingMethod

    def __post_init__(self):
        """Simplify accessing per-layer parameters; non-mappings are treated as if they are constant across
        layers."""
        for attr in (
            "reconstruction_weight",
            "downstream_reconstruction_weight",
            "balance_reconstruction_losses",
            "decoder_lr",
            "encoder_lr",
            "interaction_lr",
            "threshold_lr",
            "normalize_decoder",
        ):
            val = getattr(self, attr)
            if not isinstance(val, Mapping):
                setattr(self, attr, defaultdict(lambda v=val: v))


@dataclass(kw_only=True)
class SAECheckpoint:
    total_tokens_trained: int = 0
    step_tokens_trained: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )
    step_metrics: Dict[str, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.empty((0,), dtype=np.float32))
    )
    sae: Optional[SAE] = None
    _is_finalized: bool = field(init=False, default=False)

    def finalize(self):
        self._is_finalized = True

    def append(self, other: Dict[str, float | np.ndarray] | LayerEval):
        assert not self._is_finalized, (
            "Attempted to add metrics to a finalized checkpoint"
        )

        if isinstance(other, dict):
            other_iter = other.items
        elif isinstance(other, LayerEval):

            def other_iter() -> Iterator[Tuple[str, np.ndarray | float | None]]:
                for k in other.__class__.__dataclass_fields__:
                    v = getattr(other, k)
                    yield k, v

        for k, v in other_iter():
            if isinstance(v, np.ndarray):
                self.step_metrics[k] = np.concat((self.step_metrics[k], v))
            elif v is not None:
                self.step_metrics[k] = np.append(self.step_metrics[k], v)


class TrainingResult:
    _layer_results: Dict[int, List[SAECheckpoint]]

    def __init__(self, saes: Dict[int, SAE]):
        self._layer_results = {
            layer: [SAECheckpoint(sae=sae)] for layer, sae in saes.items()
        }

    def __getitem__(self, layer: int) -> List[SAECheckpoint]:
        return self._layer_results[layer]

    def __contains__(self, layer: int) -> bool:
        return layer in self._layer_results

    def __iter__(self):
        return iter(self._layer_results)

    def __len__(self) -> int:
        return len(self._layer_results)

    def keys(self) -> KeysView[int]:
        return self._layer_results.keys()

    def values(self) -> ValuesView[List[SAECheckpoint]]:
        return self._layer_results.values()

    def items(self) -> ItemsView[int, List[SAECheckpoint]]:
        return self._layer_results.items()

    def get(self, layer: int, default=None):
        return self._layer_results.get(layer, default)

    def __repr__(self) -> str:
        return repr(self._layer_results)

    @property
    def final_saes(self) -> Dict[int, SAE]:
        return self.checkpoint_saes(-1)

    def checkpoint_saes(self, checkpoint_index: int):
        return {
            layer: lr[checkpoint_index].sae for layer, lr in self._layer_results.items()
        }

    def clone_from_checkpoint(self, checkpoint_index: int):
        saes = self.checkpoint_saes(checkpoint_index)
        for layer, sae in list(saes.items()):
            saes[layer] = SAE(deepcopy(sae.config))
            if sae._device_tracker.device != torch.device("meta"):
                saes[layer].init_weights(sae)
        result = TrainingResult(saes)
        for layer, lr in result.items():
            lr[0].total_tokens_trained = self[layer][
                checkpoint_index
            ].total_tokens_trained
        return result


def make_optimizer(
    saes: Dict[int, SAE], layers: List[int], config: TrainingConfig
) -> torch.optim.Adam:
    param_groups = []
    param_groups.extend(
        [
            {
                "params": [param],
                "lr": config.decoder_lr[layer] or config.lr,
                "name": f"decoder_{layer}",
            }
            for layer in layers
            for param in saes[layer].decoder.decoder_params()
            if param.requires_grad
        ]
    )
    param_groups.extend(
        [
            {
                "params": [param],
                "lr": config.encoder_lr[layer] or config.lr,
                "name": f"encoder_{layer}",
            }
            for layer in layers
            for param in saes[layer].encoder.encoder_params()
            if param.requires_grad
        ]
    )
    param_groups.extend(
        [
            {
                "params": [param],
                "lr": config.interaction_lr[layer] or config.lr,
                "name": f"{pg_name}_{layer}",
            }
            for layer in layers
            for pg_name, param in saes[layer].encoder.interaction_params()
            if param.requires_grad
        ]
    )

    for pg in param_groups:
        pg["base_lr"] = pg["lr"]

    return torch.optim.Adam(param_groups, betas=config.betas)


def tune_activation_thresholds(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    num_tokens: int,
) -> None:
    """For BatchTopK SAEs, do a special training run that adjusts only the thresholds used
    in the BatchTopK activation function. This is mandatory for replacement models, since
    the additional error will almost surely have distorted the scale of inputs reaching each
    SAE, and so we will no longer hit the desired sparsity level without this step. About 1e6
    tokens seem to be sufficient.

    Note that this function adjusts the thresholds in-place, but does not save the result to
    disk.
    """
    replacement_model = make_replacement_model(model, saes)
    for sae in saes.values():
        sae.eval()
        sae.encoder.train_activations()
        sae.onload()

    progress = MultilineProgress(
        total=num_tokens,
        desc=["Tuning BatchTopK thresholds for replacement model"],
        num_header_lines=1,
    )
    num_used_tokens = 0
    for batch in make_dataloader(
        replacement_model,
        tokenizer,
        dataset,
        max_tokens=num_tokens,
        tokenizer_batch_size=tokenizer_batch_size,
        inference_batch_size=inference_batch_size,
    ):
        # NB: the activation functions handle their own exponential average update, so we
        # don't need any explicit optimizers
        batch.to(replacement_model.device)

        # Run the model until just prior to outputting logits, since this will be enough
        # to get activations to flow through each SAE.
        make_activation_batch(
            replacement_model,
            [],
            batch,
            end_layer=model.num_layers,
        )

        num_used_tokens += batch.num_tokens
        progress.total = max(num_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

    progress.close()


def tune_encoder(
    reference_model: ReplacementModel,
    tokenizer: AutoTokenizer,
    reference_saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    num_encoder_tuning_tokens: int,
    num_threshold_tuning_tokens: int = int(1e6),
    offload_after_training: bool = True,
    checkpoint_dir: Optional[str] = None,
    force_retrain: bool = False,
    train_encoders_from_scratch: bool = False,
) -> TrainingResult:
    """Do an encoder-only training run of the given SAEs, with their dictionaries fixed. This adapts the
    encoders to work within the full replacement model, without changing the semantics of the SAE features.
    """
    # TODO: now that the structure is nailed down, this obviously should use the TrainingStepper system
    try:
        first_sae_layer = min(reference_saes.keys())
        last_sae_layer = max(reference_saes.keys())
        layers_to_tune = list(range(first_sae_layer, last_sae_layer + 1))
        training_saes = {
            layer: SAE(deepcopy(sae.config)) for layer, sae in reference_saes.items()
        }
        train_result = TrainingResult(training_saes)

        # Thresholds may change for previous layers as we continue to tune later ones, so make sure
        # to apply them.
        loaded_thresholds = {}
        if checkpoint_dir and os.path.isdir(checkpoint_dir):
            max_idx = None
            thresholds_file = None
            for fname in os.listdir(checkpoint_dir):
                m = re.match(r"tuned_thresholds_(\d+)$", fname)
                if m:
                    idx = int(m.group(1))
                    if max_idx is None or idx > max_idx:
                        max_idx = idx
                        thresholds_file = fname
            if thresholds_file:
                with open(os.path.join(checkpoint_dir, thresholds_file), "rb") as f:
                    loaded_thresholds = cloudpickle.load(f)

        for layer, sae in training_saes.items():
            if checkpoint_dir and not force_retrain:
                latest_checkpoint = find_latest_checkpoint(checkpoint_dir, layer)
            else:
                latest_checkpoint = None

            if latest_checkpoint is not None:
                print("Loading checkpoint", latest_checkpoint)
                checkpoint = load_checkpoint(latest_checkpoint)
                sae = checkpoint.sae
                assert sae is not None, (
                    f"Checkpoint {latest_checkpoint} was missing SAE data"
                )
                checkpoint.sae = None

                train_result._layer_results[layer] = [
                    checkpoint,
                    SAECheckpoint(
                        sae=sae,
                        total_tokens_trained=checkpoint.total_tokens_trained,
                    ),
                ]
                training_saes[layer] = sae
                training_saes[layer].eval()
                training_saes[layer].encoder.train_activations()
                if layer in loaded_thresholds:
                    print(f"Updated thresholds for layer {layer}")
                    sae.set_activation_thresholds(loaded_thresholds[layer])
                layers_to_tune.remove(layer)
                sae.onload()
                continue
            else:
                sae.init_weights(reference_saes[layer])
                sae.decoder.requires_grad_(False)
                for i, a in enumerate(sae.encoder.activation):
                    if hasattr(a, "threshold_offset"):
                        a.threshold_offset.fill_(
                            reference_saes[layer]
                            .encoder.activation[i]
                            .threshold_offset.item()
                        )
                        a.threshold = (
                            reference_saes[layer]
                            .encoder.activation[i]
                            .threshold.detach()
                            .contiguous()
                        )
                    else:
                        a.threshold.fill_(
                            reference_saes[layer].encoder.activation[i].threshold.item()
                        )

        for sae in reference_saes.values():
            sae.eval()

        for sae_replacement_layer in reference_model.sae_layers.values():
            sae_replacement_layer.sae.eval()
            sae_replacement_layer.sae.onload()

        for layer in layers_to_tune:
            reference_sae = reference_saes[layer]
            training_sae = training_saes[layer]

            def _init_sae(init_layer: int):
                training_saes[init_layer].init_weights(None)
                training_saes[init_layer].decoder.init_weights(
                    reference_saes[init_layer].decoder
                )
                training_saes[init_layer].encoder.init_weights(
                    training_saes[init_layer].decoder
                )
                training_saes[init_layer].decoder.requires_grad_(False)

            if train_encoders_from_scratch:
                _init_sae(layer)
                if layer + 1 in training_saes:
                    _init_sae(layer + 1)

            if layer + 1 in training_saes:
                training_saes[layer + 1].onload()
                reference_saes[layer + 1].onload()

            training_sae.onload()
            reference_sae.onload()

            # Make sure we give the final layer enough time to set thresholds
            num_tokens_for_layer = (
                num_encoder_tuning_tokens + num_threshold_tuning_tokens
                if layer == last_sae_layer
                else num_encoder_tuning_tokens
            )
            progress = MultilineProgress(
                total=num_tokens_for_layer,
                desc=[f"Tuning encoder {layer}"],
                num_header_lines=1,
            )
            metrics = {
                k: []
                for k in ("total_loss", "cur_layer_loss", "next_layer_loss")
                + tuple(f"rre_{metric_layer}" for metric_layer in layers_to_tune)
            }
            num_used_tokens = 0
            num_used_encoder_tokens = None
            optimizer = make_optimizer(
                training_saes,
                [layer] + ([layer + 1] if layer + 1 in reference_saes else []),
                config,
            )

            replacement_model = make_replacement_model(
                reference_model,
                {i: training_saes[i] for i in range(first_sae_layer, layer + 1)},
                override_replacement_layers=True,
            )
            replacement_input_model = make_replacement_model(
                reference_model,
                {i: training_saes[i] for i in range(first_sae_layer, layer)},
                override_replacement_layers=True,
            )
            for batch in make_dataloader(
                reference_model,
                tokenizer,
                dataset,
                max_tokens=num_tokens_for_layer,
                tokenizer_batch_size=config.tokenizer_batch_size,
                inference_batch_size=config.training_batch_size,
            ):
                assert isinstance(batch, DataBatch)
                optimizer.zero_grad()
                batch.to(reference_model.device)
                if (
                    num_used_tokens < num_encoder_tuning_tokens
                    or layer < last_sae_layer
                ):
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type="cuda"
                            if reference_model.device.type == "cuda"
                            else "cpu",
                            dtype=torch.bfloat16,
                        ),
                    ):
                        activation_requests = [
                            (layer + 1, "layer"),
                            (first_sae_layer, "layer"),
                            (layer, "layer"),
                        ]
                        if layer + 1 in reference_model.sae_layers:
                            activation_requests.append((layer + 1, "sae"))
                        if layer in reference_model.sae_layers:
                            activation_requests.append((layer, "sae"))

                        # Get the features that our SAE would have output in the reference model
                        reference_activations = make_activation_batch(
                            reference_model,
                            activation_requests,
                            batch,
                            end_layer=layer + 2,
                        )

                        if layer + 1 in reference_saes:
                            if layer + 1 in reference_model.sae_layers:
                                expected_features = reference_activations[
                                    layer + 1
                                ].sae_features
                            else:
                                expected_features = reference_saes[layer + 1].encode(
                                    reference_activations[layer + 1].layer_output,
                                    batch.token_mask,
                                )
                        else:
                            expected_log_probs = reference_activations[
                                layer + 1
                            ].log_probs

                        if layer in reference_model.sae_layers:
                            cur_layer_expected_features = reference_activations[
                                layer
                            ].sae_features
                        else:
                            cur_layer_expected_features = reference_sae.encode(
                                reference_activations[layer].layer_output,
                                batch.token_mask,
                            )

                        # Get replacement model input as late as we can go before needing grad
                        if layer > first_sae_layer:
                            replacement_input = make_activation_batch(
                                replacement_input_model,
                                [(layer, "layer")],
                                batch,
                                start_input=reference_activations[
                                    first_sae_layer
                                ].layer_output,
                                start_layer=first_sae_layer,
                                end_layer=layer + 1,
                                start_at_sae=True,
                                stop_before_sae=True,
                            )

                            start_input = replacement_input[layer].layer_output
                            del replacement_input
                        else:
                            start_input = reference_activations[layer].layer_output

                        del reference_activations

                    with torch.autocast(
                        device_type="cuda"
                        if reference_model.device.type == "cuda"
                        else "cpu",
                        dtype=torch.bfloat16,
                    ):
                        with torch.no_grad():
                            max_feature_scale = 10 * cur_layer_expected_features.max()
                        replacement_activations = make_activation_batch(
                            replacement_model,
                            [(layer + 1, "layer"), (layer, "sae")],
                            batch,
                            start_input=start_input,
                            start_layer=layer,
                            end_layer=layer + 2,
                            start_at_sae=True,
                            additional_sae_kwargs={
                                "feature_soft_cap": max_feature_scale
                            },
                        )
                        del start_input

                        cur_layer_actual_features = replacement_activations[
                            layer
                        ].sae_features
                        assert cur_layer_actual_features is not None
                        actual_log_probs = replacement_activations[layer + 1].log_probs
                        next_layer_input = replacement_activations[
                            layer + 1
                        ].layer_output
                        del replacement_activations

                        cur_layer_loss = feature_mse_loss(
                            cur_layer_actual_features,
                            cur_layer_expected_features,
                        )
                        del cur_layer_actual_features
                        del cur_layer_expected_features

                        if layer + 1 in training_saes:
                            with torch.no_grad():
                                max_feature_scale = 10 * expected_features.max()
                            actual_features = training_saes[layer + 1].encode(
                                next_layer_input,
                                batch.token_mask,
                                feature_soft_cap=max_feature_scale,
                            )
                            del next_layer_input

                            # Using MSE loss on features here (rather than cosdist as we do elsewhere) because ideally
                            # our tuned SAE matches the original features *exactly* on the distorted input.
                            next_layer_loss = feature_mse_loss(
                                actual_features,
                                expected_features,
                            )
                            del actual_features
                            del expected_features
                        elif layer + 1 == reference_model.num_layers:
                            next_layer_loss = kl_loss(
                                actual_log_probs,
                                expected_log_probs,
                                batch,
                            )
                            del actual_log_probs
                            del expected_log_probs
                            kl_scale = cur_layer_loss.item() / (
                                next_layer_loss.item() + 1e-8
                            )
                            next_layer_loss = kl_scale * next_layer_loss
                        else:
                            next_layer_loss = cur_layer_loss.item()

                        loss = (cur_layer_loss + next_layer_loss) / 2
                    # end autocast

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            p
                            for pg in optimizer.param_groups
                            for p in pg["params"]
                            if p.requires_grad
                        ],
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    num_used_tokens += batch.num_tokens
                    metrics["total_loss"].append(loss.item())
                    metrics["cur_layer_loss"].append(cur_layer_loss.item())
                    metrics["next_layer_loss"].append(
                        next_layer_loss.item()
                        if isinstance(next_layer_loss, torch.Tensor)
                        else next_layer_loss
                    )
                    progress.total = max(num_tokens_for_layer, num_used_tokens)
                    progress.update(batch.num_tokens)

                    progress.set_postfix(
                        {
                            "loss": loss.item(),
                            "cur_layer_loss": cur_layer_loss.item(),
                            "next_layer_loss": next_layer_loss.item()
                            if isinstance(next_layer_loss, torch.Tensor)
                            else next_layer_loss,
                            "avg_loss": np.mean(metrics["total_loss"][-50:]),
                        }
                    )
                    del loss
                    del next_layer_loss
                    del cur_layer_loss

                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["base_lr"] * config.lr_schedule(
                            max(
                                min(num_used_tokens / num_encoder_tuning_tokens, 1.0),
                                0.0,
                            ),
                            pg_name=pg["name"],
                        )
                    # end for each batch
                else:
                    # Just tuning activation threshold for final layer
                    # TODO: this should be a separate function
                    with (
                        torch.no_grad(),
                        torch.autocast(
                            device_type="cuda"
                            if reference_model.device.type == "cuda"
                            else "cpu",
                            dtype=torch.bfloat16,
                        ),
                    ):
                        if num_used_encoder_tokens is None:
                            num_used_encoder_tokens = num_used_tokens

                        ab = make_activation_batch(
                            replacement_model,
                            [(last_sae_layer, "sae")],
                            batch,
                            end_layer=reference_model.num_layers,
                        )
                        assert ab[last_sae_layer].sae_features is not None
                        l0 = l0_eval(
                            ab[last_sae_layer].sae_features,
                            "float",
                        )
                        progress.set_postfix(
                            {
                                "l0": l0,
                                "avg_loss": np.mean(metrics["total_loss"][-50:]),
                            },
                            refresh=False,
                        )

                    num_used_tokens += batch.num_tokens
                    progress.total = max(num_tokens_for_layer, num_used_tokens)
                    progress.update(batch.num_tokens)
                # end for each batch

            progress.close()
            if checkpoint_dir:
                checkpoint = SAECheckpoint(
                    sae=training_sae,
                    total_tokens_trained=num_used_tokens
                    if num_used_encoder_tokens is None
                    else num_used_encoder_tokens,  # Don't count activation tuning tokens
                    step_metrics={k: np.array(lh) for k, lh in metrics.items()},
                )
                checkpoint.finalize()
                save_training_result(
                    {layer: [checkpoint]},
                    checkpoint_dir,
                    keep_in_ram=False,
                    blocking=True,
                )
                with open(f"{checkpoint_dir}/tuned_thresholds_{layer}", "wb") as f:
                    cloudpickle.dump(
                        {
                            save_layer: sae.activation_thresholds()
                            for save_layer, sae in training_saes.items()
                        },
                        f,
                    )
            training_sae.eval()
            training_sae.encoder.train_activations()
            # end for each layer

        # end training loop

        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in reference_saes.values():
                    sae.offload()
                for sae in training_saes.values():
                    sae.offload()
            except Exception:
                pass


def training_loop(
    stepper: Stepper,
    train_result: TrainingResult,
    result_layers: List[int],
    eval_fn: Callable[[TrainingBatch], Tuple[Dict[int, LayerEval], Dict[str, float]]],
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    optimizer: torch.optim.Optimizer,
    progress_desc: str,
    make_checkpoints_at: List[int] | None = None,
    previous_trained_tokens: int = 0,
    checkpoint_dir: Optional[str] = None,
    backward_fn: Optional[Callable[[torch.Tensor], Any]] = None,
) -> None:
    if make_checkpoints_at is None:
        make_checkpoints_at = []
    make_checkpoints_at = sorted(make_checkpoints_at)
    cur_checkpoint = 0

    max_tokens = config.num_train_tokens

    num_used_tokens = 0
    eval_threshold = 0
    progress = MultilineProgress(
        total=max_tokens - previous_trained_tokens,
        desc=[progress_desc],
        num_header_lines=1,
    )
    for step, batch in enumerate(
        make_dataloader(
            stepper.base_model,
            tokenizer,
            dataset,
            max_tokens=max_tokens,
            tokenizer_batch_size=config.tokenizer_batch_size,
            inference_batch_size=config.training_batch_size,
            offset=previous_trained_tokens,
        )
    ):
        optimizer.zero_grad()
        if cache_dir is not None:
            cache = load_cache(
                stepper.base_model.num_layers,
                cache_dir,
                step * config.training_batch_size,
                batch,
            )
        else:
            cache = None
        batch.to(stepper.base_model.device)

        with stepper.autocast():
            training_batch = stepper.make_batch(batch, cache)
            loss, step_result = stepper.step(training_batch, config)

        if backward_fn is not None:
            backward_fn(loss)
        else:
            loss.backward()

        def _do_step():
            nonlocal loss, num_used_tokens
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for pg in optimizer.param_groups
                    for p in pg["params"]
                    if p.requires_grad
                ],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            stepper.post_step(config)
            loss = loss.item()
            num_used_tokens += batch.num_tokens

        # After first batch, step before doing evals
        if num_used_tokens + previous_trained_tokens > 0:
            _do_step()

        if num_used_tokens >= eval_threshold:
            evals, postfix_dict = eval_fn(training_batch)
            progress.set_postfix(
                postfix_dict
                | {"loss": loss.item() if isinstance(loss, torch.Tensor) else loss},
                refresh=False,
            )
            eval_threshold = min(
                eval_threshold + config.eval_interval,
                max_tokens - previous_trained_tokens,
            )

            # Update train results, only on eval steps
            for layer, checkpoints in train_result.items():
                if layer not in result_layers:
                    continue
                checkpoints[-1].step_tokens_trained = np.append(
                    checkpoints[-1].step_tokens_trained,
                    num_used_tokens + previous_trained_tokens,
                )
                checkpoints[-1].append(evals[layer])
                checkpoints[-1].append(step_result[layer])
                checkpoints[-1].total_tokens_trained = (
                    num_used_tokens + previous_trained_tokens
                )

        # On the first batch only, we do evals before updating params
        if num_used_tokens + previous_trained_tokens == 0:
            _do_step()

        should_make_checkpoint = False
        # Handle edge case where we hit multiple checkpoint thresholds after one batch
        while (
            cur_checkpoint < len(make_checkpoints_at)
            and num_used_tokens + previous_trained_tokens
            >= make_checkpoints_at[cur_checkpoint]
        ):
            should_make_checkpoint = True
            cur_checkpoint += 1

        # Don't make a checkpoint if this is the final batch
        if (
            should_make_checkpoint
            and num_used_tokens + previous_trained_tokens < max_tokens
        ):
            for layer, checkpoints in train_result.items():
                if layer not in result_layers:
                    continue
                # Finalize current checkpoint
                checkpoint_sae = checkpoints[-1].sae
                checkpoints[-1].total_tokens_trained = (
                    num_used_tokens + previous_trained_tokens
                )
                checkpoints[-1].sae = stepper.make_checkpoint(layer)
                checkpoints[-1].finalize()

                if checkpoint_dir:
                    save_training_result(
                        {layer: [checkpoints[-1]]},
                        checkpoint_dir,
                        keep_in_ram=False,
                        blocking=False,
                    )

                # Initialize new checkpoint
                checkpoints.append(
                    SAECheckpoint(
                        sae=checkpoint_sae,
                        total_tokens_trained=num_used_tokens + previous_trained_tokens,
                    )
                )

        for pg in optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * config.lr_schedule(
                max(
                    min(
                        (num_used_tokens + previous_trained_tokens)
                        / config.num_train_tokens,
                        1.0,
                    ),
                    0.0,
                ),
                pg_name=pg["name"],
            )

        progress.total = max(max_tokens - previous_trained_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

    for layer, checkpoints in train_result.items():
        if layer not in result_layers:
            continue

        checkpoints[-1].total_tokens_trained = num_used_tokens + previous_trained_tokens
        checkpoints[-1].finalize()
    progress.close()


def _train_evals(
    base_model: ReplacementModel,
    eval_model: ReplacementModel,
    layers: List[int],
    calc_kl: bool,
) -> Callable[[TrainingBatch], Tuple[Dict[int, LayerEval], Dict[str, float]]]:
    wanted_layers = list(layers)
    target_layer = max(wanted_layers)
    if target_layer + 1 in eval_model.sae_layers:
        wanted_layers.append(target_layer + 1)
    if calc_kl:
        wanted_layers.append(base_model.num_layers)

    @torch.autocast(
        device_type="cuda" if base_model.device.type == "cuda" else "cpu",
        dtype=torch.bfloat16,
    )
    def eval_fn(
        training_batch: TrainingBatch,
    ) -> Tuple[Dict[int, LayerEval], Dict[str, float]]:
        result = run_evals(
            make_batch_for_evals(
                base_model,
                eval_model,
                training_batch,
                wanted_layers,
            ),
            # Want evals for this layer, next layer, and/or on logits
            wanted_layers,
            aggregate=True,
        )
        return result, (
            {"rre": result[target_layer].rre}
            | (
                {"next_rre": result[target_layer + 1].rre}
                if target_layer + 1 in result
                and result[target_layer + 1].rre is not None
                else {}
            )
            | {
                "L0": result[target_layer].l0,
            }
            | (
                {
                    "kl": result[base_model.num_layers].kl,
                }
                if calc_kl
                else {}
            )
            | {
                "live_features": result[target_layer].live_features,
            }
        )

    return eval_fn


def train(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    initial_saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    checkpoints_at: Optional[List[int]] = None,
    offload_after_training: bool = True,
    checkpoint_dir: Optional[str] = None,
    force_retrain: bool = False,
    fine_tune_source_dir: Optional[str] = None,
    skip_kl_eval: bool = False,
    fine_tune_in_place: bool = False,
    override_token_offset: Optional[int] = None,
    fresh_init: bool = False,
    extend_training_run: bool = False,
) -> TrainingResult:
    try:
        model.eval()

        assert override_token_offset is not None or not fine_tune_in_place, (
            "override_token_offset must be set if fine_tune_in_place"
        )

        if extend_training_run:
            assert checkpoint_dir is not None, (
                "Must specify checkpoint_dir to extend_training_run"
            )
            assert not fine_tune_in_place, (
                "Can't fine_tune_in_place with extend_training_run"
            )
            final_checkpoint_dir = (
                checkpoint_dir + f"_extended_{config.num_train_tokens}_tokens"
            )
        else:
            final_checkpoint_dir = checkpoint_dir

        if fine_tune_in_place:
            training_saes = initial_saes
        elif fine_tune_source_dir is None and not extend_training_run:
            training_saes = {
                layer: SAE(deepcopy(initial_saes[layer].config))
                for layer in config.train_layers
            }
        else:
            training_saes = {}
        train_result = TrainingResult(training_saes)

        for layer in sorted(config.train_layers, reverse=True):
            if final_checkpoint_dir and not force_retrain:
                latest_checkpoint = find_latest_checkpoint(final_checkpoint_dir, layer)
                if extend_training_run and latest_checkpoint is None:
                    latest_checkpoint = find_latest_checkpoint(checkpoint_dir, layer)
                    assert latest_checkpoint is not None, (
                        f"Couldn't find checkpoint for layer {layer}"
                    )
            else:
                latest_checkpoint = None

            if latest_checkpoint is not None:
                print("Loading checkpoint", latest_checkpoint)
                checkpoint = load_checkpoint(latest_checkpoint)
                sae = checkpoint.sae
                assert sae is not None, (
                    f"Checkpoint {latest_checkpoint} was missing SAE data"
                )
                checkpoint.sae = None

                train_result._layer_results[layer] = [
                    checkpoint,
                    SAECheckpoint(
                        sae=sae,
                        total_tokens_trained=checkpoint.total_tokens_trained,
                    ),
                ]
                training_saes[layer] = sae
                token_offset = checkpoint.total_tokens_trained
                if fine_tune_in_place:
                    with open(
                        f"{final_checkpoint_dir}/train_thresholds_{layer}", "rb"
                    ) as f:
                        loaded_thresholds = cloudpickle.load(f)
                    for update_layer, sae in training_saes.items():
                        sae.set_activation_thresholds(loaded_thresholds[update_layer])
            elif fine_tune_source_dir is not None:
                # Init weights from source checkpoint
                source_checkpoint, token_offset = find_checkpoint_after(
                    fine_tune_source_dir,
                    layer,
                    # TODO: clean this up. Encoder-tuned checkpoints currently only show as having
                    # the number of tokens used in the encoder tuning phase, instead of total.
                    -1
                    if fine_tune_in_place
                    else int(
                        (1.0 - config.finetune_fraction) * config.num_train_tokens
                    ),
                )
                if override_token_offset:
                    token_offset = override_token_offset
                print(f"Loading {source_checkpoint} for finetuning")
                sae = load_checkpoint(source_checkpoint).sae
                assert sae is not None and token_offset is not None
                training_saes[layer] = sae
                train_result._layer_results[layer] = [
                    SAECheckpoint(sae=sae, total_tokens_trained=token_offset)
                ]
                if fine_tune_in_place:
                    try:
                        with open(
                            f"{fine_tune_source_dir}/tuned_thresholds_{max(config.train_layers)}",
                            "rb",
                        ) as f:
                            loaded_thresholds = cloudpickle.load(f)
                    except Exception:
                        # Finetuning from previous fine_tune_in_place run
                        with open(
                            f"{fine_tune_source_dir}/train_thresholds_{min(config.train_layers)}",
                            "rb",
                        ) as f:
                            loaded_thresholds = cloudpickle.load(f)
                    sae.set_activation_thresholds(loaded_thresholds[layer])
            else:
                # Init weights from next layer, if it exists
                sae = training_saes[layer]
                if fresh_init:
                    sae.init_weights(None)
                else:
                    sae.init_weights(training_saes.get(layer + 1))
                token_offset = 0

            sae.set_activation_threshold_lr(config.threshold_lr[layer])

            if token_offset >= config.num_train_tokens:
                continue

            # We train all SAEs simultaneously in full_replacement, so don't start the training loop
            # until we've initialized all of them.
            if config.method is TrainingMethod.full_replacement and layer > 0:
                continue

            if checkpoints_at is not None:
                make_checkpoints_at = [t for t in checkpoints_at if t > token_offset]
            else:
                make_checkpoints_at = None

            eval_model = make_replacement_model(model, training_saes)
            stepper = TrainingMethod.to_stepper(
                config.method, model, layer, training_saes
            )

            if config.method is TrainingMethod.full_replacement:
                optimizer = make_optimizer(
                    training_saes, list(training_saes.keys()), config
                )
            else:
                if fine_tune_in_place:
                    for other_layer in training_saes.keys():
                        training_saes[other_layer].train(layer == other_layer)
                        if other_layer != layer:
                            training_saes[other_layer].encoder.train_activations()
                else:
                    # Keep SAEs used in the replacement model in train mode, so eg for BatchTopK
                    # we automatically retune the threshold based on now having an SAE at the previous layer.
                    # Unused layers should be in eval mode.
                    for other_layer in range(layer, model.num_layers):
                        if other_layer in stepper.replacement_model.sae_layers:
                            training_saes[other_layer].onload()
                            training_saes[other_layer].train()
                            training_saes[other_layer].requires_grad_(
                                layer == other_layer
                            )
                        elif other_layer in training_saes:
                            training_saes[other_layer].eval()
                            if offload_after_training and skip_kl_eval:
                                training_saes[other_layer].offload()
                # backward_fn = None
                optimizer = make_optimizer(training_saes, [layer], config)

            training_loop(
                stepper,
                train_result,
                list(range(model.num_layers))
                if config.method is TrainingMethod.full_replacement
                else [layer],
                # For consistency across methods, we always run our evals with the full replacement model
                # starting from the target layer
                _train_evals(
                    model,
                    eval_model,
                    list(range(model.num_layers))
                    if config.method is TrainingMethod.full_replacement
                    else [layer],
                    calc_kl=not skip_kl_eval,
                ),
                tokenizer,
                dataset,
                config,
                cache_dir,
                optimizer,
                f"Layer {layer}",
                previous_trained_tokens=token_offset,
                make_checkpoints_at=make_checkpoints_at,
                checkpoint_dir=final_checkpoint_dir,
                # backward_fn=backward_fn,
            )
            if final_checkpoint_dir:
                save_training_result(
                    {
                        result_layer: [train_result[result_layer][-1]]
                        for result_layer in (
                            range(model.num_layers)
                            if config.method is TrainingMethod.full_replacement
                            else [layer]
                        )
                    },
                    final_checkpoint_dir,
                    keep_in_ram=True,
                    blocking=True,
                )
                if fine_tune_in_place:
                    with open(
                        f"{final_checkpoint_dir}/train_thresholds_{layer}", "wb"
                    ) as f:
                        cloudpickle.dump(
                            {
                                save_layer: sae.activation_thresholds()
                                for save_layer, sae in training_saes.items()
                            },
                            f,
                        )

        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in train_result.final_saes.values():
                    sae.offload()
            except Exception:
                pass
