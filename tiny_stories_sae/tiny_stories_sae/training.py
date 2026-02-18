from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

from tiny_stories_sae.training_step.end_to_end_full import EndToEndFullTrainingStepper

from .activation_cache import load_cache
from .activation_data import TrainingBatch, make_batch_for_evals
from .encoder import InteractionEncoder
from .multiline_progress import MultilineProgress
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE
from .tokenization import input_generator
from .training_step import (
    EndToEndTrainingStepper,
    KLFinetuneTrainingStepper,
    NextLayerFinetunedTrainingStepper,
    NextLayerTrainingStepper,
    StandardTrainingStepper,
    Stepper,
)
from .validation import run_evals


class TrainingMethod(Enum):
    standard = "Standard"
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "KL Fine-tuning"
    next_layer_finetuned = "Next Layer + Fine-Tuning"
    e2e_full = "End-to-end Full Replacement"


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
    eval_interval: int
    train_layers: List[int]
    lr: float = 1e-3
    reconstruction_weight: Mapping[int, float] = 1.0
    downstream_reconstruction_weight: Mapping[int, float] = 1.0
    decoder_lr: Mapping[int, float | None] = None
    encoder_lr: Mapping[int, float | None] = None
    interaction_lr: Mapping[int, float | None] = None
    # Default schedule is constant
    lr_schedule: Callable[[float], float] = lambda frac_trained: 1.0
    balance_reconstruction_losses: bool | Mapping[int, bool] = True
    finetune_fraction: Optional[float] = None
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

    def clone_history(self) -> "SAECheckpoint":
        """Clones the training history, but not SAE weights, into a new LayerResult."""
        return SAECheckpoint(
            total_tokens_trained=self.total_tokens_trained,
            step_tokens_trained=deepcopy(self.step_tokens_trained),
            step_metrics=deepcopy(self.step_metrics),
            # NB: Deliberately *not* copying SAE
        )

    def finalize(self):
        self._is_finalized = True

    def append(self, other: Dict[str, float | np.ndarray]):
        assert not self._is_finalized, (
            "Attempted to add metrics to a finalized checkpoint"
        )
        for k, v in other.items():
            if isinstance(v, np.ndarray):
                self.step_metrics[k] = np.concat((self.step_metrics[k], v))
            else:
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

    def keys(self):
        return self._layer_results.keys()

    def values(self):
        return self._layer_results.values()

    def items(self):
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
            saes[layer] = SAE(sae.config)
            if sae._device_tracker.device != torch.device("meta"):
                saes[layer].init_weights(sae)
        result = TrainingResult(saes)
        for layer, lr in result.items():
            lr[0].total_tokens_trained = self[layer][
                checkpoint_index
            ].total_tokens_trained
        return result


def make_optimizer(saes: Dict[int, SAE], layers: List[int], config: TrainingConfig):
    param_groups = [
        {
            "params": [
                param
                for layer in layers
                for param in saes[layer].decoder.linear.parameters()
            ],
            "lr": config.decoder_lr or config.lr,
        },
        {
            "params": [
                param
                for layer in layers
                for param in saes[layer].encoder.linear.parameters()
            ],
            "lr": config.encoder_lr or config.lr,
        },
    ]

    if any(
        isinstance(s.encoder, InteractionEncoder)
        for layer, s in saes.items()
        if layer in layers
    ):
        param_groups.append(
            {
                "params": [
                    saes[layer].encoder.interaction
                    for layer in layers
                    if isinstance(saes[layer].encoder, InteractionEncoder)
                ],
                "lr": config.interaction_lr or config.lr,
            }
        )

    for pg in param_groups:
        pg["base_lr"] = pg["lr"]

    return torch.optim.Adam(param_groups, lr=config.lr)


def training_loop(
    stepper: Stepper,
    checkpoints: List[SAECheckpoint],
    eval_fn: Callable[[TrainingBatch], Dict[str, float]],
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    optimizer: torch.optim.Optimizer,
    progress_desc: str,
    make_checkpoints_at: List[int] = None,
    previous_trained_tokens: int = 0,
) -> None:
    sae = checkpoints[-1].sae

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
        input_generator(
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
                stepper.base_model.config.num_layers,
                cache_dir,
                step * config.training_batch_size,
                batch,
            )
        else:
            cache = None
        batch.to(stepper.base_model.device)
        training_batch = stepper.make_batch(batch, cache)

        losses = stepper.step(training_batch, config)

        # After first batch, step before doing evals
        if num_used_tokens + previous_trained_tokens > 0:
            optimizer.step()
            num_used_tokens += batch.num_tokens

        if num_used_tokens >= eval_threshold:
            evals = eval_fn(training_batch)
            progress.set_postfix(evals, refresh=False)
            eval_threshold = min(
                eval_threshold + config.eval_interval,
                max_tokens - previous_trained_tokens,
            )

            # Update train results, only on eval steps
            checkpoints[-1].step_tokens_trained = np.append(
                checkpoints[-1].step_tokens_trained,
                num_used_tokens + previous_trained_tokens,
            )
            checkpoints[-1].append(evals)
            checkpoints[-1].append(losses)
            checkpoints[-1].total_tokens_trained = (
                num_used_tokens + previous_trained_tokens
            )

        # On the first batch only, we do evals before updating params
        if num_used_tokens + previous_trained_tokens == 0:
            optimizer.step()
            num_used_tokens += batch.num_tokens

        should_make_checkpoint = False
        # Handle edge case where we hit multiple checkpoint thresholds after one batch
        while (
            cur_checkpoint < len(make_checkpoints_at)
            and num_used_tokens + previous_trained_tokens
            >= make_checkpoints_at[cur_checkpoint]
        ):
            should_make_checkpoint = True
            cur_checkpoint += 1

        if should_make_checkpoint:
            # Finalize current checkpoint
            checkpoints[-1].total_tokens_trained = (
                num_used_tokens + previous_trained_tokens
            )
            checkpoints[-1].sae = stepper.make_checkpoint()
            checkpoints[-1].finalize()

            # Initialize new checkpoint
            checkpoints.append(
                SAECheckpoint(
                    sae=sae,
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
                )
            )

        progress.total = max(max_tokens - previous_trained_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

    checkpoints[-1].total_tokens_trained = num_used_tokens
    checkpoints[-1].finalize()
    progress.close()


def _train_evals(
    base_model: torch.nn.Module, eval_model: ReplacementModel, target_layer: int
):
    def eval_fn(training_batch: TrainingBatch) -> Dict[str, float]:
        result = run_evals(
            make_batch_for_evals(
                base_model,
                eval_model,
                training_batch,
                start_layer=target_layer,
            ),
            # Want evals for this layer, next layer, and/or on logits
            list({target_layer, target_layer + 1, base_model.config.num_layers}),
            aggregate=True,
        )
        return (
            {"rre": result[target_layer].rre}
            | (
                {"next_rre": result[target_layer + 1].rre}
                if result[target_layer + 1].rre is not None
                else {}
            )
            | {
                "L0": result[target_layer].l0,
                "kl": result[base_model.config.num_layers].kl,
                "live_features": result[target_layer].live_features,
            }
        )

    return eval_fn


def fine_tune(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    train_result: TrainingResult,
    checkpoint_index: int,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    checkpoints_at: Optional[List[int]] = None,
    offload_after_training: bool = True,
):
    if config.method not in (
        TrainingMethod.finetuned,
        TrainingMethod.next_layer_finetuned,
    ):
        raise ValueError(f"{config.method.value} is not a valid method for finetuning")

    try:
        train_result = train_result.clone_from_checkpoint(checkpoint_index)
        training_saes = train_result.checkpoint_saes(0)

        # For consistency across methods, we always run our evals with the full replacement model
        # starting from the target layer
        eval_model = make_replacement_model(model, training_saes)
        for layer in sorted(config.train_layers, reverse=True):
            training_saes[layer].onload()
            optimizer = make_optimizer(training_saes, [layer], config)
            if config.method is TrainingMethod.finetuned:
                stepper = KLFinetuneTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.next_layer_finetuned:
                stepper = NextLayerFinetunedTrainingStepper(model, layer, training_saes)

            training_loop(
                stepper,
                train_result[layer],
                _train_evals(model, eval_model, layer),
                tokenizer,
                dataset,
                config,
                cache_dir,
                optimizer,
                f"Layer {layer}",
                make_checkpoints_at=checkpoints_at,
                previous_trained_tokens=train_result[layer][0].total_tokens_trained,
            )
        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in train_result.final_saes.values():
                    sae.offload()
            except Exception:
                pass


def train(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    initial_saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    checkpoints_at: Optional[List[int]] = None,
    offload_after_training: bool = True,
) -> TrainingResult:
    if config.method not in (
        TrainingMethod.standard,
        TrainingMethod.next_layer,
        TrainingMethod.e2e,
        TrainingMethod.e2e_full,
    ):
        raise ValueError(
            f"Training method {config.method.value} must be finetuned from a checkpoint of another method"
        )
    try:
        model.eval()
        training_saes = {
            layer: SAE(initial_saes[layer].config) for layer in config.train_layers
        }
        train_result = TrainingResult(training_saes)

        # For consistency across methods, we always run our evals with the full replacement model
        # starting from the target layer
        eval_model = make_replacement_model(model, training_saes)
        for layer in sorted(config.train_layers, reverse=True):
            # Init weights from next layer, if it exists
            training_saes[layer].init_weights(training_saes.get(layer + 1))
            optimizer = make_optimizer(training_saes, [layer], config)
            if config.method is TrainingMethod.standard:
                stepper = StandardTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.next_layer:
                stepper = NextLayerTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.e2e:
                stepper = EndToEndTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.e2e_full:
                stepper = EndToEndFullTrainingStepper(model, layer, training_saes)
            training_loop(
                stepper,
                train_result[layer],
                _train_evals(model, eval_model, layer),
                tokenizer,
                dataset,
                config,
                cache_dir,
                optimizer,
                f"Layer {layer}",
                previous_trained_tokens=0,
                make_checkpoints_at=checkpoints_at,
            )
        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in train_result.final_saes.values():
                    sae.offload()
            except Exception:
                pass
