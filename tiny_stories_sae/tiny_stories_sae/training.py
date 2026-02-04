from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .activation_data import TrainingBatch, make_batch_for_evals
from .encoder import InteractionEncoder
from .end_to_end_training import EndToEndTrainingStepper
from .multiline_progress import MultilineProgress
from .next_layer_training import NextLayerTrainingStepper
from .replacement_model import make_replacement_model
from .sae import SAE
from .sae_data import (
    cache_on_disk,
    init_cache,
    load_cache,
)
from .standard_training import StandardTrainingStepper
from .tokenization import CONTEXT_LENGTH, input_generator
from .training_step import Stepper
from .validation import run_evals


class TrainingMethod(Enum):
    standard = "Standard"
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "KL Fine-tuning"
    next_layer_finetuned = "Next Layer + Fine-Tuning"


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
    e2e_batch_size: int
    eval_interval: int
    reconstruction_weight: Mapping[int, float]
    downstream_reconstruction_weight: Mapping[int, float]
    train_layers: List[int]
    lr: float = 1e-3
    decoder_lr: Mapping[int, float | None] = None
    encoder_lr: Mapping[int, float | None] = None
    interaction_lr: Mapping[int, float | None] = None
    # Default schedule is constant
    main_lr_schedule: Callable[[float], float] = lambda frac_trained: 1.0
    finetune_lr_schedule: Callable[[float], float] = lambda frac_trained: 1.0
    use_downstream_saes: bool = True
    balance_reconstruction_losses: bool | Mapping[int, bool] = False
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
class TrainingResult:
    total_tokens_trained: int = 0
    step_tokens_trained: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )
    step_results: Dict[str, np.ndarray] = field(
        default_factory=lambda: defaultdict(lambda: np.empty((0,), dtype=np.float32))
    )

    def update(self, other: Dict[str, float]):
        for k, v in other.items():
            self.step_results[k] = np.append(self.step_results[k], v)


def build_cache(
    cache_dir: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    num_tokens: int,
    tokenizer_batch_size: int,
    inference_batch_size: int,
):
    progress = tqdm(total=num_tokens, desc="Building activation cache")

    init_cache(
        model,
        cache_dir,
        model.config.num_layers,
        inference_batch_size,
        1,
        CONTEXT_LENGTH,
        model.config.hidden_size,
    )
    num_used_tokens = 0

    model.eval()
    for batch in input_generator(
        model,
        tokenizer,
        dataset,
        max_tokens=num_tokens,
        tokenizer_batch_size=tokenizer_batch_size,
        inference_batch_size=inference_batch_size,
    ):
        cache_on_disk(cache_dir, model, batch)
        num_used_tokens += batch.num_tokens
        progress.total = max(num_tokens, num_used_tokens)
        progress.update(batch.num_tokens)
    progress.close()


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


def maybe_balance_reconstruction_weights(
    balance_reconstruction_losses: List[bool] | bool,
    reconstruction_weight: List[float] | float,
    downstream_reconstruction_weight: List[float] | float,
    losses: Dict[str, torch.Tensor],
    target_layer: int,
):
    balance_reconstruction_losses = (
        balance_reconstruction_losses[target_layer]
        if hasattr(balance_reconstruction_losses, "__getitem__")
        else balance_reconstruction_losses
    )
    reconstruction_weight = (
        reconstruction_weight[target_layer]
        if hasattr(reconstruction_weight, "__getitem__")
        else reconstruction_weight
    )
    downstream_reconstruction_weight = (
        downstream_reconstruction_weight[target_layer]
        if hasattr(downstream_reconstruction_weight, "__getitem__")
        else downstream_reconstruction_weight
    )
    if balance_reconstruction_losses:
        target_scale = (reconstruction_weight + downstream_reconstruction_weight) * max(
            losses["reconstruction"].item(),
            sum(losses["downstream_reconstruction"]).item(),
            1e-8,
        )
        reconstruction_weight = target_scale / max(
            losses["reconstruction"].item(), 1e-8
        )
        downstream_reconstruction_weight = target_scale / max(
            sum(losses["downstream_reconstruction"]).item(), 1e-8
        )

    return reconstruction_weight, downstream_reconstruction_weight


def training_loop(
    stepper: Stepper,
    eval_fn: Callable[[TrainingBatch], Dict[str, float]],
    tokenizer: AutoTokenizer,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    optimizer: torch.optim.Optimizer,
    progress_desc: str,
    previous_trained_tokens: int = 0,
    max_tokens: Optional[int] = None,
    post_step_hook: Optional[Callable[[int]]] = None,
):
    if max_tokens is None:
        max_tokens = config.num_train_tokens

    num_used_tokens = 0
    eval_threshold = 0
    progress = MultilineProgress(
        total=max_tokens,
        desc=[progress_desc],
        num_header_lines=1,
    )
    train_result = TrainingResult()
    for step, batch in enumerate(
        input_generator(
            stepper.base_model,
            tokenizer,
            dataset,
            max_tokens=max_tokens,
            tokenizer_batch_size=config.tokenizer_batch_size,
            inference_batch_size=config.training_batch_size,
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
        batch.token_mask = batch.token_mask.to(stepper.base_model.device)
        training_batch = stepper.make_batch(batch, cache)

        losses = stepper.step(training_batch, config)

        # After first batch, step before doing evals
        if num_used_tokens > 0:
            optimizer.step()
            num_used_tokens += batch.num_tokens

        if num_used_tokens >= eval_threshold:
            evals = eval_fn(training_batch)
            progress.set_postfix(evals, refresh=False)
            eval_threshold = min(eval_threshold + config.eval_interval, max_tokens)

            # Update train results, only on eval steps
            train_result.step_tokens_trained = np.append(
                train_result.step_tokens_trained, num_used_tokens
            )
            train_result.update(evals)
            train_result.update(losses)
            train_result.total_tokens_trained = num_used_tokens

        # On the first batch only, we do evals before updating params
        if num_used_tokens == 0:
            optimizer.step()
            num_used_tokens += batch.num_tokens

        progress.total = max(max_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

        if post_step_hook:
            post_step_hook(num_used_tokens)

    train_result.total_tokens_trained = num_used_tokens
    progress.close()
    return train_result


def _train_evals(model: torch.nn.Module, saes: Dict[int, SAE], target_layer: int):
    """For consistency across methods, we always run our evals with the full replacement model
    starting from the target layer."""
    eval_model = make_replacement_model(model, saes)

    def eval_fn(training_batch: TrainingBatch) -> Dict[str, float]:
        result = run_evals(
            make_batch_for_evals(
                model,
                eval_model,
                training_batch,
                start_layer=target_layer,
            ),
            # Want evals for this layer, next layer, and/or on logits
            list({target_layer, target_layer + 1, model.config.num_layers}),
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
                "rep_kl": result[model.config.num_layers].kl,
            }
        )

    return eval_fn


def finetune(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    reinit_weights: bool = False,
    post_step_hook: Optional[Callable[[int, int]]] = None,
):
    model.eval()


def train(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    reinit_weights: bool = False,
    post_step_hook: Optional[Callable[[int, int]]] = None,
):
    model.eval()
    train_result: Dict[int, TrainingResult] = {}
    for layer in sorted(config.train_layers, reverse=True):
        if reinit_weights:
            saes[layer].init_weights(saes.get(layer + 1))
        optimizer = make_optimizer(saes, [layer], config)
        if config.method in (TrainingMethod.standard, TrainingMethod.finetuned):
            stepper = StandardTrainingStepper(model, layer, saes)
        elif config.method in (
            TrainingMethod.next_layer,
            TrainingMethod.next_layer_finetuned,
        ):
            stepper = NextLayerTrainingStepper(model, layer, saes)
        elif config.method is TrainingMethod.e2e:
            stepper = EndToEndTrainingStepper(model, layer, saes)
        train_result[layer] = training_loop(
            stepper,
            _train_evals(model, saes, layer),
            tokenizer,
            dataset,
            config,
            cache_dir,
            optimizer,
            f"Layer {layer}",
            previous_trained_tokens=0,
            max_tokens=int((1.0 - config.finetune_fraction) * config.num_train_tokens)
            if config.finetune_fraction
            else None,
            post_step_hook=partial(post_step_hook, layer)
            if post_step_hook is not None
            else None,
        )

    return train_result
