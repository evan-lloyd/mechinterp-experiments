from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Dict, List, Mapping, Optional

import torch
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data_batch import DataBatch
from .multiline_progress import MultilineProgress
from .replacement_model import make_replacement_model
from .sae import SAE
from .sae_data import SAEData, cache_on_disk, get_sae_data, init_cache
from .tokenization import CONTEXT_LENGTH, input_generator
from .validation import sae_evals


class TrainingMethod(Enum):
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "Finetuned"


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
    e2e_batch_size: int
    eval_interval: int
    reconstruction_weight: float | Mapping[int, float]
    next_reconstruction_weight: float | Mapping[int, float]
    idempotency_weight: float | Mapping[int, float]
    dense_weight: float | Mapping[int, float]
    train_layers: List[int]
    lr: float = 1e-3
    decoder_lr: Optional[float] = None
    encoder_lr: Optional[float] = None
    dense_decoder_lr: Optional[float] = None
    dense_encoder_lr: Optional[float] = None
    inhibition_lr: Optional[float] = None
    use_next_layer_sae: bool = True
    use_kl_on_final_layer: bool = False
    balance_reconstruction_losses: bool = False | Mapping[int, float]
    use_weighted_mask: bool = False
    method: TrainingMethod
    finetune_fraction: Optional[float] = None


def sae_losses(
    batch: DataBatch,
    this_layer_baseline: SAEData,
    next_layer_baseline: SAEData,
    next_layer_replacement: SAEData,
    sae: SAE,
    use_kl: bool,
    use_next_layer_sae: bool,
    should_reconstruct_next_layer: bool,
):
    token_mask = batch.token_mask.to(sae.device)

    if should_reconstruct_next_layer:
        if use_kl:
            next_layer_reconstruction_loss = torch.nn.KLDivLoss(
                reduction="none", log_target=True
            )(
                next_layer_replacement.original.log_softmax(-1),
                next_layer_baseline.original.log_softmax(-1),
            ).mean(dim=-1)
        else:
            if use_next_layer_sae:
                next_layer_attr = "features"
                next_layer_fn = torch.abs
            else:
                next_layer_attr = "normalized_original"
                next_layer_fn = partial(torch.pow, exponent=2)

            next_layer_reconstruction_loss = next_layer_fn(
                (
                    getattr(next_layer_replacement, next_layer_attr)
                    - getattr(next_layer_baseline, next_layer_attr)
                )
            ).mean(dim=-1)

    reconstruction_loss = (
        (this_layer_baseline.reconstruction - this_layer_baseline.normalized_original)
        ** 2
    ).mean(dim=-1)

    reencode_result = sae.encode(this_layer_baseline.reconstruction)
    if this_layer_baseline.dense_decoding is not None:
        reencoded_features = reencode_result[0]
    else:
        reencoded_features = reencode_result
    idempotency_loss = ((this_layer_baseline.features - reencoded_features).abs()).mean(
        dim=-1
    )
    if this_layer_baseline.dense_decoding is not None:
        dense_loss = (this_layer_baseline.dense_decoding**2).mean(dim=-1)

    result = {
        "reconstruction": (reconstruction_loss * token_mask).sum() / batch.num_tokens,
        "next_reconstruction": (next_layer_reconstruction_loss * token_mask).sum()
        / batch.num_tokens
        if should_reconstruct_next_layer
        else 0.0,
        "idempotency": (idempotency_loss * token_mask).sum() / batch.num_tokens,
        "dense": (dense_loss * token_mask).sum() / batch.num_tokens
        if this_layer_baseline.dense_decoding is not None
        else 0.0,
    }

    if sae.normalize_activations:
        result["normalizer"] = (
            (
                (
                    sae.normalizer(batch.position_ids.to(sae.device))
                    - this_layer_baseline.norms
                )
                ** 2
            )[token_mask.bool()]
        ).mean()

    return result


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
        model.config.num_layers + 1,
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
                param for layer in layers for param in saes[layer].decoder.parameters()
            ],
            "lr": config.decoder_lr or config.lr,
        },
        {
            "params": [
                param for layer in layers for param in saes[layer].encoder.parameters()
            ],
            "lr": config.encoder_lr or config.lr,
        },
    ]
    if any(sae.with_inhibition for sae in saes.values()):
        param_groups.append(
            {
                "params": [
                    param
                    for layer in layers
                    for param in [saes[layer].inhibition]
                    if saes[layer].with_inhibition
                ],
                "lr": config.inhibition_lr or config.lr,
            }
        )
    if any(sae.d_dense is not None for sae in saes.values()):
        param_groups.append(
            {
                "params": [
                    param
                    for layer in layers
                    for param in saes[layer].dense_decoder.parameters()
                    if saes[layer].d_dense is not None
                ],
                "lr": config.dense_decoder_lr or config.lr,
            }
        )
        param_groups.append(
            {
                "params": [
                    param
                    for layer in layers
                    for param in saes[layer].dense_encoder.parameters()
                    if saes[layer].d_dense is not None
                ],
                "lr": config.dense_encoder_lr or config.lr,
            }
        )

    return torch.optim.Adam(param_groups, lr=config.lr)


def maybe_balance_reconstruction_weights(
    balance_reconstruction_losses: List[bool] | bool,
    reconstruction_weight: List[float] | float,
    next_reconstruction_weight: List[float] | float,
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
    next_reconstruction_weight = (
        next_reconstruction_weight[target_layer]
        if hasattr(next_reconstruction_weight, "__getitem__")
        else next_reconstruction_weight
    )
    if balance_reconstruction_losses:
        target_scale = (reconstruction_weight + next_reconstruction_weight) * max(
            losses["reconstruction"].item(),
            losses["next_reconstruction"].item(),
            1e-8,
        )
        reconstruction_weight = target_scale / max(
            losses["reconstruction"].item(), 1e-8
        )
        next_reconstruction_weight = target_scale / max(
            losses["next_reconstruction"].item(), 1e-8
        )

    return reconstruction_weight, next_reconstruction_weight


def train_one_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    target_layer: int,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    did_reinit: bool,
    previous_trained_tokens: int = 0,
    max_tokens: Optional[int] = None,
):
    if max_tokens is None:
        max_tokens = config.num_train_tokens

    train_result = defaultdict(list)
    replacement_model = make_replacement_model(
        model, {f"transformer.h.{layer}": sae for layer, sae in saes.items()}
    )
    optimizer = make_optimizer(saes, [target_layer], config)
    if saes[target_layer].normalize_activations:
        normalizer_optimizer = torch.optim.Adam(
            list(saes[target_layer].normalizer.parameters()), lr=0.01
        )

    progress = MultilineProgress(
        total=max_tokens,
        desc=[f"Layer {target_layer}"],
        num_header_lines=1,
    )

    num_used_tokens = 0
    model.eval()
    eval_threshold = 0
    for step, batch in enumerate(
        input_generator(
            model,
            tokenizer,
            dataset,
            max_tokens=max_tokens,
            tokenizer_batch_size=config.tokenizer_batch_size,
            inference_batch_size=config.training_batch_size,
            use_weighted_mask=config.use_weighted_mask,
        )
    ):
        optimizer.zero_grad()

        num_used_tokens += batch.num_tokens

        result, original_logits = get_sae_data(
            model,
            saes,
            batch,
            target_layer,
            target_layer + 1,
            config.use_next_layer_sae,
            cache_dir,
            step * config.training_batch_size,
            config.use_kl_on_final_layer,
        )

        # Warm up our activation normalizer on the first batch before we start using it
        if did_reinit and saes[target_layer].normalize_activations:
            did_reinit = False
            pos_ids = batch.position_ids[batch.token_mask.bool()].to(
                saes[target_layer].device
            )
            target_norms = (
                result[()][target_layer]
                .norms[batch.token_mask.to(pos_ids.device).bool()]
                .clone()
                .detach()
            )
            saes[target_layer].normalizer.heuristic_init(pos_ids, target_norms)
            for _ in range(10_000):
                normalizer_optimizer.zero_grad()
                (
                    ((saes[target_layer].normalizer(pos_ids) - target_norms) ** 2).sum()
                    / batch.num_tokens
                ).backward()
                normalizer_optimizer.step()

            # Make sure our data uses the computed normalizations
            result, original_logits = get_sae_data(
                model,
                saes,
                batch,
                target_layer,
                target_layer + 1,
                config.use_next_layer_sae,
                cache_dir,
                step * config.training_batch_size,
                config.use_kl_on_final_layer,
            )

        losses = sae_losses(
            batch,
            result[()][target_layer],
            result[()][target_layer + 1],
            result[(target_layer,)][target_layer + 1],
            saes[target_layer],
            config.use_kl_on_final_layer
            and target_layer == model.config.num_layers - 1,
            config.use_next_layer_sae and (target_layer + 1) in saes,
            config.next_reconstruction_weight > 0.0,
        )
        reconstruction_weight, next_reconstruction_weight = (
            maybe_balance_reconstruction_weights(
                config.balance_reconstruction_losses,
                config.reconstruction_weight,
                config.next_reconstruction_weight,
                losses,
                target_layer,
            )
        )
        idempotency_weight = (
            config.idempotency_weight[target_layer]
            if hasattr(config.idempotency_weight, "__getitem__")
            else config.idempotency_weight
        )
        dense_weight = (
            config.dense_weight[target_layer]
            if hasattr(config.dense_weight, "__getitem__")
            else config.dense_weight
        )

        loss = sum(
            (
                losses["reconstruction"] * reconstruction_weight,
                losses["next_reconstruction"] * next_reconstruction_weight,
                losses["idempotency"] * idempotency_weight,
                losses["dense"] * dense_weight,
            )
        )

        loss.backward()

        optimizer.step()

        if saes[target_layer].normalize_activations:
            normalizer_optimizer.zero_grad()
            losses["normalizer"].backward()
            normalizer_optimizer.step()

        if num_used_tokens >= eval_threshold:
            evals = sae_evals(
                batch,
                result[()][target_layer],
                result[()][target_layer + 1],
                result[(target_layer,)][target_layer + 1],
                model,
                saes[target_layer],
                original_logits=original_logits,
                replacement_model=replacement_model,
            )
            for k, v in evals.items():
                train_result[k].append((num_used_tokens + previous_trained_tokens, v))

            progress.set_postfix(
                {k: v for k, v in evals.items() if k not in ("mse", "idm")},
                refresh=False,
            )
            eval_threshold = min(eval_threshold + config.eval_interval, max_tokens)
        progress.total = max(max_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

    progress.close()
    return train_result, num_used_tokens


def train_e2e(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    previous_trained_tokens: int = 0,
):
    train_result = defaultdict(lambda: defaultdict(list))
    assert not any(sae.normalize_activations for sae in saes.values()), (
        "Activation normalization not implemented for e2e training"
    )
    optimizer = make_optimizer(saes, list(range(model.config.num_layers)), config)

    replacement_model = make_replacement_model(
        model, {f"transformer.h.{layer}": sae for layer, sae in saes.items()}
    )

    progress = MultilineProgress(
        total=config.num_train_tokens - previous_trained_tokens,
        desc=[
            f"Layer {layer}"
            for layer in list(range(model.config.num_layers - 1, -1, -1))
        ],
        num_header_lines=model.config.num_layers,
    )

    num_used_tokens = 0
    replacement_model.eval()
    eval_threshold = 0

    batch_size = config.e2e_batch_size
    for step, batch in enumerate(
        input_generator(
            replacement_model,
            tokenizer,
            dataset,
            # For the finetuning method, make sure it sees the tokens at the end of the training set
            # We could also accomplish the same thing by adding a "finetune" split at the dataset level
            offset=previous_trained_tokens,
            max_tokens=config.num_train_tokens,
            tokenizer_batch_size=config.tokenizer_batch_size,
            inference_batch_size=batch_size,
            use_weighted_mask=config.use_weighted_mask,
        )
    ):
        optimizer.zero_grad()
        num_used_tokens += batch.num_tokens

        result, original_logits = get_sae_data(
            replacement_model,
            saes,
            batch,
            0,
            model.config.num_layers,
            config.use_next_layer_sae,
            cache_dir,
            step * batch_size,
            config.use_kl_on_final_layer,
        )

        losses = [
            sae_losses(
                batch,
                result[()][target_layer],
                result[()][target_layer + 1],
                result[(target_layer,)][target_layer + 1],
                saes[target_layer],
                target_layer == model.config.num_layers - 1,
                False,
                target_layer
                == model.config.num_layers
                - 1,  # Reconstruct "next layer" only for the last layer, since that's when we compute the KL
            )
            for target_layer in range(model.config.num_layers)
        ]
        loss = torch.zeros((1,), device=model.device)
        # Keep track of this separately from any sparsity / other losses, so we can rescale the KL
        mse_loss = torch.zeros((1,), device=model.device)

        for target_layer in saes.keys():
            reconstruction_weight, next_reconstruction_weight = (
                maybe_balance_reconstruction_weights(
                    # We will manually balance KL vs the sum of all MSE losses
                    False,
                    config.reconstruction_weight,
                    config.next_reconstruction_weight,
                    losses[target_layer],
                    target_layer,
                )
            )
            idempotency_weight = (
                config.idempotency_weight[target_layer]
                if hasattr(config.idempotency_weight, "__getitem__")
                else config.idempotency_weight
            )
            dense_weight = (
                config.dense_weight[target_layer]
                if hasattr(config.dense_weight, "__getitem__")
                else config.dense_weight
            )
            loss += sum(
                (
                    losses[target_layer]["reconstruction"] * reconstruction_weight,
                    losses[target_layer]["idempotency"] * idempotency_weight,
                    losses[target_layer]["dense"] * dense_weight,
                )
            )
            with torch.no_grad():
                mse_loss += (
                    losses[target_layer]["reconstruction"] * reconstruction_weight
                )

            if target_layer == model.config.num_layers - 1:
                kl_loss = losses[target_layer]["next_reconstruction"]

        # Balance KL loss with MSE ala (Karvonen 2025)
        with torch.no_grad():
            kl_scale = mse_loss.item() / (kl_loss.item() + 1e-8)
        loss += kl_loss * kl_scale

        loss.backward()
        optimizer.step()

        if num_used_tokens >= eval_threshold:
            evals = [
                sae_evals(
                    batch,
                    result[()][target_layer],
                    result[()][target_layer + 1],
                    result[(target_layer,)][target_layer + 1],
                    model,
                    saes[target_layer],
                    original_logits=original_logits,
                    replacement_model=replacement_model,
                )
                for target_layer in range(model.config.num_layers - 1, -1, -1)
            ]
            for i, e in enumerate(evals):
                for k, v in e.items():
                    train_result[i][k].append(
                        (num_used_tokens + previous_trained_tokens, v)
                    )
            progress.set_postfix(
                [{k: v for k, v in e.items() if k not in ("mse",)} for e in evals],
                refresh=False,
            )
            eval_threshold = min(
                eval_threshold + config.eval_interval,
                config.num_train_tokens - previous_trained_tokens,
            )
        progress.total = max(
            config.num_train_tokens - previous_trained_tokens, num_used_tokens
        )
        progress.update(batch.num_tokens)

    progress.close()
    return train_result, num_used_tokens


def train(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    reinit_weights: bool = False,
):
    train_result = {layer: defaultdict(list) for layer in saes.keys()}
    num_tokens = 0
    if config.method in (TrainingMethod.next_layer, TrainingMethod.finetuned):
        # Always train later layers first
        for layer in sorted(config.train_layers, reverse=True):
            if reinit_weights:
                saes[layer].init_weights(saes.get(layer + 1))
            # NB: yeah, we're overwriting num_tokens, we have the same number for each layer though
            # so it doesn't matter.
            result, num_tokens = train_one_layer(
                model,
                tokenizer,
                saes,
                layer,
                dataset,
                config,
                cache_dir,
                reinit_weights,
                previous_trained_tokens=0,
                max_tokens=int(
                    (1.0 - config.finetune_fraction) * config.num_train_tokens
                ) if config.finetune_fraction else None,
            )
            for key, value in result.items():
                train_result[layer][key].extend(value)

    if config.method in (TrainingMethod.e2e, TrainingMethod.finetuned):
        e2e_result, num_tokens = train_e2e(
            model,
            tokenizer,
            saes,
            dataset,
            config,
            cache_dir,
            previous_trained_tokens=num_tokens,
        )
        for layer, result in e2e_result.items():
            for key, value in result.items():
                train_result[layer][key].extend(value)

    return train_result
