from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional

import torch
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data_batch import DataBatch
from .multiline_progress import MultilineProgress
from .replacement_model import make_replacement_model
from .sae import SAE
from .sae_data import (
    SAEData,
    cache_on_disk,
    get_sae_data,
    init_cache,
    get_sae_data_with_replacement,
)
from .tokenization import CONTEXT_LENGTH, input_generator
from .validation import sae_evals


class TrainingMethod(Enum):
    standard = "Standard"
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "Finetuned"
    full = "Full Replacement"
    full_finetuned = "Full Replacement Finetuning"


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
    e2e_batch_size: int
    eval_interval: int
    reconstruction_weight: float | Mapping[int, float]
    downstream_reconstruction_weight: float | Mapping[int, float]
    idempotency_weight: float | Mapping[int, float]
    dense_weight: float | Mapping[int, float]
    train_layers: List[int]
    lr: float = 1e-3
    # Default schedule is constant
    main_lr_schedule: Callable[[float], float] = lambda frac_trained: 1.0
    finetune_lr_schedule: Callable[[float], float] = lambda frac_trained: 1.0
    decoder_lr: Optional[float] = None
    encoder_lr: Optional[float] = None
    dense_decoder_lr: Optional[float] = None
    dense_encoder_lr: Optional[float] = None
    inhibition_lr: Optional[float] = None
    use_downstream_saes: bool = True
    balance_reconstruction_losses: bool = False | Mapping[int, bool]
    use_weighted_mask: bool = False
    method: TrainingMethod
    finetune_fraction: Optional[float] = None


def sae_losses(
    batch: DataBatch,
    this_layer_baseline: SAEData,
    downstream_layers_baseline: List[SAEData],
    downstream_layers_replacement: List[SAEData],
    sae: SAE,
    use_downstream_saes: bool,
    want_idempotency: bool = False,
    want_mse: bool = True,
):
    token_mask = batch.token_mask.to(sae.device)

    downstream_reconstruction_loss = []
    for baseline, replacement in zip(
        downstream_layers_baseline, downstream_layers_replacement
    ):
        if baseline.reconstruction_eval == "KL":
            downstream_reconstruction_loss.append(
                (
                    torch.nn.KLDivLoss(reduction="none", log_target=True)(
                        replacement.original.log_softmax(-1),
                        baseline.original.log_softmax(-1),
                    ).mean(dim=-1)
                    * token_mask
                ).sum()
                / batch.num_tokens
            )
        else:
            if use_downstream_saes:
                next_layer_attr = "features"
                next_layer_fn = torch.abs
            else:
                next_layer_attr = "normalized_original"
                next_layer_fn = partial(torch.pow, exponent=2)

            downstream_reconstruction_loss.append(
                (
                    next_layer_fn(
                        (
                            getattr(replacement, next_layer_attr)
                            - getattr(baseline, next_layer_attr)
                        )
                    ).mean(dim=-1)
                    * token_mask
                ).sum()
                / batch.num_tokens
            )

    if want_mse:
        reconstruction_loss = (
            (
                this_layer_baseline.reconstruction
                - this_layer_baseline.normalized_original
            )
            ** 2
        ).mean(dim=-1)

    reencode_result = sae.encode(this_layer_baseline.reconstruction)

    if want_idempotency:
        if this_layer_baseline.dense_decoding is not None:
            reencoded_features = reencode_result[0]
        else:
            reencoded_features = reencode_result
        idempotency_loss = (
            (this_layer_baseline.features - reencoded_features).abs()
        ).mean(dim=-1)
    else:
        idempotency_loss = None
    if this_layer_baseline.dense_decoding is not None:
        dense_loss = (this_layer_baseline.dense_decoding**2).mean(dim=-1)

    result = {
        "reconstruction": (reconstruction_loss * token_mask).sum() / batch.num_tokens
        if want_mse
        else torch.zeros((1,), device=sae.device),
        "downstream_reconstruction": downstream_reconstruction_loss,
        "idempotency": (idempotency_loss * token_mask).sum() / batch.num_tokens
        if want_idempotency
        else torch.zeros((1,), device=sae.device),
        "dense": (dense_loss * token_mask).sum() / batch.num_tokens
        if this_layer_baseline.dense_decoding is not None
        else torch.zeros((1,), device=sae.device),
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
        model,
        {layer: sae for layer, sae in saes.items() if layer >= target_layer},
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
            config.use_downstream_saes,
            cache_dir,
            step * config.training_batch_size,
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
                config.use_downstream_saes,
                cache_dir,
                step * config.training_batch_size,
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
        losses = sae_losses(
            batch,
            result[()][target_layer],
            [result[()][target_layer + 1]]
            if config.method is TrainingMethod.next_layer
            else [],
            [result[(target_layer,)][target_layer + 1]]
            if config.method is TrainingMethod.next_layer
            else [],
            saes[target_layer],
            config.use_downstream_saes,
            want_idempotency=idempotency_weight > 0.0,
            want_mse=True,
        )
        reconstruction_weight, downstream_reconstruction_weight = (
            maybe_balance_reconstruction_weights(
                config.balance_reconstruction_losses,
                config.reconstruction_weight,
                config.downstream_reconstruction_weight,
                losses,
                target_layer,
            )
        )

        loss = sum(
            (
                losses["reconstruction"] * reconstruction_weight,
                sum(losses["downstream_reconstruction"])
                * downstream_reconstruction_weight,
                losses["idempotency"] * idempotency_weight,
                losses["dense"] * dense_weight,
            )
        )

        loss.backward()

        optimizer.step()

        for pg in optimizer.param_groups:
            pg["lr"] = pg["base_lr"] * config.main_lr_schedule(
                min(num_used_tokens / max_tokens, 1.0)
            )

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
            train_result["raw_loss.mse"].append(losses["reconstruction"].item())
            train_result["raw_loss.next_layer_mse"].append(
                losses["downstream_reconstruction"][-1].item()
                if config.method is TrainingMethod.next_layer
                and target_layer <= model.config.num_layers - 1
                else 0.0
            )
            train_result["raw_loss.kl"].append(
                losses["downstream_reconstruction"][-1].item()
                if config.method is TrainingMethod.next_layer
                and target_layer == model.config.num_layers - 1
                else 0.0
            )
            train_result["weighted_loss.mse"].append(
                train_result["raw_loss.mse"][-1] * reconstruction_weight
            )
            train_result["weighted_loss.next_layer_mse"].append(
                train_result["raw_loss.next_layer_mse"][-1]
                * downstream_reconstruction_weight
            )
            train_result["weighted_loss.kl"].append(
                train_result["raw_loss.kl"][-1] * downstream_reconstruction_weight
            )

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
    target_layer: int,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    previous_trained_tokens: int = 0,
):
    train_result = defaultdict(list)
    assert not any(sae.normalize_activations for sae in saes.values()), (
        "Activation normalization not implemented for e2e training"
    )
    optimizer = make_optimizer(saes, [target_layer], config)

    replacement_model = make_replacement_model(
        model,
        {layer: sae for layer, sae in saes.items() if layer >= target_layer},
    )

    progress = MultilineProgress(
        total=config.num_train_tokens - previous_trained_tokens,
        desc=[f"Layer {target_layer}"],
        num_header_lines=1,
    )

    num_used_tokens = 0
    model.eval()
    eval_threshold = 0

    batch_size = config.e2e_batch_size
    for step, batch in enumerate(
        input_generator(
            model,
            tokenizer,
            dataset,
            # For the finetuning method, make sure it only sees the tokens at the end of the training set
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
            model,
            saes,
            batch,
            target_layer,
            model.config.num_layers,
            use_downstream_saes=False,
            cache_dir=cache_dir,
            cache_offset=step * batch_size,
        )
        losses = sae_losses(
            batch,
            result[()][target_layer],
            # e2e looks at reconstruction at every later layer; fine-tuning only looks at this layer
            # and final KL
            [
                result[()][layer]
                for layer in sorted(result[()].keys())
                if layer > target_layer
            ]
            if config.method is TrainingMethod.e2e
            else [result[()][model.config.num_layers]],
            [
                result[(target_layer,)][layer]
                for layer in sorted(result[(target_layer,)].keys())
                if layer > target_layer
            ]
            if config.method is TrainingMethod.e2e
            else [result[(target_layer,)][model.config.num_layers]],
            saes[target_layer],
            use_downstream_saes=False,
            want_idempotency=False,
            # Only finetuning looks at MSE of *this* layer
            want_mse=config.method is TrainingMethod.finetuned,
        )
        loss = torch.zeros((1,), device=model.device)
        # Keep track of this separately from any sparsity / other losses, so we can rescale the KL
        mse_loss = torch.zeros((1,), device=model.device)

        reconstruction_weight, downstream_reconstruction_weight = (
            maybe_balance_reconstruction_weights(
                # We will manually balance KL vs the sum of all MSE losses
                False,
                config.reconstruction_weight,
                config.downstream_reconstruction_weight,
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
        loss += sum(
            (
                losses["idempotency"] * idempotency_weight,
                losses["dense"] * dense_weight,
            )
        )
        mse_loss += losses["reconstruction"]
        # Last downstream recon term is actually KL
        mse_loss += sum(losses["downstream_reconstruction"][:-1])
        kl_loss = losses["downstream_reconstruction"][-1]

        with torch.no_grad():
            # Balance KL loss with MSE ala (Karvonen 2025).
            # Last layer for e2e is a special case, since it has no MSE term.
            if (
                config.method is TrainingMethod.e2e
                and target_layer == model.config.num_layers - 1
            ):
                kl_scale = 1.0
            else:
                # For e2e, make KL be the same scale as the *average* MSE loss for a layer
                # (this factor will be 1 for finetuned)
                kl_scale = (
                    mse_loss.item() / len(losses["downstream_reconstruction"])
                ) / (kl_loss.item() + 1e-8)

        loss += mse_loss * reconstruction_weight + kl_loss * kl_scale

        loss.backward()

        optimizer.step()

        for pg in optimizer.param_groups:
            if config.method is TrainingMethod.e2e:
                schedule_fn = config.main_lr_schedule
            else:
                schedule_fn = config.finetune_lr_schedule

            pg["lr"] = pg["base_lr"] * schedule_fn(
                min(
                    num_used_tokens
                    / (config.num_train_tokens - previous_trained_tokens),
                    1.0,
                )
            )

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
            train_result["raw_loss.mse"].append(losses["reconstruction"].item())
            train_result["raw_loss.kl"].append(kl_loss.item())
            train_result["raw_loss.downstream"].append(
                [ds.item() for ds in losses["downstream_reconstruction"]]
            )
            train_result["weighted_loss.mse"].append(
                losses["reconstruction"].item() * reconstruction_weight
            )
            train_result["weighted_loss.kl"].append(kl_loss.item() * kl_scale)
            train_result["weighted_loss.downstream"].append(
                [
                    ds.item() * reconstruction_weight
                    for ds in losses["downstream_reconstruction"]
                ]
            )

            progress.set_postfix(
                {k: v for k, v in evals.items() if k not in ("mse", "idm")},
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


def train_full(
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
        "Activation normalization not implemented for full training"
    )
    optimizer = make_optimizer(saes, list(range(model.config.num_layers)), config)

    replacement_model = make_replacement_model(model, saes)

    progress = MultilineProgress(
        total=config.num_train_tokens - previous_trained_tokens,
        desc=[
            f"Layer {layer}"
            for layer in list(range(model.config.num_layers - 1, -1, -1))
        ],
        num_header_lines=model.config.num_layers,
    )

    num_used_tokens = 0
    model.eval()
    eval_threshold = 0

    batch_size = config.e2e_batch_size
    for step, batch in enumerate(
        input_generator(
            model,
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
        token_mask = batch.token_mask.to(model.device)
        optimizer.zero_grad()
        num_used_tokens += batch.num_tokens

        baseline_data, replacement_data = get_sae_data_with_replacement(
            model,
            replacement_model,
            saes,
            batch,
            cache_dir=cache_dir,
            cache_offset=step * batch_size,
        )

        mse_loss = torch.zeros((1,), device=model.device)
        for layer in range(model.config.num_layers):
            mse_loss += (
                (
                    (
                        replacement_data[layer].reconstruction
                        - baseline_data[layer].original
                    )
                    ** 2
                ).mean(dim=-1)
                * token_mask
            ).sum() / batch.num_tokens

        kl_loss = (
            torch.nn.KLDivLoss(reduction="none", log_target=True)(
                replacement_data[model.config.num_layers].original.log_softmax(-1),
                baseline_data[model.config.num_layers].original.log_softmax(-1),
            ).mean(dim=-1)
            * token_mask
        ).sum() / batch.num_tokens

        # Balance KL loss with MSE ala (Karvonen 2025)
        with torch.no_grad():
            kl_scale = (mse_loss.item() / model.config.num_layers) / (
                kl_loss.item() + 1e-8
            )
        loss = mse_loss + kl_loss * kl_scale

        loss.backward()
        optimizer.step()

        for pg in optimizer.param_groups:
            if config.method is TrainingMethod.full:
                schedule_fn = config.main_lr_schedule
            else:
                schedule_fn = config.finetune_lr_schedule

            pg["lr"] = pg["base_lr"] * schedule_fn(
                min(
                    num_used_tokens
                    / (config.num_train_tokens - previous_trained_tokens),
                    1.0,
                )
            )

        if num_used_tokens >= eval_threshold:
            evals = {
                target_layer: sae_evals(
                    batch,
                    baseline_data[target_layer],
                    baseline_data[target_layer + 1],
                    replacement_data[target_layer + 1],
                    model,
                    saes[target_layer],
                    original_logits=baseline_data[model.config.num_layers].original,
                    replacement_model=replacement_model,
                )
                for target_layer in range(model.config.num_layers - 1, -1, -1)
            }
            for layer, e in evals.items():
                for k, v in e.items():
                    train_result[layer][k].append(
                        (num_used_tokens + previous_trained_tokens, v)
                    )
            train_result["all"]["raw_loss.mse"].append(mse_loss.item())
            train_result["all"]["raw_loss.kl"].append(kl_loss.item())
            train_result["all"]["weighted_loss.mse"].append(mse_loss.item())
            train_result["all"]["weighted_loss.kl"].append(kl_loss.item() * kl_scale)

            progress.set_postfix(
                [
                    {k: v for k, v in evals[layer].items() if k not in ("mse", "idm")}
                    for layer in range(model.config.num_layers - 1, -1, -1)
                ],
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
    start_at_finetune: bool = False,
):
    train_result = {layer: defaultdict(list) for layer in saes.keys()}
    train_result["all"] = defaultdict(list)

    num_tokens = 0
    if (
        config.method
        in (
            TrainingMethod.standard,
            TrainingMethod.next_layer,
            TrainingMethod.finetuned,
            TrainingMethod.full_finetuned,
        )
        and not start_at_finetune
    ):
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
                )
                if config.finetune_fraction
                else None,
            )
            for key, value in result.items():
                train_result[layer][key].extend(value)

    if start_at_finetune:
        num_tokens = int((1.0 - config.finetune_fraction) * config.num_train_tokens)

    if config.method in (TrainingMethod.e2e, TrainingMethod.finetuned):
        for layer in sorted(config.train_layers, reverse=True):
            if reinit_weights and config.method is not TrainingMethod.finetuned:
                saes[layer].init_weights(saes.get(layer + 1))
            result, _ = train_e2e(
                model,
                tokenizer,
                saes,
                layer,
                dataset,
                config,
                cache_dir,
                previous_trained_tokens=num_tokens,
            )
            for key, value in result.items():
                train_result[layer][key].extend(value)

    if config.method in (TrainingMethod.full, TrainingMethod.full_finetuned):
        if reinit_weights and config.method is not TrainingMethod.full_finetuned:
            for layer in sorted(config.train_layers, reverse=True):
                saes[layer].init_weights(None)

        full_result, _ = train_full(
            model,
            tokenizer,
            saes,
            dataset,
            config,
            cache_dir,
            previous_trained_tokens=num_tokens,
        )
        for layer, result in full_result.items():
            for key, value in result.items():
                train_result[layer][key].extend(value)

    return train_result
