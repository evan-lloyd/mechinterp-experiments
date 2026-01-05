from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional

import torch
from datasets import IterableDataset
from tqdm.auto import tqdm

from .multiline_progress import MultilineProgress
from .sae_data import SAEData, cache_on_disk, get_sae_data, init_cache
from .tokenization import CONTEXT_LENGTH, input_generator
from .validation import sae_evals

if TYPE_CHECKING:
    from transformers import AutoTokenizer

    from .data_batch import DataBatch
    from .sae import SAE


class TrainingMethod(Enum):
    next_layer = "Next Layer"
    e2e = "End-to-end"
    finetuned = "Finetuned"


@dataclass(kw_only=True)
class TrainingConfig:
    num_train_tokens: int
    tokenizer_batch_size: int
    training_batch_size: int
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


def train_one_layer(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    target_layer: int,
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str],
    did_reinit: bool,
):
    param_groups = [
        {
            "params": list(saes[target_layer].decoder.parameters()),
            "lr": config.decoder_lr or config.lr,
        },
        {
            "params": list(saes[target_layer].encoder.parameters()),
            "lr": config.encoder_lr or config.lr,
        },
    ]
    if saes[target_layer].with_inhibition:
        param_groups.append(
            {
                "params": saes[target_layer].inhibition,
                "lr": config.inhibition_lr or config.lr,
            }
        )
    if saes[target_layer].d_dense is not None:
        param_groups.append(
            {
                "params": list(saes[target_layer].dense_decoder.parameters()),
                "lr": config.dense_decoder_lr or config.lr,
            }
        )
        param_groups.append(
            {
                "params": list(saes[target_layer].dense_encoder.parameters()),
                "lr": config.dense_encoder_lr or config.lr,
            }
        )

    optimizer = torch.optim.Adam(param_groups, lr=config.lr)
    if saes[target_layer].normalize_activations:
        normalizer_optimizer = torch.optim.Adam(
            list(saes[target_layer].normalizer.parameters()), lr=0.01
        )

    progress = MultilineProgress(
        total=config.num_train_tokens,
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
            max_tokens=config.num_train_tokens,
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
        balance_reconstruction_losses = (
            config.balance_reconstruction_losses[target_layer]
            if hasattr(config.balance_reconstruction_losses, "__getitem__")
            else config.balance_reconstruction_losses
        )
        reconstruction_weight = (
            config.reconstruction_weight[target_layer]
            if hasattr(config.reconstruction_weight, "__getitem__")
            else config.reconstruction_weight
        )
        next_reconstruction_weight = (
            config.next_reconstruction_weight[target_layer]
            if hasattr(config.next_reconstruction_weight, "__getitem__")
            else config.next_reconstruction_weight
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
        # lr_scheduler.step()

        if saes[target_layer].normalize_activations:
            normalizer_optimizer.zero_grad()
            losses["normalizer"].backward()
            normalizer_optimizer.step()

        if num_used_tokens >= eval_threshold:
            progress.set_postfix(
                {
                    k: v
                    for k, v in sae_evals(
                        batch,
                        result[()][target_layer],
                        result[()][target_layer + 1],
                        result[(target_layer,)][target_layer + 1],
                        model,
                        saes[target_layer],
                        original_logits=original_logits,
                    ).items()
                    if k not in ("mse",)
                },
                refresh=False,
            )
            eval_threshold = min(
                eval_threshold + config.eval_interval, config.num_train_tokens
            )
        progress.total = max(config.num_train_tokens, num_used_tokens)
        progress.update(batch.num_tokens)

    progress.close()

def train(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    cache_dir: Optional[str] = None,
    reinit_weights: bool = False,
):
    if config.method in (TrainingMethod.next_layer, TrainingMethod.finetuned):
        # Always train later layers first
        for layer in sorted(config.train_layers, reverse=True):
            if reinit_weights:
                saes[layer].init_weights(saes.get(layer + 1))
            train_one_layer(
                model,
                tokenizer,
                saes,
                layer,
                dataset,
                config,
                cache_dir,
                reinit_weights,
            )
    elif config.method is TrainingMethod.e2e:
        pass
