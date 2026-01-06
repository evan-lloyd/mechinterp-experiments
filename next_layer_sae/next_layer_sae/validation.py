from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .data_batch import DataBatch
from .ops import generate
from .replacement_model import make_replacement_model
from .sae import SAE
from .sae_data import SAEData, get_logits, get_sae_data
from .tokenization import CONTEXT_LENGTH, input_generator


@torch.no_grad
def validate_saes(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    num_tokens: Optional[int] = None,
    num_batches: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_next_layer_sae: bool = False,
):
    model.eval()
    num_tokens_consumed = 0
    position_ids = np.empty((0,), dtype=np.float32)
    token_influence = np.zeros((CONTEXT_LENGTH,))
    num_inputs = 0
    for step, batch in enumerate(
        input_generator(
            model,
            tokenizer,
            dataset,
            max_tokens=num_tokens,
            tokenizer_batch_size=tokenizer_batch_size,
            inference_batch_size=inference_batch_size,
            max_batches=num_batches,
        )
    ):
        if "progress" not in locals():
            progress = tqdm(
                total=num_tokens
                or num_batches * inference_batch_size * batch.position_ids.shape[1],
                desc="Running SAE evals",
            )
        num_inputs += batch.num_dataset_rows
        for i in range(batch.batch_size):
            for j in range(CONTEXT_LENGTH):
                pos_id = batch.position_ids[i, j]
                if pos_id >= 0:
                    token_influence[pos_id] += batch.token_mask[i, j]

        position_ids = np.concatenate(
            (
                position_ids,
                batch.position_ids[batch.token_mask.bool()].flatten().numpy(),
            ),
            axis=0,
        )
        data, base_logits = get_sae_data(
            model,
            saes,
            batch,
            use_next_layer_sae=use_next_layer_sae,
            cache_dir=cache_dir,
            cache_offset=step * inference_batch_size,
            use_kl_on_final_layer=True,
        )
        batch_replacement_evals = run_replacement_evals(model, saes, batch, base_logits)
        if "replacement_evals" not in locals():
            replacement_evals = {key: [] for key in batch_replacement_evals.keys()}
        for key, value in batch_replacement_evals.items():
            if isinstance(value, np.ndarray):
                replacement_evals[key] = np.concat(
                    (batch_replacement_evals[key], value)
                )
            else:
                replacement_evals[key].append(value)
        for layer, sae in saes.items():
            batch_evals = sae_evals(
                batch,
                data[()][layer],
                data[()][layer + 1],
                data[(layer,)][layer + 1],
                model,
                sae,
                aggregate=False,
                original_logits=base_logits,
            )

            # Combine results across rows
            if "all_evals" not in locals():
                if len(saes) == 1:
                    all_evals = {key: [] for key in batch_evals.keys()}
                else:
                    all_evals = {key: defaultdict(list) for key in batch_evals.keys()}

            for key, value in batch_evals.items():
                if isinstance(value, np.ndarray):
                    if len(saes) == 1:
                        all_evals[key] = np.concat((all_evals[key], value))
                    else:
                        all_evals[key][layer] = np.concat(
                            (all_evals[key][layer], value)
                        )
                else:
                    if len(saes) == 1:
                        all_evals[key].append(value)
                    else:
                        all_evals[key][layer].append(value)

        num_tokens_consumed += batch.num_tokens
        progress.update(batch.num_tokens)

    progress.total = num_tokens_consumed
    progress.refresh()
    progress.close()
    return all_evals, replacement_evals, position_ids


@torch.no_grad
def run_replacement_evals(
    base_model: torch.nn.Module,
    saes: Dict[int, SAE],
    batch: DataBatch,
    base_logits: torch.Tensor,
):
    token_mask = batch.token_mask.to(base_model.device)
    replacement_model = make_replacement_model(
        base_model, {f"transformer.h.{layer}": sae for layer, sae in saes.items()}
    )
    replacement_logits = replacement_model(
        input_ids=batch.input_ids.to(replacement_model.device),
        position_ids=batch.position_ids.to(replacement_model.device),
        attention_mask=batch.attention_mask.to(replacement_model.device),
        use_cache=False,
    ).logits
    replacement_log_probs = replacement_logits.log_softmax(-1)
    base_log_probs = base_logits.log_softmax(-1)
    kl = (
        torch.nn.KLDivLoss(reduction="none", log_target=True)(
            replacement_log_probs, base_log_probs
        )
        .sum(-1)[token_mask.bool()]
        .flatten()
        .cpu()
        .numpy()
    )

    return {"kl": kl}


@torch.no_grad
def sae_evals(
    batch: DataBatch,
    this_layer_baseline: SAEData,
    next_layer_baseline: SAEData,
    next_layer_replacement: SAEData,
    model: torch.nn.Module,
    sae: SAE,
    aggregate: bool = True,
    original_logits: Optional[torch.Tensor] = None,
    replacement_model: Optional[torch.nn.Module] = None,
):
    # TODO: we should refactor SAEData to use replacement_model; it should compute the logits for us
    # on steps where we're running evals
    if replacement_model is None:
        replacement_model = model
    token_mask = batch.token_mask.to(sae.device).bool()
    mse = ((next_layer_replacement.original - next_layer_baseline.original) ** 2).sum(
        dim=-1
    ) * token_mask
    next_norm = (
        torch.linalg.vector_norm(
            next_layer_replacement.original - next_layer_baseline.original,
            dim=-1,
        )
        / torch.linalg.vector_norm(next_layer_baseline.original, dim=-1)[
            token_mask.bool()
        ].mean()
    ) * token_mask
    relative_norm = (
        torch.linalg.vector_norm(
            this_layer_baseline.denormalized_reconstruction
            - this_layer_baseline.original,
            dim=-1,
        )
        / torch.linalg.vector_norm(this_layer_baseline.original, dim=-1)[
            token_mask.bool()
        ].mean()
    ) * token_mask
    l0 = (this_layer_baseline.features > 0).to(torch.float32).sum(dim=-1)
    reencode_result = sae.encode(this_layer_baseline.reconstruction)
    if this_layer_baseline.dense_decoding is not None:
        reencoded_features = reencode_result[0]
    else:
        reencoded_features = reencode_result
    idempotency = ((this_layer_baseline.features - reencoded_features) ** 2).sum(
        dim=-1
    ) * token_mask

    if original_logits is None:
        original_logits = get_logits(
            model,
            next_layer_baseline,
            batch.attention_mask.to(model.device),
        )

    replacement_output = get_logits(
        replacement_model,
        next_layer_replacement,
        batch.attention_mask.to(model.device),
    )

    rep_kl = (
        torch.nn.KLDivLoss(reduction="none", log_target=True)(
            replacement_output.log_softmax(-1),
            original_logits.log_softmax(-1),
        ).sum(dim=-1)
        * token_mask
    )

    if aggregate:
        next_norm = next_norm.sum().item() / batch.num_tokens
        relative_norm = relative_norm.sum().item() / batch.num_tokens
        l0 = l0[token_mask.bool()].sum().item() / batch.num_tokens
        idempotency = idempotency.sum().item() / batch.num_tokens
        rep_kl = rep_kl.sum().item() / batch.num_tokens
        mse = mse.sum().item() / batch.num_tokens
        if this_layer_baseline.dense_decoding is not None:
            dense_l2 = (
                (this_layer_baseline.dense_decoding**2).sum(dim=-1) * token_mask
            ).sum().item() / batch.num_tokens
    else:
        next_norm = next_norm[token_mask.bool()].flatten().cpu().numpy()
        relative_norm = relative_norm[token_mask.bool()].flatten().cpu().numpy()
        l0 = l0[token_mask.bool()].flatten().cpu().numpy()
        idempotency = idempotency[token_mask.bool()].flatten().cpu().numpy()
        rep_kl = rep_kl[token_mask.bool()].flatten().cpu().numpy()
        mse = mse[token_mask.bool()].flatten().cpu().numpy()
        if this_layer_baseline.dense_decoding is not None:
            dense_l2 = (
                (this_layer_baseline.dense_decoding**2)
                .sum(dim=-1)[token_mask.bool()]
                .flatten()
                .cpu()
                .numpy()
            )

    result = {
        "rcn": relative_norm,
        "next_rcn": next_norm,
        "mse": mse,
        "idm": idempotency,
        "L0": l0,
        "rep_kl": rep_kl,
    }
    if this_layer_baseline.dense_decoding is not None:
        result["dense"] = dense_l2

    if sae.normalize_activations:
        normalizer_loss = (
            (
                sae.normalizer(batch.position_ids.to(sae.device))
                - this_layer_baseline.norms
            )
            ** 2
        )[token_mask.bool()]
        if aggregate:
            result["nrm"] = normalizer_loss.sum().item() / batch.num_tokens
        else:
            result["nrm"] = normalizer_loss.flatten().cpu().numpy()
    return result


def generate_with_replacement(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input: str | List[str],
    saes: Dict[int, SAE],
    do_sample: bool = False,
):
    replacement_model = make_replacement_model(
        model, {f"transformer.h.{layer}": sae for layer, sae in saes.items()}
    )
    generate(input, replacement_model, tokenizer, do_sample=do_sample, temperature=0.5)
