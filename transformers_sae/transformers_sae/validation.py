from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from bitarray import bitarray
from datasets import IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .activation_cache import load_cache
from .activation_data import (
    ActivationBatch,
    TrainingBatch,
    make_activation_batch,
    make_batch_for_evals,
)
from .metrics import kl_eval, l0_eval, live_features_eval, rre_eval
from .ops import generate
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE
from .tokenization import make_dataloader
from .training_step import KLFinetuneTrainingStepper


@dataclass(kw_only=True)
class LayerEval:
    rre: np.ndarray | float | None = None
    kl: np.ndarray | float | None = None
    l0: np.ndarray | float | None = None
    live_features: bitarray | float | None = None

    def update(self, other: "LayerEval"):
        for attr in self.__class__.__dataclass_fields__:
            if getattr(self, attr) is None:
                setattr(self, attr, getattr(other, attr))
            elif getattr(other, attr) is not None:
                if attr == "live_features":
                    assert isinstance(getattr(self, attr), bitarray) and isinstance(
                        getattr(other, attr), bitarray
                    ), "Trying to combine non-aggregated LayerEval"
                    self.live_features |= other.live_features
                    continue
                assert isinstance(getattr(self, attr), np.ndarray) and isinstance(
                    getattr(other, attr), np.ndarray
                ), "Trying to combine non-aggregated LayerEval"
                setattr(
                    self, attr, np.concat((getattr(self, attr), getattr(other, attr)))
                )


@dataclass(kw_only=True)
class ValidationResult:
    layer_results: Dict[int, LayerEval]
    position_ids: np.ndarray


@torch.no_grad()
def run_validations(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    num_tokens: Optional[int] = None,
    num_batches: Optional[int] = None,
    cache_dir: Optional[str] = None,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    offload: bool = True,
):
    try:
        if end_layer is None:
            end_layer = model.num_layers + 1
        full_replacement_model = make_replacement_model(
            model,
            {
                layer: saes[layer]
                for layer in range(start_layer, end_layer)
                if layer in saes
            },
        )
        model.eval()
        num_tokens_consumed = 0
        results = ValidationResult(
            layer_results={
                layer: LayerEval() for layer in range(start_layer, end_layer)
            },
            position_ids=np.empty((0,)),
        )

        for layer in range(start_layer, end_layer):
            if layer in saes:
                saes[layer].eval()
                saes[layer].onload()

        for step, batch in enumerate(
            make_dataloader(
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
            results.position_ids = np.concatenate(
                (
                    results.position_ids,
                    batch.position_ids[batch.token_mask.bool()].flatten().cpu().numpy(),
                ),
                axis=0,
            )

            batch.to(model.device)
            if cache_dir is not None:
                baseline_run = load_cache(
                    model.num_layers,
                    cache_dir,
                    step * inference_batch_size,
                    batch,
                )
                baseline_activations = {}
                for k, v in baseline_run.items():
                    if k in range(start_layer, end_layer):
                        baseline_activations[k] = ActivationBatch(
                            layer_output=v.to(model.device)
                        )
            else:
                baseline_activations = make_activation_batch(
                    model,
                    [(layer, "layer") for layer in range(start_layer, end_layer)],
                    batch,
                    start_layer=-1,  # not cached, so we start from raw input
                    end_layer=end_layer,
                )
            evals = run_evals(
                make_batch_for_evals(
                    model,
                    full_replacement_model,
                    TrainingBatch(
                        batch, {}, baseline_activations, replacement_layers=[]
                    ),
                    start_layer,
                    end_layer,
                ),
                list(range(start_layer, end_layer)),
                aggregate=False,
            )

            for layer in evals.keys():
                results.layer_results[layer].update(evals[layer])

            num_tokens_consumed += batch.num_tokens
            progress.update(batch.num_tokens)

        progress.total = num_tokens_consumed
        progress.refresh()
        progress.close()

        return results

    finally:
        if offload:
            try:
                for layer in range(start_layer, end_layer):
                    if layer in saes:
                        saes[layer].offload()
            except Exception:
                pass


@torch.no_grad()
def run_single_layer_replacements(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    num_tokens: Optional[int] = None,
    num_batches: Optional[int] = None,
    cache_dir: Optional[str] = None,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
    offload: bool = True,
):
    try:
        if end_layer is None:
            end_layer = model.num_layers + 1

        results = ValidationResult(
            layer_results={
                layer: LayerEval() for layer in range(start_layer, end_layer)
            },
            position_ids=np.empty((0,)),
        )

        for layer in range(start_layer, end_layer):
            if layer in saes:
                saes[layer].onload()

        for layer in range(start_layer, end_layer):
            if layer not in saes:
                continue

            # This happens to run exactly the right replacement models we need
            stepper = KLFinetuneTrainingStepper(model, layer, saes)

            num_tokens_consumed = 0

            for step, batch in enumerate(
                make_dataloader(
                    model,
                    tokenizer,
                    dataset,
                    max_tokens=num_tokens,
                    tokenizer_batch_size=tokenizer_batch_size,
                    inference_batch_size=inference_batch_size,
                    max_batches=num_batches,
                )
            ):
                if cache_dir is not None:
                    cache = load_cache(
                        stepper.base_model.num_layers,
                        cache_dir,
                        step * inference_batch_size,
                        batch,
                    )
                else:
                    cache = None
                if "progress" not in locals():
                    progress = tqdm(
                        total=num_tokens
                        or num_batches
                        * inference_batch_size
                        * batch.position_ids.shape[1],
                        desc=f"Running SAE evals (layer {layer})",
                    )
                results.position_ids = np.concatenate(
                    (
                        results.position_ids,
                        batch.position_ids[batch.token_mask.bool()]
                        .flatten()
                        .cpu()
                        .numpy(),
                    ),
                    axis=0,
                )

                batch.to(model.device)
                evals = run_evals(
                    stepper.make_batch(batch, cache),
                    [model.num_layers],
                    aggregate=False,
                )

                results.layer_results[layer].update(evals[model.num_layers])

                num_tokens_consumed += batch.num_tokens
                progress.update(batch.num_tokens)

            progress.total = num_tokens_consumed
            progress.refresh()
            progress.close()
            del progress

        return results
    finally:
        if offload:
            try:
                for layer in range(start_layer, end_layer):
                    if layer in saes:
                        saes[layer].offload()
            except Exception:
                pass


@torch.no_grad()
def run_evals(
    batch: TrainingBatch,
    layers: List[int],
    aggregate: bool = True,
) -> Dict[int, LayerEval]:
    assert set(batch.baseline_activations.keys()) == set(
        batch.replacement_activations.keys()
    ), (
        f"Missing activations for evaluation: {batch.baseline_activations.keys()} != {batch.replacement_activations.keys()}"
    )

    if aggregate:
        eval_type = "float"
    else:
        eval_type = "np"

    result = {}
    for layer in layers:
        replacement = batch.replacement_activations[layer]
        baseline = batch.baseline_activations[layer]
        if baseline.log_probs is not None:
            result[layer] = LayerEval(
                kl=kl_eval(
                    replacement.log_probs,
                    baseline.log_probs,
                    batch.input_data,
                    eval_type,
                )
            )
        else:
            result[layer] = LayerEval(
                rre=rre_eval(
                    replacement.sae_output,
                    baseline.layer_output,
                    batch.input_data,
                    eval_type,
                ),
                l0=l0_eval(
                    replacement.sae_features,
                    None,
                    batch.input_data,
                    eval_type,
                ),
                live_features=live_features_eval(
                    replacement.sae_features,
                    None,
                    batch.input_data,
                    eval_type,
                ),
            )

    return result


def generate_with_replacement(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input: str | List[str],
    saes: Dict[int, SAE],
    do_sample: bool = False,
    stream: bool = True,
    offload: bool = True,
    max_new_tokens: int = 100,
):
    for sae in saes.values():
        sae.onload()

    replacement_model = make_replacement_model(
        model, {layer: sae for layer, sae in saes.items()}
    )
    result = generate(
        input,
        replacement_model,
        tokenizer,
        do_sample=do_sample,
        temperature=0.5,
        stream=stream,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    if offload:
        for sae in saes.values():
            sae.offload()

    return result
