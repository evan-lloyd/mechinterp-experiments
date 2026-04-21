from __future__ import annotations
from tqdm.auto import tqdm
from contextlib import ExitStack
from copy import deepcopy
from transformers_sae.metrics import mse_loss, cos_dist_loss, l1_loss

from collections import defaultdict
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
    ParamSpec,
    Tuple,
    Unpack,
    ValuesView,
    Protocol,
)

import numpy as np
import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

from .activation_cache import load_cache
from .activation_data import TrainingBatch, make_activation_batch, make_batch_for_evals
from .encoder import LISTA, Encoder
from .multiline_progress import MultilineProgress
from .ops import (
    find_checkpoint_after,
    find_latest_checkpoint,
    get_state_dict_from_checkpoint,
    load_checkpoint,
    save_training_result,
)
from .replacement_model import ReplacementModel, make_replacement_model
from .sae import SAE
from .tokenization import make_dataloader
from .training_step import (
    EndToEndFullTrainingStepper,
    EndToEndTrainingStepper,
    FullReplacementTrainingStepper,
    KLFinetuneTrainingStepper,
    NextLayerFinetunedTrainingStepper,
    NextLayerTrainingStepper,
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


def make_optimizer(saes: Dict[int, SAE], layers: List[int], config: TrainingConfig):
    param_groups = [
        {
            "params": [
                param
                for layer in layers
                for param in saes[layer].decoder.decoder_params()
                if param.requires_grad
            ],
            "lr": config.decoder_lr or config.lr,
            "name": "decoder",
        },
        {
            "params": [
                param
                for layer in layers
                for param in saes[layer].encoder.encoder_params()
                if param.requires_grad
            ],
            "lr": config.encoder_lr or config.lr,
            "name": "encoder",
        },
    ]

    if any(
        isinstance(s.encoder, LISTA) for layer, s in saes.items() if layer in layers
    ):
        param_groups.extend(
            [
                {
                    "params": [param],
                    "lr": config.interaction_lr or config.lr,
                    "name": pg_name,
                }
                for layer in layers
                for pg_name, param in saes[layer].encoder.interaction_params()
                if isinstance(saes[layer].encoder, LISTA) and param.requires_grad
            ]
        )

    for pg in param_groups:
        pg["base_lr"] = pg["lr"]

    return torch.optim.Adam(param_groups, lr=config.lr, betas=config.betas)


def alista_encoder(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    baseline_saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    num_tokens: int,
    offload_after_training: bool = True,
) -> TrainingResult:
    training_saes = {
        layer: SAE(deepcopy(sae.config)) for layer, sae in baseline_saes.items()
    }
    train_result = TrainingResult(training_saes)
    for layer, sae in training_saes.items():
        sae.init_weights(baseline_saes[layer])

    for layer, sae in training_saes.items():
        # Appendix E, Algorithm 1 of Liu et al. 2019
        with torch.no_grad():
            sae.decoder.linear.weight /= sae.decoder.linear.weight.norm(
                dim=1, keepdim=True
            )
        sae.encoder.linear.weight = torch.nn.Parameter(
            sae.decoder.linear.weight.T.detach().clone()
        )
        sae.encoder.linear.bias = torch.nn.Parameter(
            torch.zeros_like(sae.encoder.linear.bias)
        )
        step_size = 1e-5
        square_dict = sae.decoder.linear.weight.T @ sae.decoder.linear.weight
        with torch.no_grad():
            progress = tqdm(range(100000), desc=f"Compute ALISTA weights {layer}")
            for _ in progress:
                updated_weight = (
                    sae.encoder.linear.weight
                    - step_size * square_dict @ sae.encoder.linear.weight
                )

                t_diff = 1 - (
                    torch.einsum("ji,ij->i", sae.decoder.linear.weight, updated_weight)
                )
                proj_weight = t_diff.unsqueeze(-1) * updated_weight
                new_weight = updated_weight + proj_weight
                progress.set_postfix(
                    # {}, refresh=False
                    {
                        "delta": (new_weight - sae.encoder.linear.weight).norm().item(),
                    },
                    refresh=False,
                )
                sae.encoder.linear.weight = torch.nn.Parameter(new_weight)

    tune_activation_thresholds(
        model,
        tokenizer,
        training_saes,
        dataset,
        config.tokenizer_batch_size,
        4,  # TODO
        num_tokens,
        offload_after_training,
    )

    return train_result


def tune_encoder(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    baseline_saes: Dict[int, SAE],
    dataset: IterableDataset,
    config: TrainingConfig,
    num_tokens: int,
    offload_after_training: bool = True,
) -> TrainingResult:
    """Do an encoder-only training run of the given SAEs, with their dictionaries fixed. This adapts the
    encoders to work within the full replacement model, without changing the semantics of the SAE features.
    """
    try:
        first_sae_layer = min(baseline_saes.keys())
        last_sae_layer = max(baseline_saes.keys())
        layers_to_tune = range(first_sae_layer, last_sae_layer + 1)
        training_saes = {
            layer: SAE(deepcopy(sae.config)) for layer, sae in baseline_saes.items()
        }
        train_result = TrainingResult(training_saes)

        for layer, sae in training_saes.items():
            # Finetune
            # sae.init_weights(baseline_saes[layer])
            # sae.decoder.requires_grad_(False)
            # for i, a in enumerate(sae.encoder.activation):
            #     a.threshold.fill_(
            #         baseline_saes[layer].encoder.activation[i].threshold.item()
            #     )

            # From scratch
            if layer == first_sae_layer:
                sae.init_weights(baseline_saes[layer])
                for i, a in enumerate(sae.encoder.activation):
                    a.threshold.fill_(
                        baseline_saes[layer].encoder.activation[i].threshold.item()
                    )
                continue
            sae.init_weights(None)
            sae.decoder.init_weights(baseline_saes[layer].decoder)
            sae.encoder.init_weights(sae.decoder)
            sae.onload()
            sae.decoder.requires_grad_(False)

        for sae in baseline_saes.values():
            sae.onload()

        for layer in layers_to_tune:
            baseline_sae = baseline_saes[layer]
            training_sae = training_saes[layer]

            # We don't need to train the first SAE, because there are no SAEs before it in the replacement
            # model that could distort its inputs.
            if layer == first_sae_layer:
                training_sae.eval()
                continue

            progress = MultilineProgress(
                total=num_tokens,
                desc=[f"Tuning encoder {layer}"],
                num_header_lines=1,
            )
            num_used_tokens = 0
            optimizer = make_optimizer(training_saes, [layer], config)

            # NB: *not* including the current SAE, because we're training it. This is used
            # to collect the expected features, which must be done with the baseline SAE.
            replacement_model = make_replacement_model(
                model,
                {i: training_saes[i] for i in range(first_sae_layer, layer)},
            )
            for batch in make_dataloader(
                model,
                tokenizer,
                dataset,
                max_tokens=num_tokens,
                tokenizer_batch_size=config.tokenizer_batch_size,
                inference_batch_size=config.training_batch_size,
            ):
                optimizer.zero_grad()
                batch.to(model.device)

                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type="cuda" if model.device.type == "cuda" else "cpu",
                        dtype=torch.bfloat16,
                    ),
                ):
                    # Get the features that our SAE would have output in the base model
                    baseline_activations = make_activation_batch(
                        model,
                        [(layer, "layer"), (first_sae_layer, "layer")],
                        batch,
                        end_layer=layer + 1,
                    )

                    # Get the actual input our SAE will receive in the replacement model
                    replacement_activations = make_activation_batch(
                        replacement_model,
                        [(layer, "layer")],
                        batch,
                        start_input=baseline_activations[first_sae_layer].layer_output,
                        start_layer=first_sae_layer,
                        end_layer=layer + 1,
                        start_at_sae=True,
                    )

                    expected_features = baseline_sae.encode(
                        baseline_activations[layer].layer_output,
                        batch.token_mask,
                    )

                with torch.autocast(
                    device_type="cuda" if model.device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                ):
                    actual_features = training_sae.encode(
                        replacement_activations[layer].layer_output,
                        batch.token_mask,
                    )
                    # Using MSE loss on features here (rather than cosdist as we do elsewhere) because ideally
                    # our tuned SAE matches the original features *exactly* on the distorted input.
                    loss = (
                        mse_loss(
                            actual_features,
                            expected_features,
                            batch,
                        )
                        # Rescale loss to be "per active feature"
                        * training_sae.config.d_sae
                        / training_sae.encoder.activation[-1].config.k
                    )
                loss.backward()

                progress.set_postfix({"loss": loss.item()})

                optimizer.step()
                num_used_tokens += batch.num_tokens
                progress.total = max(num_tokens, num_used_tokens)
                progress.update(batch.num_tokens)

                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * config.lr_schedule(
                        max(
                            min(
                                num_used_tokens / config.num_train_tokens,
                                1.0,
                            ),
                            0.0,
                        ),
                        pg_name=pg["name"],
                    )
                # end for each batch

            training_sae.eval()
            progress.close()
            # end for each layer

        # end training loop

        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in baseline_saes.values():
                    sae.offload()
                for sae in training_saes.values():
                    sae.offload()
            except Exception:
                pass


def tune_activation_thresholds(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    saes: Dict[int, SAE],
    dataset: IterableDataset,
    tokenizer_batch_size: int,
    inference_batch_size: int,
    num_tokens: int,
    offload_after_training: bool = True,
    lr_schedule: Optional[Callable[[float, int], float]] = None,
    threshold_lr: float = 0.01,
    interaction_lr: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> None:
    """For BatchTopK SAEs, do a special training run that adjusts only the thresholds used
    in the BatchTopK activation function. This is mandatory for replacement models, since
    the additional error will almost surely have distorted the scale of inputs reaching each
    SAE, and so we will no longer hit the desired sparsity level without this step. About 1e6
    tokens seem to be sufficient.

    Note that this function adjusts the thresholds in-place, but does not save the result to
    disk.
    """
    try:
        replacement_model = make_replacement_model(model, saes)
        for layer, sae in saes.items():
            sae.onload()
            sae.eval()
            sae.encoder.train_activations()
            # sae.decoder.linear.weight /= sae.decoder.linear.weight.norm(dim=0, keepdim=True)
            sae.set_activation_threshold_lr(threshold_lr)
            # if hasattr(sae.encoder, "scale"):
            #     print(f"scale {layer}", sae.encoder.scale)
            # print(f"thresh {layer}", sae.activation_thresholds())

        first_sae_layer = min(saes.keys())

        progress = MultilineProgress(
            total=num_tokens,
            desc=["Tuning BatchTopK thresholds for replacement model"],
            num_header_lines=1,
        )
        num_used_tokens = 0

        # Only train scale
        params = [
            param
            for sae in saes.values()
            for _, param in sae.encoder.interaction_params()
        ]
        if params:
            optimizer = torch.optim.Adam(
                params,
                lr=interaction_lr,
                betas=betas,
            )
        for step, batch in enumerate(
            make_dataloader(
                replacement_model,
                tokenizer,
                dataset,
                max_tokens=num_tokens,
                tokenizer_batch_size=tokenizer_batch_size,
                inference_batch_size=inference_batch_size,
            )
        ):
            if params:
                optimizer.zero_grad()
            batch.to(replacement_model.device)

            # Run the model until just prior to outputting logits, since this will be enough
            # to get activations to flow through each SAE.
            if lr_schedule is not None:
                for layer, sae in saes.items():
                    sae.set_activation_threshold_lr(
                        lr_schedule(min(num_used_tokens / num_tokens, 1.0), layer)
                    )
            # for pg in optimizer.param_groups:
            #     pg["lr"] =
            if params:
                baseline_activations = make_activation_batch(
                    model,
                    [(layer, "layer") for layer in saes.keys()],
                    # [(model.num_layers - 1, "layer"), (first_sae_layer, "layer")],
                    batch,
                    end_layer=model.num_layers,
                )
                replacement_activations = make_activation_batch(
                    replacement_model,
                    [(layer, "sae") for layer in saes.keys()],
                    # [(model.num_layers - 1, "sae")],
                    batch,
                    start_input=baseline_activations[first_sae_layer].layer_output,
                    start_layer=first_sae_layer,
                    end_layer=model.num_layers,
                    start_at_sae=True,
                )
                # print("max f", replacement_activations[model.num_layers-1].sae_features.max().item())

                # loss = mse_loss(
                #     replacement_activations[model.num_layers - 1].sae_output,
                #     baseline_activations[model.num_layers - 1].layer_output,
                #     batch,
                # )
                loss = torch.zeros((1,), dtype=torch.float32, device=model.device)
                for layer in saes.keys():
                    loss += mse_loss(
                        replacement_activations[layer].sae_output,
                        baseline_activations[layer].layer_output,
                        batch,
                    )
                loss.backward()

                optimizer.step()

            else:
                with torch.no_grad():
                    ab = make_activation_batch(
                        replacement_model,
                        [(model.num_layers - 1, "sae")],
                        batch,
                        end_layer=model.num_layers,
                    )
                    # print("max f", ab[model.num_layers-1].sae_features.max().item())

            num_used_tokens += batch.num_tokens
            progress.total = max(num_tokens, num_used_tokens)
            progress.update(batch.num_tokens)

        progress.close()
        # for layer, sae in saes.items():
        #     if hasattr(sae.encoder, "scale"):
        #         print(f"scale {layer}", sae.encoder.scale)
        #     print(f"thresh {layer}", sae.activation_thresholds())
    finally:
        if offload_after_training:
            try:
                for sae in saes.values():
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
            optimizer.step()
            optimizer.zero_grad()
            loss = None
            num_used_tokens += batch.num_tokens

        # After first batch, step before doing evals
        if num_used_tokens + previous_trained_tokens > 0:
            _do_step()

        if num_used_tokens >= eval_threshold:
            evals, postfix_dict = eval_fn(training_batch)
            progress.set_postfix(postfix_dict, refresh=False)
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
) -> TrainingResult:
    try:
        model.eval()
        training_saes = {
            layer: SAE(deepcopy(initial_saes[layer].config))
            for layer in config.train_layers
        }
        train_result = TrainingResult(training_saes)

        for layer in sorted(config.train_layers, reverse=True):
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
                sae.onload()
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
            elif fine_tune_source_dir is not None:
                # Init weights from source checkpoint
                source_checkpoint, token_offset = find_checkpoint_after(
                    fine_tune_source_dir,
                    layer,
                    int((1.0 - config.finetune_fraction) * config.num_train_tokens),
                )
                print(f"Loading {source_checkpoint} for finetuning")
                sae = training_saes[layer]
                sae.load_state_dict(
                    get_state_dict_from_checkpoint(source_checkpoint), assign=True
                )
                sae.onload()
            else:
                # Init weights from next layer, if it exists
                sae = training_saes[layer]
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

            if config.method is TrainingMethod.standard:
                stepper = StandardTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.next_layer:
                stepper = NextLayerTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.e2e:
                stepper = EndToEndTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.e2e_full:
                stepper = EndToEndFullTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.finetuned:
                stepper = KLFinetuneTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.next_layer_finetuned:
                stepper = NextLayerFinetunedTrainingStepper(model, layer, training_saes)
            elif config.method is TrainingMethod.full_replacement:
                stepper = FullReplacementTrainingStepper(model, training_saes)

            eval_model = make_replacement_model(model, training_saes)

            if config.method is TrainingMethod.full_replacement:
                optimizer = make_optimizer(
                    training_saes, list(training_saes.keys()), config
                )
            else:
                # Keep SAEs used in the replacement model in train mode, so eg for BatchTopK
                # we automatically retune the threshold based on now having an SAE at the previous layer.
                # Unused layers should be in eval mode.
                for other_layer in range(layer, model.num_layers):
                    if other_layer in stepper.replacement_model.sae_layers:
                        training_saes[other_layer].train()
                        training_saes[other_layer].requires_grad_(layer == other_layer)
                    elif other_layer in training_saes:
                        training_saes[other_layer].eval()
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
                checkpoint_dir=checkpoint_dir,
                # backward_fn=backward_fn,
            )
            if checkpoint_dir:
                save_training_result(
                    {
                        result_layer: [train_result[result_layer][-1]]
                        for result_layer in (
                            range(model.num_layers)
                            if config.method is TrainingMethod.full_replacement
                            else [layer]
                        )
                    },
                    checkpoint_dir,
                    keep_in_ram=True,
                    blocking=True,
                )

        return train_result
    finally:
        if offload_after_training:
            try:
                for sae in train_result.final_saes.values():
                    sae.offload()
            except Exception:
                pass
