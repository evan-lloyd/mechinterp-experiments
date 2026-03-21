from transformers_sae.tokenization import make_dataloader
from functools import partial
from math import ceil
from typing import List, Literal, Optional

import torch
from attr import dataclass
from datasets import IterableDataset
from sae_bench.evals.core.main import (
    ActivationsStore,
    CoreEvalConfig,
    run_evals,
)
from sae_bench.sae_bench_utils.general_utils import _standardize_sae_cfg
from transformer_lens import utils as transformer_lens_utils
from transformer_lens.HookedTransformer import Output as TransformerLensOutput
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask

from .data_batch import DataBatch
from .replacement_model import ReplacementModel
from .sae import SAE
from .truncated_model import truncated_model


class WrappedSAEConfig:
    def __init__(self, sae: SAE, layer: int):
        self.hook_name = f"model.layers.{layer}"
        self.hook_layer = layer
        self.context_size = 1024  # how my SAEs were trained
        self.hook_head_index = None
        self.model_name = "google/gemma-2-2b"
        self.architecture = "batch_topk"
        self.prepend_bos = True
        self.d_in = sae.config.d_model
        self.d_sae = sae.config.d_sae
        self.normalize_activations = "none"
        self.apply_b_dec_to_input = False
        self.dtype = "bfloat16"
        self.device = "cuda"


class WrappedSAE:
    sae: SAE
    cfg: WrappedSAEConfig
    dtype: str
    device: str

    def __init__(self, sae: SAE, layer: int):
        self.sae = sae
        self.layer = layer
        self.cfg = WrappedSAEConfig(sae, layer)
        self.dtype = str(sae.config.inference_dtype).replace("torch.", "")
        self.device = str(sae.config.device)

    def forward(self, x: torch.Tensor):
        return self.sae.forward(
            pass_through_positions=torch.empty((0,), dtype=torch.long, device=x.device),
            token_mask=torch.ones(
                (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            ),
        )

    def encode(self, x: torch.Tensor):
        return self.sae.encode(
            x,
            token_mask=torch.ones(
                (x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device
            ),
        )

    def decode(self, x: torch.Tensor):
        return self.sae.decode(x)

    def to(self, *args, **kwargs):
        return self

    @property
    def W_enc(self):
        return self.sae.encoder.linear.weight.T

    @property
    def W_dec(self):
        return self.sae.decoder.linear.weight.T


@dataclass
class WrapperCfg:
    device: "str"


class BatchWrapper(torch.Tensor):
    @staticmethod
    def __new__(cls, x, extra_data, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, t: torch.Tensor, batch: DataBatch):
        super().__init__()
        self.tensor = t
        self.batch = batch

    def __getattr__(self, name):
        if name in ("batch", "tensor"):
            return object.__getattribute__(self, name)
        return self.tensor.__getattr__(name)

    def __setattr__(self, name, value):
        if name in ("batch", "tensor"):
            return object.__setattr__(self, name, value)
        return self.tensor.__setattr__(name, value)


class ReplacementModelWrapper:
    model: ReplacementModel
    tokenizer: AutoTokenizer
    cfg: WrapperCfg

    def __init__(self, model: ReplacementModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = WrapperCfg(model.device)

    def to_tokens(self, input: str | List[str], *args, **kwargs):
        return self.tokenizer(
            input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.context_length,
        )["input_ids"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        input,
        return_type: Optional[str] = None,
        loss_per_token: bool = False,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[Literal["left", "right"]] = None,
        start_at_layer: Optional[int] = None,
        tokens: torch.Tensor | None = None,
        shortformer_pos_embed=None,
        attention_mask: Optional[torch.Tensor] = None,
        stop_at_layer: Optional[int] = None,
        past_kv_cache=None,
    ):
        """Doing the bare minimum to work with SAEBench."""
        # Assume input already tokenized
        assert isinstance(input, torch.Tensor)
        if start_at_layer is None:
            tokens = input
        else:
            raise NotImplementedError(
                "haven't validated starting from intermediate layers yet"
            )

        # ActivationsStore concatenates multiple independent inputs into the same batch row, separated
        # by bos tokens. We need to account for this in the position ids.
        # bos_mask = tokens == self.tokenizer.bos_token_id
        # bos_idx = bos_mask.nonzero(as_tuple=False)
        # print(bos_idx)
        # position_ids = torch.zeros_like(tokens)
        # for cur_idx, next_idx in zip(
        #     bos_idx[:-1],
        #     torch.cat(
        #         (
        #             bos_idx[1:],
        #             torch.tensor(
        #                 [[tokens.shape[0], tokens.shape[1]]],
        #                 device=bos_idx.device,
        #                 dtype=bos_idx.dtype,
        #             ),
        #         ),
        #     ),
        # ):
        #     # Finish off this row?
        #     if cur_idx[0] != next_idx[0]:
        #         position_ids[cur_idx[0], cur_idx[1] + 1 :] = torch.arange(
        #             0, tokens.shape[1] - cur_idx[1] - 1
        #         )
        #     # Else, partial row
        #     else:
        #         position_ids[cur_idx[0], cur_idx[1] + 1 : next_idx[1]] = torch.arange(
        #             0, next_idx[1] - cur_idx[1] - 1
        #         )

        # [batch_dim, 1, token_dim, token_dim]
        # transformers_attention_mask = create_causal_mask(
        #     config=self.model.config,
        #     inputs_embeds=tokens,  # Doesn't actually need the embedding, just uses the shape/device
        #     attention_mask=attention_mask,
        #     cache_position=torch.arange(0, tokens.shape[1], device=tokens.device),
        #     past_key_values=None,
        #     position_ids=position_ids,
        # )
        # print(
        #     "replacementmodelwrapper",
        #     transformers_attention_mask[0, 0, 0:10, 0:10].bool(),
        # )

        # batch = DataBatch(
        #     input_ids=tokens,
        #     position_ids=position_ids,
        #     attention_mask=transformers_attention_mask,
        #     num_tokens=tokens.numel()
        #     - bos_idx.shape[0],  # Don't count sequence separators
        #     batch_size=tokens.shape[0],
        #     num_dataset_rows=1,
        #     input_lens=[tokens.shape[1]] * tokens.shape[0],
        #     token_mask=attention_mask,
        #     special_token_indices=torch.empty(
        #         (0,), dtype=torch.long, device=tokens.device
        #     ),
        # )
        if start_at_layer is None:
            start_layer = -1
        else:
            start_layer = start_at_layer

        if stop_at_layer is None:
            if return_type is not None:
                end_layer = self.model.num_layers + 1
            else:
                end_layer = self.model.num_layers
        elif stop_at_layer < 0:
            end_layer = self.model.num_layers - stop_at_layer
        else:
            end_layer = stop_at_layer

        input_args, input_kwargs = self.model.get_base_model_args(
            tokens.batch, input, start_layer == -1
        )
        with truncated_model(
            self.model, start_layer, end_layer, start_at_sae=False, sae_kwargs={}
        ) as model_to_run:
            logits = model_to_run(*input_args, **input_kwargs, use_cache=False)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
        if return_type == "logits":
            return logits
        if return_type is None:
            return None

        # TransformerLens attention_mask is not for token->token attention; it's a mask on individual
        # tokens, like our token_mask
        loss = self.loss_fn(
            logits, tokens, tokens.batch.token_mask, per_token=loss_per_token
        )
        if return_type == "loss":
            return loss

        return TransformerLensOutput(logits, loss)

    def loss_fn(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        per_token: bool = False,
    ):
        """Wrapper around `utils.lm_cross_entropy_loss`.

        Used in forward() with return_type=="loss" or "both".
        """
        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        return transformer_lens_utils.lm_cross_entropy_loss(
            logits, tokens, attention_mask, per_token
        )

    def run_with_hooks(
        self,
        *model_args,
        fwd_hooks=None,
        bwd_hooks=None,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        **model_kwargs,
    ):
        """Doing the bare minimum to work with SAEBench; assumes we get a single hook only."""
        if fwd_hooks is None:
            fwd_hooks = []
        if bwd_hooks is None:
            bwd_hooks = []
        assert len(fwd_hooks) == 1
        hook_name = fwd_hooks[0][0]
        hook_fn = fwd_hooks[0][1]
        submodule = self.model.get_submodule(hook_name)

        assert submodule is not None

        def _hook_wrapper(_module, module_input, module_output):
            return hook_fn(module_output, hook=None)

        with submodule.register_forward_hook(_hook_wrapper):
            out = self.forward(*model_args, **model_kwargs)

        return out

    def run_with_cache(
        self,
        *model_args,
        names_filter: List[str] | None = None,
        device=None,
        remove_batch_dim: bool = False,
        incl_bwd: bool = False,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        pos_slice=None,
        return_cache_object=False,
        **model_kwargs,
    ):
        """Doing the bare minimum to work with SAEBench; assumes we get a single hook only."""
        assert bool(names_filter)
        hook_name = names_filter[0]
        submodule = self.model.get_submodule(hook_name)

        assert submodule is not None

        activation_cache = {}

        def _hook_output(probe_key, _module, _args, out):
            activation_cache[probe_key] = out

        with submodule.register_forward_hook(partial(_hook_output, hook_name)):
            out = self.forward(*model_args, **model_kwargs)

        return out, activation_cache


class TokenizationWrapper:
    def __init__(
        self,
        model: ReplacementModel,
        tokenizer: AutoTokenizer,
        dataset: IterableDataset,
        max_tokens: int | None,
        tokenizer_batch_size: int,
        inference_batch_size: int,
        offset: int = 0,
        max_batches: int | None = None,
    ):
        self.store_batch_size_prompts = inference_batch_size
        self.context_size = model.context_length
        self.dataloader_args = {
            "model": model,
            "tokenizer": tokenizer,
            "dataset": dataset,
            "max_tokens": max_tokens,
            "tokenizer_batch_size": tokenizer_batch_size,
            "inference_batch_size": inference_batch_size,
            "offset": offset,
            "max_batches": max_batches,
        }
        self.reset_input_dataset()

    def reset_input_dataset(self):
        self.dataloader = make_dataloader(**self.dataloader_args)
        self.data_iter = self.dataloader.__iter__()

    def shuffle_input_dataset(self, seed: int):
        self.dataloader_args["dataset"] = self.dataloader_args["dataset"].shuffle(
            seed=seed
        )
        self.reset_input_dataset()

    def get_batch_tokens(self, eval_batch_size_prompts: int):
        batch = next(self.data_iter)
        batch.to(self.dataloader_args["model"].device)
        return BatchWrapper(batch.input_ids, batch)


def run_sae_bench_evals(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    sae: SAE,
    layer: int,
    batch_size: int,
    num_tokens: int,
    dataset: IterableDataset,
):
    sae.eval()
    sae.onload()
    wrapped_model = ReplacementModelWrapper(model, tokenizer)

    num_batches = ceil(num_tokens / (batch_size * model.context_length))

    core_eval_config = CoreEvalConfig(
        model_name=model.name_or_path,
        batch_size_prompts=batch_size,
        n_eval_reconstruction_batches=num_batches,
        n_eval_sparsity_variance_batches=num_batches,
        exclude_special_tokens_from_reconstruction=True,
        dataset="",  # not actually used
        context_size=1024,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        llm_dtype=str(model.dtype).replace("torch.", ""),
    )

    wrapped_sae = WrappedSAE(sae, layer)
    _standardize_sae_cfg(wrapped_sae.cfg)

    activation_store = TokenizationWrapper(
        model, tokenizer, dataset, None, 256, batch_size, 0, None
    )
    # activation_store = ActivationsStore.from_sae(
    #     wrapped_model,
    #     wrapped_sae,
    #     context_size=model.context_length,
    #     dataset=dataset,
    # )
    # # Seems undesirable for evaluation!
    # activation_store.activations_mixing_fraction = 0.0

    with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=True):
        core_results = run_evals(
            sae=wrapped_sae,
            activation_store=activation_store,
            model=wrapped_model,
            eval_config=core_eval_config,
            ignore_tokens={
                tokenizer.pad_token_id,  # type: ignore
                tokenizer.eos_token_id,  # type: ignore
                tokenizer.bos_token_id,  # type: ignore
            },
            verbose=True,
        )

    sae.offload()
    return core_results
