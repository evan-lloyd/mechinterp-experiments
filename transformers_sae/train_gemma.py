import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import make_replacement_model, ReplacementModel

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = "cuda:0"
    TRAINING_BATCH_SIZE = 1
elif torch.mps.is_available():
    TRAINING_DEVICE = "mps:0"
    TRAINING_BATCH_SIZE = 2
else:
    TRAINING_DEVICE = "cpu"
    TRAINING_BATCH_SIZE = 2

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
training_dataset = load_dataset(
    "monology/pile-uncopyrighted-parquet",
    split="train",
    streaming=True,
    columns=["text"],
)
validation_dataset = load_dataset(
    "monology/pile-test-val",
    split="validation",
    revision="refs/convert/parquet",
    streaming=True,
    columns=["text"],
)


class GemmaReplacement(ReplacementModel):
    def get_model_args(self, batch, model_input, start_at_embedding):
        input_args, input_kwargs = super().get_model_args(
            batch, model_input, start_at_embedding
        )
        if not start_at_embedding:
            input_kwargs["position_embeddings"] = self.model.rotary_emb(
                model_input, batch.position_ids
            )
        return input_args, input_kwargs

    def get_layer_args(self, *args, **kwargs):
        layer_args, layer_kwargs = super().get_layer_args(*args, **kwargs)
        layer_kwargs["position_embeddings"] = kwargs.get("position_embeddings")
        return layer_args, layer_kwargs


with MemoryTrackingMode() as mtm:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=TRAINING_DEVICE,
        dtype=torch.bfloat16,
        use_safetensors=True,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        # ),
    )
    model = make_replacement_model(
        model,
        {},
        num_layers=model.config.num_hidden_layers,
        context_length=1024,  # model.config.max_position_embeddings,
        d_model=model.config.hidden_size,
        norm_path="model.norm",
        layer_path="model.layers",
        replacement_class=GemmaReplacement,
    )

print(model)
print(mtm.memory_max)
print(mtm.memory_cur)

TRAINING_CACHE_DIR = None if torch.cuda.is_available() else ".training_cache"
VALIDATION_CACHE_DIR = None if torch.cuda.is_available() else ".validation_cache"
NUM_TRAINING_TOKENS = int(1e7) if torch.cuda.is_available() else int(1e6)
EVAL_INTERVAL = int(1e5)
NUM_VALIDATION_TOKENS = int(1e6) if torch.cuda.is_available() else int(1e5)
D_SAE = model.d_model * 4
TOPK = 100
TOKENIZER_BATCH_SIZE = 256
FINETUNE_FRACTION = 0.1
# Note this will use up ~1.8GB of space, set to False if you want to skip
SAVE_FINAL_RESULTS = True

import numpy as np

from transformers_sae.sae import SAE, make_sae_config
from transformers_sae.training import TrainingConfig, TrainingMethod, fine_tune, train
from transformers_sae.validation import run_validations


def SAE_SPECS():
    return TrainingMethod.__iter__()


empty_saes = {
    method: {
        layer: SAE(
            make_sae_config(
                d_model=model.d_model,
                d_sae=D_SAE,
                device=TRAINING_DEVICE,
                # dtype=torch.float32,
                dtype=torch.bfloat16,
                encoder_kind="topk",
                top_k=TOPK,
            )
        )
        for layer in range(model.num_layers)
    }
    for method in SAE_SPECS()
}


def linear_decay_during_finetune(frac_trained: float):
    if frac_trained < (1 - FINETUNE_FRACTION):
        return 1.0
    return 1.0 - (frac_trained - (1 - FINETUNE_FRACTION)) / FINETUNE_FRACTION


training_config = {
    method: TrainingConfig(
        tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_train_tokens=NUM_TRAINING_TOKENS,
        eval_interval=EVAL_INTERVAL,
        train_layers=list(range(model.num_layers)),
        lr=1e-3,
        interaction_lr=1e-3,
        lr_schedule=linear_decay_during_finetune,  # per Karvonen (2025)
        downstream_reconstruction_weight=1.0,
        reconstruction_weight=1.0,
        balance_reconstruction_losses=True,
        method=method,
        finetune_fraction=FINETUNE_FRACTION
        if method in (TrainingMethod.finetuned, TrainingMethod.next_layer_finetuned)
        else None,
    )
    for method in SAE_SPECS()
}

training_results = {}
validation_results = {}

import os

from transformers_sae.activation_cache import build_cache

if TRAINING_CACHE_DIR and (
    not os.path.exists(TRAINING_CACHE_DIR) or not os.listdir(TRAINING_CACHE_DIR)
):
    build_cache(
        TRAINING_CACHE_DIR,
        model,
        tokenizer,
        training_dataset,
        tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
        inference_batch_size=TRAINING_BATCH_SIZE,
        num_tokens=NUM_TRAINING_TOKENS,
    )

if VALIDATION_CACHE_DIR and (
    not os.path.exists(VALIDATION_CACHE_DIR) or not os.listdir(VALIDATION_CACHE_DIR)
):
    build_cache(
        VALIDATION_CACHE_DIR,
        model,
        tokenizer,
        validation_dataset,
        tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
        inference_batch_size=TRAINING_BATCH_SIZE,
        num_tokens=NUM_VALIDATION_TOKENS,
    )

training_results = train(
    model,
    tokenizer,
    empty_saes[TrainingMethod.next_layer],
    training_dataset,
    training_config[TrainingMethod.next_layer],
    cache_dir=TRAINING_CACHE_DIR,
    checkpoints_at=list(
        range(
            int(1e6),
            training_config[TrainingMethod.next_layer].num_train_tokens,
            int(1e6),
        )
    ),
    checkpoint_dir="/workspace/gemma_2_2b/next_layer/",
)
