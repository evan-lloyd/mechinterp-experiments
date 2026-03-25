import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import make_replacement_model

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = "cuda:0"
    TRAINING_BATCH_SIZE = 16
elif torch.mps.is_available():
    TRAINING_DEVICE = "mps:0"
    TRAINING_BATCH_SIZE = 8
else:
    TRAINING_DEVICE = "cpu"
    TRAINING_BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
training_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
validation_dataset = load_dataset(
    "roneneldan/TinyStories", split="validation", streaming=True
)

with MemoryTrackingMode() as mtm:
    model = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M",
        device_map=TRAINING_DEVICE,
        dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     # load_in_8bit=True,
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # ),
    )
    model = make_replacement_model(
        model,
        {},
        num_layers=model.config.num_hidden_layers,
        context_length=512,  # model.config.max_position_embeddings,
        d_model=model.config.hidden_size,
    )

print(model)
print(mtm.memory_max, mtm.memory_cur)
TRAINING_CACHE_DIR = None if torch.cuda.is_available() else ".training_cache"
VALIDATION_CACHE_DIR = None if torch.cuda.is_available() else ".validation_cache"
NUM_TRAINING_TOKENS = int(1e7) if torch.cuda.is_available() else int(1e6)
EVAL_INTERVAL = int(1e5)
NUM_VALIDATION_TOKENS = int(1e6) if torch.cuda.is_available() else int(1e5)
D_SAE = model.d_model * 4
TOPK = 100
TOKENIZER_BATCH_SIZE = 128
FINETUNE_FRACTION = 0.1
# Note this will use up ~1.8GB of space, set to False if you want to skip
SAVE_FINAL_RESULTS = True

import numpy as np

from transformers_sae.sae import (
    SAE,
    make_sae_config,
)
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
                train_dtype=torch.float32,
                inference_dtype=torch.bfloat16,
                encoder_kind="batch_topk",
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

for spec in (
    # TrainingMethod.standard,
    TrainingMethod.next_layer,
    # TrainingMethod.e2e,
    # TrainingMethod.e2e_full,
):
    print(f"Training {spec.value}")
    training_results[spec] = train(
        model,
        tokenizer,
        empty_saes[spec],
        training_dataset,
        training_config[spec],
        cache_dir=TRAINING_CACHE_DIR,
        # checkpoints_at=[int((1.0 - FINETUNE_FRACTION) * NUM_TRAINING_TOKENS)]
        # if spec in (TrainingMethod.standard, TrainingMethod.next_layer)
        # else None,
    )
    validation_results[spec] = run_validations(
        model,
        tokenizer,
        training_results[spec].final_saes,
        validation_dataset,
        num_tokens=NUM_VALIDATION_TOKENS,
        tokenizer_batch_size=training_config[spec].tokenizer_batch_size,
        inference_batch_size=training_config[spec].training_batch_size,
        cache_dir=VALIDATION_CACHE_DIR,
    )
    validations = validation_results[spec]
    print(
        f"mean rre={ {k: np.mean(v.rre).item() for k, v in validations.layer_results.items() if v.rre is not None} }"
    )
    print(
        f"mean l0={ {k: np.mean(v.l0).item() for k, v in validations.layer_results.items() if v.l0 is not None} }"
    )
    print(
        f"geom mean kl={ {k: np.exp(np.mean(np.log(np.clip(v.kl, min=1e-9)))).item() for k, v in validations.layer_results.items() if v.kl is not None} }"
    )
    print(
        f"arith mean kl={ {k: np.mean(v.kl).item() for k, v in validations.layer_results.items() if v.kl is not None} }"
    )
    print(
        f"live features={ {k: sum(v.live_features) / D_SAE for k, v in validations.layer_results.items() if v.live_features is not None} }"
    )
