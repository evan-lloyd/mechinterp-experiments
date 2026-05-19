import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import MemoryTrackingMode, load_saes
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.training import (
    TrainingConfig,
    TrainingMethod,
    train,
)
from transformers_sae.validation import generate_with_replacement, run_validations

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = "cuda:0"
    TRAINING_BATCH_SIZE = 2
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


with MemoryTrackingMode() as mtm:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=TRAINING_DEVICE,
        dtype=torch.bfloat16,
        use_safetensors=True,
    )
    model.eval()
    model.requires_grad_(False)
    model = make_replacement_model(
        model,
        {},
        num_layers=model.config.num_hidden_layers,
        context_length=1024,  # also used by Gemma scope
        d_model=model.config.hidden_size,
        layer_path="model.layers",
        replacement_class=GemmaReplacement,
    )

print(model)
print(mtm.memory_max)
print(mtm.memory_cur)

NUM_TRAINING_TOKENS = int(6e7)
NUM_FINETUNE_TOKENS = int(1e7)
TOTAL_TOKENS = NUM_TRAINING_TOKENS + NUM_FINETUNE_TOKENS
FINETUNE_FRACTION = NUM_FINETUNE_TOKENS / TOTAL_TOKENS
EVAL_INTERVAL = int(1e5)
NUM_VALIDATION_TOKENS = int(1e6)
TOKENIZER_BATCH_SIZE = 256

CHECKPOINT_BASE_PATH = f"{os.getenv('HF_BUCKET_LOCAL')}/gemma_2_2b/"
saes = load_saes(
    f"{CHECKPOINT_BASE_PATH}/next_layer_finetuned_lista_onsager",
    model.num_layers,
)

# TODO: we should refactor the fine tune logic in training.py to handle this
for sae in saes.values():
    sae.onload()
    # We will re-load the training version when we get to it
    sae.eval()


def linear_decay_during_finetune(frac_trained: float, **kwargs):
    if frac_trained < (1 - FINETUNE_FRACTION):
        return 1.0
    return 1.0 - (frac_trained - (1 - FINETUNE_FRACTION)) / FINETUNE_FRACTION


training_config = TrainingConfig(
    tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
    training_batch_size=TRAINING_BATCH_SIZE,
    num_train_tokens=TOTAL_TOKENS,
    eval_interval=EVAL_INTERVAL,
    train_layers=list(range(0, model.num_layers)),
    # train_layers=list(range(model.num_layers - 2, model.num_layers)),
    betas=(
        0.0,
        0.999,
    ),  # TODO: is this actually good for our training method? not for tinystories anyway
    lr=1e-4,
    interaction_lr=1e-4,
    threshold_lr=1e-2,
    lr_schedule=linear_decay_during_finetune,  # per Karvonen (2025)
    downstream_reconstruction_weight=1.0,
    reconstruction_weight=1.0,
    balance_reconstruction_losses=True,
    method=TrainingMethod.in_place_finetuned,
    finetune_fraction=FINETUNE_FRACTION,
)

training_results = train(
    model,
    tokenizer,
    saes,
    training_dataset,
    training_config,
    checkpoint_dir=f"{CHECKPOINT_BASE_PATH}/next_layer_finetuned_lista_onsager_7e7",
    fine_tune_source_dir=f"{CHECKPOINT_BASE_PATH}/next_layer_finetuned_lista_onsager",
    force_retrain=False,
    offload_after_training=False,
    fine_tune_in_place=True,
    override_token_offset=NUM_TRAINING_TOKENS,
)

validations = run_validations(
    model,
    tokenizer,
    training_results.final_saes,
    validation_dataset,
    TOKENIZER_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    NUM_VALIDATION_TOKENS,
    start_layer=training_config.train_layers[0],
    offload=False,
)

print(
    f"mean rre={ {k: np.mean(v.rre).item() for k, v in validations.layer_results.items() if v.rre is not None} }"
)
print(
    f"geom mean rre={ {k: np.exp(np.mean(np.log(np.clip(v.rre, a_min=1e-9, a_max=None)))).item() for k, v in validations.layer_results.items() if v.rre is not None} }"
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
    f"live features={ {k: sum(v.live_features) / saes[k].config.d_sae for k, v in validations.layer_results.items() if v.live_features is not None} }"
)

with torch.autocast(
    device_type="cuda" if model.device.type == "cuda" else "cpu",
    dtype=torch.bfloat16,
):
    generate_with_replacement(
        model,
        tokenizer,
        "The capital of France,",
        {
            layer: sae
            for layer, sae in training_results.final_saes.items()
            if layer >= training_config.train_layers[0]
        },
    )
