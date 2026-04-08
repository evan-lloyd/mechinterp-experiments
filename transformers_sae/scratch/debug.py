import cloudpickle
from transformers_sae.ops import load_checkpoint, load_validations

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.sae import SAE, make_sae_config
from transformers_sae.training import (
    TrainingConfig,
    TrainingMethod,
    train,
    tune_activation_thresholds,
)
from transformers_sae.validation import generate_with_replacement, run_validations

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


sae = load_checkpoint(
    "/workspace/sae_checkpoints/gemma_2_2b/next_layer_finetuned_lista/layer_11_tokens_50000194.checkpoint"
).sae

empty_saes = {
    layer: SAE(
        make_sae_config(
            d_model=model.d_model,
            d_sae=sae.config.d_sae,
            device=TRAINING_DEVICE,
            train_dtype=torch.float32,
            inference_dtype=torch.bfloat16,
            encoder_kind="batch_topk",
            top_k=sae.config.encoder.activation_function.k,
            with_interaction=True,
            n_iterations=sae.config.encoder.n_iterations,
        )
    )
    for layer in range(model.num_layers)
}


validations = load_validations(
    "/workspace/sae_checkpoints/validations/gemma_2_2b/next_layer_finetuned_lista"
)

ats = cloudpickle.load(
    open(
        "/workspace/sae_checkpoints/validations/gemma_2_2b/next_layer_finetuned_lista/11.activation_thresholds", "rb"
    )
)
