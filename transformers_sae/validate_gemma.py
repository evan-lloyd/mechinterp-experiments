import os
from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import (
    MemoryTrackingMode,
    find_latest_checkpoint,
    load_checkpoint,
    save_validations,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.training import tune_activation_thresholds
from transformers_sae.validation import run_validations

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
    model = make_replacement_model(
        model,
        {},
        num_layers=model.config.num_hidden_layers,
        context_length=1024,  # model.config.max_position_embeddings,
        d_model=model.config.hidden_size,
        layer_path="model.layers",
        replacement_class=GemmaReplacement,
    )
    model.eval()
    model.requires_grad_(False)

print(model)
print(mtm.memory_max)
print(mtm.memory_cur)

VALIDATION_BASE_PATH = "/workspace/sae_checkpoints/validations/gemma_2_2b"
CHECKPOINT_BASE_PATH = "/workspace/sae_checkpoints/gemma_2_2b"
TOKENIZER_BATCH_SIZE = 256
NUM_VALIDATION_TOKENS = int(1e6)
NUM_THRESHOLD_TUNING_TOKENS = int(1e6)
NUM_TRAINING_TOKENS = int(1e8)


def load_saes(checkpoint_dir: str):
    saes = {}

    def load_layer_checkpoint(layer):
        checkpoint = find_latest_checkpoint(checkpoint_dir, layer)
        if checkpoint is not None:
            cp = load_checkpoint(checkpoint)
            if cp.total_tokens_trained < NUM_TRAINING_TOKENS:
                print(f"No checkpoint found for layer {layer}")
                return layer, None
            sae = load_checkpoint(checkpoint).sae
            sae.eval()
            sae.onload()
            print(f"Loaded checkpoint for layer {layer}")
            return layer, sae
        else:
            print(f"No checkpoint found for layer {layer}")
            return layer, None

    # Load the latest checkpoints for each layer in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            load_layer_checkpoint, range(model.num_layers - 1, -1, -1)
        )
        # results = executor.map(load_gemma_scope, range(model.num_layers - 1, -1, -1))
        for layer, sae in results:
            if sae is not None:
                saes[layer] = sae

    return saes


for training_method in (
    "next_layer_finetuned_interaction",
    "next_layer",
    "next_layer_interaction",
    "next_layer_finetuned",
):
    results_path = f"{VALIDATION_BASE_PATH}/{training_method}"

    # Check if all validation files already exist
    existing_validations = set(
        layer
        for layer in range(model.num_layers)
        if os.path.isfile(f"{results_path}/{layer}.validation.cloudpickle")
    )
    if len(existing_validations) == model.num_layers:
        print(f"Skipping {training_method}, validations already complete")

    saes = load_saes(f"{CHECKPOINT_BASE_PATH}/{training_method}")
    assert len(saes) == model.num_layers, (
        f"Missing SAEs for {training_method}, only had {set(saes.keys())}"
    )
    orig_thresholds = {
        layer: tuple(a.threshold.item() for a in sae.encoder.activation)
        for layer, sae in saes.items()
    }
    for start_layer in set(saes.keys()) - existing_validations:
        print(
            f"Running validations for {training_method} replacement starting at {start_layer}"
        )
        for layer, sae in saes.items():
            for i, a in enumerate(sae.encoder.activation):
                a.threshold.fill_(orig_thresholds[layer][i])

        tune_activation_thresholds(
            model,
            tokenizer,
            {layer: sae for layer, sae in saes.items() if layer >= start_layer},
            training_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_THRESHOLD_TUNING_TOKENS,
        )
        new_thresholds = {
            layer: tuple(a.threshold.item() for a in sae.encoder.activation)
            for layer, sae in saes.items()
            if layer >= start_layer
        }

        validations = run_validations(
            model,
            tokenizer,
            saes,
            validation_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_VALIDATION_TOKENS,
            start_layer=start_layer,
        )
        save_validations({start_layer: validations}, results_path)
        with open(f"{results_path}/{start_layer}.activation_thresholds", "wb") as f:
            cloudpickle.dump(new_thresholds, f)

        print(
            f"{training_method} start layer {start_layer} KL: ",
            np.exp(
                np.mean(
                    np.log(
                        np.clip(
                            validations.layer_results[model.num_layers].kl,
                            min=1e-9,
                        )
                    )
                )
            ).item(),
        )
