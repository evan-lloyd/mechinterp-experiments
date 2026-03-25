import os
import re
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import cloudpickle
import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import (
    MemoryTrackingMode,
    save_validations,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.sae_lens_wrapper import wrap_sae_lens_pretrained
from transformers_sae.training import tune_activation_thresholds
from transformers_sae.validation import run_validations

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

gemma_release = "gemma-scope-2b-pt-res-canonical"

with files("sae_lens").joinpath("pretrained_saes.yaml").open("r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
yaml_data = [
    row for row in yaml_data[gemma_release]["saes"] if "width_16k" in row["id"]
]
gemma_scope_sae_target_l0 = {
    layer: int(re.match(r".+?average_l0_(\d+)", yd["path"]).group(1))
    for layer in range(model.num_layers)
    for yd in yaml_data
    if yd["id"] == f"layer_{layer}/width_16k/canonical"
}


def load_saes(checkpoint_dir: str):
    saes = {}

    def load_gemma_scope(layer):
        sae = wrap_sae_lens_pretrained(
            gemma_scope_sae_target_l0[layer],
            release=gemma_release,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=TRAINING_DEVICE,
        )
        return layer, sae

    # Load the latest checkpoints for each layer in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(load_gemma_scope, range(model.num_layers - 1, -1, -1))
        for layer, sae in results:
            if sae is not None:
                saes[layer] = sae

    return saes


MAX_THRESHOLD_LR = 1e-2
MIN_THRESHOLD_LR = 1e-5


def linear_decay(frac_trained: float):
    return (1.0 - frac_trained) * MAX_THRESHOLD_LR + frac_trained * MIN_THRESHOLD_LR


saes = {}
for training_method in (
    "gemma_scope_canonical_l0",
    "gemma_scope_100_l0",
):
    if "canonical" in training_method:
        target_l0 = gemma_scope_sae_target_l0
    else:
        target_l0 = {layer: 100 for layer in range(model.num_layers)}

    results_path = f"{VALIDATION_BASE_PATH}/{training_method}"

    # Check if all validation files already exist
    existing_validations = set(
        layer
        for layer in range(model.num_layers)
        if os.path.isfile(f"{results_path}/{layer}.validation.cloudpickle")
    )
    if len(existing_validations) == model.num_layers:
        print(f"Skipping {training_method}, validations already complete")
        continue

    if not saes:
        saes = load_saes(f"{CHECKPOINT_BASE_PATH}/{training_method}")
    assert len(saes) == model.num_layers, (
        f"Missing SAEs for {training_method}, only had {set(saes.keys())}"
    )
    for start_layer in set(saes.keys()) - existing_validations:
        print(
            f"Running validations for {training_method} replacement starting at {start_layer}"
        )
        for layer, sae in saes.items():
            sae.threshold_offset.fill_(0.0)
            sae.target_l0 = target_l0[layer]

        tune_activation_thresholds(
            model,
            tokenizer,
            {layer: sae for layer, sae in saes.items() if layer >= start_layer},
            training_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_THRESHOLD_TUNING_TOKENS,
            lr_schedule=linear_decay,
        )
        new_thresholds = {
            layer: sae.threshold_offset.item()
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
        mean_l0 = {
            k: np.mean(v.l0).item()
            for k, v in validations.layer_results.items()
            if v.l0 is not None
        }
        l0_diff = {k: target_l0[k] - mean_l0[k] for k in mean_l0.keys()}
        print(f"mean l0={mean_l0}")
        print(f"target l0={target_l0}")
        print(f"l0 diff={l0_diff}")
