import os
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import yaml

from transformers_sae.sae_lens_wrapper import convert_sae_lens_pretrained

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import re

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import (
    MemoryTrackingMode,
    load_saes,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.training import TrainingConfig, TrainingMethod, tune_encoder
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
NUM_ENCODER_TUNING_TOKENS = int(1e6)
NUM_THRESHOLD_TUNING_TOKENS = int(1e6)
NUM_TRAINING_TOKENS = int(5e7)
# NUM_TRAINING_TOKENS = 0
NUM_WARMUP_STEPS = 100

FINETUNE_FRACTION = 0.2


def linear_decay_during_finetune(frac_trained: float, **kwargs):
    # num_steps = frac_trained * NUM_ENCODER_TUNING_TOKENS
    # if num_steps < NUM_WARMUP_STEPS:
    #     return num_steps / NUM_WARMUP_STEPS
    if frac_trained < (1 - FINETUNE_FRACTION):
        return 1.0
    return 1.0 - (frac_trained - (1 - FINETUNE_FRACTION)) / FINETUNE_FRACTION


training_config = TrainingConfig(
    tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
    training_batch_size=TRAINING_BATCH_SIZE,
    num_train_tokens=NUM_TRAINING_TOKENS,
    eval_interval=int(1e5),
    # train_layers=list(range(10, model.num_layers)),
    train_layers=list(range(0, model.num_layers)),
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
    method=TrainingMethod.next_layer,
)

START_LAYER = 0
END_LAYER = model.num_layers - 1

parser = argparse.ArgumentParser(
    description="Tune encoder(s) for specified training method(s)"
)
parser.add_argument(
    "-m",
    "--method",
    dest="training_methods",
    action="append",
    required=True,
    help="Training method to use (may be specified multiple times, e.g. -m next_layer_lista_onsager -m next_layer)",
)
args = parser.parse_args()


GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"
def load_gemma_scope_saes(start_layer: int, end_layer: int, target_l0: int | None = None):
    saes = {}

    if target_l0 is None:
        with files("sae_lens").joinpath("pretrained_saes.yaml").open("r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        yaml_data = [
            row for row in yaml_data[GEMMA_SCOPE_RELEASE]["saes"] if "width_16k" in row["id"]
        ]
        l0_by_layer = {
            layer: int(re.match(r".+?average_l0_(\d+)", yd["path"]).group(1))
            for layer in range(model.num_layers)
            for yd in yaml_data
            if yd["id"] == f"layer_{layer}/width_16k/canonical"
        }
    else:
        l0_by_layer = {layer: target_l0 for layer in range(start_layer, end_layer + 1)}

    def load_gemma_scope(layer):
        sae = convert_sae_lens_pretrained(
            l0_by_layer[layer],
            release=GEMMA_SCOPE_RELEASE,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=TRAINING_DEVICE,
        )
        print(f"Loaded gemma scope {layer} with target L0={l0_by_layer[layer]}")
        return layer, sae

    # Load the latest checkpoints for each layer in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            load_gemma_scope, range(start_layer, end_layer + 1)
        )
        for layer, sae in results:
            if sae is not None:
                saes[layer] = sae

    return saes


for training_method in args.training_methods:
    results_path = f"{VALIDATION_BASE_PATH}/{training_method}"

    if "gemma_scope" in training_method:
        if "canonical" in training_method:
            target_l0 = None
        else:
            # gemma_scope_{target_l0}_l0
            target_l0 = int(training_method.split("_")[2])
       
        saes = load_gemma_scope_saes(START_LAYER, END_LAYER, target_l0)
    else:
        saes = load_saes(
            f"{CHECKPOINT_BASE_PATH}/{training_method}",
            END_LAYER + 1,
            START_LAYER,
        )
    for start_layer in (START_LAYER,):
        print(
            f"Tuning encoders for {training_method} replacement starting at {start_layer}"
        )

        tr = tune_encoder(
            model,
            tokenizer,
            {
                layer: sae
                for layer, sae in saes.items()
                if layer >= start_layer and layer <= END_LAYER
            },
            training_dataset,
            training_config,
            num_encoder_tuning_tokens=NUM_ENCODER_TUNING_TOKENS,
            num_threshold_tuning_tokens=NUM_THRESHOLD_TUNING_TOKENS,
            checkpoint_dir=f"{CHECKPOINT_BASE_PATH}/{training_method}_tuned_encoder_{START_LAYER}",
            force_retrain=False,
            train_encoders_from_scratch=False,
            # run_full_evals=True,
            # num_previous_replacement_layers=2,
        )
        saes = tr.final_saes

        validations = run_validations(
            model,
            tokenizer,
            saes,
            validation_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_VALIDATION_TOKENS,
            start_layer=start_layer,
            end_layer=END_LAYER + 1
            if END_LAYER < model.num_layers - 1
            else model.num_layers + 1,
            offload=False,
        )

        print(
            f"{training_method} start layer {start_layer} metrics",
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
                saes,
                offload=False,
            )
