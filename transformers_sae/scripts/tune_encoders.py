import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import (
    MemoryTrackingMode,
    method_to_saes,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.training import TrainingConfig, TrainingMethod, tune_encoder
from transformers_sae.validation import generate_with_replacement, run_validations

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = torch.device("cuda:0")
    TRAINING_BATCH_SIZE = 2
elif torch.mps.is_available():
    TRAINING_DEVICE = torch.device("mps:0")
    TRAINING_BATCH_SIZE = 2
else:
    TRAINING_DEVICE = torch.device("cpu")
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
NUM_ENCODER_TUNING_TOKENS = int(1e5)
NUM_THRESHOLD_TUNING_TOKENS = int(1e6)
NUM_TRAINING_TOKENS = int(5e7)
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
parser.add_argument(
    "-l",
    "--start-layer",
    type=int,
    default=0,
    help="Starting layer for SAE replacement in the tuned model (default: 0)",
)
parser.add_argument(
    "-rl",
    "--reference-start-layer",
    type=int,
    help="Starting layer for SAE replacement in the reference model (default: None)",
)
args = parser.parse_args()


for training_method in args.training_methods:
    results_path = f"{VALIDATION_BASE_PATH}/{training_method}"

    if args.reference_start_layer is None:
        load_start_layer = args.start_layer
    else:
        load_start_layer = min(args.reference_start_layer, args.start_layer)

    saes = method_to_saes(
        CHECKPOINT_BASE_PATH,
        training_method,
        range(load_start_layer, END_LAYER + 1),
        TRAINING_DEVICE,
    )

    if args.reference_start_layer is not None:
        reference_model = make_replacement_model(
            model,
            {
                layer: sae
                for layer, sae in saes.items()
                if layer >= args.reference_start_layer and layer <= END_LAYER
            },
        )
    else:
        reference_model = model

    for start_layer in (args.start_layer,):
        print(
            f"Tuning encoders for {training_method} replacement starting at {start_layer}"
        )

        tr = tune_encoder(
            reference_model,
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
            checkpoint_dir=f"{CHECKPOINT_BASE_PATH}/{training_method}_tuned_encoder_{start_layer}_1e5",
            force_retrain=False,
            train_encoders_from_scratch=False,
            # run_full_evals=True,
            # num_previous_replacement_layers=2,
        )
        saes = {
            layer: sae
            for layer, sae in tr.final_saes.items()
            if layer >= start_layer and layer <= END_LAYER
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
