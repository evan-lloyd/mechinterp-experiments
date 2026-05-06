import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from datasets import load_dataset
from deepeval.benchmarks.mmlu.task import MMLUTask
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.benchmark import BenchmarkModel, MMLUBenchmark
from transformers_sae.ops import (
    MemoryTrackingMode,
    load_saes,
    save_validations,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.training import (
    TrainingConfig,
    TrainingMethod,
    tune_activation_thresholds,
    tune_encoder,
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
NUM_TRAINING_TOKENS = int(5e7)
# NUM_TRAINING_TOKENS = 0


MMLU_TASKS = [
    MMLUTask.BUSINESS_ETHICS,
    MMLUTask.CLINICAL_KNOWLEDGE,
    MMLUTask.MEDICAL_GENETICS,
    MMLUTask.HIGH_SCHOOL_PHYSICS,
    MMLUTask.VIROLOGY,
    MMLUTask.HIGH_SCHOOL_MICROECONOMICS,
    MMLUTask.ECONOMETRICS,
    MMLUTask.COLLEGE_COMPUTER_SCIENCE,
    MMLUTask.HIGH_SCHOOL_BIOLOGY,
    MMLUTask.ABSTRACT_ALGEBRA,
]
MMLU_BATCH_SIZE = 16


FINETUNE_FRACTION = 0.2


def linear_decay_during_finetune(frac_trained: float, **kwargs):
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

for training_method in (
    # "next_layer_finetuned_interaction",
    # "next_layer",
    # "next_layer_interaction",
    # "next_layer_finetuned",
    "next_layer_finetuned_lista",
    # "next_layer_lista",
    # "next_layer_lista_normalized_decoder",
    # "next_layer_finetuned_lista_normalized_decoder",
    # "next_layer_lista_feature_rescaling",
    # "next_layer_finetuned_lista_feature_rescaling",
    # "next_layer_lista_spectral_norm",
    # "next_layer_finetuned_lista",
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
        continue

    saes = load_saes(
        f"{CHECKPOINT_BASE_PATH}/{training_method}", model.num_layers, START_LAYER
    )
    # saes = load_saes(
    #     f"{CHECKPOINT_BASE_PATH}/{training_method}_tuned_encoder_{START_LAYER}_densebtk",
    #     START_LAYER,
    # )
    # assert len(saes) == model.num_layers, (
    #     f"Missing SAEs for {training_method}, only had {set(saes.keys())}"
    # )
    for start_layer in (START_LAYER,):
        # for start_layer in sorted(set(saes.keys()) - existing_validations, reverse=False):
        print(
            f"Running validations for {training_method} replacement starting at {start_layer}"
        )

        tr = tune_encoder(
            model,
            tokenizer,
            {layer: sae for layer, sae in saes.items() if layer >= start_layer},
            training_dataset,
            training_config,
            NUM_THRESHOLD_TUNING_TOKENS,
            offload_after_training=False,
            checkpoint_dir=f"{CHECKPOINT_BASE_PATH}/{training_method}_tuned_encoder_{START_LAYER}",
        )
        saes = tr.final_saes

        # tune_activation_thresholds(
        #     model,
        #     tokenizer,
        #     {layer: sae for layer, sae in saes.items() if layer >= start_layer},
        #     training_dataset,
        #     TOKENIZER_BATCH_SIZE,
        #     TRAINING_BATCH_SIZE,
        #     NUM_THRESHOLD_TUNING_TOKENS,
        #     offload_after_training=False,
        # )
        # new_thresholds = {
        #     layer: tuple(a.threshold.item() for a in sae.encoder.activation)
        #     for layer, sae in saes.items()
        #     if layer >= start_layer
        # }

        validations = run_validations(
            model,
            tokenizer,
            saes,
            validation_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_VALIDATION_TOKENS,
            start_layer=start_layer,
            offload=False,
        )
        # save_validations({start_layer: validations}, results_path)
        # with open(f"{results_path}/{start_layer}.activation_thresholds", "wb") as f:
        #     cloudpickle.dump(new_thresholds, f)

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
        # if start_layer == 0:
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
            # mmlu = MMLUBenchmark(
            #     tokenizer,
            #     model.context_length,
            #     tasks=MMLU_TASKS,
            # )
            # mmlu.evaluate(
            #     model=BenchmarkModel(make_replacement_model(model, saes), tokenizer),
            #     batch_size=MMLU_BATCH_SIZE,
            # )
