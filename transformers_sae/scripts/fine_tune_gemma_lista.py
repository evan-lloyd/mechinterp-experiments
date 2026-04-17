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

TRAINING_CACHE_DIR = None
VALIDATION_CACHE_DIR = None
NUM_TRAINING_TOKENS = int(5e7)
EVAL_INTERVAL = int(1e5)
NUM_VALIDATION_TOKENS = int(1e6)
# to match Gemma Scope
D_SAE = 16384
D_MODEL = model.d_model
TOPK = 100
TOKENIZER_BATCH_SIZE = 256
FINETUNE_FRACTION = 0.2
N_ITERATIONS = 5

empty_saes = {
    layer: SAE(
        make_sae_config(
            d_model=model.d_model,
            d_sae=D_SAE,
            device=TRAINING_DEVICE,
            train_dtype=torch.float32,
            inference_dtype=torch.bfloat16,
            activation_kind="batch_topk",
            top_k=list(range(TOPK // N_ITERATIONS, TOPK + 1, TOPK // N_ITERATIONS)),
            encoder_kind="lista",
            n_iterations=N_ITERATIONS,
        )
    )
    for layer in range(model.num_layers)
}
print(empty_saes[0])
print([a.config.k for a in empty_saes[0].encoder.activation if hasattr(a.config, "k")])


def linear_decay_during_finetune(frac_trained: float, **kwargs):
    if frac_trained < (1 - FINETUNE_FRACTION):
        return 1.0
    return 1.0 - (frac_trained - (1 - FINETUNE_FRACTION)) / FINETUNE_FRACTION


training_config = TrainingConfig(
    tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
    training_batch_size=TRAINING_BATCH_SIZE,
    num_train_tokens=NUM_TRAINING_TOKENS,
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
    method=TrainingMethod.next_layer_finetuned,
    finetune_fraction=FINETUNE_FRACTION,
)

training_results = {}
validation_results = {}

training_results = train(
    model,
    tokenizer,
    empty_saes,
    training_dataset,
    training_config,
    cache_dir=TRAINING_CACHE_DIR,
    checkpoints_at=list(
        range(
            int(1e7),
            training_config.num_train_tokens,
            int(1e7),
        )
    ),
    checkpoint_dir="/workspace/sae_checkpoints/gemma_2_2b/next_layer_finetuned_lista/",
    force_retrain=False,
    fine_tune_source_dir="/workspace/sae_checkpoints/gemma_2_2b/next_layer_lista/",
    offload_after_training=False,
)

tune_activation_thresholds(
    model,
    tokenizer,
    training_results.final_saes,
    training_dataset,
    TOKENIZER_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    int(1e6),
    offload_after_training=False,
)

validations = run_validations(
    model,
    tokenizer,
    training_results.final_saes,
    validation_dataset,
    TOKENIZER_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    NUM_VALIDATION_TOKENS,
    cache_dir=VALIDATION_CACHE_DIR,
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
    f"live features={ {k: sum(v.live_features) / D_SAE for k, v in validations.layer_results.items() if v.live_features is not None} }"
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
