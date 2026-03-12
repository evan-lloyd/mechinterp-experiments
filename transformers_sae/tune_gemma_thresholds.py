from transformers_sae.multiline_progress import MultilineProgress
from transformers_sae.metrics import l0_eval
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import (
    MemoryTrackingMode,
    find_latest_checkpoint,
    load_checkpoint,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.validation import generate_with_replacement, run_validations
from transformers_sae.tokenization import make_dataloader
from transformers_sae.activation_data import make_activation_batch

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

TOKENIZER_BATCH_SIZE = 256
NUM_VALIDATION_TOKENS = int(1e6)
NUM_TUNE_TOKENS = int(1e6)

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
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        # ),
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


checkpoint_dir = "/workspace/sae_checkpoints/gemma_2_2b/next_layer_old/"

saes = {}


def load_layer_checkpoint(layer):
    checkpoint = find_latest_checkpoint(checkpoint_dir, layer)
    if checkpoint is not None:
        sae = load_checkpoint(checkpoint).sae
        sae.train()
        sae.onload()
        print(f"Loaded checkpoint for layer {layer}")
        return layer, sae
    else:
        print(f"No checkpoint found for layer {layer}")
        return layer, None


# Load the latest checkpoints for each layer in parallel
with ThreadPoolExecutor() as executor:
    results = executor.map(load_layer_checkpoint, range(model.num_layers - 1, -1, -1))
    for layer, sae in results:
        if sae is not None:
            saes[layer] = sae

full_replacement_model = make_replacement_model(model, saes)
num_used_tokens = 0

print("pre-tuned thresholds", {layer: sae.encoder.batch_topk_threshold.item() for layer, sae in saes.items()})

with torch.no_grad():
    progress = MultilineProgress(
        total=NUM_TUNE_TOKENS,
        desc=["Tuning SAE thresholds in replacement model"],
        num_header_lines=1,
    )
    for step, batch in enumerate(
        make_dataloader(
            model,
            tokenizer,
            training_dataset,
            max_tokens=NUM_TUNE_TOKENS,
            tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
            inference_batch_size=TRAINING_BATCH_SIZE,
        )
    ):
        batch.to(model.device)
        sae_activations = make_activation_batch(
            full_replacement_model,
            [],
            # [(layer, "sae") for layer in range(model.num_layers)],
            batch,
            # We don't need logits
            end_layer=model.num_layers,
        )
        progress.total = max(NUM_TUNE_TOKENS, num_used_tokens)
        progress.update(batch.num_tokens)

print("post-tuned thresholds", {layer: sae.encoder.batch_topk_threshold.item() for layer, sae in saes.items()})

for sae in saes.values():
    sae.eval()

validations = run_validations(
    model,
    tokenizer,
    saes,
    validation_dataset,
    TOKENIZER_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    NUM_VALIDATION_TOKENS,
    cache_dir=None,
)

print(
    f"mean rre={ {k: np.mean(v.rre).item() for k, v in validations.layer_results.items() if v.rre is not None} }"
)
print(
    f"mean l0={ {k: np.mean(v.l0).item() for k, v in validations.layer_results.items() if v.l0 is not None} }"
)
print(
    f"geom mean kl={ {k: np.exp(np.mean(np.log(np.clip(v.kl, min=1e-9)))).item() for k, v in validations.layer_results.items() if v.kl is not None} })"
)
print(
    f"arith mean kl={ {k: np.mean(v.kl).item() for k, v in validations.layer_results.items() if v.kl is not None} }"
)
print(
    f"live features={ {k: sum(v.live_features) / saes[0].config.d_sae for k, v in validations.layer_results.items() if v.live_features is not None} }"
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
    )
