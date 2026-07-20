import argparse
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import MemoryTrackingMode, method_to_saes, save_validations
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
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

VALIDATION_BASE_PATH = f"{os.getenv('HF_BUCKET_LOCAL')}/validations/gemma_2_2b"
CHECKPOINT_BASE_PATH = f"{os.getenv('HF_BUCKET_LOCAL')}/gemma_2_2b"
TOKENIZER_BATCH_SIZE = 256
NUM_VALIDATION_TOKENS = int(1e6)
NUM_THRESHOLD_TUNING_TOKENS = int(1e6)
NUM_TRAINING_TOKENS = int(5e7)
FINETUNE_FRACTION = 0.2
END_LAYER = model.num_layers

parser = argparse.ArgumentParser(
    description="Validate SAEs for specified training method(s)"
)
parser.add_argument(
    "-m",
    "--method",
    dest="training_methods",
    action="append",
    required=True,
    help="Training method to validate (may be specified multiple times, e.g. -m next_layer_lista_onsager -m next_layer)",
)
parser.add_argument(
    "-l",
    "--start-layer",
    type=int,
    default=0,
    help="Starting layer for SAE replacement (default: 0)",
)
parser.add_argument(
    "-t",
    "--after-tokens",
    type=int,
    required=False,
    help="Number of training tokens for target checkpoint (default: latest checkpoint)",
)
args = parser.parse_args()

start_layer = args.start_layer

for training_method in args.training_methods:
    train_activations = "_train_activations" in training_method
    results_path = f"{VALIDATION_BASE_PATH}/{training_method}"
    validation_file = os.path.join(
        results_path, f"{start_layer}.validation.cloudpickle"
    )
    if os.path.exists(validation_file):
        print(
            f"Validation for start layer {start_layer} already exists at {validation_file}, skipping."
        )
        continue

    saes = method_to_saes(
        CHECKPOINT_BASE_PATH,
        training_method.replace("_train_activations", ""),
        range(start_layer, model.num_layers),
        TRAINING_DEVICE,
        after_tokens=args.after_tokens,
    )
    if set(saes.keys()) != set(range(start_layer, END_LAYER)):
        raise ValueError(f"Didn't find full range of SAEs for {training_method}")

    for start_layer in (start_layer,):
        print(
            f"Running validations for {training_method} replacement starting at {start_layer}"
        )

        validations = run_validations(
            model,
            tokenizer,
            saes,
            validation_dataset,
            TOKENIZER_BATCH_SIZE,
            TRAINING_BATCH_SIZE,
            NUM_VALIDATION_TOKENS,
            start_layer=start_layer,
            offload=True,
            eval_layers=list(saes.keys()) + [model.num_layers],
            use_train_activations=train_activations,
        )
        save_validations({start_layer: validations}, results_path)

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
        # TODO: figure out why this breaks for train_activations gemma scope
        # with torch.autocast(
        #     device_type="cuda" if model.device.type == "cuda" else "cpu",
        #     dtype=torch.bfloat16,
        # ):
        #     generate_with_replacement(
        #         model,
        #         tokenizer,
        #         "The capital of France,",
        #         saes,
        #         offload=False,
        #     )
