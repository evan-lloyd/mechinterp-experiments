import argparse
import os
from functools import partial
from typing import Any

import numpy as np
import torch
from datasets import interleave_datasets, load_dataset
from datasets.iterable_dataset import IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.benchmark.cqa import CQA
from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.tokenization import make_dataloader
from transformers_sae.training import TrainingConfig, TrainingMethod

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

RANDOM_SEED = 179081059
NUM_TRAINING_TOKENS = int(5e7)
NUM_FINETUNE_TOKENS = int(1e7)
CQA_FRACTION = 0.10
TOTAL_TOKENS = NUM_TRAINING_TOKENS + NUM_FINETUNE_TOKENS
FINETUNE_FRACTION = NUM_FINETUNE_TOKENS / TOTAL_TOKENS
EVAL_INTERVAL = int(1e5)
NUM_VALIDATION_TOKENS = int(1e6)
TOKENIZER_BATCH_SIZE = 256
CHECKPOINT_BASE_PATH = f"{os.getenv('HF_BUCKET_LOCAL')}/gemma_2_2b"

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
main_training_dataset = load_dataset(
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

cqa = CQA(
    n_shots=0,
    subsets=["all"],
    max_context=model.context_length,
    debiasing_sample_fraction=0.0,
    max_samples=None,
    run_permutations=False,
    split="train",
)


def _format_with_answers(preamble: str, example: dict[str, Any]):
    example["text"] = preamble + cqa.format_example(example, 0, True)
    return example


base_dataset = cqa.prepare_dataset()[0]
base_dataset = base_dataset.dataset.map(
    partial(_format_with_answers, base_dataset.preamble)
).to_list()


def _make_cqa_dataset():
    # Yield variable-sized concatenated examples as fine-tuning set
    cur_offset = 0
    while cur_offset < len(base_dataset):
        num_examples = np.random.randint(5, 25)
        yield {
            "text": "\n\n".join(
                e["text"] for e in base_dataset[cur_offset : cur_offset + num_examples]
            )
        }
        cur_offset += num_examples


training_dataset = interleave_datasets(
    [
        IterableDataset.from_generator(_make_cqa_dataset).repeat(None),
        main_training_dataset,
    ],
    probabilities=[CQA_FRACTION, 1.0 - CQA_FRACTION],
    seed=RANDOM_SEED,
)


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

num_cqa_tokens = 0
num_pile_tokens = 0
for batch in make_dataloader(
    model,
    tokenizer,
    training_dataset,
    max_tokens=int(1e7),
    tokenizer_batch_size=training_config.tokenizer_batch_size,
    inference_batch_size=training_config.training_batch_size,
    include_example_info=True,
):
    for ex in batch.example_info:
        if ex.original_example["text"].startswith("Question:"):
            num_cqa_tokens += ex.token_range[-1] - ex.token_range[0]
        else:
            num_pile_tokens += ex.token_range[-1] - ex.token_range[0]

print(
    f"CQA fraction: {num_cqa_tokens / (num_cqa_tokens + num_pile_tokens)} (CQA {num_cqa_tokens}, pile {num_pile_tokens}, total {num_cqa_tokens + num_pile_tokens})"
)
