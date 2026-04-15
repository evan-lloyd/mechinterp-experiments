import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import nullcontext
from math import sqrt

import torch
from accelerate import init_empty_weights
from datasets import IterableDataset, load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM

from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.sae import SAE, make_sae_config
from transformers_sae.training import TrainingConfig, TrainingMethod, train


def mock_dataset():
    yield {"text": "foo"}


# training_dataset = IterableDataset.from_generator(mock_dataset)
training_dataset = load_dataset(
    "monology/pile-uncopyrighted-parquet",
    split="train",
    streaming=True,
    columns=["text"],
)

if torch.cuda.is_available():
    TRAINING_DEVICE = "cuda:0"
elif torch.mps.is_available():
    TRAINING_DEVICE = "mps:0"
else:
    TRAINING_DEVICE = "cpu"
TRAINING_BATCH_SIZE = 1

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)

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

with (
    # torch._subclasses.fake_tensor.FakeTensorMode(
    #     allow_non_fake_inputs=True,
    #     shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(
    #         should_record_events=False
    #     ),
    # ),
    nullcontext(),
):
    TRAINING_CACHE_DIR = None
    VALIDATION_CACHE_DIR = None
    NUM_TRAINING_TOKENS = int(1e4)
    EVAL_INTERVAL = int(1e3)
    NUM_VALIDATION_TOKENS = int(1e3)
    # to match Gemma Scope
    D_SAE = 16384
    D_MODEL = model.d_model
    BASE_SAE_PARAMETERS = 2 * D_SAE * D_MODEL + D_SAE + D_MODEL
    D_SAE_INTERACTION = int(
        (
            sqrt(((2 * D_MODEL + 1) ** 2 + 4 * (BASE_SAE_PARAMETERS - D_MODEL)))
            - (2 * D_MODEL + 1)
        )
        / 2
    )
    INTERACTION_SAE_PARAMETERS = (
        D_SAE_INTERACTION**2
        + 2 * D_SAE_INTERACTION * D_MODEL
        + D_SAE_INTERACTION
        + D_MODEL
    )
    TOPK = 100
    TOKENIZER_BATCH_SIZE = 256
    FINETUNE_FRACTION = 0.1

    empty_saes = {
        layer: SAE(
            make_sae_config(
                d_model=model.d_model,
                d_sae=D_SAE_INTERACTION,
                device=TRAINING_DEVICE,
                train_dtype=torch.float32,
                inference_dtype=torch.bfloat16,
                activation_kind="batch_topk",
                top_k=TOPK,
                activation_kind=True,
            )
        )
        for layer in range(model.num_layers)
    }

    def linear_decay_during_finetune(frac_trained: float):
        if frac_trained < (1 - FINETUNE_FRACTION):
            return 1.0
        return 1.0 - (frac_trained - (1 - FINETUNE_FRACTION)) / FINETUNE_FRACTION

    training_config = TrainingConfig(
        tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        num_train_tokens=NUM_TRAINING_TOKENS,
        eval_interval=EVAL_INTERVAL,
        train_layers=list(range(0, model.num_layers)),
        betas=(
            0.0,
            0.999,
        ),  # TODO: is this actually good for our training method? not for tinystories anyway
        lr=1e-4,
        interaction_lr=1e-4,
        lr_schedule=linear_decay_during_finetune,  # per Karvonen (2025)
        downstream_reconstruction_weight=1.0,
        reconstruction_weight=1.0,
        balance_reconstruction_losses=True,
        method=TrainingMethod.full_replacement,
    )

    training_results = {}
    validation_results = {}

    with (
        nullcontext(),
        # MemoryTrackingMode() as mm,
    ):
        training_results = train(
            model,
            tokenizer,
            empty_saes,
            training_dataset,
            training_config,
        )

    # print(mm.memory_max, mm.memory_cur)
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory usage: {peak_mem:.2f} MB")
