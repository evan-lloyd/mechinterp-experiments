import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.ops import MemoryTrackingMode
from transformers_sae.replacement_model import (
    make_replacement_model,
)
from transformers_sae.sae import SAE, make_sae_config
from transformers_sae.validation import run_validations

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = "cuda:0"
    TRAINING_BATCH_SIZE = 16
elif torch.mps.is_available():
    TRAINING_DEVICE = "mps:0"
    TRAINING_BATCH_SIZE = 8
else:
    TRAINING_DEVICE = "cpu"
    TRAINING_BATCH_SIZE = 8

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
training_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
validation_dataset = load_dataset(
    "roneneldan/TinyStories", split="validation", streaming=True
)

with MemoryTrackingMode() as mtm:
    model = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M",
        dtype=torch.bfloat16,
        device_map=TRAINING_DEVICE,
        use_safetensors=True,
    )
    model.requires_grad_(False)
    model = make_replacement_model(
        model,
        {},
        num_layers=model.config.num_hidden_layers,
        context_length=512,
        d_model=model.config.hidden_size,
    )

print(model)
print(mtm.memory_max, mtm.memory_cur)


class NoisySAE(SAE):
    def __init__(self, *args, **kwargs):
        self.layer = kwargs.pop("layer")
        super().__init__(*args, **kwargs)
        self.encoder = torch.nn.Identity()

    def forward(self, *args, **kwargs):
        original_dtype = args[0].dtype
        original_output = args[0].float()
        reconstruction = (
            original_output
            + torch.rand_like(original_output)
            * (
                torch.linalg.vector_norm(
                    original_output, dim=-1, keepdim=True
                )
                + 1e-8
            )
            * 0.2
        )
        self.encoder(
            torch.ones(
                (args[0].shape[0], args[0].shape[1], 1),
                device=TRAINING_DEVICE,
            )
        )
        return reconstruction.to(original_dtype)


sae_config = make_sae_config(
    d_model=1,
    d_sae=1,
    device=model.device,
    train_dtype=torch.float32,
    inference_dtype=torch.bfloat16,
    encoder_kind="relu",
)

TRAINING_BATCH_SIZE = 1
TOKENIZER_BATCH_SIZE = 256
NUM_TOKENS = int(1e4)

validations = run_validations(
    model,
    tokenizer,
    {layer: NoisySAE(sae_config, layer=layer) for layer in range(model.num_layers)},
    training_dataset,
    TOKENIZER_BATCH_SIZE,
    TRAINING_BATCH_SIZE,
    NUM_TOKENS,
)

print(
    f"mean rre={ {k: np.mean(v.rre).item() for k, v in validations.layer_results.items() if v.rre is not None} })"
)
print(
    f"mean l0={ {k: np.mean(v.l0).item() for k, v in validations.layer_results.items() if v.l0 is not None} }"
)
print(
    f"geom mean kl={ {k: np.exp(np.mean(np.log(np.clip(v.kl, min=1e-9)))).item() for k, v in validations.layer_results.items() if v.kl is not None} })"
)
