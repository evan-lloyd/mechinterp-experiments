import os
import torch
import numpy as np
from transformers_sae.ops import load_saes

# Path to your checkpoints (edit this if needed)
HF_BUCKET_LOCAL = os.environ.get("HF_BUCKET_LOCAL")
CHECKPOINT_BASE_PATH = f"{HF_BUCKET_LOCAL}/gemma_2_2b"
CHECKPOINT_DIR = f"{CHECKPOINT_BASE_PATH}/next_layer_lista_iters_10_tuned_encoder_0"

# Load all SAEs in this checkpoint directory. Guess the layer range if necessary.
# We'll attempt to load SAEs for layers 0-31, edit as needed:
NUM_LAYERS = 26
saes = load_saes(CHECKPOINT_DIR, list(range(NUM_LAYERS)))

print(f"Loaded {len(saes)} SAEs from {CHECKPOINT_DIR}")
for layer, sae in saes.items():
    scale = sae.encoder.scale
    if isinstance(scale, torch.nn.Parameter):
        scale_value = scale.detach().cpu().numpy()
    else:
        scale_value = np.array(scale)
    print(f"Layer {layer} encoder.scale: {scale_value}")
