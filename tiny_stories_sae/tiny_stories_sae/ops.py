import os

import torch
from transformers import AutoModel, AutoTokenizer, TextStreamer

# TinyStories33M max_position_embeddings
MAX_GENERATION = 2048


def generate(
    inputs,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    stream=True,
    stream_callback=None,
    **kwargs,
):
    if isinstance(inputs, str):
        inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    class TextStreamerWithCallback(TextStreamer):
        def put(self, value):
            super().put(value)

            if stream_callback:
                stream_callback(value[0])

    streamer = TextStreamerWithCallback(tokenizer, skip_prompt=True) if stream else None

    try:
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_length=MAX_GENERATION,
                streamer=streamer,
                tokenizer=tokenizer,
                **kwargs,
            )
    except KeyboardInterrupt:
        print("\n*** Generation aborted by user ***")


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)
