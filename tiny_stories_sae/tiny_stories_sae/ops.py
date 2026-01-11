import os
from collections import defaultdict

import cloudpickle
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as save_file_numpy
from safetensors.torch import save_file as save_file_torch
from transformers import AutoModel, AutoTokenizer, TextStreamer

# TinyStories33M max_position_embeddings
MAX_GENERATION = 2048


def load_demo_run(from_dir, sae_args, sae_kwargs):
    from .training import TrainingMethod
    from .sae import SAE

    saes = defaultdict(lambda: defaultdict(dict))
    loaded_saes = set()
    with safe_open(
        f"{from_dir}/e2e_demo_saes.safetensors",
        framework="pt",
        device=sae_kwargs["device"],
    ) as f:
        for key in f.keys():
            dict_path, _ = key.split(":")
            dict_path = dict_path.split(".")
            dict_key = ".".join(dict_path)

            # We'll see each root multiple times
            if dict_key in loaded_saes:
                continue

            sae = saes[getattr(TrainingMethod, dict_path[0])][int(dict_path[1])] = SAE(
                *sae_args, **sae_kwargs
            )
            sae.load_state_dict(
                {
                    k.split(":")[1]: f.get_tensor(k)
                    for k in f.keys()
                    if k.startswith(dict_key)
                }
            )
            loaded_saes.add(dict_key)

    training_config = cloudpickle.load(
        open(f"{from_dir}/e2e_demo_training_config.cloudpickle", "rb")
    )
    training_results = cloudpickle.load(
        open(f"{from_dir}/e2e_demo_training_results.cloudpickle", "rb")
    )

    validation_evals = {}
    with safe_open(
        f"{from_dir}/e2e_demo_validation_evals.safetensors",
        framework="np",
        device="cpu",
    ) as f:
        for key in f.keys():
            parts = key.split(".")
            method = getattr(TrainingMethod, parts[0])
            if method not in validation_evals:
                validation_evals[method] = {}
            eval_key = parts[1]
            if eval_key not in validation_evals[method]:
                validation_evals[method][eval_key] = {}
            layer = int(parts[2])

            validation_evals[method][eval_key][layer] = f.get_tensor(key)

    replacement_evals = {}
    with safe_open(
        f"{from_dir}/e2e_demo_replacement_evals.safetensors",
        framework="np",
        device="cpu",
    ) as f:
        for key in f.keys():
            parts = key.split(".")
            method = getattr(TrainingMethod, parts[0])
            eval_key = parts[1]

            if method not in replacement_evals:
                replacement_evals[method] = {}

            replacement_evals[method][eval_key] = f.get_tensor(key)

    return saes, training_config, training_results, validation_evals, replacement_evals


def save_demo_run(
    out_dir,
    saes,
    training_config,
    training_results,
    validation_evals,
    replacement_evals,
):
    sae_dict = {}
    for method, saes_by_layer in saes.items():
        for layer, sae in saes_by_layer.items():
            for k, v in sae.state_dict().items():
                sae_dict[f"{method.name}.{layer}:{k}"] = v
    save_file_torch(sae_dict, f"{out_dir}/e2e_demo_saes.safetensors")

    cloudpickle.dump(
        training_config, open(f"{out_dir}/e2e_demo_training_config.cloudpickle", "wb")
    )
    cloudpickle.dump(
        training_results, open(f"{out_dir}/e2e_demo_training_results.cloudpickle", "wb")
    )

    validation_evals_dict = {}
    for method, evals in validation_evals.items():
        for eval_key, eval_dict in evals.items():
            for layer, values in eval_dict.items():
                if isinstance(values, np.ndarray):
                    validation_evals_dict[f"{method.name}.{eval_key}.{layer}"] = values
                else:
                    validation_evals_dict[f"{method.name}.{eval_key}.{layer}"] = (
                        np.full((1,), np.nan)
                    )
    save_file_numpy(
        validation_evals_dict, f"{out_dir}/e2e_demo_validation_evals.safetensors"
    )

    replacement_evals_dict = {}
    for method, evals in replacement_evals.items():
        for eval_key, values in evals.items():
            if isinstance(values, np.ndarray):
                replacement_evals_dict[f"{method.name}.{eval_key}"] = values
            else:
                replacement_evals_dict[f"{method.name}.{eval_key}"] = np.full(
                    (1,), np.nan
                )

    save_file_numpy(
        replacement_evals_dict, f"{out_dir}/e2e_demo_replacement_evals.safetensors"
    )


def clone_sae(sae):
    from .sae import SAE

    result = SAE(
        sae.d_model,
        sae.d_sae,
        device=sae.device,
        kind=sae.kind,
        topk=sae.topk,
        init_from=sae,
        with_inhibition=sae.with_inhibition,
        d_dense=sae.d_dense,
        normalize_activations=sae.normalize_activations,
    )

    return result


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
