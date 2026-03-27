import os
import re
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import cloudpickle
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.benchmark import BenchmarkModel, MMLUBenchmark, MMLUTask
from transformers_sae.ops import (
    MemoryTrackingMode,
    find_latest_checkpoint,
    load_checkpoint,
)
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model
from transformers_sae.sae_lens_wrapper import (
    SAELensSAEWrapper,
    wrap_sae_lens_pretrained,
)

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

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

HF_BUCKET_LOCAL = os.environ.get("HF_BUCKET_LOCAL", "/workspace/sae_checkpoints")
VALIDATION_BASE_PATH = f"{HF_BUCKET_LOCAL}/validations/gemma_2_2b"
CHECKPOINT_BASE_PATH = f"{HF_BUCKET_LOCAL}/gemma_2_2b"
BENCHMARK_BASE_PATH = f"{HF_BUCKET_LOCAL}/benchmarks"
NUM_TRAINING_TOKENS = int(1e8)

# Somewhat arbitrary list of tasks; these are the first 10 from the DeepEval enum that
# result in tokenizations that are short enough for our SAE's context window.
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
# Validations are saved per start_layer; benchmarks use full replacement (all layers).
START_LAYER = 0

CHECKPOINT_TRAINING_METHODS = (
    "next_layer_finetuned_interaction",
    "next_layer",
    "next_layer_interaction",
    "next_layer_finetuned",
)

GEMMA_SCOPE_TRAINING_METHODS = (
    "gemma_scope",
    "gemma_scope_canonical_l0",
    "gemma_scope_100_l0",
)

gemma_release = "gemma-scope-2b-pt-res-canonical"

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

with files("sae_lens").joinpath("pretrained_saes.yaml").open("r") as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)
yaml_rows = [
    row for row in yaml_data[gemma_release]["saes"] if "width_16k" in row["id"]
]
gemma_scope_sae_target_l0 = {
    layer: int(re.match(r".+?average_l0_(\d+)", yd["path"]).group(1))
    for layer in range(model.num_layers)
    for yd in yaml_rows
    if yd["id"] == f"layer_{layer}/width_16k/canonical"
}


def load_saes_from_checkpoints(checkpoint_dir: str):
    saes = {}

    def load_layer_checkpoint(layer):
        checkpoint = find_latest_checkpoint(checkpoint_dir, layer)
        if checkpoint is not None:
            cp = load_checkpoint(checkpoint)
            if cp.total_tokens_trained < NUM_TRAINING_TOKENS:
                print(f"Checkpoint below token budget for layer {layer}")
                return layer, None
            sae = load_checkpoint(checkpoint).sae
            sae.eval()
            sae.onload()
            print(f"Loaded checkpoint for layer {layer}")
            return layer, sae
        else:
            print(f"No checkpoint found for layer {layer}")
            return layer, None

    with ThreadPoolExecutor() as executor:
        results = executor.map(
            load_layer_checkpoint, range(model.num_layers - 1, -1, -1)
        )
        for layer, sae in results:
            if sae is not None:
                saes[layer] = sae

    return saes


def load_saes_gemma_scope():
    saes = {}

    def load_gemma_scope(layer):
        sae = wrap_sae_lens_pretrained(
            gemma_scope_sae_target_l0[layer],
            release=gemma_release,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=TRAINING_DEVICE,
        )
        return layer, sae

    with ThreadPoolExecutor() as executor:
        results = executor.map(load_gemma_scope, range(model.num_layers - 1, -1, -1))
        for layer, sae in results:
            if sae is not None:
                saes[layer] = sae

    return saes


def apply_replacement_thresholds(saes, replacement_thresholds):
    for layer, sae in saes.items():
        th = replacement_thresholds[layer]
        if isinstance(sae, SAELensSAEWrapper):
            sae.threshold_offset.fill_(th)
        else:
            for i, a in enumerate(sae.encoder.activation):
                a.threshold.fill_(th[i])


def load_activation_thresholds(training_method: str):
    path = (
        f"{VALIDATION_BASE_PATH}/{training_method}/{START_LAYER}.activation_thresholds"
    )
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def run_baseline_benchmark():
    out_path = f"{BENCHMARK_BASE_PATH}/baseline"
    if os.path.isfile(out_path):
        print("Skipping baseline, benchmark file already exists")
        return

    os.makedirs(BENCHMARK_BASE_PATH, exist_ok=True)
    mmlu = MMLUBenchmark(
        tokenizer,
        model.context_length,
        tasks=MMLU_TASKS,
    )
    mmlu.evaluate(model=BenchmarkModel(model, tokenizer), batch_size=MMLU_BATCH_SIZE)
    with open(out_path, "wb") as f:
        cloudpickle.dump(
            {"answer_stats": mmlu.answer_stats, "predictions": mmlu.predictions}, f
        )
    print(f"Wrote {out_path}")


def run_sae_benchmark(training_method: str, saes):
    out_path = f"{BENCHMARK_BASE_PATH}/{training_method}"
    if os.path.isfile(out_path):
        print(f"Skipping {training_method}, benchmark file already exists")
        return

    assert len(saes) == model.num_layers, (
        f"Missing SAEs for {training_method}, only had {set(saes.keys())}"
    )

    # Try a variant with no threshold replacement
    if training_method != "gemma_scope":
        replacement_thresholds = load_activation_thresholds(training_method)
        apply_replacement_thresholds(saes, replacement_thresholds)

    os.makedirs(BENCHMARK_BASE_PATH, exist_ok=True)
    mmlu = MMLUBenchmark(
        tokenizer,
        model.context_length,
        tasks=MMLU_TASKS,
    )
    mmlu.evaluate(
        model=BenchmarkModel(make_replacement_model(model, saes), tokenizer),
        batch_size=MMLU_BATCH_SIZE,
    )
    with open(out_path, "wb") as f:
        cloudpickle.dump(
            {"answer_stats": mmlu.answer_stats, "predictions": mmlu.predictions}, f
        )
    print(f"Wrote {out_path}")


run_baseline_benchmark()

gemma_scope_saes = None
for training_method in CHECKPOINT_TRAINING_METHODS:
    out_path = f"{BENCHMARK_BASE_PATH}/{training_method}"
    if os.path.isfile(out_path):
        print(f"Skipping {training_method}, benchmark file already exists")
        continue

    print(f"Running benchmarks for {training_method}")
    saes = load_saes_from_checkpoints(f"{CHECKPOINT_BASE_PATH}/{training_method}")
    run_sae_benchmark(training_method, saes)

for training_method in GEMMA_SCOPE_TRAINING_METHODS:
    out_path = f"{BENCHMARK_BASE_PATH}/{training_method}"
    if os.path.isfile(out_path):
        print(f"Skipping {training_method}, benchmark file already exists")
        continue
    print(f"Running benchmarks for {training_method}")
    saes = load_saes_gemma_scope()
    run_sae_benchmark(training_method, saes)
