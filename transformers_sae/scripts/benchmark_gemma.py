import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers_sae.benchmark import (
    BenchmarkKind,
    BenchmarkSpec,
    run_benchmark,
)
from transformers_sae.ops import MemoryTrackingMode, method_to_saes
from transformers_sae.replacement_model import GemmaReplacement, make_replacement_model

# Tweak TRAINING_BATCH_SIZE for your hardware if necessary
if torch.cuda.is_available():
    TRAINING_DEVICE = torch.device("cuda:0")
elif torch.mps.is_available():
    TRAINING_DEVICE = torch.device("mps:0")
else:
    TRAINING_DEVICE = torch.device("cpu")

model_id = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

HF_BUCKET_LOCAL = os.environ.get("HF_BUCKET_LOCAL")
VALIDATION_BASE_PATH = f"{HF_BUCKET_LOCAL}/validations/gemma_2_2b"
CHECKPOINT_BASE_PATH = f"{HF_BUCKET_LOCAL}/gemma_2_2b"
BENCHMARK_BASE_PATH = f"{HF_BUCKET_LOCAL}/benchmarks/gemma_2_2b"
TOKENIZER_BATCH_SIZE = 256
INFERENCE_BATCH_SIZE = 16
START_LAYER = 0

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


BENCHMARK_SPECS = {
    "mmlu": BenchmarkSpec(
        BenchmarkKind.mmlu,
        n_shots=5,
        # subsets=["abstract_algebra"],
        # subsets=[
        #     "business_ethics",
        #     "clinical_knowledge",
        #     "medical_genetics",
        #     "high_school_physics",
        #     "virology",
        #     "high_school_microeconomics",
        #     "econometrics",
        #     "college_computer_science",
        #     "high_school_biology",
        #     "abstract_algebra",
        # ],
        subsets=[
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ],
    ),
    "arc-e": BenchmarkSpec(BenchmarkKind.arc_e, 10, []),
    "cqa": BenchmarkSpec(BenchmarkKind.cqa, 10, []),
}

# For dev purposes, allow forcing re-run of specific benchmarks
OVERRIDE_EXISTING_BENCHMARKS = {"cqa"}
# BENCHMARK_SUBSET = set(BENCHMARK_SPECS.keys())
BENCHMARK_SUBSET = {"cqa"}
# For dev, allow running truncated dataset
MAX_SAMPLES = 400
# MAX_SAMPLES = None

# DEBIASING_SAMPLE_FRACTION = 0.0
DEBIASING_SAMPLE_FRACTION = 0.05


def run_benchmarks(training_method: str):
    def _out_path(name: str):
        return f"{BENCHMARK_BASE_PATH}/{training_method}_{name}.parquet"

    specs_to_run = {
        k: v
        for k, v in BENCHMARK_SPECS.items()
        if (not os.path.exists(_out_path(k)) or k in OVERRIDE_EXISTING_BENCHMARKS)
        and k in BENCHMARK_SUBSET
    }
    if not specs_to_run:
        print(f"Skipping benchmarks for {training_method}; all already exist")
        return

    print(f"Running benchmarks for {training_method}: {list(specs_to_run.keys())}")
    saes = method_to_saes(
        CHECKPOINT_BASE_PATH,
        training_method,
        range(START_LAYER, model.num_layers),
        TRAINING_DEVICE,
    )
    if len(saes) != model.num_layers - START_LAYER and training_method != "baseline":
        raise RuntimeError(f"Missing SAEs for {training_method}, aborting run")

    replacement_model = make_replacement_model(model, saes)
    for sae in saes.values():
        sae.eval()
        sae.onload()

    for name, spec in specs_to_run.items():
        result = run_benchmark(
            replacement_model,
            tokenizer,
            spec,
            tokenizer_batch_size=TOKENIZER_BATCH_SIZE,
            inference_batch_size=INFERENCE_BATCH_SIZE,
            debiasing_sample_fraction=DEBIASING_SAMPLE_FRACTION,
            max_samples=MAX_SAMPLES,
            run_permutations=True,
        )

        os.makedirs(BENCHMARK_BASE_PATH, exist_ok=True)
        result.to_parquet(_out_path(name), index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks for specified training method(s)"
    )
    parser.add_argument(
        "-m",
        "--method",
        dest="training_methods",
        action="append",
        required=True,
        help="Training method to use (may be specified multiple times, e.g. -m next_layer_lista_onsager -m next_layer)",
    )
    args = parser.parse_args()

    for training_method in args.training_methods:
        run_benchmarks(training_method)
