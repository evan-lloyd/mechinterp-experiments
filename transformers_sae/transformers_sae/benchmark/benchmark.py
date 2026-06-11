from dataclasses import dataclass
from enum import Enum

import pandas as pd
import torch
from transformers import AutoTokenizer

from ..activation_data import make_activation_batch
from ..multiline_progress import MultilineProgress
from ..replacement_model import ReplacementModel
from ..tokenization import make_dataloader
from .arc_e import ArcE
from .benchmark_runner import BenchmarkRunner
from .mmlu import MMLU


class BenchmarkKind(Enum):
    mmlu = "MMLU"
    arc_e = "Arc-E"


_BENCHMARK_RUNNER: dict[BenchmarkKind, type[BenchmarkRunner]] = {
    BenchmarkKind.mmlu: MMLU,
    BenchmarkKind.arc_e: ArcE,
}


@dataclass
class BenchmarkSpec:
    kind: BenchmarkKind
    n_shots: int
    subsets: list[str]


@dataclass
class BenchmarkExampleResult:
    prompt: str
    subset: str
    correct_answer: str
    actual_answer: str
    forced_valid_answer: str
    answer_probs: tuple[float]
    correct_answer_prob: float
    valid_answer_prob: float
    conditional_correct_answer_prob: float


@torch.no_grad()
def run_benchmark(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    spec: BenchmarkSpec,
    tokenizer_batch_size: int,
    inference_batch_size: int,
) -> pd.DataFrame:
    runner = _BENCHMARK_RUNNER[spec.kind](
        n_shots=spec.n_shots, subsets=spec.subsets, max_context=model.context_length
    )

    answer_token_ids = (
        tokenizer(runner.valid_answers, return_tensors="pt")
        .input_ids[:, 1:]
        .flatten()
        .to(model.device)
    )

    progress = MultilineProgress(
        total=runner.num_examples,
        desc=[f"Running benchmark: {spec.kind.value}"],
        num_header_lines=1,
    )
    example_results = []
    with torch.autocast(
        device_type="cuda" if model.device.type == "cuda" else "cpu",
        dtype=torch.bfloat16,
    ):
        for batch in make_dataloader(
            model,
            tokenizer,
            runner.dataset,
            None,  # Run until dataset is exhausted
            tokenizer_batch_size=tokenizer_batch_size,
            inference_batch_size=inference_batch_size,
            include_example_info=True,
            skip_long_examples=True,
        ):
            batch.to(model.device)
            # Get final residual rather than running full model, so we can only compute the logits
            # for the final token and save a little time / a lot of memory.
            ab = make_activation_batch(
                model,
                [
                    (
                        model.num_layers - 1,
                        "sae" if model.num_layers - 1 in model.sae_layers else "layer",
                    )
                ],
                batch,
                end_layer=model.num_layers,
            )[model.num_layers - 1]
            if (model.num_layers - 1) in model.sae_layers:
                final_residual = ab.sae_output
            else:
                final_residual = ab.layer_output
            assert final_residual is not None

            # We may have multiple inputs per "batch" row, so find the indices of the final tokens for
            # each individual example and compute their logits in parallel.
            batch_indices = torch.tensor(
                [ex.batch_index for ex in batch.example_info],
                device=final_residual.device,
            )
            token_indices = torch.tensor(
                [ex.token_range[1] - 1 for ex in batch.example_info],
                device=final_residual.device,
            )
            selected_residuals = (
                final_residual[batch_indices, token_indices, :]
                .flatten()
                .view(1, len(batch.example_info), -1)
            )
            answer_logprobs = (
                model.get_logits(selected_residuals).log_softmax(-1).squeeze(0)
            )
            valid_answer_logprobs = answer_logprobs[:, answer_token_ids]
            correct_answer_probs = valid_answer_logprobs.gather(
                1,
                torch.tensor(
                    [
                        runner.get_answer_index(ex.original_example)
                        for ex in batch.example_info
                    ],
                    device=model.device,
                ).unsqueeze(-1),
            ).exp()
            valid_answer_probs = valid_answer_logprobs.exp()
            any_valid_answer_probs = valid_answer_probs.sum(dim=-1)

            top_logit_indices = answer_logprobs.argmax(-1)
            top_logit_answers = [
                tokenizer.decode(idx.item()) for idx in top_logit_indices
            ]

            forced_valid_indices = valid_answer_logprobs.argmax(-1)
            forced_valid_answers = [
                tokenizer.decode(answer_token_ids[idx])
                # When the probabilities for valid answers are very close, we can get inconsistent values
                # for the argmax, so make the "forced" answer be the actual one if it was valid
                if top_logit_answers[i] not in runner.valid_answers
                else top_logit_answers[i]
                for i, idx in enumerate(forced_valid_indices)
            ]

            for i, ex in enumerate(batch.example_info):
                example_results.append(
                    BenchmarkExampleResult(
                        prompt=ex.original_example["text"],
                        subset=runner.get_subset(ex.original_example),
                        correct_answer=runner.valid_answers[
                            runner.get_answer_index(ex.original_example)
                        ],
                        actual_answer=top_logit_answers[i],
                        forced_valid_answer=forced_valid_answers[i],
                        answer_probs=tuple(valid_answer_probs[i, :].tolist()),
                        correct_answer_prob=correct_answer_probs[i].item(),
                        valid_answer_prob=any_valid_answer_probs[i].item(),
                        conditional_correct_answer_prob=correct_answer_probs[i].item()
                        / (any_valid_answer_probs[i].item() + 1e-8),
                    )
                )
            progress.set_postfix(
                {
                    "subjects": set(
                        runner.get_subset(ex.original_example)
                        for ex in batch.example_info
                    )
                },
                refresh=False,
            )
            progress.update(batch.num_dataset_rows)
    progress.close()

    df = pd.DataFrame([vars(er) for er in example_results])

    overall_accuracy = (df["actual_answer"] == df["correct_answer"]).mean()
    forced_valid_accuracy = (df["forced_valid_answer"] == df["correct_answer"]).mean()

    mean_correct_answer_prob = df["correct_answer_prob"].mean()
    mean_valid_answer_prob = df["valid_answer_prob"].mean()
    mean_conditional_correct_prob = df["conditional_correct_answer_prob"].mean()

    print(f"Overall accuracy (top logit): {overall_accuracy}")
    print(f"Forced valid answer accuracy: {forced_valid_accuracy}")
    print(f"Mean correct answer prob: {mean_correct_answer_prob}")
    print(f"Mean valid answer prob (any valid): {mean_valid_answer_prob}")
    print(f"Mean conditional correct answer prob: {mean_conditional_correct_prob}")

    return df
