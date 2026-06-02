from dataclasses import dataclass
from enum import Enum

import torch
from transformers import AutoTokenizer

from ..activation_data import make_activation_batch
from ..multiline_progress import MultilineProgress
from ..replacement_model import ReplacementModel
from ..tokenization import make_dataloader
from .benchmark_runner import BenchmarkRunner
from .mmlu import MMLU


class BenchmarkKind(Enum):
    mmlu = "MMLU"


_BENCHMARK_TO_DATASET: dict[BenchmarkKind, type[BenchmarkRunner]] = {
    BenchmarkKind.mmlu: MMLU
}


@dataclass
class BenchmarkSpec:
    kind: BenchmarkKind
    n_shots: int
    subsets: list[str]


@torch.no_grad()
def run_benchmark(
    model: ReplacementModel,
    tokenizer: AutoTokenizer,
    spec: BenchmarkSpec,
    tokenizer_batch_size: int,
    inference_batch_size: int,
):
    runner = _BENCHMARK_TO_DATASET[spec.kind](
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
    num_correct = 0
    total = 0
    for batch in make_dataloader(
        model,
        tokenizer,
        runner.dataset,
        None,  # Run until dataset is exhausted
        tokenizer_batch_size=tokenizer_batch_size,
        inference_batch_size=inference_batch_size,
        include_example_info=True,
    ):
        batch.to(model.device)
        # Get final residual rather than running full model, so we can only compute the logits
        # for the final token and save a little time.
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
        indices = [(ex.batch_index, ex.token_range[1] - 1) for ex in batch.example_info]
        batch_indices = torch.tensor(
            [idx[0] for idx in indices], device=final_residual.device
        )
        token_indices = torch.tensor(
            [idx[1] for idx in indices], device=final_residual.device
        )
        selected_residuals = (
            final_residual[batch_indices, token_indices, :]
            .flatten()
            .view(1, len(indices), -1)
        )
        answer_logits = model.get_logits(selected_residuals)

        # Compute logits and probabilities for valid answers
        probs = answer_logits.softmax(-1).squeeze(0)  # shape: (n_examples, vocab_size)

        # Get the logits for valid answers only
        # valid_logits = probs[:, answer_token_ids]
        # valid_logits shape: (n_examples, n_valid_answers)

        # Top logit answer (unrestricted, by vocab)
        top_logit_indices = probs.argmax(-1)
        top_logit_answers = [tokenizer.decode(idx.item()) for idx in top_logit_indices]

        # Probability of any valid answer (sum over valid answer indices)
        # any_valid_answer_probs = valid_logits.sum(dim=-1)  # shape: (n_examples,)

        # # Probability of the correct answer (by index into valid answers)
        # correct_indices = [
        #     answer_token_ids[ex.original_example["answer"]] for ex in batch.example_info
        # ]
        # correct_indices_tensor = torch.tensor(correct_indices, device=probs.device)
        # correct_answer_probs = valid_logits.gather(
        #     1, correct_indices_tensor.unsqueeze(-1)
        # ).squeeze(-1)

        # # Top logit answer restricted to valid answers
        # force_mc_indices = valid_logits.argmax(-1)  # index into valid answers
        # force_mc_answers = [runner.valid_answers[i.item()] for i in force_mc_indices]
        total += batch.num_dataset_rows
        num_correct += sum(
            top_logit_answers[i] == runner.valid_answers[ex.original_example["answer"]]
            for i, ex in enumerate(batch.example_info)
        )
        progress.update(batch.num_dataset_rows)
        progress.set_postfix(
            {
                "subjects": set(
                    ex.original_example["subject"] for ex in batch.example_info
                )
            }
        )
    progress.close()

    print(f"Overall accuracy: {num_correct / total}")
