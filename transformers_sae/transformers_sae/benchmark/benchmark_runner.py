from copy import copy
from dataclasses import dataclass
from functools import partial, cached_property
from typing import Any

from datasets import IterableDataset, concatenate_datasets


@dataclass
class DatasetInfo:
    dataset: IterableDataset
    preamble: str
    num_examples: int


class BenchmarkRunner:
    dataset: IterableDataset
    subsets: list[str]
    max_context: int
    valid_answers: list[str]
    num_examples: int
    num_debiasing_samples: int
    debiasing_sample_fraction: float

    @cached_property
    def num_valid_answers(self):
        return len(self.valid_answers)

    def __init__(
        self,
        *,
        n_shots: int,
        subsets: list[str],
        max_context: int,
        debiasing_sample_fraction: float,
        max_samples: int | None,
    ):
        self.n_shots = n_shots
        self.subsets = copy(subsets)
        self.max_context = max_context
        self.debiasing_sample_fraction = debiasing_sample_fraction

        datasets = self.prepare_dataset()
        self.num_examples = sum([d.num_examples for d in datasets])
        if max_samples is not None:
            self.num_examples = min(self.num_examples, max_samples)
        self.num_debiasing_samples = int(
            self.debiasing_sample_fraction * self.num_examples
        )
        debiasing_datasets = []
        if self.num_debiasing_samples > 0:
            debiasing_samples_gathered = 0
            for dataset in datasets:
                if debiasing_samples_gathered >= self.num_debiasing_samples:
                    break
                num_samples = min(
                    dataset.num_examples,
                    self.num_debiasing_samples - debiasing_samples_gathered,
                )
                debiasing_datasets.append(
                    DatasetInfo(
                        dataset.dataset.take(num_samples).map(
                            partial(
                                self.add_label_permutations, debiasing_samples_gathered
                            ),
                            batched=True,
                            batch_size=num_samples,
                        ),
                        dataset.preamble,
                        num_samples,
                    )
                )
                debiasing_samples_gathered += dataset.num_examples
            self.debiasing_dataset = concatenate_datasets(
                [
                    d.dataset.map(partial(self.format_dataset, d.preamble))
                    for d in debiasing_datasets
                ]
            )
        else:
            self.debiasing_dataset = None

        self.dataset = concatenate_datasets(
            [d.dataset.map(partial(self.format_dataset, d.preamble)) for d in datasets]
        )

        if max_samples is not None:
            self.dataset = self.dataset.take(max_samples)

    def add_label_permutations(
        self, index_offset: int, examples: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        # Clone each example so we have a copy with each cyclic permutation of valid answers
        num_examples = 0
        for k, v in examples.items():
            num_examples = len(v)
            examples[k] = examples[k] * self.num_valid_answers
        examples["_benchmark_runner_permutation_offset"] = [
            i for i in range(self.num_valid_answers) for _ in range(num_examples)
        ]

        examples["_benchmark_runner_example_index"] = (
            list(range(index_offset, index_offset + num_examples))
            * self.num_valid_answers
        )
        return examples

    def format_dataset(self, preamble: str, example: dict[str, Any]) -> dict[str, Any]:
        example["text"] = preamble + self.format_example(
            example,
            example.get("_benchmark_runner_permutation_offset", 0),
            False,
        )
        return example

    def format_example(
        self,
        example: dict[str, Any],
        permutation_offset: int = 0,
        with_answer: bool = False,
    ) -> str:
        raise NotImplementedError("Benchmark runner must define format_example")

    def get_answer_index(self, example: dict[str, Any]) -> int:
        raise NotImplementedError("Benchmark runner must define get_answer")

    def get_subset(self, example: dict[str, Any]) -> str:
        raise NotImplementedError("Benchmark runner must define get_subset")

    def prepare_dataset(self) -> list[DatasetInfo]:
        raise NotImplementedError("Benchmark runner must define prepare_dataset")
