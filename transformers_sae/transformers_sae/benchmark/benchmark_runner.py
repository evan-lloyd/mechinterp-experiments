from copy import copy
from typing import Any

from datasets import IterableDataset


class BenchmarkRunner:
    dataset: IterableDataset
    subsets: list[str]
    max_context: int
    valid_answers: list[str]
    num_examples: int

    def __init__(self, n_shots: int, subsets: list[str], max_context: int):
        self.n_shots = n_shots
        self.subsets = copy(subsets)
        self.max_context = max_context

        self.dataset = self.prepare_dataset()

    def get_answer_index(self, example: dict[str, Any]) -> int:
        raise NotImplementedError("Benchmark runner must define get_answer")

    def get_subset(self, example: dict[str, Any]) -> str:
        raise NotImplementedError("Benchmark runner must define get_subset")

    def prepare_dataset(self) -> IterableDataset:
        raise NotImplementedError("Benchmark runner must define prepare_dataset")
