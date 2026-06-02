from datasets import IterableDataset
from copy import copy


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

    def prepare_dataset(self) -> IterableDataset:
        raise NotImplementedError("Benchmark runner must define prepare_dataset")
