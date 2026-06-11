from functools import partial
from typing import Any

from datasets import IterableDataset, load_dataset

from .benchmark_runner import BenchmarkRunner


class ArcE(BenchmarkRunner):
    @property
    def valid_answers(self) -> list[str]:
        return [" A", " B", " C", " D", " E"]

    def get_answer_index(self, example: dict[str, Any]) -> int:
        try:
            index = self.valid_answers.index(" " + example["answerKey"])
        except ValueError:
            # Some answers are labeled "1", "2", etc
            index = int(example["answerKey"]) - 1
        return index

    def get_subset(self, example: dict[str, Any]) -> str:
        return "all"

    def format_dataset(self, preamble: str, example: dict[str, Any]):
        example["text"] = preamble + "\n\n" + self.format_example(example)
        return example

    def format_example(self, example: dict[str, Any], with_answer: bool = False) -> str:
        prompt = "Question: " + example["question"] + "\n"
        for i, a in enumerate(self.valid_answers):
            if i >= len(example["choices"]["text"]):
                break
            prompt += f"\n{a.strip()}. {example['choices']['text'][i]}"
        prompt += "\nAnswer:"
        if with_answer:
            prompt += self.valid_answers[self.get_answer_index(example)]
        return prompt

    def prepare_dataset(self) -> IterableDataset:
        base_dataset = load_dataset(
            "allenai/ai2_arc",
            "ARC-Easy",
            streaming=True,
        )["test"]
        self.num_examples = base_dataset.info.splits["test"].num_examples - self.n_shots
        preamble = ""
        dataset = base_dataset.skip(self.n_shots)
        return dataset.map(partial(self.format_dataset, preamble))
