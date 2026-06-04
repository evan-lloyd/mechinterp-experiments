from functools import partial
from typing import Any

from datasets import IterableDataset, concatenate_datasets, load_dataset

from .benchmark_runner import BenchmarkRunner


class MMLU(BenchmarkRunner):
    @property
    def valid_answers(self):
        return [" A", " B", " C", " D"]

    def format_example(self, example: dict[str, Any], with_answer: bool) -> str:
        prompt = example["question"]
        for i, a in enumerate(self.valid_answers):
            prompt += f"\n{a.strip()}. {example['choices'][i]}"
        prompt += "\nAnswer:"
        if with_answer:
            prompt += self.valid_answers[example["answer"]] + "\n\n"
        return prompt

    def make_prompt(self, prelude: str, example: dict[str, Any]) -> dict[str, Any]:
        prompt = prelude + "\n\n" + example["question"]
        prompt += self.format_example(example, False)

        example["text"] = prompt
        return example

    def prepare_dataset(self) -> IterableDataset:
        self.num_examples = 0
        datasets = []
        for subject in self.subsets:
            base_dataset = load_dataset(
                "cais/mmlu",
                subject,
                streaming=True,
            )["test"]
            dev_dataset = load_dataset("cais/mmlu", subject, streaming=False)["dev"]
            prelude = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
            for i in range(self.n_shots):
                prelude += self.format_example(dev_dataset[i], True)
            datasets.append(base_dataset.map(partial(self.make_prompt, prelude)))
            self.num_examples += base_dataset.info.splits["test"].num_examples

        return concatenate_datasets(datasets)
