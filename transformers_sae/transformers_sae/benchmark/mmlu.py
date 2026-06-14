from typing import Any

from datasets import load_dataset

from .benchmark_runner import BenchmarkRunner, DatasetInfo


class MMLU(BenchmarkRunner):
    @property
    def valid_answers(self):
        return [" A", " B", " C", " D"]

    def get_answer_index(self, example: dict[str, Any]) -> int:
        return example["answer"]

    def get_subset(self, example: dict[str, Any]) -> str:
        return example["subject"]

    def format_example(
        self,
        example: dict[str, Any],
        permutation_offset: int = 0,
        with_answer: bool = False,
    ) -> str:
        prompt = "Question: " + example["question"] + "\n"
        for i, a in enumerate(self.valid_answers):
            prompt += f"\n{a.strip()}. {example['choices'][(i + permutation_offset) % self.num_valid_answers]}"
        prompt += "\nAnswer:"
        if with_answer:
            prompt += self.valid_answers[
                (example["answer"] + permutation_offset) % self.num_valid_answers
            ]
        return prompt

    def prepare_dataset(self) -> list[DatasetInfo]:
        datasets = []
        for subject in self.subsets:
            base_dataset = load_dataset(
                "cais/mmlu",
                subject,
                streaming=True,
            )["test"]
            dev_dataset = load_dataset("cais/mmlu", subject, streaming=False)["dev"]
            preamble = (
                "The following are multiple choice questions (with answers)"
                f" about {subject.replace('_', ' ')}.\n\n"
            )
            preamble += (
                "\n\n".join(
                    self.format_example(e, with_answer=True)
                    for e in dev_dataset.to_list()
                )
                + "\n\n"
            )
            datasets.append(
                DatasetInfo(
                    base_dataset,
                    preamble,
                    base_dataset.info.splits["test"].num_examples,
                )
            )

        return datasets
