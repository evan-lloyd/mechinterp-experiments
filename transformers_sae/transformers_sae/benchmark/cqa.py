from typing import Any

from datasets import load_dataset

from .benchmark_runner import BenchmarkRunner, DatasetInfo


class CQA(BenchmarkRunner):
    @property
    def valid_answers(self) -> list[str]:
        return [" A", " B", " C", " D", " E"]

    def get_answer_index(self, example: dict[str, Any]) -> int:
        return self.valid_answers.index(" " + example["answerKey"])

    def get_subset(self, example: dict[str, Any]) -> str:
        return "all"

    def format_example(
        self,
        example: dict[str, Any],
        permutation_offset: int = 0,
        with_answer: bool = False,
    ) -> str:
        prompt = "Question: " + example["question"] + "\n"
        for i, a in enumerate(self.valid_answers):
            prompt += f"\n{a.strip()}. {example['choices']['text'][(i + permutation_offset) % self.num_valid_answers]}"
        prompt += "\nAnswer:"
        if with_answer:
            prompt += self.valid_answers[
                (self.get_answer_index(example) + permutation_offset)
                % self.num_valid_answers
            ]
        return prompt

    def prepare_dataset(self) -> list[DatasetInfo]:
        base_dataset = load_dataset(
            "tau/commonsense_qa",
            streaming=True,
        )["train"]
        num_examples = base_dataset.info.splits["train"].num_examples - self.n_shots
        shot_dataset = base_dataset.take(self.n_shots)
        preamble = (
            "\n\n".join(
                self.format_example(e, with_answer=True) for e in shot_dataset.to_list()
            )
            + "\n\n"
        )
        dataset = base_dataset.skip(self.n_shots)
        return [
            DatasetInfo(
                dataset,
                preamble,
                num_examples,
            )
        ]
