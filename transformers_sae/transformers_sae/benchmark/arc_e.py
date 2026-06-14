from typing import Any

from datasets import load_dataset

from .benchmark_runner import BenchmarkRunner, DatasetInfo


class ArcE(BenchmarkRunner):
    @property
    def valid_answers(self) -> list[str]:
        # return [" A", " B", " C", " D", " E"]
        return [" A", " B", " C", " D"]

    def get_answer_index(self, example: dict[str, Any]) -> int:
        try:
            index = self.valid_answers.index(" " + example["answerKey"])
        except ValueError:
            # Some answers are labeled "1", "2", etc
            index = int(example["answerKey"]) - 1
        return index

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
            if i >= len(example["choices"]["text"]):
                break
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
            "allenai/ai2_arc",
            "ARC-Easy",
            streaming=True,
        )["test"]
        num_examples = base_dataset.info.splits["test"].num_examples - self.n_shots
        shot_dataset = base_dataset.take(self.n_shots)
        preamble = (
            "\n\n".join(
                self.format_example(e, with_answer=True) for e in shot_dataset.to_list()
            )
            + "\n\n"
        )
        dataset = base_dataset.skip(self.n_shots)
        # Arc-E has a small number of entries with a variable number of possible answers, which would throw
        # off "valid answer" probability calculation. There's not many, so just discard them.
        return [
            DatasetInfo(
                dataset.filter(lambda ex: len(ex["choices"]["label"]) == 4),
                preamble,
                num_examples,
            )
        ]
