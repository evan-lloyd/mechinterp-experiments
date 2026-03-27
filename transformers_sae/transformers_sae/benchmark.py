from dataclasses import dataclass
from functools import cache
from typing import List

import torch
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu import mmlu as mmlu_python_module
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.dataset import Golden
from deepeval.models.base_model import DeepEvalBaseLLM
from tqdm.auto import tqdm as tqdm_auto

from .ops import generate


@dataclass
class AnswerWrapper:
    answer: str


class BenchmarkModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, top_logit_scoring: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.answer_token_ids = (
            self.tokenizer([" A B C D"], return_tensors="pt")
            .input_ids[0, 1:]
            .to(model.device)
        )
        self.special_ids = torch.tensor(tokenizer.all_special_ids).to(model.device)
        self.top_logit_scoring = top_logit_scoring

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "BenchmarkModelWrapper"

    def generate(self, prompt: str, schema) -> str:
        return self.batch_generate([prompt], [schema])[0]

    @torch.inference_mode()
    def batch_generate(self, prompts: List[str], schemas) -> List[str]:
        with torch.autocast(
            device_type="cuda" if self.model.device.type == "cuda" else "cpu",
            dtype=torch.bfloat16,
        ):
            # An easier version of the task, return whichever of the multiple-choice answers has
            # the highest logits. This (putatively) lets us see if there are remaining latent
            # capabilities in models too badly damaged to reliably return a properly formatted
            # response on their own.
            if self.top_logit_scoring:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                    self.model.device
                )

                special_token_indices = (
                    (inputs.input_ids.view(-1).unsqueeze(-1) == self.special_ids)
                    .any(dim=-1)
                    .nonzero()
                ).squeeze(-1)
                token_mask = torch.ones_like(inputs.input_ids)
                logits = self.model(
                    **inputs,
                    use_cache=False,
                    token_mask=token_mask,
                    pass_through_positions=special_token_indices,
                )[0][:, -1, self.answer_token_ids]
                token_to_answer = ("A", "B", "C", "D")
                return [
                    AnswerWrapper(token_to_answer[i])
                    for i in logits.max(dim=-1).indices
                ]

            output_token_ids = generate(
                prompts,
                self.model,
                self.tokenizer,
                stream=False,
                use_cache=True,
                strip_input=True,
                max_new_tokens=1,
            )
            return [
                AnswerWrapper(a.strip())
                for a in self.tokenizer.batch_decode(output_token_ids)
            ]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


class MMLUBenchmark(MMLU):
    def __init__(self, tokenizer, max_context, **kwargs):
        self.tokenizer = tokenizer
        self.max_context = max_context
        super().__init__(**kwargs)

    def evaluate(self, *args, **kwargs):
        mmlu_python_module.tqdm = tqdm_auto
        self.filter_tasks()
        result = super().evaluate(*args, **kwargs)
        self.load_benchmark_dataset.cache_clear()
        return result

    def filter_tasks(self):
        for task in list(self.tasks):
            self.load_benchmark_dataset(task)

    @cache
    def load_benchmark_dataset(self, task: MMLUTask) -> List[Golden]:
        from datasets import load_dataset

        dataset = load_dataset(
            "cais/mmlu",
            task.value,
        )
        self.dataset = dataset

        # Seems like a bug in the parent class; it sets the shots_dataset to
        # whatever the first loaded task was, but they can give very different
        # peformance. So here we instead update it to the current task.
        self.shots_dataset = list(dataset["dev"])

        # Construct test set. Filter out any prompts whose tokenizations
        # exceed the maximum context window.
        goldens = []
        choices = ["A", "B", "C", "D"]
        for data in dataset["test"]:
            input = MMLUTemplate.format_question(data, include_answer=False)
            golden = Golden(input=input, expected_output=choices[data["answer"]])
            prompt = MMLUTemplate.generate_output(
                train_set=self.shots_dataset,
                input=golden.input,
                task=task,
                n_shots=self.n_shots,
            )
            tokenization = self.tokenizer(prompt, return_tensors="pt").input_ids
            if tokenization.shape[1] <= self.max_context:
                goldens.append(golden)
        if len(goldens) == 0:
            self.tasks.remove(task)

        return goldens
