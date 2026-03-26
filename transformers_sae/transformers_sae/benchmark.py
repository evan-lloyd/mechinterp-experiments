from dataclasses import dataclass
from typing import List

import torch
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu import mmlu as mmlu_python_module
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.models.base_model import DeepEvalBaseLLM
from tqdm.auto import tqdm as tqdm_auto

from .ops import generate


@dataclass
class AnswerWrapper:
    answer: str


class BenchmarkModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def get_model_name(self):
        return "BenchmarkModelWrapper"

    def generate(self, prompt: str, schema) -> str:
        return self.batch_generate([prompt], [schema])[0]

    @torch.inference_mode()
    def batch_generate(self, prompts: List[str], schemas) -> List[str]:
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
    def __init__(
        self,
        num_examples_per_task: int | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_examples_per_task = num_examples_per_task

    def load_benchmark_dataset(self, task: MMLUTask):
        return super().load_benchmark_dataset(task)[slice(self.num_examples_per_task)]

    def evaluate(self, *args, **kwargs):
        mmlu_python_module.tqdm = tqdm_auto
        return super().evaluate(*args, **kwargs)
