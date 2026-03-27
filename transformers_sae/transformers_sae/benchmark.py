from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import List

import pandas as pd
import torch
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu import mmlu as mmlu_python_module
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.dataset import Golden
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.scorer.scorer import Scorer
from tqdm.auto import tqdm as tqdm_auto

from .ops import generate


@dataclass
class AnswerProbs:
    top_logit_answer: str
    multiple_choice_probs: torch.Tensor


@dataclass
class ScoreWrapper:
    top_logit_answer_is_correct: bool
    any_multiple_choice_answer_prob: float
    correct_answer_prob: float
    conditional_correct_answer_prob: float

    def __bool__(self):
        return self.top_logit_answer_is_correct


@dataclass
class AnswerWrapper:
    answer: str


class BenchmarkModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.answer_token_ids = (
            self.tokenizer([" A B C D"], return_tensors="pt")
            .input_ids[0, 1:]
            .to(model.device)
        )
        self.special_ids = torch.tensor(tokenizer.all_special_ids).to(model.device)

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
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                self.model.device
            )

            special_token_indices = (
                (inputs.input_ids.view(-1).unsqueeze(-1) == self.special_ids)
                .any(dim=-1)
                .nonzero()
            ).squeeze(-1)
            token_mask = torch.ones_like(inputs.input_ids)
            probs = self.model(
                **inputs,
                use_cache=False,
                token_mask=token_mask,
                pass_through_positions=special_token_indices,
            )[0][:, -1, :].softmax(-1)
            answer_probs = []
            top_logit_answers = probs.max(-1).indices
            mc_probs = probs[:, self.answer_token_ids]
            for i in range(probs.shape[0]):
                answer_probs.append(
                    AnswerProbs(
                        self.tokenizer.decode(top_logit_answers[i].item()),
                        mc_probs[i].to("cpu"),
                    )
                )

            return [AnswerWrapper(ap) for ap in answer_probs]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


class BenchmarkScorer(Scorer):
    @classmethod
    def exact_match_score(cls, target: str, prediction: AnswerProbs) -> ScoreWrapper:
        """Overriding this function so we can compute some additional metrics based on logits."""
        total_mc_probs = prediction.multiple_choice_probs.sum().item()
        correct_answer_prob = prediction.multiple_choice_probs[
            ord(target) - ord("A")
        ].item()
        return ScoreWrapper(
            target.strip() == prediction.top_logit_answer.strip(),
            total_mc_probs,
            correct_answer_prob,
            correct_answer_prob / (total_mc_probs + 1e-9),
        )


class MMLUBenchmark(MMLU):
    def __init__(self, tokenizer, max_context, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.scorer = BenchmarkScorer()

    def evaluate(self, *args, **kwargs):
        mmlu_python_module.tqdm = tqdm_auto
        self.filter_tasks()
        result = super().evaluate(*args, **kwargs)
        self.load_benchmark_dataset.cache_clear()

        self.answer_stats = pd.DataFrame(
            [
                (
                    sw.top_logit_answer_is_correct,
                    sw.any_multiple_choice_answer_prob,
                    sw.correct_answer_prob,
                    sw.conditional_correct_answer_prob,
                )
                for sw in self.predictions["Correct"]
            ],
            columns=[
                "top_logit",
                "any_mc_answer",
                "correct_answer",
                "conditional_correct_answer",
            ],
        )
        print(
            "Mean probability for any multiple choice answer: ",
            self.answer_stats["any_mc_answer"].mean().item(),
        )
        print(
            "Mean probability for correct answer: ",
            self.answer_stats["correct_answer"].mean().item(),
        )
        print(
            "Mean probability for correct answer, conditional on giving a multiple choice answer: ",
            self.answer_stats["conditional_correct_answer"].mean().item(),
        )
        print(
            "Distribution of top logit answers: ",
            Counter([p.top_logit_answer for p in self.predictions["Prediction"]]),
        )
        print(
            "Distribution of correct answers: ",
            Counter(self.predictions["Expected Output"]),
        )

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
