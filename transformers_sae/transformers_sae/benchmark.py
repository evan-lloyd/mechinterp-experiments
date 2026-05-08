from deepeval.benchmarks.bool_q.template import BoolQTemplate
from deepeval.benchmarks.schema import AffirmationSchema
from collections import Counter
from dataclasses import dataclass
from functools import cache
from typing import List

import pandas as pd
import torch
from deepeval.benchmarks import MMLU, BoolQ
from deepeval.benchmarks.mmlu import mmlu as mmlu_python_module
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.tasks import MMLUTask
from deepeval.dataset import Golden
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.scorer.scorer import Scorer
from tqdm.auto import tqdm as tqdm_auto


class AnswerProbs(str):
    """Derive from string so we can still pass around extra info even though DeepEval will be calling str()
    on this object."""

    top_logit_answer: str
    multiple_choice_probs: torch.Tensor
    force_mc_answer: str

    def __new__(cls, value, **answer_data):
        instance = super().__new__(cls, value)
        instance.top_logit_answer = value
        for k, v in answer_data.items():
            setattr(instance, k, v)
        return instance

    def __str__(self):
        return self


@dataclass
class ScoreWrapper:
    top_logit_answer_is_correct: bool
    any_multiple_choice_answer_prob: float
    correct_answer_prob: float
    conditional_correct_answer_prob: float
    force_mc_answer_is_correct: bool

    def __bool__(self):
        return self.top_logit_answer_is_correct


@dataclass
class AnswerWrapper:
    answer: AnswerProbs


class BenchmarkModel(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, is_multiple_choice: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        if is_multiple_choice:
            self.answer_token_ids = (
                self.tokenizer([" A B C D"], return_tensors="pt")
                .input_ids[0, 1:]
                .to(model.device)
            )
        else:
            self.answer_token_ids = (
                self.tokenizer(["No", "Yes"], return_tensors="pt")
                .input_ids[:, 1]
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
            inputs = self.tokenizer(
                prompts,
                # [prefix + prompt for prefix, prompt in zip(prefixes, prompts)],
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

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
            force_mc_answers = probs[:, self.answer_token_ids].max(-1).indices
            for i in range(probs.shape[0]):
                answer_probs.append(
                    AnswerProbs(
                        self.tokenizer.decode(top_logit_answers[i].item()),
                        multiple_choice_probs=mc_probs[i].to("cpu"),
                        force_mc_answer=self.tokenizer.decode(
                            self.answer_token_ids[force_mc_answers[i].item()]
                        ),
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
        if target in ("A", "B", "C", "D"):
            correct_answer_prob = prediction.multiple_choice_probs[
                ord(target) - ord("A")
            ].item()
        else:
            correct_answer_prob = prediction.multiple_choice_probs[int(target == "Yes")]
        return ScoreWrapper(
            target.strip() == prediction.top_logit_answer.strip(),
            total_mc_probs,
            correct_answer_prob,
            correct_answer_prob / (total_mc_probs + 1e-9),
            target.strip() == prediction.force_mc_answer.strip(),
        )


class BoolQBenchmark(BoolQ):
    def __init__(self, tokenizer, max_context, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_context = max_context
        self.scorer = BenchmarkScorer()

    def load_benchmark_dataset(self) -> List[Golden]:
        from datasets import load_dataset

        # Load dataset
        if self.dataset:
            dataset = self.dataset
        else:
            dataset = load_dataset("boolq", "default")
            self.dataset = dataset

        # Construct test set
        goldens: List[Golden] = []
        for data in dataset["validation"]:
            # TODO: if we have to override their template to get good performance, why even use DeepEval?
            # Adapted from "FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS"
            input = f"""{data["passage"]}

Can we conclude that {data["question"]}? Yes or no?

Answer:
"""
            expected_output = BoolQTemplate.format_answer(data)
            golden = Golden(input=input, expected_output=expected_output)
            goldens.append(golden)

        return goldens

    def evaluate(self, *args, **kwargs):
        mmlu_python_module.tqdm = tqdm_auto
        result = super().evaluate(*args, **kwargs)

        self.answer_stats = pd.DataFrame(
            [
                (
                    sw.top_logit_answer_is_correct,
                    sw.any_multiple_choice_answer_prob,
                    sw.correct_answer_prob,
                    sw.conditional_correct_answer_prob,
                    sw.force_mc_answer_is_correct,
                )
                for sw in self.predictions["Correct"]
            ],
            columns=[
                "top_logit",
                "any_mc_answer",
                "correct_answer",
                "conditional_correct_answer",
                "force_mc",
            ],
        )
        print(
            "Accuracy for top logit among valid answers: ",
            self.answer_stats["force_mc"].mean().item(),
        )
        print(
            "Mean probability for any Yes/No answer: ",
            self.answer_stats["any_mc_answer"].mean().item(),
        )
        print(
            "Mean probability for correct answer: ",
            self.answer_stats["correct_answer"].mean().item(),
        )
        print(
            "Mean probability for correct answer, conditional on giving a Yes/No answer: ",
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
                    sw.force_mc_answer_is_correct,
                )
                for sw in self.predictions["Correct"]
            ],
            columns=[
                "top_logit",
                "any_mc_answer",
                "correct_answer",
                "conditional_correct_answer",
                "force_mc",
            ],
        )
        print(
            "Accuracy for top logit among valid answers: ",
            self.answer_stats["force_mc"].mean().item(),
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
