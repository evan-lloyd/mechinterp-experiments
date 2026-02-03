from typing import Callable, Literal

import numpy as np
import torch

from .activation_data import ActivationBatch
from .sae_data import DataBatch

"""NB: The losses here are deliberately taking a mean, rather than sum, on the final dimension, to factor out
the implicit dependence on d_model (mse) or d_vocab (kl). This is probably non-standard, but is effectively
just an arbitrary scaling factor that shouldn't affect anything other than the specific value of the optimal
learning rate.
"""

_ReturnType = Literal["float", "tensor", "np"]


def _handle_batch(fn):
    def _inner(
        actual: torch.Tensor,
        target: torch.Tensor,
        batch: DataBatch,
        return_type: _ReturnType = "tensor",
    ) -> torch.Tensor | np.ndarray | float:
        """If aggregating, return a tensor or float which is the mean over non-masked tokens. Otherwise, convert to
        a numpy array of the raw values returned by the wrapped function, for only the non-masked tokens.
        """
        if return_type == "np":
            return fn(actual, target)[batch.token_mask.bool()].flatten().cpu().numpy()
        else:
            result = (fn(actual, target) * batch.token_mask).sum() / batch.num_tokens
            if return_type == "tensor":
                return result
            return result.item()

    return _inner


@_handle_batch
def cos_dist_loss(actual: torch.Tensor, target: torch.Tensor):
    return 1 - torch.nn.functional.cosine_similarity(actual, target, dim=-1)


@_handle_batch
def mse_loss(actual: torch.Tensor, target: torch.Tensor):
    return ((actual - target) ** 2).mean(dim=-1)


@_handle_batch
def kl_loss(actual: torch.Tensor, target: torch.Tensor):
    return torch.nn.KLDivLoss(reduction="none", log_target=True)(
        actual.log_softmax(-1),
        target.log_softmax(-1),
    ).mean(dim=-1)


@_handle_batch
def kl_eval(actual: torch.Tensor, target: torch.Tensor):
    return torch.nn.KLDivLoss(reduction="none", log_target=True)(
        actual.log_softmax(-1),
        target.log_softmax(-1),
    ).sum(dim=-1)


@_handle_batch
def rre_eval(actual: torch.Tensor, target: torch.Tensor):
    return torch.linalg.vector_norm(actual - target, dim=-1) / (
        torch.linalg.vector_norm(target, dim=-1) + 1e-8
    )


@_handle_batch
def l0_eval(features: torch.Tensor, _):
    return (features > 0).to(torch.float32).sum(dim=-1)


def downstream_loss(
    replacement: ActivationBatch,
    baseline: ActivationBatch,
    batch: DataBatch,
    downstream_attr: str,
    downstream_fn: Callable[[torch.Tensor, torch.Tensor, DataBatch], torch.Tensor],
) -> torch.Tensor:
    return downstream_fn(
        getattr(replacement, downstream_attr),
        getattr(baseline, downstream_attr),
        batch,
    )
