from functools import partial
from typing import Callable, Literal, overload

import numpy as np
import torch
from bitarray import bitarray

from .activation_data import DataBatch
from .ops import tensor_to_numpy

"""NB: The losses here are deliberately taking a mean, rather than sum, on the final dimension, to factor out
the implicit dependence on d_model (mse) or d_vocab (kl). This is probably non-standard, but is effectively
just an arbitrary scaling factor that shouldn't affect anything other than the specific value of the optimal
learning rate.
"""

_ReturnType = Literal["float", "tensor", "np"]


@overload
def _batch_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch],
    torch.Tensor,
]: ...


@overload
def _batch_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, Literal["tensor"]],
    torch.Tensor,
]: ...


@overload
def _batch_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, Literal["float"]],
    float,
]: ...


@overload
def _batch_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, Literal["np"]],
    np.ndarray,
]: ...


def _batch_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, _ReturnType],
    torch.Tensor | np.ndarray | float,
]:
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
            return tensor_to_numpy(
                fn(actual, target)[batch.token_mask.bool()].flatten().cpu()
            )
        else:
            result = (fn(actual, target) * batch.token_mask).sum() / batch.num_tokens
            if return_type == "tensor":
                return result
            return result.item()

    return _inner


@overload
def _batch_gmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch],
    torch.Tensor,
]: ...


@overload
def _batch_gmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, Literal["tensor"]],
    torch.Tensor,
]: ...


@overload
def _batch_gmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, Literal["float"]],
    float,
]: ...


def _batch_gmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, _ReturnType],
    torch.Tensor | np.ndarray | float,
]:
    def _inner(
        actual: torch.Tensor,
        target: torch.Tensor,
        batch: DataBatch,
        return_type: _ReturnType = "tensor",
    ) -> torch.Tensor | np.ndarray | float:
        """If aggregating, return a tensor or float which is the geometric mean over non-masked tokens. Otherwise, convert to
        a numpy array of the raw values returned by the wrapped function, for only the non-masked tokens.
        """
        if return_type == "np":
            return tensor_to_numpy(
                fn(actual, target)[batch.token_mask.bool()].flatten().cpu()
            )
        else:
            result = fn(actual, target)[batch.token_mask.bool()]
            result = result.clip(min=1e-9).log().mean().exp()
            if return_type == "tensor":
                return result
            return result.item()

    return _inner


@_batch_mean
def cos_dist_loss(actual: torch.Tensor, target: torch.Tensor):
    return 1 - torch.nn.functional.cosine_similarity(actual, target, dim=-1)


@_batch_mean
def mse_loss(actual: torch.Tensor, target: torch.Tensor):
    return ((actual - target) ** 2).mean(dim=-1)


@_batch_gmean
def gmse_loss(actual: torch.Tensor, target: torch.Tensor):
    return ((actual - target) ** 2).sum(dim=-1)


@_batch_mean
def l1_loss(actual: torch.Tensor, target: torch.Tensor):
    return (actual - target).abs().mean(dim=-1)


# Not using decorator, since we want the geometric mean.
def kl_loss(
    actual: torch.Tensor,
    target: torch.Tensor,
    batch: DataBatch,
    return_type: _ReturnType = "tensor",
    overwrite_inputs: bool = False,
) -> torch.Tensor | np.ndarray | float:
    # Save ~50% memory by overwriting our input tensors. This is of course potentially
    # risky, but we don't always need to save log_probs after this, and this is by far
    # the most memory-expensive operation.
    # from torch source (aten/src/ATen/native/Loss.cpp): output = at::exp(target) * (target - input);
    # TODO: it seems like we should be able to write a more memory-efficient version of this with
    # a custom backward function that lets us do some operations in-place.
    if overwrite_inputs:
        torch.subtract(target, actual, out=actual)
        target.exp_()
        torch.mul(target, actual, out=actual)
        result = actual.sum(dim=-1)[batch.token_mask.bool()]
    else:
        result = torch.nn.functional.kl_div(
            actual,
            target,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)[batch.token_mask.bool()]

    if return_type == "np":
        return tensor_to_numpy(result.flatten().cpu())
    else:
        result = result.clip(min=1e-9).log().mean().exp()
        if return_type == "tensor":
            return result
        return result.item()


kl_eval = partial(kl_loss, overwrite_inputs=True)


@_batch_mean
def rre_eval(actual: torch.Tensor, target: torch.Tensor):
    return torch.linalg.vector_norm(actual - target, dim=-1, dtype=torch.float32) / (
        torch.linalg.vector_norm(target, dim=-1, dtype=torch.float32) + 1e-8
    )


@_batch_mean
def l0_eval(features: torch.Tensor, _):
    return (features > 0).to(torch.float32).sum(dim=-1)


# Not using decorator, since aggregation logic is different.
def live_features_eval(
    features: torch.Tensor,
    _: torch.Tensor,
    batch: DataBatch,
    return_type: _ReturnType = "float",
):
    # TODO: handle this more cleanly, but for now detect if we should mock the result for profiling
    if isinstance(features, torch._subclasses.FakeTensor):
        if return_type == "float":
            return 1.0
        else:
            return bitarray([1] * features.shape[1])
    result = bitarray((features[batch.token_mask.bool()].sum(dim=0) > 0).tolist())
    if return_type == "float":
        return sum(result) / features.shape[-1]
    return result
