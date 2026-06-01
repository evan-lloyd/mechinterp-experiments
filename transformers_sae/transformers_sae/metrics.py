from functools import partial
from typing import Callable, Literal, Protocol, cast, overload

import numpy as np
import torch
from bitarray import bitarray

from .activation_data import DataBatch
from .ops import tensor_to_numpy

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
            return tensor_to_numpy(fn(actual, target)[batch.token_mask].flatten().cpu())
        else:
            result = (fn(actual, target)[batch.token_mask]).sum() / batch.num_tokens
            if return_type == "tensor":
                return result
            return result.item()

    return _inner


class FeatureSpaceMetric(Protocol):
    @overload
    def __call__(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: Literal["tensor"] = ...,
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: Literal["np"],
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: Literal["float"],
    ) -> float: ...

    @overload
    def __call__(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: _ReturnType,
    ) -> torch.Tensor | np.ndarray | float: ...

    def __call__(
        self,
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: _ReturnType = "tensor",
    ) -> torch.Tensor | np.ndarray | float: ...


def _feature_space_mean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> FeatureSpaceMetric:
    """Decorator to return the batch mean of the given function. Equivalent to _batch_mean except we assume
    that the token mask has already been applied, which will happen when the inputs are in feature space, since
    we mask the input to our SAEs.
    """

    def _inner(
        actual: torch.Tensor,
        target: torch.Tensor,
        return_type: _ReturnType = "tensor",
    ) -> torch.Tensor | np.ndarray | float:
        if return_type == "np":
            return tensor_to_numpy(fn(actual, target).flatten().cpu())
        else:
            result = fn(actual, target).mean()
            if return_type == "tensor":
                return result
            return result.item()

    return cast(FeatureSpaceMetric, _inner)


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
    """Geometric mean loss over a batch of tokens."""

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
            return tensor_to_numpy(fn(actual, target)[batch.token_mask].flatten().cpu())
        else:
            result = fn(actual, target)[batch.token_mask]
            result = result.clip(min=1e-9).log().mean().exp()
            if return_type == "tensor":
                return result
            return result.item()

    return _inner


@overload
def _batch_tmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, float],
    torch.Tensor,
]: ...


@overload
def _batch_tmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, float, Literal["tensor"]],
    torch.Tensor,
]: ...


@overload
def _batch_tmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, float, Literal["float"]],
    float,
]: ...


def _batch_tmean(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[
    [torch.Tensor, torch.Tensor, DataBatch, float, _ReturnType],
    torch.Tensor | np.ndarray | float,
]:
    """Trimmed mean loss over a batch of tokens"""

    def _inner(
        actual: torch.Tensor,
        target: torch.Tensor,
        batch: DataBatch,
        trim_fraction: float,
        return_type: _ReturnType = "tensor",
    ) -> torch.Tensor | np.ndarray | float:
        result = fn(actual, target)[batch.token_mask].flatten()
        if return_type == "np":
            top_k = result.topk(int(trim_fraction * batch.num_tokens), sorted=False)
            result[top_k.indices] = 0.0
            return tensor_to_numpy(result.cpu())

        top_k = result.topk(
            int((1.0 - trim_fraction) * batch.num_tokens), sorted=False, largest=False
        )
        result = result[top_k.indices].mean()
        if return_type == "tensor":
            return result
        return result.item()

    return _inner


def cauchy_loss(c: float | torch.Tensor):

    @_batch_mean
    def _inner(actual: torch.Tensor, target: torch.Tensor):
        return (0.5 * ((actual - target) / c) ** 2 + 1).log().mean(dim=-1)

    return _inner


@_feature_space_mean
def cos_dist_loss(actual: torch.Tensor, target: torch.Tensor):
    return 1 - torch.nn.functional.cosine_similarity(actual, target, dim=-1)


def _mse(actual: torch.Tensor, target: torch.Tensor):
    return ((actual - target) ** 2).mean(dim=-1)


mse_loss = _batch_mean(_mse)
feature_mse_loss = _feature_space_mean(_mse)


@_batch_tmean
def tmse_loss(actual: torch.Tensor, target: torch.Tensor):
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
        result = actual.sum(dim=-1)[batch.token_mask]
    else:
        result = torch.nn.functional.kl_div(
            actual,
            target,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)[batch.token_mask]

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


def _target_only(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def _inner(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fn(target)

    return _inner


@_feature_space_mean
def _l0(_, features: torch.Tensor):
    return (features > 0).to(torch.float32).sum(dim=-1)


l0_eval = partial(_l0, torch.empty((0,)))


# Not using decorator, since aggregation logic is different.
def live_features_eval(
    features: torch.Tensor,
    return_type: _ReturnType = "float",
):
    # TODO: handle this more cleanly, but for now detect if we should mock the result for profiling
    if isinstance(features, torch._subclasses.FakeTensor):
        if return_type == "float":
            return 1.0
        else:
            return bitarray([1] * features.shape[1])
    result = bitarray((features.squeeze(0).sum(dim=0) > 0).tolist())
    if return_type == "float":
        return sum(result) / features.shape[-1]
    return result
