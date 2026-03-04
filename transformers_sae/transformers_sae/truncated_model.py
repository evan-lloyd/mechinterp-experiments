from contextlib import contextmanager

import torch

from .ops import ensure_tensor
from .replacement_model import ReplacementModel


@contextmanager
def truncated_model(
    model: ReplacementModel, start_layer: int, end_layer: int, start_at_sae: bool
):
    """Modifies a transformer model in place for the duration of the context manager,
    such that only the layers between start_layer and end_layer are executed.
    """

    if (
        start_layer == -1
        or start_layer >= model.num_layers
        or start_layer not in model.sae_layers
    ) and start_at_sae:
        raise ValueError(
            f"Can't start_at_sae for layer {start_layer}, which has no SAE"
        )

    class _EarlyStopping(Exception):
        pass

    class _StopLayer(torch.nn.Module):
        def __init__(self, orig_layer: torch.nn.Module):
            # Some implementations (gemma2) look at attributes on layers before calling them,
            # so we need to proxy those
            super().__init__()
            object.__setattr__(self, "orig_layer", orig_layer)

        def __getattr__(self, name: str):
            if name != "forward":
                return getattr(object.__getattribute__(self, "orig_layer"), name)
            return super().__getattr__(name)

        def forward(self, *args, **kwargs):
            raise _EarlyStopping()

    class _StartAtSAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sae = model.get_layer(start_layer).sae

        def __getattr__(self, name: str):
            if name not in ("forward", "sae"):
                return getattr(model.get_layer(start_layer), name)
            return super().__getattr__(name)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            result = self.sae(x.float(), should_cast=False)
            return (result.to(x.dtype),)

    patched_layers = torch.nn.ModuleList(
        model.get_submodule(model.layer_path)[max(start_layer, 0) : end_layer]
        + (
            [_StopLayer(model.get_layer(end_layer))]
            if end_layer < model.num_layers
            else []
        )
    )
    if start_at_sae:
        patched_layers[0] = _StartAtSAE()

    class _TruncatedTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = patched_layers

        def forward(self, x: torch.Tensor, *args, **kwargs):
            residual = x
            for i, layer in enumerate(self.layers):
                layer_args, layer_kwargs = model.get_layer_args(
                    i + max(start_layer, 0), layer, *args, **kwargs
                )
                residual = ensure_tensor(layer(residual, *layer_args, **layer_kwargs))
            if end_layer > model.num_layers:
                residual = model.get_logits(residual)
            return (residual,)

    try:
        orig_layers = model.get_submodule(model.layer_path)

        # If starting from embedding, we only need to handle early stopping
        if start_layer == -1:
            model.set_submodule(model.layer_path, patched_layers)
            truncated_model = model
        else:
            truncated_model = _TruncatedTransformer()
        try:
            yield truncated_model
        except _EarlyStopping:
            pass
    finally:
        model.set_submodule(model.layer_path, orig_layers)
