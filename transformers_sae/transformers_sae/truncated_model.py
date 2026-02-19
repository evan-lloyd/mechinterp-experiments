from contextlib import contextmanager

import torch

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
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise _EarlyStopping()

    class _StartAtSAE(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return model.transformer.h[start_layer].sae(x)

    patched_layers = torch.nn.ModuleList(
        model.transformer.h[max(start_layer, 0) : end_layer]
        + ([_StopLayer()] if end_layer < model.num_layers else [])
    )
    if start_at_sae:
        patched_layers[0] = _StartAtSAE()

    class _TruncatedTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, *args, **kwargs):
            residual = x
            for layer in patched_layers:
                residual = layer(
                    residual,
                    attention_mask=kwargs.get("attention_mask"),
                    use_cache=kwargs.get("use_cache"),
                )[0]
            if end_layer > model.num_layers:
                residual = model.transformer.ln_f(residual)
                residual = model.lm_head(
                    residual.view(-1, residual.shape[-2], residual.shape[-1])
                )
            return (residual,)

    try:
        orig_layers = model.transformer.h

        # If starting from embedding, we only need to handle early stopping
        if start_layer == -1:
            model.transformer.h = patched_layers
            truncated_model = model
        else:
            truncated_model = _TruncatedTransformer()
        try:
            yield truncated_model
        except _EarlyStopping:
            pass
    finally:
        model.transformer.h = orig_layers
