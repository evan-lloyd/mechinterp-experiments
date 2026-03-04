import torch
from sae_lens import SAE as SAELens

from .sae import SAE, SAEConfig


class SAELensSAE(torch.nn.Module):
    def __init__(
        self,
        sae_lens_config: dict,
        sae_lens_sae: SAELens,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.sae_lens_config = sae_lens_config
        self.sae_lens_sae = sae_lens_sae
        self.device = device
        self.dtype = dtype

    @property
    def encoder(self):
        return self.sae_lens_sae.hook_sae_acts_post

    def forward(self, x: torch.Tensor, *args, **kwargs):
        orig_dtype = x.dtype
        return self.sae_lens_sae(x.to(self.dtype)).to(orig_dtype)

    def offload(self):
        self.to(torch.device("cpu"))

    def onload(self):
        self.to(self.device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.requires_grad_(mode)

    def decode(self, x: torch.Tensor, should_cast: bool = True):
        orig_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self.sae_lens_sae.decode(x)
        if should_cast:
            result = result.to(orig_dtype)
        return result

    def encode(
        self,
        x: torch.Tensor,
        should_cast: bool = True,
    ):
        out_dtype = x.dtype
        if should_cast:
            x = x.to(self.dtype)
        result = self.sae_lens_sae.decode(x)
        if should_cast:
            result = result.to(out_dtype)
        return result


def wrap_sae_lens_pretrained(**kwargs):
    saelens, saelens_config, _ = SAELens.from_pretrained_with_cfg_and_sparsity(**kwargs)
    return SAELensSAE(
        saelens_config, saelens, kwargs.get("device"), kwargs.get("dtype")
    )
