from typing import Optional

import torch


class SAE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_sae: int,
        device="cpu",
        kind="standard",
        topk=None,
        init_from: Optional["SAE"] = None,
    ):
        super().__init__()
        self.device = device
        self.kind = kind
        self.topk = topk
        self.d_model = d_model
        self.d_sae = d_sae
        self.init_weights(init_from)

    @torch.no_grad
    def init_weights(
        self,
        init_from: Optional["SAE"] = None,
    ):
        if init_from is None:
            self.decoder = torch.nn.Linear(self.d_sae, self.d_model, device=self.device)
            self.encoder = torch.nn.Linear(self.d_model, self.d_sae, device="meta")
            self.encoder.weight = torch.nn.Parameter(
                self.decoder.weight.T.clone().detach().contiguous()
            )
            self.encoder.bias = torch.nn.Parameter(
                torch.zeros(self.d_sae, device=self.device)
            )
        else:
            self.decoder = torch.nn.Linear(self.d_sae, self.d_model, device="meta")
            self.encoder = torch.nn.Linear(self.d_model, self.d_sae, device="meta")
            self.decoder.weight = torch.nn.Parameter(
                init_from.decoder.weight.clone().detach().contiguous()
            )
            self.decoder.bias = torch.nn.Parameter(
                init_from.decoder.bias.clone().detach().contiguous()
            )
            self.encoder.weight = torch.nn.Parameter(
                init_from.encoder.weight.clone().detach().contiguous()
            )
            self.encoder.bias = torch.nn.Parameter(
                init_from.encoder.bias.clone().detach().contiguous()
            )

    def activation_fn(self, x: torch.Tensor):
        if self.kind == "standard":
            return x.relu()
        elif self.kind == "topk":
            topk = torch.topk(x, k=self.topk, dim=-1)
            result = torch.zeros_like(x)
            result.scatter_(-1, topk.indices, topk.values.relu())
            return result
        else:
            raise NotImplementedError(f'"{self.kind}" SAE kind not implemented')

    def decode(self, x: torch.Tensor):
        return self.decoder(x)

    def encode(self, x: torch.Tensor):
        return self.activation_fn(self.encoder(x))

    def forward(self, x: torch.Tensor, *args, position_ids: torch.Tensor, **kwargs):
        return (self.decode(self.encode(x)),)
