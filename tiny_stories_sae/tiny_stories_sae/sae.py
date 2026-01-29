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
        use_interaction=False,
        init_from: Optional["SAE"] = None,
        n_interaction_iterations=1,
    ):
        super().__init__()
        self.device = device
        self.kind = kind
        self.topk = topk
        self.d_model = d_model
        self.d_sae = d_sae
        self.use_interaction = use_interaction
        self.n_interaction_iterations = n_interaction_iterations
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
            if self.use_interaction:
                self.interaction = torch.nn.Parameter(
                    torch.eye(self.d_sae, device=self.device)
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
            if self.use_interaction:
                self.interaction = torch.nn.Parameter(
                    init_from.interaction.clone().detach().contiguous()
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
        encoder_output = self.encoder(x)
        if self.use_interaction:
            features = self.activation_fn(encoder_output)
            for _ in range(self.n_interaction_iterations):
                features = self.activation_fn(
                    encoder_output + features @ self.interaction
                )
            encoder_output = features
        else:
            encoder_output = self.activation_fn(encoder_output)

        return encoder_output

    def forward(self, x: torch.Tensor, *args, position_ids: torch.Tensor, **kwargs):
        return (self.decode(self.encode(x)),)
