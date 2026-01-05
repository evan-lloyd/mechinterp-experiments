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
        with_inhibition: bool = False,
        d_dense: Optional[int] = None,
        normalize_activations: bool = False,
    ):
        super().__init__()
        self.device = device
        self.kind = kind
        self.topk = topk
        self.d_model = d_model
        self.d_sae = d_sae
        self.d_dense = d_dense
        self.with_inhibition = with_inhibition
        self.normalize_activations = normalize_activations
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
            if self.d_dense is not None:
                self.dense_decoder = torch.nn.Linear(
                    self.d_dense, self.d_model, device=self.device
                )
                self.dense_encoder = torch.nn.Linear(
                    self.d_model, self.d_dense, device="meta"
                )
                self.dense_encoder.weight = torch.nn.Parameter(
                    self.dense_decoder.weight.T.clone().detach().contiguous()
                )
                self.dense_encoder.bias = torch.nn.Parameter(
                    torch.zeros(self.d_dense, device=self.device)
                )
            if self.with_inhibition:
                self.inhibition = torch.nn.Parameter(
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

            if self.d_dense is not None:
                self.dense_decoder = torch.nn.Linear(
                    self.d_dense, self.d_model, device="meta"
                )
                self.dense_encoder = torch.nn.Linear(
                    self.d_model, self.d_dense, device="meta"
                )
                self.dense_decoder.weight = torch.nn.Parameter(
                    init_from.dense_decoder.weight.clone().detach().contiguous()
                )
                self.dense_decoder.bias = torch.nn.Parameter(
                    init_from.dense_decoder.bias.clone().detach().contiguous()
                )
                self.dense_encoder.weight = torch.nn.Parameter(
                    init_from.dense_encoder.weight.clone().detach().contiguous()
                )
                self.dense_encoder.bias = torch.nn.Parameter(
                    init_from.dense_encoder.bias.clone().detach().contiguous()
                )

            if self.with_inhibition:
                self.inhibition = torch.nn.Parameter(
                    init_from.inhibition.clone().detach().contiguous()
                )

        if self.normalize_activations:
            self.normalizer = ActivationNormalizer(self.device)
            self.normalizer.init_weights()
        else:
            self.normalizer = DummyNormalizer()

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

    def decode(self, *args):
        if self.d_dense is None:
            return self.decoder(args[0])
        dense_part = self.dense_decoder(args[1])

        return self.decoder(args[0]) + dense_part, dense_part

    def encode(self, x: torch.Tensor):
        encoder_output = self.activation_fn(self.encoder(x))

        if self.with_inhibition:
            encoder_output = self.activation_fn(encoder_output @ self.inhibition)

        if self.d_dense is None:
            return encoder_output

        return encoder_output, self.dense_encoder(x)

    def forward(self, x: torch.Tensor, *args, position_ids: torch.Tensor, **kwargs):
        normalization = self.normalizer(position_ids).unsqueeze(-1)
        if self.d_dense is None:
            return (self.decode(self.encode(x / normalization)) * normalization,)
        return (self.decode(*self.encode(x / normalization))[0] * normalization,)


class DummyNormalizer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return torch.ones_like(args[0])


class ActivationNormalizer(torch.nn.Module):
    """Models activation magnitudes as a function of token position, assumed to be an exponential decay
    down to a steady state "offset":
        f(i) = scale * exp(-rate * i) + offset.
    This qualitatively fits what I saw after plotting the magnitudes of a bunch of examples, but I don't
    know if this holds in general. Note that this is only used if "normalize_activations" is set to True
    in the SAE initialization. Also note that this becomes the "standard" SAE activation normalizer (the
    mean activation magnitude) when scale and rate are fixed to 1 and 0 respectively.
    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.init_weights()

    def init_weights(self):
        self.scale = torch.nn.Parameter(torch.tensor(0.5, device=self.device))
        self.rate = torch.nn.Parameter(torch.tensor(0.0, device=self.device))
        self.offset = torch.nn.Parameter(torch.tensor(0.5, device=self.device))

    @torch.no_grad
    def heuristic_init(self, position_ids: torch.Tensor, target_norms: torch.Tensor):
        max_token_position = position_ids.max().item()
        gathered_norms = [list() for _ in range(max_token_position + 1)]
        for i in range(target_norms.shape[0]):
            gathered_norms[position_ids[i]].append(target_norms[i])
        gathered_norms = torch.tensor(
            [torch.mean(torch.stack(g)) for g in gathered_norms],
            device=target_norms.device,
        )

        offset = target_norms.mean().item()
        scale, peak_idx = gathered_norms.max(0)

        self.scale.fill_(max(scale - offset, 0.0))
        # Find where peak decays by ~one factor of e
        for decay_idx in range(peak_idx, gathered_norms.shape[0]):
            if gathered_norms[decay_idx] - offset <= (scale - offset) * 0.63:
                break
        self.rate.fill_(decay_idx)
        self.offset.fill_(offset)

    def forward(self, position_ids: torch.Tensor):
        return (
            self.scale * torch.exp(-self.rate * position_ids) + self.offset
        ).relu() + 1e-6
