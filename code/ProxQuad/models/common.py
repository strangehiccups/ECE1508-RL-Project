"""Building blocks shared by the networks."""

import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}   # all 1-Lipschitz


def mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str,
        out_act: bool = False) -> nn.Sequential:
    dims, act, layers = [in_dim, *hidden, out_dim], ACTIVATIONS[activation], []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


class LipschitzMLP(nn.Module):
    """MLP with L2 Lipschitz constant <= `lipschitz`: each weight is spectrally normalized
    (sigma_max = 1, one power iteration per training forward) and scaled by
    lipschitz^(1 / n_layers); activations are 1-Lipschitz. The output is multiplied by a
    learned gain tanh(rho), |.| <= 1, starting at init_out_gain (spectral normalization
    undoes a small init of the last weights). Biases start at 0: f(0) = 0."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], out_dim: int, activation: str,
                 lipschitz: float, init_out_gain: float = 1.0):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.linears = nn.ModuleList(spectral_norm(nn.Linear(a, b))
                                     for a, b in zip(dims[:-1], dims[1:]))
        for lin in self.linears:
            nn.init.zeros_(lin.bias)
        self.act = ACTIVATIONS[activation]()
        self.gain = lipschitz ** (1.0 / len(self.linears))
        self.out_rho = nn.Parameter(torch.tensor(math.atanh(min(init_out_gain, 0.999))))

    def out_gain(self) -> torch.Tensor:
        return torch.tanh(self.out_rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            x = self.gain * lin(x)
            if i < len(self.linears) - 1:
                x = self.act(x)
        return self.out_gain() * x

    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """|out gain| x product of the exact per-layer spectral norms (power iteration only
        estimates them, so this can exceed `lipschitz` slightly)."""
        bound = abs(self.out_gain().item())
        for lin in self.linears:
            bound *= self.gain * torch.linalg.matrix_norm(lin.weight, ord=2).item()
        return bound
