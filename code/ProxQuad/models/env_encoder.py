"""Environment-factor encoder mu: e_t -> z_t (Phase 1 only; trained jointly with pi).

RMA: 3-layer MLP (256, 128), e_t in R^17 -> z_t in R^8.
Here e_t is env.priv_layout (24 dims; 29 with rotor lag, see envs/observations.py).
"""

import torch
import torch.nn as nn

from ..configs import EncoderCfg
from ..utils.normalizer import RunningNorm
from .common import mlp


class EnvFactorEncoder(nn.Module):
    """Running mean/std normalizer on e_t -> MLP -> linear z."""

    def __init__(self, priv_dim: int, z_dim: int, cfg: EncoderCfg):
        super().__init__()
        self.priv_dim, self.z_dim = priv_dim, z_dim
        self.norm = RunningNorm(priv_dim)
        self.net = mlp(priv_dim, cfg.hidden, z_dim, cfg.activation)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e (B, priv_dim) -> z (B, z_dim)."""
        return self.net(self.norm(e))

    def update_norm(self, e: torch.Tensor) -> None:
        self.norm.update(e)
