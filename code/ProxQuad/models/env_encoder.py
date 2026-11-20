"""Environment-factor encoder mu: e_t -> z_t (Phase 1 only; trained jointly with pi).

RMA reference: 3-layer MLP (256, 128), e_t in R^17 -> z_t in R^8.
Here e_t is env.priv_layout (24 dims; 29 with rotor lag, see envs/observations.py).
Architecture TBD.
"""

import torch
import torch.nn as nn


class EnvFactorEncoder(nn.Module):
    def __init__(self, priv_dim: int, z_dim: int):
        super().__init__()
        self.priv_dim, self.z_dim = priv_dim, z_dim
        raise NotImplementedError("architecture of mu not decided yet")

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e (B, priv_dim) -> z (B, z_dim)."""
        raise NotImplementedError
