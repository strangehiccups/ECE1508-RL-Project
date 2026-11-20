"""Base policy pi(x~_t, a_{t-1}, o_t^top, z_t) -> a_t and critic V. Shared by both phases.
x~_t = bottom drone's tracking errors, o_t^top = top drone's state + previous command.
    Phase 1:              z_t = mu(e_t)
    Phase 2 / deployment: z_hat_t = phi(bottom history, top history)

RMA reference: 3-layer MLP, 128 hidden units. Architecture and the critic's inputs
(z_t, or e_t directly as an asymmetric critic) are TBD.
"""

import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    def __init__(self, tracking_dim: int, action_dim: int, top_dim: int, z_dim: int):
        super().__init__()
        self.tracking_dim, self.action_dim = tracking_dim, action_dim
        self.top_dim, self.z_dim = top_dim, z_dim
        raise NotImplementedError("architecture of pi not decided yet")

    def forward(self, err: torch.Tensor, a_prev: torch.Tensor, top: torch.Tensor,
                z: torch.Tensor) -> torch.distributions.Distribution:
        """err (B, tracking_dim), a_prev (B, 4), top (B, top_dim), z (B, z_dim)
        -> action distribution. The env clamps actions to [-1, 1]."""
        raise NotImplementedError


class Critic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        raise NotImplementedError("critic inputs/architecture not decided yet")

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
