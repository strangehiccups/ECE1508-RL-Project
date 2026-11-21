"""Base policy pi(x~_t, a_{t-1}, o_t^top, z_t) -> a_t and critic V. Shared by both phases.
x~_t = bottom drone's tracking errors, o_t^top = top drone's state + previous command.
    Phase 1:              z_t = mu(e_t)
    Phase 2 / deployment: z_hat_t = phi(bottom history, top history)
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from ..configs import CriticCfg, PolicyCfg
from ..utils.normalizer import RunningNorm
from .common import LipschitzMLP, mlp


class BasePolicy(nn.Module):
    """Diagonal Gaussian, mean = a_hover + f([obs_norm(obs), z]); std: learned
    state-independent vector.
    cfg.spectral_norm: f is a LipschitzMLP, |mean(x) - mean(x')| <= lipschitz |x - x'|
        (L2, obs in SI units): obs_norm only centers, so the bound does not change with
        the running statistics.
    else: plain MLP, obs_norm centers and scales."""

    def __init__(self, obs_dims: dict[str, int], z_dim: int, action_dim: int,
                 cfg: PolicyCfg, a_hover: float):
        super().__init__()
        self.keys = list(obs_dims)
        obs_dim = sum(obs_dims.values())
        self.obs_norm = RunningNorm(obs_dim, scale=not cfg.spectral_norm)
        if cfg.spectral_norm:
            # Uses a spectrally normlized MLP if turned on
            self.net = LipschitzMLP(obs_dim + z_dim, cfg.hidden, action_dim, cfg.activation,
                                    cfg.lipschitz, cfg.init_out_gain)
        else:
            # MLP as actor
            self.net = mlp(obs_dim + z_dim, cfg.hidden, action_dim, cfg.activation)
            with torch.no_grad():
                # Initialize the network weights with a small gain to prevent initial crashes
                self.net[-1].weight.mul_(cfg.init_out_gain)
                self.net[-1].bias.zero_()
        
        # Setup the log of std of each action distribution
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(cfg.init_std)))
        self.log_std_bounds = cfg.log_std_bounds
        self.register_buffer("a_hover", torch.tensor(float(a_hover)))

    def _obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs[k] for k in self.keys], dim=-1)

    def act_mean(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Deterministic action (Phase 2 rollouts, deployment)."""
        return self.a_hover + self.net(torch.cat([self.obs_norm(self._obs(obs)), z], dim=-1))

    def forward(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> Independent:
        """obs: env obs dict, z (B, z_dim) -> distribution; log_prob / entropy are (B,)."""
        # Returns the normal distribution (mean, std^2) of the action
        mean = self.act_mean(obs, z)
        std = self.log_std.clamp(*self.log_std_bounds).exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def update_norm(self, obs: dict[str, torch.Tensor]) -> None:
        self.obs_norm.update(self._obs(obs))

    @torch.no_grad()
    def lipschitz_bound(self) -> float:
        """Certified L2 bound of f w.r.t. its input [obs_norm(obs), z]: SI units with
        spectral_norm, normalized units otherwise (x max 1/std for SI units)."""
        if isinstance(self.net, LipschitzMLP):
            return self.net.lipschitz_bound()
        return math.prod(torch.linalg.matrix_norm(m.weight, ord=2).item()
                         for m in self.net if isinstance(m, nn.Linear))


class Critic(nn.Module):
    """V(x~_t, a_{t-1}, o_t^top, e_t). Asymmetric: raw e_t, no mu, so the value loss does
    not shape z. Own MLP and normalizer, nothing shared with pi."""

    def __init__(self, obs_dims: dict[str, int], priv_dim: int, cfg: CriticCfg):
        super().__init__()
        self.keys = [*obs_dims, "priv"]
        in_dim = sum(obs_dims.values()) + priv_dim
        self.norm = RunningNorm(in_dim)
        # MLP critic network that outputs the predicted value give inputs (observations)
        self.net = mlp(in_dim, cfg.hidden, 1, cfg.activation)

    def _inp(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Builds the input vector by concatenation
        return torch.cat([obs[k] for k in self.keys], dim=-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """-> V (B,)."""
        return self.net(self.norm(self._inp(obs))).squeeze(-1)

    def update_norm(self, obs: dict[str, torch.Tensor]) -> None:
        self.norm.update(self._inp(obs))
