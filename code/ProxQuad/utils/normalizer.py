"""Running mean/std normalizer (as in code/quad/policy.py), as an nn.Module: the
statistics are buffers in the owning network's state_dict. Updated by the trainer in
Phase 1, frozen afterwards."""

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0, scale: bool = True):
        """scale=False: subtract the mean only, keeping the input's units."""
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.clip, self.scale = clip, scale

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        w_new, w_old = (b_n / tot).float(), (self.count / tot).float()
        self.mean += delta * w_new
        self.var.copy_(self.var * w_old + b_var * w_new + delta**2 * w_old * w_new)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.mean
        if self.scale:
            x = x / torch.sqrt(self.var + 1e-8)
        return x.clamp(-self.clip, self.clip)
