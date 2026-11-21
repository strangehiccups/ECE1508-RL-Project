"""Adaptation module phi (Phase 2): state-action histories of BOTH drones -> z_hat_t,
regressed onto z_t = mu(e_t) of the frozen Phase 1 encoder.

Inputs at time t (env.get_history(), oldest first, zero-padded after a reset):
    bottom: (B, L, 15 + 4)   own [x_k, a_k]: true state (v, R, w) + action
    top:    (B, L, 22)       top drone [s_k, u_{k-1}] (TOP_LAYOUT)
The top history lets phi tell "heavy / weak motors" from "in the wake" (both look like a
sag in the bottom history) and predict the downwash from where the top drone is.

RMA: per-step 2-layer MLP embedding (32) -> 3 x 1-D conv over time -> flatten -> linear.
Here each drone has its own embedding; the two are stacked as channels, so every conv
layer mixes both drones at every time step. GRU memory: README "Planned".
"""

import torch
import torch.nn as nn

from ..configs import AdaptationCfg
from ..utils.normalizer import RunningNorm
from .common import ACTIVATIONS, mlp


class AdaptationModule(nn.Module):
    def __init__(self, bottom_dim: int, bottom_len: int, top_dim: int, top_len: int,
                 z_dim: int, cfg: AdaptationCfg):
        super().__init__()
        assert bottom_len == top_len, "phi stacks both drones as channels: equal lengths"
        # Define running norms of both observations
        self.bottom_norm, self.top_norm = RunningNorm(bottom_dim), RunningNorm(top_dim)
        
        # Define embeddings of observation histories
        E = cfg.embed_dim
        self.embed_bottom = mlp(bottom_dim, (E,), E, cfg.activation, out_act=True)
        self.embed_top = mlp(top_dim, (E,), E, cfg.activation, out_act=True)
        
        # Defines the CNN
        # Input channels = 2*E, k = kernel size, s = stride
        layers, ch, L = [], 2 * E, bottom_len
        for out_ch, k, s in cfg.conv:
            layers += [nn.Conv1d(ch, out_ch, k, s), ACTIVATIONS[cfg.activation]()]
            ch, L = out_ch, (L - k) // s + 1  # input of next layer = output of current layer; length of embeddings remaining
            assert L > 0, f"history of {bottom_len} steps too short for conv {cfg.conv}"
        self.conv = nn.Sequential(*layers)
        
        # MLP head, outputs z_hat
        self.head = nn.Linear(ch * L, z_dim)

    def forward(self, bottom_hist: torch.Tensor, top_hist: torch.Tensor) -> torch.Tensor:
        """bottom_hist (B, L, bottom_dim), top_hist (B, L, top_dim) -> z_hat (B, z_dim)."""
        # Embed the observation histories and use CNN to give prediction z_hat
        h = torch.cat([self.embed_bottom(self.bottom_norm(bottom_hist)),
                       self.embed_top(self.top_norm(top_hist))], dim=-1)   # (B, L, 2E)
        return self.head(self.conv(h.transpose(1, 2)).flatten(1))

    def update_norm(self, bottom_hist: torch.Tensor, top_hist: torch.Tensor) -> None:
        self.bottom_norm.update(bottom_hist)
        self.top_norm.update(top_hist)
