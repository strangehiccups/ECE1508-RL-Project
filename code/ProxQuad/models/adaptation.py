"""Adaptation module phi (Phase 2): state-action histories of BOTH drones -> z_hat_t,
regressed onto z_t = mu(e_t) of the frozen Phase 1 encoder.

Inputs at time t (env.get_history(), oldest first, zero-padded after a reset):
    bottom: (B, L_bottom, 15 + 4)   own [x_k, a_k]: true state (v, R, w) + action
    top:    (B, L_top, 22)          top drone [s_k, u_{k-1}] (TOP_LAYOUT)
The top history lets phi tell "heavy / weak motors" from "in the wake" (both look like a
sag in the bottom history) and predict the downwash from where the top drone is.
RMA reference: per-step 2-layer MLP embedding (32) -> 3 x 1-D conv over time -> linear.
Open: architecture (one encoder over the concatenated streams, or two streams fused
late; GRU memory, README "Planned").
"""

import torch
import torch.nn as nn


class AdaptationModule(nn.Module):
    def __init__(self, bottom_dim: int, bottom_len: int, top_dim: int, top_len: int,
                 z_dim: int):
        super().__init__()
        self.bottom_dim, self.bottom_len = bottom_dim, bottom_len
        self.top_dim, self.top_len, self.z_dim = top_dim, top_len, z_dim
        raise NotImplementedError("Phase 2: architecture of phi not decided yet")

    def forward(self, bottom_hist: torch.Tensor, top_hist: torch.Tensor) -> torch.Tensor:
        """bottom_hist (B, L_bottom, bottom_dim), top_hist (B, L_top, top_dim)
        -> z_hat (B, z_dim)."""
        raise NotImplementedError
