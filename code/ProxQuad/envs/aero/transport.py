"""Wake transport delay (AeroCfg.transport_delay). Air reaching point p from jet k left the
source tau = s_tau * t(s) earlier: t(s) is the travel time along the jet
(WakeModel.travel_time), s_tau a random per-episode scale (in e_t). Frozen wake: jet k is
evaluated with the source position, axis and thrust it had at emission, from a ring buffer
of the top drone's wake sources per control step; `transport_delay_passes` fixed-point
passes recompute s from that past source. A new episode fills the buffer with its first
state (a steady wake, as if the top drone had always been there).
"""

import torch

from ...configs import AeroCfg
from ...utils.math import uniform
from ..drone import DroneParams
from .base import WakeModel


class WakeTransport:
    def __init__(self, cfg: AeroCfg, dt: float, num_envs: int, num_sources: int,
                 device: torch.device):
        self.cfg, self.dt, self.dev = cfg, dt, device
        self.L = round(cfg.transport_delay_max_s / dt) + 1
        L, N, K = self.L, num_envs, num_sources
        self.src = torch.zeros(L, N, K, 3, device=device)
        self.axis = torch.zeros(L, N, 3, device=device)
        self.thrust = torch.zeros(L, N, K, device=device)
        self.head = 0                                    # slot of the newest step
        self.scale = torch.full((N,), sum(cfg.transport_delay_scale) / 2, device=device)
        self.pending: torch.Tensor | None = torch.arange(N, device=device)   # to fill
        self.n_idx = torch.arange(N, device=device)[:, None, None]
        self.k_idx = torch.arange(K, device=device)[None, None, :]
        self.steps: torch.Tensor | None = None   # delays of the last call [steps], (N, P, K)

    def resample(self, ids: torch.Tensor) -> None:
        self.scale[ids] = uniform(self.cfg.transport_delay_scale, len(ids), self.dev)

    def reset(self, ids: torch.Tensor) -> None:
        self.pending = ids if self.pending is None else torch.cat([self.pending, ids])

    def push(self, src: torch.Tensor, axis: torch.Tensor, thrust: torch.Tensor) -> None:
        """This control step's sources: (N, K, 3), (N, 3), (N, K)."""
        self.head = (self.head + 1) % self.L
        self.src[self.head], self.axis[self.head], self.thrust[self.head] = src, axis, thrust
        if self.pending is not None:
            ids = self.pending
            self.src[:, ids], self.axis[:, ids] = src[ids], axis[ids]
            self.thrust[:, ids] = thrust[ids]
            self.pending = None

    def delayed_sources(self, points: torch.Tensor, wake: WakeModel, tp: DroneParams):
        """Per point (N, P, 3) and jet: emission-time source (N, P, K, 3), axis
        (N, P, K, 3), thrust (N, P, K)."""
        h = self.head
        src, d, T = self.src[h][:, None], self.axis[h][:, None, None], self.thrust[h][:, None]
        for _ in range(1 + self.cfg.transport_delay_passes):
            s = ((points[:, :, None, :] - src) * d).sum(-1)                   # (N, P, K)
            tau = self.scale[:, None, None] * wake.travel_time(s, T, tp)
            k = (tau / self.dt).round().long().clamp(0, self.L - 1)
            slot = (h - k) % self.L
            src = self.src[slot, self.n_idx, self.k_idx]
            d = self.axis[slot, self.n_idx]
            T = self.thrust[slot, self.n_idx, self.k_idx]
        self.steps = k
        return src, d, T

    def episode_params(self) -> torch.Tensor:
        """(N, 1) delay scale / its range midpoint (privileged info)."""
        return (self.scale / (sum(self.cfg.transport_delay_scale) / 2))[:, None]
