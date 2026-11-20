"""Wake (downwash velocity field) models of the top drone."""

import torch

from ...configs import WakeCfg
from ...utils.math import uniform
from ..drone import DroneParams, DroneState
from .base import WakeModel


class GaussianJetWake(WakeModel):
    """Axisymmetric turbulent round jet with a self-similar Gaussian profile.

    For each wake source k (each rotor's jet, or one merged jet of the whole drone):
        d     = -z_top                                wake axis (follows the top's tilt)
        s     = (p - p_k) . d                         distance downstream of the source
        rho^2 = |p - p_k|^2 - s^2                     squared distance from the axis
        sig   = sig0 + k_s * s                        jet width grows linearly
        w_c   = w0 * sig0 / sig                       w_c^2 sig^2 = const: momentum flux
                                                      conserved, so w_c ~ 1/s far away
        w_k   = w_c * exp(-rho^2 / (2 sig^2)) * gate(s)
        w0    = k_w * sqrt(T_k / (2 rho_air A_k))     momentum-theory induced velocity
    Sources are merged by momentum (root-sum-square of speeds) along the common axis d.
    Random per episode: spread rate k_s and speed gain k_w.

    sources = "rotors": four jets at the rotor hubs, T = rotor thrust, A = one disk,
                        sig0 = sigma0_rotor * prop radius. Resolves partial overlap of
                        individual rotors -- the default for a 1 m-span X650.
    sources = "body":   one jet at the top COM, T = total thrust, A = four disks,
                        sig0 = sigma0_body * arm length.
    """

    def __init__(self, cfg: WakeCfg, air_density: float, num_envs: int,
                 device: torch.device):
        assert cfg.sources in ("body", "rotors"), cfg.sources
        self.cfg, self.rho, self.dev = cfg, air_density, device
        self.spread = torch.full((num_envs,), sum(cfg.spread_rate) / 2, device=device)
        self.gain = torch.full((num_envs,), sum(cfg.speed_gain) / 2, device=device)

    def resample(self, ids: torch.Tensor) -> None:
        self.spread[ids] = uniform(self.cfg.spread_rate, len(ids), self.dev)
        self.gain[ids] = uniform(self.cfg.speed_gain, len(ids), self.dev)

    def episode_params(self) -> torch.Tensor:
        mid = lambda r: sum(r) / 2  # noqa: E731
        return torch.stack([self.gain / mid(self.cfg.speed_gain),
                            self.spread / mid(self.cfg.spread_rate)], dim=-1)

    def velocity(self, points, top: DroneState, top_params: DroneParams, top_thrust):
        cfg, tp = self.cfg, top_params
        d = -top.R[:, :, 2]                                           # (N, 3)
        if cfg.sources == "body":
            src = top.pos[:, None, :]                                 # (N, 1, 3)
            T = top_thrust.sum(-1, keepdim=True)                      # (N, 1)
            area = 4 * torch.pi * tp.prop_radius**2
            sig0 = cfg.sigma0_body * tp.arm
        else:
            src = top.pos[:, None, :] + torch.einsum("nij,kj->nki", top.R, tp.rotor_pos)
            T = top_thrust                                            # (N, 4)
            area = torch.pi * tp.prop_radius**2
            sig0 = cfg.sigma0_rotor * tp.prop_radius
        w0 = self.gain[:, None] * torch.sqrt(T.clamp_min(0.0) / (2 * self.rho * area))

        r = points[:, :, None, :] - src[:, None, :, :]                # (N, P, K, 3)
        s = (r * d[:, None, None, :]).sum(-1)                         # (N, P, K)
        rho2 = ((r * r).sum(-1) - s * s).clamp_min(0.0)
        sig = sig0 + self.spread[:, None, None] * s.clamp_min(0.0)
        gate = (s / cfg.onset_ramp).clamp(0.0, 1.0)
        w = w0[:, None, :] * (sig0 / sig) * torch.exp(-rho2 / (2 * sig * sig)) * gate
        speed = torch.sqrt((w * w).sum(-1))                           # (N, P)
        return speed[..., None] * d[:, None, :]
