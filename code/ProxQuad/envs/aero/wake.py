"""Wake (downwash velocity field) models of the top drone."""

import torch

from ...configs import WakeCfg
from ...utils.math import normalize, uniform
from ..drone import DroneParams, DroneState
from .base import WakeModel


class GaussianJetWake(WakeModel):
    """Axisymmetric turbulent round jet with a Gaussian profile and centerline decay.

    For each wake source k (each rotor's jet, or one merged jet of the whole drone):
        d     = -z_top                                wake axis (follows the top's tilt)
        s     = (p - p_k) . d                         distance downstream of the source
        rho^2 = |p - p_k|^2 - s^2                     squared distance from the axis
        sig   = sig0 + k_s * s                        jet width grows linearly
        w_c   = w0 * sig0 / sig                       w_c^2 sig^2 = const: momentum flux
                                                      conserved, so w_c ~ 1/s far away
        w_k   = w_c * exp(-rho^2 / (2 sig^2)) * gate(s)
        w0    = k_w * sqrt(T_k / (2 rho_air A_k))     momentum-theory induced velocity
    Sources are merged by momentum: root-sum-square of speeds, along the w^2-weighted mean
    axis (the common axis d unless a transport delay gives each jet its emission axis).
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
        self.cfg, self.dev = cfg, device
        self.rho = air_density   # air density [kg/m^3]
        
        # per-episode random parameters
        self.spread = torch.full((num_envs,), sum(cfg.spread_rate) / 2, device=device)  # linear growth of the jet width
        self.gain = torch.full((num_envs,), sum(cfg.speed_gain) / 2, device=device)  # multiplier of the momentum-theory wake speed

    def resample(self, ids: torch.Tensor) -> None:
        # Resample per-episode random parameters for env ids.
        self.spread[ids] = uniform(self.cfg.spread_rate, len(ids), self.dev)
        self.gain[ids] = uniform(self.cfg.speed_gain, len(ids), self.dev)

    def episode_params(self) -> torch.Tensor:
        # Returns the episode's wake settings, each divided by its range midpoint (privileged info).
        mid = lambda r: sum(r) / 2  # noqa: E731
        return torch.stack([self.gain / mid(self.cfg.speed_gain),
                            self.spread / mid(self.cfg.spread_rate)], dim=-1)

    def sources(self, top: DroneState, top_params: DroneParams, top_thrust: torch.Tensor):
        """The top drone's wake sources now: positions (N, K, 3), common axis d = -z_top
        (N, 3), thrust (N, K). K = 4 rotor hubs, or 1 (the COM, total thrust)."""
        d = -top.R[:, :, 2]
        if self.cfg.sources == "body":
            return top.pos[:, None, :], d, top_thrust.sum(-1, keepdim=True)
        hubs = top.pos[:, None, :] + torch.einsum("nij,kj->nki", top.R, top_params.rotor_pos)
        return hubs, d, top_thrust

    def _area_sig0(self, tp: DroneParams) -> tuple[float, float]:
        """Disk area per source and initial jet width."""
        if self.cfg.sources == "body":
            return 4 * torch.pi * tp.prop_radius**2, self.cfg.sigma0_body * tp.arm
        return torch.pi * tp.prop_radius**2, self.cfg.sigma0_rotor * tp.prop_radius

    def _w0(self, T: torch.Tensor, tp: DroneParams) -> torch.Tensor:
        """Initial jet speed, momentum theory x gain. T (N, ..., K)."""
        area, _ = self._area_sig0(tp)
        g = self.gain.view(-1, *([1] * (T.dim() - 1)))
        return g * torch.sqrt(T.clamp_min(0.0) / (2 * self.rho * area))

    def travel_time(self, s: torch.Tensor, T: torch.Tensor, tp: DroneParams) -> torch.Tensor:
        """Time for air to travel from the source to downstream distance s [s] along the
        centerline, w_c = w0 sig0 / (sig0 + k_s s): t = s / w0 + k_s s^2 / (2 w0 sig0).
        s, T (N, P, K)."""
        _, sig0 = self._area_sig0(tp)
        w0 = self._w0(T, tp).clamp_min(1e-3)
        s = s.clamp_min(0.0)
        return s / w0 + self.spread[:, None, None] * s * s / (2 * w0 * sig0)

    def field(self, points, src, d, T, tp: DroneParams) -> torch.Tensor:
        """Wake velocity (N, P, 3) at points (N, P, 3) from jets k with source src
        (N, P, K, 3), axis d (N, P, K, 3) and thrust T (N, P, K), per point and jet
        (size-1 dims broadcast). Jets are merged by momentum: speed = root-sum-square,
        direction = the w^2-weighted mean axis."""
        _, sig0 = self._area_sig0(tp)
        w0 = self._w0(T, tp)                                        # (N, P|1, K)
        r = points[:, :, None, :] - src                             # (N, P, K, 3)
        s = (r * d).sum(-1)                                         # downstream distance
        rho2 = ((r * r).sum(-1) - s * s).clamp_min(0.0)             # squared distance from the axis
        sig = sig0 + self.spread[:, None, None] * s.clamp_min(0.0)  # jet width
        gate = (s / self.cfg.onset_ramp).clamp(0.0, 1.0)            # starts at the rotor plane
        w = w0 * (sig0 / sig) * torch.exp(-rho2 / (2 * sig * sig)) * gate
        w2 = w * w
        axis = normalize((w2[..., None] * d).sum(-2))
        return torch.sqrt(w2.sum(-1))[..., None] * axis

    def velocity(self, points, top: DroneState, top_params: DroneParams, top_thrust):
        """Instantaneous wake (no transport delay)."""
        src, d, T = self.sources(top, top_params, top_thrust)
        return self.field(points, src[:, None], d[:, None, None], T[:, None], top_params)
