"""Composed downwash model: wake + rotor model + airframe drag + turbulence."""

import math

import torch

from ...configs import AeroCfg
from ...utils.math import uniform
from ..drone import DroneParams, DroneState
from .base import AeroOutput, RotorModel, WakeModel
from .rotor import BEMTRotor, EfficiencyRotor
from .surrogate import LearnedWake
from .wake import GaussianJetWake

WAKE_MODELS = {"gaussian_jet": GaussianJetWake, "learned": LearnedWake}
ROTOR_MODELS = {"efficiency": EfficiencyRotor, "bemt": BEMTRotor}


class DownwashModel:
    """Effect of the top drone's wake on the bottom drone.

    Evaluated once per control step, before the policy acts, so the privileged vector
    e_t describes the disturbance over the next interval [t, t + dt):
      1. wake velocity at the 4 bottom rotor hubs and at the bottom geometric center
      2. rotor thrust ratio from the axial inflow (rotor model).
      3. airframe drag  F = 0.5 rho CdA |w| w  at the center
      4. turbulence: OU force/torque, std scaled by intensity |w_center| / v_h
    The top drone feels nothing from the bottom one.
    """

    def __init__(self, cfg: AeroCfg, num_envs: int, device: torch.device):
        if cfg.transport_delay:
            raise NotImplementedError("wake transport delay (README, downwash fidelity)")
        self.cfg, self.n, self.dev = cfg, num_envs, device
        rho = cfg.air_density
        self.wake: WakeModel = WAKE_MODELS[cfg.wake.model](cfg.wake, rho, num_envs, device)
        self.rotor: RotorModel = ROTOR_MODELS[cfg.rotor.model](cfg.rotor, rho, num_envs, device)
        self.cda = torch.full((num_envs,), sum(cfg.airframe_cda) / 2, device=device)   # Drag coeffcient * area
        self.turb_f = torch.zeros(num_envs, 3, device=device)   # Turbulence force
        self.turb_t = torch.zeros(num_envs, 3, device=device)   # Turbulence torque

    def resample(self, ids: torch.Tensor) -> None:
        # Resample per-episode random parameters for env ids.
        self.wake.resample(ids)
        self.rotor.resample(ids)
        self.cda[ids] = uniform(self.cfg.airframe_cda, len(ids), self.dev)

    def episode_params(self) -> torch.Tensor:
        """(N, 4) this episode's wake settings [wake gain, wake spread, rotor coefficient,
        airframe CdA], each / its range midpoint: privileged info (e_t)."""
        cda = (self.cda / (sum(self.cfg.airframe_cda) / 2))[:, None]
        return torch.cat([self.wake.episode_params(), self.rotor.episode_params(), cda], -1)

    def reset(self, ids: torch.Tensor) -> None:
        self.turb_f[ids] = 0.0
        self.turb_t[ids] = 0.0

    def compute(self, bot: DroneState, bp: DroneParams, bot_omega: torch.Tensor,
                top: DroneState, tp: DroneParams, top_thrust: torch.Tensor,
                dt: float) -> AeroOutput:
        """bot_omega (N, 4): bottom rotor speeds; top_thrust (N, 4): top rotor thrusts."""
        if not self.cfg.enabled:
            return AeroOutput.none(self.n, self.dev)
        cfg, rho = self.cfg, self.cfg.air_density

        # Compute wake jets at query points
        rotors = bot.pos[:, None, :] + torch.einsum("nij,kj->nki", bot.R, bp.rotor_pos)  # Rotor inertial pos
        pts = torch.cat([rotors, bot.pos[:, None, :]], dim=1)             # Query points (N, 5, 3)
        v = self.wake.velocity(pts, top, tp, top_thrust)                  # Wake velocities at 5 query points (N, 5, 3)

        # Compute effect of wake jets on rotors
        z_b = bot.R[:, :, 2]                                             # Bottom drone rotor axis
        inflow = (-(v[:, :4] * z_b[:, None, :]).sum(-1)).clamp_min(0.0)  # Inflow speed (N, 4)
        ratio = self.rotor.thrust_ratio(inflow, bp, bot_omega)           # Compute thrust ratio based on rotor model

        # Compute effect wake jets on body
        v_center = v[:, 4]                                               # Wake velocity at body
        w_center = v_center.norm(dim=-1)                                 # Wake speed at body
        drag = (0.5 * rho * self.cda * w_center)[:, None] * v_center     # Drag induced by inflow at body

        # Ornstein-Uhlenbeck turbulence, exact discretization, unit stationary std
        a = math.exp(-dt / cfg.turb_corr_time)
        b = math.sqrt(1.0 - a * a)
        self.turb_f = a * self.turb_f + b * torch.randn_like(self.turb_f)
        self.turb_t = a * self.turb_t + b * torch.randn_like(self.turb_t)
        intensity = (w_center / bp.hover_induced_velocity(rho))[:, None]      # Intensity of OU turbulence determined by wake speed/hover induced vel
        W = bp.weight_nom
        f_turb = cfg.turb_force_std * W * intensity * self.turb_f             # Turbulence force
        t_turb = cfg.turb_torque_std * W * bp.arm * intensity * self.turb_t   # Turbulence torque

        return AeroOutput(thrust_ratio=ratio, force=drag + f_turb, torque=t_turb, drag=drag,
                          rotor_inflow=inflow, center_wake=v_center)
