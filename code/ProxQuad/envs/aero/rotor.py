"""Rotor (propeller) models: thrust of a bottom rotor under axial inflow from the wake.
Both are analytic and cost a few elementwise ops per rotor."""

import math

import torch

from ...configs import RotorCfg
from ...utils.math import uniform
from ..drone import DroneParams
from .base import RotorModel


class EfficiencyRotor(RotorModel):
    """Tier A. Linear thrust loss in the inflow, normalized by the hover induced velocity:

        ratio = clip(1 - k_eta * w / v_h, eta_min, 1),   k_eta random per episode.

    Independent of rotor speed. Tier B gives k_eta ~ 0.4 near hover.
    """

    def __init__(self, cfg: RotorCfg, air_density: float, num_envs: int,
                 device: torch.device):
        self.cfg, self.rho, self.dev = cfg, air_density, device
        self.k = torch.full((num_envs,), sum(cfg.efficiency_gain) / 2, device=device)

    def resample(self, ids):
        self.k[ids] = uniform(self.cfg.efficiency_gain, len(ids), self.dev)

    def episode_params(self):
        return (self.k / (sum(self.cfg.efficiency_gain) / 2))[:, None]

    def thrust_ratio(self, inflow, params: DroneParams, omega):
        v_h = params.hover_induced_velocity(self.rho)
        return (1.0 - self.k[:, None] * inflow / v_h).clamp(self.cfg.efficiency_min, 1.0)


class BEMTRotor(RotorModel):
    """Tier B. Blade-element momentum theory, ideal twist, uniform inflow, at the actual
    rotor speed (the lagged w from the motor model).

    With solidity sig, lift slope a, tip pitch th and climb inflow ratio
    lam_c = w / (Omega R) (the wake acts like climbing at speed w), the total inflow
    ratio and thrust coefficient have closed forms (Leishman, Principles of Helicopter
    Aerodynamics, ch. 3):
        lam(lam_c) = sqrt((sig a/16 - lam_c/2)^2 + sig a th / 8) - (sig a/16 - lam_c/2)
        C_T        = (sig a / 4) (th - lam)
    th is set so that C_T(lam_c = 0) matches the bench fit, C_T0 = k_f / (rho pi R^4).
    ratio = C_T(lam_c) / C_T0. Random per episode: solidity.
    """

    def __init__(self, cfg: RotorCfg, air_density: float, num_envs: int,
                 device: torch.device):
        self.cfg, self.rho, self.dev = cfg, air_density, device
        self.sigma = torch.full((num_envs,), sum(cfg.solidity) / 2, device=device)

    def resample(self, ids):
        self.sigma[ids] = uniform(self.cfg.solidity, len(ids), self.dev)

    def episode_params(self):
        return (self.sigma / (sum(self.cfg.solidity) / 2))[:, None]

    def thrust_ratio(self, inflow, params: DroneParams, omega):
        R = params.prop_radius
        ct0 = params.k_f / (self.rho * math.pi * R**4)
        sa = (self.sigma * self.cfg.lift_slope)[:, None]                  # (N, 1)
        lam_h = math.sqrt(ct0 / 2)
        theta = 4 * ct0 / sa + lam_h
        lam_c = inflow / (omega * R).clamp_min(1.0)
        b = sa / 16 - lam_c / 2
        lam = torch.sqrt(b * b + sa * theta / 8) - b
        return ((theta - lam) / (theta - lam_h)).clamp(0.0, 1.5)
