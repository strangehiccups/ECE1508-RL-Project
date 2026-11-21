"""Rotor (propeller) models: thrust of a bottom rotor under axial inflow from the wake.
Both are analytic."""

import math

import torch

from ...configs import RotorCfg
from ...utils.math import uniform
from ..drone import DroneParams
from .base import RotorModel


class EfficiencyRotor(RotorModel):
    """Option A. Linear thrust loss in the inflow, normalized by the hover induced velocity:

        ratio = clip(1 - k_eta * w / v_h, eta_min, 1),   k_eta random per episode.

    Independent of rotor speed. Option B linearized at hover: k_eta = sig a / (sig a + 16 lam_h),
    0.30-0.39 over the solidity range.
    """

    def __init__(self, cfg: RotorCfg, air_density: float, num_envs: int,
                 device: torch.device):
        self.cfg, self.dev = cfg, device
        self.rho = air_density  # air density [kg/m^3]
        
        # Efficiency gain: relates wake inflow directly to thrust ratio, sampled per env
        self.k = torch.full((num_envs,), sum(cfg.efficiency_gain) / 2, device=device)

    def resample(self, ids):
        # Resample the efficency gain
        self.k[ids] = uniform(self.cfg.efficiency_gain, len(ids), self.dev)

    def episode_params(self):
        # Return this episode's efficiency setting
        return (self.k / (sum(self.cfg.efficiency_gain) / 2))[:, None]

    def thrust_ratio(self, inflow, params: DroneParams, omega):
        # Determine how fast is the wake relative to the hovering jet of the bottom drone
        v_h = params.hover_induced_velocity(self.rho)
        return (1.0 - self.k[:, None] * inflow / v_h).clamp(self.cfg.efficiency_min, 1.0)


class BEMTRotor(RotorModel):
    """Option B. Blade-element momentum theory, ideal twist, uniform inflow, at the actual
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
        self.cfg, self.dev = cfg, device
        self.rho = air_density # air density [kg/m^3]
        
        # Solidity: blade area / disk area, sampled per env
        self.sigma = torch.full((num_envs,), sum(cfg.solidity) / 2, device=device)

    def resample(self, ids):
        # Resample per-episode random parameters for env ids.
        self.sigma[ids] = uniform(self.cfg.solidity, len(ids), self.dev)

    def episode_params(self):
        # Return this episode's solidity setting
        return (self.sigma / (sum(self.cfg.solidity) / 2))[:, None]

    def thrust_ratio(self, inflow, params: DroneParams, omega):
        R = params.prop_radius                                            # Prop radius
        ct0 = params.k_f / (self.rho * math.pi * R**4)                    # Hover thrust coefficient
        sa = (self.sigma * self.cfg.lift_slope)[:, None]                  # Solidity * lift slope (N, 1)
        lam_h = math.sqrt(ct0 / 2)                                        # Hover inflow ratio: inflow / blade tip speed
        theta = 4 * ct0 / sa + lam_h                                      # Fitted pitch angle of prop
        lam_c = inflow / (omega * R).clamp_min(1.0)                       # Inflow ratio in flight
        
        # Solving balde element and momentum
        b = sa / 16 - lam_c / 2
        lam = torch.sqrt(b * b + sa * theta / 8) - b
        return ((theta - lam) / (theta - lam_h)).clamp(0.0, 1.5)
