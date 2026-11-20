"""Interfaces of the aerodynamic-interaction models.

The downwash on the bottom drone is split into two independent, swappable parts:

  WakeModel   "what air velocity does the top drone produce at point p?"
              (top state, top rotor thrusts, query points) -> velocity (N, P, 3)
  RotorModel  "how much thrust does a bottom rotor make under that inflow?"
              (axial inflow per rotor, commanded thrust) -> thrust ratio (N, 4)

DownwashModel (downwash.py) composes them, adds airframe drag and turbulence, and
returns an AeroOutput. Higher-fidelity tiers (README "Downwash fidelity") plug in by
implementing one of these two interfaces; the env does not change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ..drone import DroneParams, DroneState


@dataclass
class AeroOutput:
    """Effect of the top drone on the bottom drone over one control interval."""
    thrust_ratio: torch.Tensor  # (N, 4) thrust in wake / still-air thrust, per rotor
    force: torch.Tensor         # (N, 3) world-frame force, drag + turbulence [N]
    torque: torch.Tensor        # (N, 3) world-frame torque, turbulence [N m]
    rotor_inflow: torch.Tensor  # (N, 4) axial wake speed through each bottom rotor [m/s]
    com_wake: torch.Tensor      # (N, 3) wake velocity at the bottom COM [m/s]
    drag: torch.Tensor          # (N, 3) deterministic part of `force`: airframe drag [N]

    @staticmethod
    def none(n: int, device) -> "AeroOutput":
        z = lambda *s: torch.zeros(n, *s, device=device)  # noqa: E731
        return AeroOutput(torch.ones(n, 4, device=device), z(3), z(3), z(4), z(3), z(3))


class WakeModel(ABC):
    def resample(self, ids: torch.Tensor) -> None:
        """Draw per-episode random parameters for envs `ids` (default: none)."""

    def episode_params(self) -> torch.Tensor:
        """(N, 2) per-episode random parameters [speed gain, spread rate], each divided by
        its range midpoint (privileged info)."""
        raise NotImplementedError

    @abstractmethod
    def velocity(self, points: torch.Tensor, top: DroneState, top_params: DroneParams,
                 top_thrust: torch.Tensor) -> torch.Tensor:
        """points (N, P, 3) world; top_thrust (N, 4) [N] -> wake velocity (N, P, 3)."""


class RotorModel(ABC):
    def resample(self, ids: torch.Tensor) -> None:
        """Draw per-episode random parameters for envs `ids` (default: none)."""

    def episode_params(self) -> torch.Tensor:
        """(N, 1) the per-episode random coefficient divided by its range midpoint."""
        raise NotImplementedError

    @abstractmethod
    def thrust_ratio(self, inflow: torch.Tensor, params: DroneParams,
                     omega: torch.Tensor) -> torch.Tensor:
        """inflow (..., N, 4): axial wake speed through each rotor disk, positive =
        downward through the disk [m/s]. omega (..., N, 4): actual rotor speed [rad/s].
        Returns thrust in the wake / thrust in still air at the same speed (..., N, 4)."""
