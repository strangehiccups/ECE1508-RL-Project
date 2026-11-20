"""Batched rotor model: quadratic thrust curve, with the optional rotor lag (MotorCfg.rotor_lag).

    w_cmd clamped to [w_idle, w_max] 
    rotor_lag = False:  w = w_cmd
    rotor_lag = True:   w[k+1] = w_cmd + (w[k] - w_cmd) exp(-lambda dt)  (exact ZOH)
    f = k_f * motor_factor * w^2                        thrust along rotor +z
    yaw reaction torque = kappa * f * rot_dir           (MJCF actuator gear)

With the lag, w is updated once per PHYSICS step, as in Pegasus: the env precomputes
the rotor speeds of all `decimation` substeps of a control step (the command is held)
and MuJoCo replays them one substep at a time.
"""

import torch

from ..configs import MotorCfg
from .drone import DroneParams


class LaggedQuadraticMotors:
    def __init__(self, cfg: MotorCfg, params: DroneParams, num_envs: int,
                 device: torch.device):
        self.cfg, self.p = cfg, params
        self.omega = torch.full((num_envs, 4), params.omega_hover, device=device)

    def reset(self, ids: torch.Tensor, omega0: float | torch.Tensor) -> None:
        self.omega[ids] = omega0

    def rollout(self, omega_cmd: torch.Tensor, dt: float, substeps: int) -> torch.Tensor:
        """Advance `substeps` physics steps of length dt with the command held.
        Fills in the rotor speeds for each physics step in Mujoco. Save the last rotor speed. """
        cmd = omega_cmd.clamp(self.cfg.omega_idle, self.cfg.omega_max)

        # Quadratic thrust curve, no lag
        if not self.cfg.rotor_lag:                  
            self.omega = cmd    # Save the last rotor speed
            return cmd.expand(substeps, *cmd.shape)

        # Lagged rotor model
        decay = torch.exp(-self.p.rotor_lambda * dt)[:, None]
        w = self.omega
        seq = []
        for _ in range(substeps):
            w = (cmd + (w - cmd) * decay).clamp(self.cfg.omega_idle, self.cfg.omega_max)
            seq.append(w)
        self.omega = w    # Save the last rotor speed
        return torch.stack(seq)

    def thrust(self, omega: torch.Tensor) -> torch.Tensor:
        """Rotor thrust [N] for rotor speeds (..., N, 4)."""
        return self.p.k_f * self.p.motor_factor * omega * omega
