"""Geometric tracking controller (Lee et al. 2010), batched over envs (torch).
Flies the scripted top drone. Uses nominal parameters only.
"""

import math
from dataclasses import dataclass

import torch

from ..configs import ControllerCfg
from ..utils.math import normalize_with_rate, vee
from .drone import DroneParams, DroneState


@dataclass
class FlatRef:
    """Flat-output reference, batched over envs. Terms left as None count as zero."""
    pos: torch.Tensor                     # (N, 3) m
    vel: torch.Tensor                     # (N, 3) m/s
    acc: torch.Tensor | None = None       # (N, 3) m/s^2  feedforward in F_d
    jerk: torch.Tensor | None = None      # (N, 3) m/s^3  F_d_dot ~ m jerk -> Omega_d
    yaw: torch.Tensor | None = None       # (N,)   rad
    yaw_dot: torch.Tensor | None = None   # (N,)   rad/s  -> Omega_d


class GeometricController:
    def __init__(self, cfg: ControllerCfg, params: DroneParams, dt: float, num_envs: int,
                 device: torch.device):
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.kp, self.kd, self.ki = t(cfg.kp_pos), t(cfg.kd_pos), t(cfg.ki_pos)
        self.k_R, self.k_omega = t(cfg.k_R), t(cfg.k_omega)
        self.int_limit = cfg.int_limit
        self.tan_tilt = math.tan(math.radians(cfg.max_tilt_deg))
        self.p, self.dt = params, dt
        self.e3 = t([0.0, 0.0, 1.0])
        self.integral = torch.zeros(num_envs, 3, device=device)
        self.omega_max_sq = params.motor.omega_max ** 2
        self.J = torch.diag(params.inertia_nom)                    # (3, 3) nominal inertia

    def reset(self, ids: torch.Tensor) -> None:
        self.integral[ids] = 0.0

    def __call__(self, s: DroneState, ref: FlatRef) -> torch.Tensor:
        """Returns rotor speed commands (N, 4) [rad/s]. Call once per control step
        (advances the integrator)."""
        p = self.p
        # ---- position loop -> desired force (inertial frame)
        e_p, e_v = ref.pos - s.pos, ref.vel - s.vel
        self.integral = (self.integral + e_p * self.dt).clamp(-self.int_limit, self.int_limit)
        acc = self.kp * e_p + self.kd * e_v + self.ki * self.integral
        if ref.acc is not None:
            acc = acc + ref.acc
        F_raw = p.mass_nom * (acc + p.g * self.e3)
        Fz_min = 0.2 * p.weight_nom
        Fz = F_raw[:, 2].clamp(min=Fz_min)                         # keep pushing up
        Fxy = F_raw[:, :2]
        scale = (Fz * self.tan_tilt / Fxy.norm(dim=-1).clamp_min(1e-9)).clamp(max=1.0)
        F = torch.cat([Fxy * scale[:, None], Fz[:, None]], dim=-1)  # within tilt limit
        saturated = (F_raw[:, 2] < Fz_min) | (scale < 1.0)

        # ---- desired attitude R_d and its body rate Omega_d
        F_dot = p.mass_nom * ref.jerk if ref.jerk is not None else None
        R_d, w_d = self.desired_attitude(F, F_dot, ref.yaw, ref.yaw_dot)
        # F_dot is the derivative of the unclipped F_d: invalid while clipping is active
        w_d = torch.where(saturated[:, None], torch.zeros_like(w_d), w_d)

        # ---- thrust: desired force projected on the current body z axis
        T = (F * s.R[:, :, 2]).sum(-1).clamp(min=0.0)              # positive thrust only

        # ---- attitude loop (Lee eq. 16, Omega_dot_d = 0):
        #      tau = -k_R e_R - k_w e_w + w x J w - J (w x R^T R_d w_d)
        R, w = s.R, s.omega
        RtRd = R.transpose(1, 2) @ R_d
        e_R = 0.5 * vee(RtRd.transpose(1, 2) - RtRd)               # 0.5 vee(R_d^T R - R^T R_d)
        w_d_body = (RtRd @ w_d[..., None]).squeeze(-1)             # R^T R_d w_d
        e_w = w - w_d_body
        Jw = w @ self.J.T                                          # (N, 3) = J w per env
        tau = (-self.k_R * e_R - self.k_omega * e_w + torch.cross(w, Jw, dim=-1)
               - torch.cross(w, w_d_body, dim=-1) @ self.J.T)

        # ---- control allocation: [F, tau] -> w^2 -> clip -> w
        w_sq = (torch.cat([T[:, None], tau], dim=-1) @ p.alloc_pinv.T).clamp(min=0.0)
        peak = w_sq.max(dim=-1, keepdim=True).values
        w_sq = w_sq / (peak / self.omega_max_sq).clamp(min=1.0)   # clip the maximum w^2
        return torch.sqrt(w_sq)

    @staticmethod
    def desired_attitude(F: torch.Tensor, F_dot: torch.Tensor | None,
                         yaw: torch.Tensor | None, yaw_dot: torch.Tensor | None
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """R_d = [x_d, y_d, z_d] from the thrust direction and heading (Lee b1c
        convention), and Omega_d = vee(R_d^T R_d_dot) in the desired body frame.
        F (N, 3), F_dot (N, 3) or None, yaw / yaw_dot (N,) or None (= 0)."""
        zero = torch.zeros_like(F[:, 0])
        yaw = zero if yaw is None else yaw
        yaw_dot = zero if yaw_dot is None else yaw_dot
        F_dot = torch.zeros_like(F) if F_dot is None else F_dot  # Desired force derivative
        c, s = torch.cos(yaw), torch.sin(yaw)

        # Build desired body axes and their derivatives from the desired force and yaw.
        z_d, z_d_dot = normalize_with_rate(F, F_dot)
        x_c = torch.stack([c, s, zero], dim=-1)
        x_c_dot = yaw_dot[:, None] * torch.stack([-s, c, zero], dim=-1)
        C = torch.cross(z_d, x_c, dim=-1)
        C_dot = torch.cross(z_d_dot, x_c, dim=-1) + torch.cross(z_d, x_c_dot, dim=-1)
        y_d, y_d_dot = normalize_with_rate(C, C_dot)
        x_d = torch.cross(y_d, z_d, dim=-1)
        x_d_dot = torch.cross(y_d_dot, z_d, dim=-1) + torch.cross(y_d, z_d_dot, dim=-1)

        # Desired rotation matrix and body rates
        R_d = torch.stack([x_d, y_d, z_d], dim=-1)
        R_d_dot = torch.stack([x_d_dot, y_d_dot, z_d_dot], dim=-1)
        W = R_d.transpose(1, 2) @ R_d_dot                          # skew-symmetric
        w_d = vee(0.5 * (W - W.transpose(1, 2)))                   # symmetrize round-off
        return R_d, w_d


def flat_attitude(ref: FlatRef, g: float) -> tuple[torch.Tensor, torch.Tensor]:
    """R_ref (N, 3, 3) and w_ref (N, 3) implied by a flat-output reference."""
    up = torch.zeros_like(ref.pos)
    up[:, 2] = g
    acc = up if ref.acc is None else ref.acc + up
    return GeometricController.desired_attitude(acc, ref.jerk, ref.yaw, ref.yaw_dot)
