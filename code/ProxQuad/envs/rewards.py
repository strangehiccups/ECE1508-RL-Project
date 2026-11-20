"""Reward and termination terms for the bottom drone."""

import torch

from ..configs import RewardCfg, TerminationCfg
from .drone import DroneState


def compute_terminations(bot: DroneState, top: DroneState, pos_err: torch.Tensor,
                         cfg: TerminationCfg) -> tuple[torch.Tensor, dict]:
    reasons = {
        "ground": bot.pos[:, 2] < cfg.min_height,
        "flipped": bot.R[:, 2, 2] < cfg.min_upright,
        "lost": pos_err.norm(dim=-1) > cfg.max_pos_err,
        "collision": _inside_ellipsoid(top.pos - bot.pos, cfg.collision_radii),
        "nan": ~(torch.isfinite(bot.pos).all(-1) & torch.isfinite(bot.vel).all(-1)),
    }
    crashed = torch.stack(list(reasons.values())).any(0)
    return crashed, reasons


def _inside_ellipsoid(d: torch.Tensor, radii: tuple[float, float]) -> torch.Tensor:
    r_xy, r_z = radii
    return (d[:, :2] / r_xy).pow(2).sum(-1) + (d[:, 2] / r_z).pow(2) < 1.0


def compute_reward(err: dict[str, torch.Tensor], action: torch.Tensor,
                   prev_action: torch.Tensor, a_hover: float, crashed: torch.Tensor,
                   cfg: RewardCfg, dt: float) -> tuple[torch.Tensor, dict]:
    """Tracking reward on env._tracking_errors. Returns (N,) and the terms.

    Rate terms (weights per second) are integrated over the control step dt, so returns
    do not depend on the control rate. action_rate: |da/dt| x dt = |a - a_prev|; crash:
    one-off event (see RewardCfg)."""
    trace = err["rot_err"].diagonal(dim1=1, dim2=2).sum(-1)
    per_second = {
        "survival": torch.full_like(trace, cfg.survival),
        "pos": -cfg.pos * err["pos_err"].norm(dim=-1),
        "vel": -cfg.vel * err["vel_err"].norm(dim=-1),
        "att": -cfg.att * (3.0 - trace),
        "ang_vel": -cfg.ang_vel * err["ang_vel_err"].norm(dim=-1),
        "action": -cfg.action * (action - a_hover).norm(dim=-1),
    }
    terms = {k: v * dt for k, v in per_second.items()}
    terms["action_rate"] = -cfg.action_rate * (action - prev_action).norm(dim=-1)
    terms["crash"] = -cfg.crash * crashed.float()
    reward = torch.stack(list(terms.values())).sum(0)
    return torch.nan_to_num(reward, nan=-cfg.crash), terms
