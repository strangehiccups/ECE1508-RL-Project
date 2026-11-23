"""One full episode per env with a given controller, recording per-step signals.

Reseeding before env.reset() gives every controller the same episodes (scenario, paths,
randomization, gusts, start states): comparisons are paired. The env auto-resets on done,
so `alive` marks the steps of each env's first episode. Signals are those of s_t, read
before env.step() (afterwards a finished env already holds its next episode).
Memory: T x N x 4 B per signal, ~60 MB at 4096 envs x 15 s; moved to the CPU at the end.
"""

import math
from dataclasses import dataclass, field

import torch

from ..envs import ProxQuadEnv
from .controllers import Controller

SIGNALS = (
    "pos_err",       # |p - p_ref| [m]
    "z_err",         # p_z - p_ref,z [m], signed
    "vel_err",       # |v - v_ref| [m/s]
    "ang_vel_err",   # |omega - omega_ref| [rad/s]
    "tilt_deg",      # angle between body z and reference z
    "yaw_deg",       # heading error about the reference z, signed
    "act_rate",      # |a_t - a_{t-1}|, clamped actions
    "sat",           # share of motors at the command limit (|a| >= 1)
    "thrust_ratio",  # min over the 4 bottom rotors: thrust in wake / still air
    "thrust_loss",   # mean over the 4 bottom rotors of 1 - thrust ratio
    "collective",    # commanded sum w^2 / (4 w_hover^2), clamped action
    "dz",            # top - bottom height [m]
    "dxy",           # horizontal distance between the drones [m]
    "top_err",       # top drone: |p - p_ref| [m]
    "reward",
)


@dataclass
class Recording:
    name: str
    signals: dict[str, torch.Tensor]   # (T, N)
    alive: torch.Tensor                # (T, N) bool: step belongs to the first episode
    kind: torch.Tensor                 # (N,) scenario index
    crashed: torch.Tensor              # (N,) bool
    reasons: dict[str, torch.Tensor]   # (N,) bool, crash reason at the first done
    dt: float
    collision_radii: tuple[float, float]   # (r_xy, r_z) of the collision ellipsoid [m]
    cache: dict = field(default_factory=dict, repr=False)   # derived results (metrics)


@torch.no_grad()
def rollout(env: ProxQuadEnv, ctrl: Controller, seed: int, steps: int | None = None) -> Recording:
    T, N, dev = steps or env.max_steps, env.n, env.dev
    torch.manual_seed(seed)
    obs = env.reset()
    ctrl.reset()
    sig = {k: torch.empty(T, N, device=dev) for k in SIGNALS}
    alive_T = torch.empty(T, N, dtype=torch.bool, device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    crashed = torch.zeros(N, dtype=torch.bool, device=dev)
    reasons: dict[str, torch.Tensor] = {}
    kind = env.scenario.kind.clone()
    prev = obs["prev_action"].clone()
    w_hover_sq = 4 * env.bottom.omega_hover ** 2

    for t in range(T):
        bot, top = env._bot_state, env._top_state
        err = env._tracking_errors(bot)
        E = err["rot_err"]                                   # R_ref^T R
        a = ctrl.act(env, obs)
        ac = a.clamp(-1.0, 1.0)
        sig["pos_err"][t] = err["pos_err"].norm(dim=-1)
        sig["z_err"][t] = err["pos_err"][:, 2]
        sig["vel_err"][t] = err["vel_err"].norm(dim=-1)
        sig["ang_vel_err"][t] = err["ang_vel_err"].norm(dim=-1)
        sig["tilt_deg"][t] = torch.rad2deg(torch.acos(E[:, 2, 2].clamp(-1.0, 1.0)))
        sig["yaw_deg"][t] = torch.rad2deg(torch.atan2(E[:, 1, 0], E[:, 0, 0]))
        sig["act_rate"][t] = (ac - prev).norm(dim=-1)
        sig["sat"][t] = (a.abs() >= 1.0).float().mean(-1)
        sig["thrust_ratio"][t] = env._aero.thrust_ratio.min(-1).values
        sig["thrust_loss"][t] = (1.0 - env._aero.thrust_ratio).mean(-1)
        sig["collective"][t] = env.action_to_omega(ac).pow(2).sum(-1) / w_hover_sq
        d = top.pos - bot.pos
        sig["dz"][t], sig["dxy"][t] = d[:, 2], d[:, :2].norm(dim=-1)
        top_ref = env.scenario.top_reference(env.steps.float() * env.dt).pos
        sig["top_err"][t] = (top.pos - top_ref).norm(dim=-1)
        for k, v in ctrl.extras().items():
            sig.setdefault(k, torch.full((T, N), math.nan, device=dev))[t] = v

        obs, reward, done, info = env.step(a)
        sig["reward"][t] = reward
        alive_T[t] = alive
        first = done & alive
        crashed |= first & info["crashed"]
        for k, v in info["crash_reasons"].items():
            reasons[k] = reasons.get(k, torch.zeros_like(alive)) | (first & v)
        alive &= ~done
        prev = ac

    cpu = lambda x: x.cpu()  # noqa: E731
    return Recording(ctrl.name, {k: cpu(v) for k, v in sig.items()}, cpu(alive_T), cpu(kind),
                     cpu(crashed), {k: cpu(v) for k, v in reasons.items()}, env.dt,
                     tuple(env.cfg.termination.collision_radii))
