"""Tabulate the downwash model on a geometry grid, and time each model variant.

Run from code/:  python -m ProxQuad.scripts.inspect_downwash

Top X650 hovers level at the origin at hover thrust; the bottom X650 hovers level at
(dx, 0, -dz). Random parameters are fixed at the middle of their ranges.
Reported per cell: "lift loss" = vertical force lost, in % of the bottom drone's weight
(rotor thrust loss + downward drag), and the tilt torque from uneven rotor loss.
Use the table to calibrate against measured data before training.
"""

import copy
import itertools
import time

import torch

from ..configs import AeroCfg, EnvCfg, ObsCfg, SimCfg
from ..envs.aero import DownwashModel
from ..envs.drone import DroneState
from ..envs.prox_quad_env import ProxQuadEnv
from ..utils.math import quat_to_rotmat

DZ = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
DX = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]


def state(pos: torch.Tensor) -> DroneState:
    n = pos.shape[0]
    q = torch.zeros(n, 4, device=pos.device)
    q[:, 0] = 1.0
    z = torch.zeros(n, 3, device=pos.device)
    return DroneState(pos=pos, quat=q, R=quat_to_rotmat(q), vel=z, omega=z.clone())


def build(n: int, device, sources: str, rotor: str) -> DownwashModel:
    cfg = AeroCfg()
    cfg.wake.sources, cfg.rotor.model = sources, rotor
    cfg.turb_force_std = cfg.turb_torque_std = 0.0
    cfg.transport_delay = False            # steady field: the delay does not change it
    return DownwashModel(cfg, n, device, 0.004)   # random params start at range midpoints


def nominal(p, k: int):
    """Copy of DroneParams with k envs and all random factors at 1."""
    q = copy.copy(p)
    q.n = k
    q.mass_factor = torch.ones(k, device=p.dev)
    q.inertia_factor = torch.ones(k, 2, device=p.dev)
    q.motor_factor = torch.ones(k, 4, device=p.dev)
    q.com_offset = torch.zeros(k, 3, device=p.dev)
    return q


def table(env: ProxQuadEnv, sources: str, rotor: str) -> None:
    grid = list(itertools.product(DZ, DX))
    k = len(grid)
    model = build(k, env.dev, sources, rotor)
    bp, tp = nominal(env.bottom, k), nominal(env.top, k)
    top_pos = torch.zeros(k, 3, device=env.dev)
    bot_pos = torch.tensor([[dx, 0.0, -dz] for dz, dx in grid], device=env.dev)
    hover_w = torch.full((k, 4), bp.omega_hover, device=env.dev)
    out = model.compute(state(bot_pos), bp, hover_w, state(top_pos), tp,
                        torch.full((k, 4), tp.f_hover, device=env.dev), env.dt)
    f = out.thrust_ratio * bp.f_hover
    lift = (bp.f_hover - f).sum(-1) - out.drag[:, 2]
    loss = (100 * lift / bp.weight_nom).reshape(len(DZ), len(DX))
    wrench = f @ (bp.alloc / bp.k_f).T                      # [T, tau_x, tau_y, tau_z]
    tilt = wrench[:, 1:3].norm(dim=-1).reshape(len(DZ), len(DX))

    print(f"\n== wake sources={sources}, rotor={rotor}")
    print("   lift loss [% of weight]                 | tilt torque from uneven rotors [N m]")
    hdr = "".join(f"{dx:6.2f}" for dx in DX)
    print(f"   dz\\dx {hdr} | {hdr}")
    for i, dz in enumerate(DZ):
        a = "".join(f"{v:6.1f}" for v in loss[i].tolist())
        b = "".join(f"{v:6.2f}" for v in tilt[i].tolist())
        print(f"   {dz:5.2f} {a} | {b}")


def benchmark(env: ProxQuadEnv, sources: str, rotor: str, reps: int = 200) -> float:
    model = build(env.n, env.dev, sources, rotor)
    args = (env._bot_state, env.bottom, env.bot_motors.omega, env._top_state, env.top,
            env.top_motors.thrust(env.top_motors.omega), env.dt)
    for _ in range(10):
        model.compute(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model.compute(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    n = 4096
    env = ProxQuadEnv(EnvCfg(sim=SimCfg(num_envs=n), obs=ObsCfg()))
    print(f"X650: weight {env.bottom.weight_nom:.2f} N, arm {env.bottom.arm:.3f} m, "
          f"prop radius {env.bottom.prop_radius:.4f} m, hover induced velocity "
          f"{env.bottom.hover_induced_velocity(1.225):.2f} m/s")
    tiers = [("rotors", "efficiency"), ("body", "efficiency"), ("rotors", "bemt"),
             ("body", "bemt")]
    for sources, rotor in tiers:
        table(env, sources, rotor)

    zero = torch.full((n, 4), env.a_hover, device=env.dev)
    for _ in range(10):
        env.step(zero)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        env._simulate()
    physics = (time.perf_counter() - t0) / 100
    print(f"\n== cost per control step, {n} envs")
    print(f"   MuJoCo physics ({env.cfg.sim.decimation} substeps): {1e3 * physics:.3f} ms")
    for sources, rotor in tiers:
        print(f"   downwash sources={sources:6s} rotor={rotor:10s}: "
              f"{1e3 * benchmark(env, sources, rotor):.3f} ms")


if __name__ == "__main__":
    main()
