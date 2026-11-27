"""Why does a Phase 2 policy crash? pi on z_hat ("zhat") and pi on the true z ("ztrue") fly
the same seeded episodes (sensors on, all scenarios equally likely, one episode per env).
Prints crash reasons, the pairing (crashes that the true z avoids), crash % per sensor
delay of the bottom drone, crash timing, crash % per episode-factor quintile, and the
flips' last seconds (oscillation, saturation, wake, z_hat error).
Output: <out>/crash_diag.npz (per-env factors, crash info, last 1.6 s before each crash).

Run from code/:
    python -m ProxQuad.scripts.crash_diag --checkpoint runs/phase2_hifi_z8_circles/model.pt
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from ..configs import EnvCfg
from ..envs import ProxQuadEnv
from ..envs.scenarios import SCENARIOS
from ..eval import Phase1Policy, Phase2Policy, load_phase2, match_obs

SIG = ["tilt", "wx", "wy", "wz", "sat", "thrust_ratio", "z_mse", "dxy", "dz"]
WIN = 400   # steps kept before each crash (1.6 s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Phase 2 checkpoint")
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output dir (default: <checkpoint dir>/crash_diag)")
    return p.parse_args()


@torch.no_grad()
def fly(env: ProxQuadEnv, ctrl, seed: int) -> dict[str, np.ndarray]:
    torch.manual_seed(seed)
    obs = env.reset()
    ctrl.reset()
    N, T, dev = env.n, env.max_steps, env.dev
    buf = torch.zeros(WIN, N, len(SIG), device=dev)
    snap = torch.zeros(WIN, N, len(SIG), device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    crashed = torch.zeros(N, dtype=torch.bool, device=dev)
    t_end = torch.full((N,), T, device=dev)
    zsum = torch.zeros(N, device=dev)
    reasons = {}
    start = {"priv": obs["priv"].clone(), "kind": env.scenario.kind.clone(),
             "delay_b": env.sensors.delay["bottom"].clone(),
             "delay_t": env.sensors.delay["top"].clone()}
    for t in range(T):
        bot = env._bot_state
        a = ctrl.act(env, obs)
        zm = ctrl.extras().get("z_mse", torch.full((N,), float("nan"), device=dev))
        d = env._top_state.pos - bot.pos
        buf[t % WIN] = torch.stack([
            torch.rad2deg(torch.acos(bot.R[:, 2, 2].clamp(-1, 1))), *bot.omega.T,
            (a.abs() >= 1).float().mean(-1), env._aero.thrust_ratio.min(-1).values, zm,
            d[:, :2].norm(dim=-1), d[:, 2]], -1)
        zsum += torch.where(alive, zm.nan_to_num(0), 0)
        obs, _, done, info = env.step(a)
        first = done & alive
        if first.any():
            snap[:, first] = buf[(torch.arange(WIN, device=dev) + t + 1) % WIN][:, first]
            t_end[first] = t + 1
        crashed |= first & info["crashed"]
        for k, v in info["crash_reasons"].items():
            reasons[k] = reasons.get(k, torch.zeros_like(alive)) | (first & v)
        alive &= ~done
    out = {"crashed": crashed, "t_end": t_end, "snap": snap, "zmse_mean": zsum / t_end,
           **{f"r_{k}": v for k, v in reasons.items()}, **start}
    return {k: v.cpu().numpy() for k, v in out.items()}


def factors(priv: np.ndarray, layout, delay_b, delay_t) -> dict[str, np.ndarray]:
    F, off = {}, 0
    for k, d in layout:
        v = priv[:, off:off + d]
        off += d
        if k in ("rotor_thrust_ratio", "wake_drag", "sensor_delay"):
            continue                      # per-step, or the delay itself (added below)
        if d == 1:
            F[k] = v[:, 0]
        elif k == "motor_factor":
            F["motor_min"], F["motor_spread"] = v.min(1), v.max(1) - v.min(1)
        elif k == "inertia_ratio":
            F["inertia_xy"], F["inertia_z"] = v[:, 0], v[:, 1]
        else:
            F[k + "_norm"] = np.linalg.norm(v, axis=1)
    F["delay_bottom"], F["delay_top"] = delay_b, delay_t
    return F


def report(R: dict, F: dict, dt: float) -> None:
    a, b = R["zhat"]["crashed"], R["ztrue"]["crashed"]
    for c in ("zhat", "ztrue"):
        r = R[c]
        print(f"{c}: crash {100 * r['crashed'].mean():.2f} %  " + ", ".join(
            f"{k[2:]} {100 * r[k].mean():.2f}" for k in r if k.startswith("r_") and r[k].any()))
    print(f"paired: both {np.sum(a & b)}, zhat only {np.sum(a & ~b)}, ztrue only {np.sum(~a & b)}"
          f" of {len(a)}")

    db = F["delay_bottom"]
    print("\ncrash % per bottom sensor delay [steps] (zhat / ztrue): " + "; ".join(
        f"{int(u)}: {100 * a[db == u].mean():.1f} / {100 * b[db == u].mean():.1f}"
        for u in np.unique(db)))
    te = R["zhat"]["t_end"] * dt
    edges = [0, 1, 2, 3, 5, 10, 15.01]
    print(f"crash time [s] {edges[:-1]}: zhat {np.histogram(te[a], edges)[0].tolist()}, "
          f"ztrue {np.histogram(R['ztrue']['t_end'][b] * dt, edges)[0].tolist()}")
    kind = R["zhat"]["kind"]
    print("crash % per scenario (zhat / ztrue): " + "; ".join(
        f"{s} {100 * a[kind == i].mean():.1f}/{100 * b[kind == i].mean():.1f}"
        for i, s in enumerate(SCENARIOS)))

    rows = []
    for k, v in F.items():
        if k.startswith("delay") or np.ptp(v) == 0:
            continue
        q = np.quantile(v, [0, .2, .4, .6, .8, 1])
        bins = np.clip(np.searchsorted(q, v, side="right") - 1, 0, 4)
        r = [100 * a[bins == j].mean() for j in range(5)]
        rows.append((max(r) - min(r), k, r, [100 * b[bins == j].mean() for j in range(5)], q))
    print("\ncrash % per factor quintile, low -> high (zhat | ztrue), largest spread first:")
    for _, k, r, r2, q in sorted(rows, reverse=True)[:6]:
        print(f"  {k:16s} " + " ".join(f"{x:5.1f}" for x in r) + "  | " +
              " ".join(f"{x:4.1f}" for x in r2) + f"   ({q[0]:.2f} .. {q[-1]:.2f})")
    if "rotor_lambda" in F:
        slow = F["rotor_lambda"] < np.quantile(F["rotor_lambda"], 1 / 3)
        print("  slowest third of rotor_lambda vs rest, per bottom delay (zhat crash %): " +
              "; ".join(f"{int(u)}: {100 * a[(db == u) & slow].mean():.1f} vs "
                        f"{100 * a[(db == u) & ~slow].mean():.1f}" for u in np.unique(db)))

    S = {k: i for i, k in enumerate(SIG)}
    snap, flips = R["zhat"]["snap"], np.where(a & R["zhat"]["r_flipped"])[0]
    if len(flips) == 0:
        return
    st = []
    for i in flips:
        v = snap[WIN - min(WIN, int(R["zhat"]["t_end"][i])):, i]     # valid steps only
        b20 = np.where(v[:, S["tilt"]] < 20)[0]
        j = b20[-1] if len(b20) else 0                              # tilt leaves 20 deg
        w = v[max(0, j - round(1 / dt)):j + 1]                        # the second before
        wxy = w[:, [S["wx"], S["wy"]]]
        q = max(1, len(w) // 4)
        st.append([(len(v) - 1 - j) * dt,
                   (np.diff(np.sign(wxy), axis=0) != 0).sum(0).max() / (2 * len(w) * dt),
                   np.sqrt((wxy[-q:] ** 2).mean()) / max(np.sqrt((wxy[:q] ** 2).mean()), 1e-6),
                   w[:, S["sat"]].mean(), v[-q:, S["sat"]].mean(),
                   w[:, S["thrust_ratio"]].min(), np.nanmean(w[:, S["z_mse"]]), v[j, S["dxy"]]])
    st = np.array(st)
    med = np.nanmedian(st, 0)
    zs = np.median(R["zhat"]["zmse_mean"][~a])
    print(f"\nflips: {len(flips)}. Median: tilt 20 deg -> flip {med[0]:.2f} s; roll/pitch rate "
          f"oscillation {med[1]:.1f} Hz, growing x{med[2]:.1f} over the second before "
          f"({100 * (st[:, 2] > 2).mean():.0f} % of flips > x2); motors at the limit "
          f"{100 * med[3]:.0f} % of the second before, {100 * med[4]:.0f} % at the end; "
          f"in the wake (min rotor thrust ratio < 0.9) {100 * (st[:, 5] < 0.9).mean():.0f} %; "
          f"|z_hat - z|^2 {med[6]:.0f} (survivors' episode mean {zs:.1f}); horizontal "
          f"distance to the top drone {med[7]:.2f} m")
    for nm, m in (("before 3 s", te[flips] < 3), ("after 3 s", te[flips] >= 3)):
        if m.any():
            print(f"  {nm}: {m.sum()} flips, in the wake {100 * (st[m, 5] < 0.9).mean():.0f} %, "
                  f"true z survives {100 * (~b[flips][m]).mean():.0f} %, bottom delay 6+ "
                  f"{100 * (db[flips][m] >= 6).mean():.0f} %")


def main() -> None:
    args = parse_args()
    cfg = EnvCfg()
    cfg.sim.num_envs = args.envs
    cfg.sim.stagger_first_episode = False
    cfg.scenario.top.scenario_probs = {s: 1.0 / len(SCENARIOS) for s in SCENARIOS}
    cfg.sensors.enabled = True
    match_obs(cfg.obs, args.checkpoint)
    env = ProxQuadEnv(cfg)
    enc, pi, phi, dec = load_phase2(args.checkpoint, env)
    R = {"zhat": fly(env, Phase2Policy(enc, pi, phi, dec), args.seed),
         "ztrue": fly(env, Phase1Policy(enc, pi, name="z=true"), args.seed)}
    F = factors(R["zhat"]["priv"], env.priv_layout, R["zhat"]["delay_b"], R["zhat"]["delay_t"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty bins / windows
        report(R, F, env.dt)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "crash_diag"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "crash_diag.npz", sig=np.array(SIG), dt=env.dt,
             **{f"{c}_{k}": v for c, r in R.items() for k, v in r.items()})
    print(f"\nwrote {out}/crash_diag.npz")


if __name__ == "__main__":
    main()
