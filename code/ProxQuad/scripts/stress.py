"""Stress tests of a Phase 2 checkpoint against geometric+UDE: named conditions x
controllers, every controller on the same seeded episodes (one per env, the chosen
scenarios equally likely). Controllers: pi on z_hat ("policy"), the same pi on the true
z ("z=true"), and geometric+UDE at each filter pair in --ude-taus; "UDE best" is the pair
with the lowest position RMS over all scenarios in that condition.
Output: console tables, <out>/stress.json ({condition: {controller: summarize()}}),
stress.csv. Results are written after every condition.

Run from code/:
    python -m ProxQuad.scripts.stress --checkpoint runs/phase2_circles_3000/model.pt
    python -m ProxQuad.scripts.stress --checkpoint ... --conditions nominal latency noise_x2
"""

import argparse
import csv
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..configs import EnvCfg
from ..envs import ProxQuadEnv
from ..envs.scenarios import SCENARIOS
from ..eval import (Phase1Policy, Phase2Policy, RobustGeometricBaseline, load_phase2, rollout,
                    summarize)


def widen(cfg: EnvCfg, s: float) -> None:
    """Randomization ranges x s around their centres; CoM offset and gust x s."""
    r = cfg.randomization
    w = lambda lo_hi: ((lo_hi[0] + lo_hi[1]) / 2 - s * (lo_hi[1] - lo_hi[0]) / 2,  # noqa: E731
                       (lo_hi[0] + lo_hi[1]) / 2 + s * (lo_hi[1] - lo_hi[0]) / 2)
    r.mass_factor, r.motor_factor = w(r.mass_factor), w(r.motor_factor)
    r.inertia_factor_xy, r.inertia_factor_z = w(r.inertia_factor_xy), w(r.inertia_factor_z)
    r.com_offset = tuple(s * c for c in r.com_offset)
    r.gust_force *= s


def gap(cfg: EnvCfg, lo: float, hi: float) -> None:
    cfg.scenario.top.dz = (lo, hi)
    cfg.scenario.start_clearance = min(cfg.scenario.start_clearance,
                                       0.99 * lo / cfg.termination.collision_radii[1])


def sensors(cfg: EnvCfg, scale: float) -> None:
    cfg.sensors.enabled, cfg.sensors.noise_scale = True, scale


def combined(cfg: EnvCfg) -> None:
    gap(cfg, 0.5, 0.8)
    cfg.aero.wake.speed_gain = (2.0, 3.0)
    widen(cfg, 1.5)


CONDITIONS = {
    "nominal": lambda c: None,
    "rand_x2": lambda c: widen(c, 2.0),
    "wake_gain_2.5-3": lambda c: setattr(c.aero.wake, "speed_gain", (2.5, 3.0)),
    "gap_0.38-0.45": lambda c: gap(c, 0.38, 0.45),
    "combined": combined,          # gap 0.5-0.8 m, wake gain 2-3, randomization x1.5
    "latency": lambda c: sensors(c, 0.0),
    "noise_x1": lambda c: sensors(c, 1.0),
    "noise_x2": lambda c: sensors(c, 2.0),
    "noise_x5": lambda c: sensors(c, 5.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True, help="Phase 2 checkpoint")
    p.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--ude-taus", nargs="+", default=["0.03/0.02", "0.05/0.03", "0.1/0.05", "0.2/0.1"],
                   help="UDE force/torque filter time constants [s]")
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settle-s", type=float, default=2.0)
    p.add_argument("--out", help="output dir (default: <checkpoint dir>/stress)")
    return p.parse_args()


def run_condition(args, name: str) -> dict[str, dict]:
    cfg = EnvCfg()
    cfg.sim.num_envs = args.envs
    cfg.sim.stagger_first_episode = False
    cfg.scenario.top.scenario_probs = {s: 1.0 / len(args.scenarios) for s in args.scenarios}
    obs2 = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"].env.obs
    cfg.obs.bottom_history_len, cfg.obs.top_history_len = (obs2.bottom_history_len,
                                                            obs2.top_history_len)
    CONDITIONS[name](cfg)
    env = ProxQuadEnv(cfg)
    encoder, pi, phi, dec = load_phase2(args.checkpoint, env)
    ctrls = [Phase2Policy(encoder, pi, phi, dec), Phase1Policy(encoder, pi, name="z=true")]
    for tau in args.ude_taus:
        tf, tt = (float(x) for x in tau.split("/"))
        ude_cfg = replace(cfg.robust_controller, ude_tau_force=tf, ude_tau_torque=tt)
        ctrls.append(RobustGeometricBaseline(env, f"UDE {tau}", ude_cfg))
    out = {}
    for ctrl in ctrls:
        t0 = time.perf_counter()
        out[ctrl.name] = summarize(rollout(env, ctrl, args.seed), args.scenarios, args.settle_s)
        print(f"  {name} / {ctrl.name}: {time.perf_counter() - t0:.0f} s", flush=True)
    del env, ctrls
    gc.collect()
    torch.cuda.empty_cache()
    return out


def best_ude(res: dict[str, dict]) -> str:
    udes = [k for k in res if k.startswith("UDE ")]
    return min(udes, key=lambda k: res[k]["all"]["pos_rms_cm_mean"])


def print_tables(results: dict[str, dict]) -> None:
    cols = ("policy", "z=true", "UDE best")
    print("\nall scenarios: survival % (position RMS cm, p95) | action rate |da|")
    print(f"{'condition':18s}" + "".join(f"{c:>28s}" for c in cols) + "   best UDE T")
    for cond, res in results.items():
        b = best_ude(res)
        row = []
        for c in cols:
            d = res[b if c == "UDE best" else c]["all"]
            row.append(f"{100 - d['crash_pct']:6.2f} ({d['pos_rms_cm_mean']:5.2f}, "
                       f"{d['pos_rms_cm_p95']:5.2f}) {d['act_rate_mean']:.4f}")
        print(f"{cond:18s}" + "".join(f"{r:>28s}" for r in row) + f"   {b[4:]}")
    print("\nall scenarios, encounters: onset peak p95 / exit peak p95 [cm], recover [s], "
          "lead [ms]; margin p5")
    print(f"{'condition':18s}" + "".join(f"{c:>36s}" for c in cols))
    for cond, res in results.items():
        b = best_ude(res)
        row = []
        for c in cols:
            d = res[b if c == "UDE best" else c]["all"]
            row.append(f"{d['enc_onset_peak_cm_p95']:5.2f} / {d['enc_exit_peak_cm_p95']:5.2f}, "
                       f"{d['enc_recover_s_mean']:4.2f}, {d['onset_lead_ms']:+4.0f}; "
                       f"{d['margin_min_p5']:.2f}")
        print(f"{cond:18s}" + "".join(f"{r:>36s}" for r in row))


def save(results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress.json").write_text(json.dumps(results, indent=2))
    with open(out / "stress.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["condition", "controller", "scenario", "metric", "value"])
        for cond, res in results.items():
            for ctrl, summ in res.items():
                for g, d in summ.items():
                    for k, v in d.items():
                        if k != "wake":
                            wr.writerow([cond, ctrl, g, k, v])


def main() -> None:
    args = parse_args()
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "stress"
    results = {}
    for name in args.conditions:
        print(f"condition {name}", flush=True)
        results[name] = run_condition(args, name)
        save(results, out)
    print_tables(results)
    print(f"\nwrote {out}/stress.json, stress.csv")


if __name__ == "__main__":
    main()
