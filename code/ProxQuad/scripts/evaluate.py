"""Evaluate a trained policy on every scenario, against the geometric+UDE baseline.

Env: the current default EnvCfg (training randomization ranges, downwash on), the chosen
scenarios equally likely, one full episode per env. Every controller flies the same
episodes (same seed). Output: console table, <out>/summary.json, summary.csv, PNG plots.

Run from code/:
    python -m ProxQuad.scripts.evaluate --checkpoint runs/phase1_pilot/model.pt
    python -m ProxQuad.scripts.evaluate --checkpoint ... --ablate-z --scenarios hover_over far
    python -m ProxQuad.scripts.evaluate --phase 2 --checkpoint runs/phase2_circles/model.pt --ablate-z
Phase 2: pi on z_hat from phi ("policy"), against the same pi on the true z = mu(e_t)
("z=true"); --ablate-z adds z=0 and z=mean. --sensors / --noise-scale: every controller
flies on delayed, noisy state estimates of both drones (envs/sensors.py); metrics stay on
the true state.
"""

import argparse
import time

import torch
from pathlib import Path

from ..configs import EnvCfg
from ..envs import ProxQuadEnv
from ..envs.scenarios import SCENARIOS
from ..eval import (Phase1Policy, Phase2Policy, RobustGeometricBaseline,
                    load_phase1, load_phase2, print_summary, rollout, save_plots, save_summary,
                    summarize)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase", type=int, choices=(1, 2), default=1)
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--no-baseline", action="store_true", help="skip the geometric+UDE baseline")
    p.add_argument("--ablate-z", action="store_true", help="also run pi with z = 0 and z = mean z")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--no-downwash", action="store_true")
    p.add_argument("--sensors", action="store_true",
                   help="controllers see delayed, noisy states (SensorCfg defaults)")
    p.add_argument("--noise-scale", type=float, default=None,
                   help="SensorCfg.noise_scale (implies --sensors; 0: latency only)")
    p.add_argument("--settle-s", type=float, default=2.0,
                   help="tracking metrics skip the first seconds (start-state transient)")
    p.add_argument("--out", help="output dir (default: <checkpoint dir>/eval)")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EnvCfg()
    cfg.sim.num_envs = args.envs
    cfg.sim.stagger_first_episode = False
    cfg.scenario.top.scenario_probs = {s: 1.0 / len(args.scenarios) for s in args.scenarios}
    cfg.aero.enabled = not args.no_downwash
    cfg.sensors.enabled = args.sensors or args.noise_scale is not None
    if args.noise_scale is not None:
        cfg.sensors.noise_scale = args.noise_scale
    if args.phase == 2:   # phi's history lengths
        obs2 = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"].env.obs
        cfg.obs.bottom_history_len, cfg.obs.top_history_len = (obs2.bottom_history_len,
                                                                obs2.top_history_len)
    env = ProxQuadEnv(cfg)

    if args.phase == 2:
        encoder, pi, phi, dec = load_phase2(args.checkpoint, env)
        policy = Phase2Policy(encoder, pi, phi, dec)
        expert = Phase1Policy(encoder, pi, name="z=true")   # same pi with mu(e_t)
        controllers = [policy, expert]
    else:
        encoder, pi = load_phase1(args.checkpoint, env)
        policy = expert = Phase1Policy(encoder, pi, stochastic=args.stochastic)
        controllers = [policy]
    if not args.no_baseline:
        controllers.append(RobustGeometricBaseline(env))

    recs = []
    for ctrl in controllers:
        t0 = time.perf_counter()
        recs.append(rollout(env, ctrl, args.seed))
        print(f"{ctrl.name}: {time.perf_counter() - t0:.0f} s", flush=True)
        if ctrl is expert and args.ablate_z:
            for mode, name in (("zero", "z=0"), ("fixed", "z=mean")):
                abl = Phase1Policy(encoder, pi, z_mode=mode, z_fixed=expert.mean_z(),
                                   stochastic=args.stochastic, name=name)
                recs.append(rollout(env, abl, args.seed))
                print(f"{name}: done", flush=True)

    results = {r.name: summarize(r, args.scenarios, args.settle_s) for r in recs}
    print(f"\ntracking errors over t >= {args.settle_s:g} s; crash% and return over the whole episode")
    print_summary(results)
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "eval"
    save_summary(results, out)
    if not args.no_plots:
        save_plots(recs, args.scenarios, out, settle_s=args.settle_s)
    print(f"\nwrote {out}/summary.json, summary.csv" + ("" if args.no_plots else ", *.png"))


if __name__ == "__main__":
    main()
