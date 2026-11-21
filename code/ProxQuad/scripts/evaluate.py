"""Evaluate a trained policy on every scenario, against the geometric baseline.

Env: the current default EnvCfg (training randomization ranges, downwash on), the chosen
scenarios equally likely, one full episode per env. Every controller flies the same
episodes (same seed). Output: console table, <out>/summary.json, summary.csv, PNG plots.

Run from code/:
    python -m ProxQuad.scripts.evaluate --checkpoint runs/phase1_pilot/model.pt
    python -m ProxQuad.scripts.evaluate --checkpoint ... --ablate-z --scenarios hover_over far
"""

import argparse
import time
from pathlib import Path

from ..configs import EnvCfg
from ..envs import ProxQuadEnv
from ..envs.scenarios import SCENARIOS
from ..eval import (GeometricBaseline, Phase1Policy, Phase2Policy, load_phase1, print_summary,
                    rollout, save_plots, save_summary, summarize)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--phase", type=int, choices=(1, 2), default=1)
    p.add_argument("--envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--no-baseline", action="store_true", help="skip the geometric controller")
    p.add_argument("--ablate-z", action="store_true", help="also run pi with z = 0 and z = mean z")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--no-downwash", action="store_true")
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
    env = ProxQuadEnv(cfg)

    if args.phase == 2:
        policy = Phase2Policy(args.checkpoint, env)
    else:
        encoder, pi = load_phase1(args.checkpoint, env)
        policy = Phase1Policy(encoder, pi, stochastic=args.stochastic)
    controllers = [policy]
    if not args.no_baseline:
        controllers.append(GeometricBaseline(env))

    recs = []
    for ctrl in controllers:
        t0 = time.perf_counter()
        recs.append(rollout(env, ctrl, args.seed))
        print(f"{ctrl.name}: {time.perf_counter() - t0:.0f} s", flush=True)
        if ctrl is policy and args.ablate_z and args.phase == 1:
            for mode, name in (("zero", "z=0"), ("fixed", "z=mean")):
                abl = Phase1Policy(encoder, pi, z_mode=mode, z_fixed=policy.mean_z(),
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
