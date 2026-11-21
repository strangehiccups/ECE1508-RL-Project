"""Phase 1 entry point. Run from code/:  python -m ProxQuad.scripts.train_phase1
Sanity run first (no downwash, top drone always far: plain hover):
    python -m ProxQuad.scripts.train_phase1 --sanity --iterations 200
With the circle scenarios (bottom drone flies circles too):
    python -m ProxQuad.scripts.train_phase1 --with-circles --iterations 2000 --log-dir runs/phase1_circles
"""

import argparse

from ..algos.phase1 import Phase1Trainer
from ..configs import Phase1Cfg

CIRCLE_MIX = {"hover_over": 0.2, "pass_over": 0.2, "approach_hover_leave": 0.2, "far": 0.1,
              "circle_same": 0.15, "circle_opposite": 0.15}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--log-dir")
    p.add_argument("--sanity", action="store_true", help="aero off, top drone always far")
    p.add_argument("--with-circles", action="store_true", help="scenario mix CIRCLE_MIX")
    args = p.parse_args()

    cfg = Phase1Cfg()
    if args.num_envs:
        cfg.env.sim.num_envs = args.num_envs
    if args.iterations:
        cfg.iterations = args.iterations
    if args.with_circles:
        cfg.env.scenario.top.scenario_probs = dict(CIRCLE_MIX)
    if args.sanity:
        cfg.env.aero.enabled = False
        cfg.env.scenario.top.scenario_probs = {"far": 1.0}
        cfg.log_dir, cfg.checkpoint = "runs/phase1_sanity", "runs/phase1_sanity/model.pt"
    if args.log_dir:
        cfg.log_dir, cfg.checkpoint = args.log_dir, f"{args.log_dir}/model.pt"
    Phase1Trainer(cfg).train()
