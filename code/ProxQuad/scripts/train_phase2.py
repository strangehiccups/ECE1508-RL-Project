"""Phase 2 entry point. Run from code/:
    python -m ProxQuad.scripts.train_phase2 --phase1 runs/phase1_circles/model.pt --with-circles --log-dir runs/phase2_circles
"""

import argparse

from ..algos.phase2 import Phase2Trainer
from ..configs import Phase2Cfg
from .train_phase1 import CIRCLE_MIX, FIGURE8_MIX

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase1", help="Phase 1 checkpoint (frozen mu and pi)")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--log-dir")
    mix = p.add_mutually_exclusive_group()
    mix.add_argument("--with-circles", action="store_true", help="scenario mix CIRCLE_MIX")
    mix.add_argument("--with-figure8", action="store_true", help="scenario mix FIGURE8_MIX")
    args = p.parse_args()

    cfg = Phase2Cfg()
    if args.phase1:
        cfg.phase1_checkpoint = args.phase1
    if args.num_envs:
        cfg.env.sim.num_envs = args.num_envs
    if args.iterations:
        cfg.iterations = args.iterations
    if args.with_circles:
        cfg.env.scenario.top.scenario_probs = dict(CIRCLE_MIX)
    if args.with_figure8:
        cfg.env.scenario.top.scenario_probs = dict(FIGURE8_MIX)
    if args.log_dir:
        cfg.log_dir, cfg.checkpoint = args.log_dir, f"{args.log_dir}/model.pt"
    Phase2Trainer(cfg).train()
