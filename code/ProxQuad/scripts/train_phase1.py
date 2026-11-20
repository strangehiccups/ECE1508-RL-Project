"""Phase 1 entry point. Run from code/:  python -m ProxQuad.scripts.train_phase1"""

from ..algos.phase1 import Phase1Trainer
from ..configs import Phase1Cfg

if __name__ == "__main__":
    Phase1Trainer(Phase1Cfg()).train()
