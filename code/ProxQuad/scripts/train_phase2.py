"""Phase 2 entry point. Run from code/:  python -m ProxQuad.scripts.train_phase2"""

from ..algos.phase2 import Phase2Trainer
from ..configs import Phase2Cfg

if __name__ == "__main__":
    Phase2Trainer(Phase2Cfg())
