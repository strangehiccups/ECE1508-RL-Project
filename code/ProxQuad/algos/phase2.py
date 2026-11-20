"""Phase 2: train the adaptation module phi by supervised regression, RMA style.

    z_t     = mu(e_t)                                (frozen Phase 1 encoder, target)
    z_hat_t = phi(bottom history, top history)       (both drones' state-action histories)
    loss    = || z_hat_t - z_t ||^2

On-policy data collection (DAgger-like, as in RMA): roll out the frozen base policy
driven by z_hat_t from the CURRENT phi (not by z_t), so phi is trained on the states
its own estimation errors lead to.

Rates: pi acts every env step (250 Hz); phi refreshes z_hat every
cfg.adaptation_decimation = 2 steps (125 Hz) and for envs that just reset; pi reuses the
latest z_hat in between.
"""

import torch

from ..configs import Phase2Cfg
from ..envs import ProxQuadEnv
from ..models import AdaptationModule, BasePolicy, EnvFactorEncoder


class Phase2Trainer:
    def __init__(self, cfg: Phase2Cfg):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        oc = cfg.env.obs
        assert oc.bottom_history_len > 0 and oc.top_history_len > 0, "Phase 2 needs both histories"
        self.env = E = ProxQuadEnv(cfg.env)

        # frozen Phase 1 networks
        ck = torch.load(cfg.phase1_checkpoint, map_location=E.dev, weights_only=False)
        p1 = ck["cfg"]
        self.encoder = EnvFactorEncoder(E.priv_dim, p1.z_dim)
        self.policy = BasePolicy(E.tracking_dim, E.action_dim, E.top_dim, p1.z_dim)
        for name in ("encoder", "policy"):
            net = getattr(self, name)
            net.load_state_dict(ck[name])
            net.to(E.dev).eval().requires_grad_(False)

        # trainable adaptation module
        self.phi = AdaptationModule(E.bottom_hist_dim, oc.bottom_history_len, E.top_dim,
                                    oc.top_history_len, p1.z_dim).to(E.dev)
        self.opt = torch.optim.Adam(self.phi.parameters(), lr=cfg.lr)

        self.z_hat = torch.zeros(E.n, p1.z_dim, device=E.dev)   # held between refreshes
        self.k = 0   # env steps since start, for the refresh rate

    @torch.no_grad()
    def target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression target z_t from the frozen encoder."""
        return self.encoder(obs["priv"])

    @torch.no_grad()
    def estimate(self, just_reset: torch.Tensor | None = None) -> torch.Tensor:
        """Refresh z_hat every adaptation_decimation steps; call once per env step, before
        acting. just_reset (N,) bool: envs whose episode just restarted get a fresh z_hat
        instead of carrying the previous episode's."""
        h = self.env.get_history()
        if self.k % self.cfg.adaptation_decimation == 0:
            self.z_hat = self.phi(h["bottom"], h["top"])
        elif just_reset is not None and just_reset.any():
            ids = just_reset.nonzero().squeeze(-1)
            self.z_hat[ids] = self.phi(h["bottom"][ids], h["top"][ids])
        self.k += 1
        return self.z_hat

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], just_reset: torch.Tensor | None = None):
        z_hat = self.estimate(just_reset)
        return self.policy(obs["tracking_err"], obs["prev_action"], obs["top"], z_hat)

    def collect_rollout(self):
        """Run `rollout_steps` env steps with act(); store env.get_history() (clone it:
        the history is a view into a circular buffer) and target(obs) at every step (phi learns from every step, whatever its deployment
        refresh rate)."""
        raise NotImplementedError("Phase 2 comes after Phase 1 is settled")

    def update(self, rollout) -> dict[str, float]:
        """MSE of phi(bottom, top) vs z_t over the rollout, one Adam step."""
        raise NotImplementedError("Phase 2 comes after Phase 1 is settled")
