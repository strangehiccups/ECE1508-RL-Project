"""Phase 1: jointly train mu (env-factor encoder) and pi (base policy) with model-free
RL on ProxQuadEnv, using the privileged e_t.

    z_t = mu(e_t);   a_t ~ pi(x~_t, a_{t-1}, o_t^top, z_t);   gradients flow into mu through pi.

x~_t = bottom drone's tracking errors, o_t^top = top drone's state + previous command.

The RL algorithm (PPO / TRPO / ...) is still open. This skeleton fixes only the data
flow between env and networks so the algorithm can be dropped into `update()`.
"""

import torch

from ..configs import Phase1Cfg
from ..envs import ProxQuadEnv
from ..models import BasePolicy, Critic, EnvFactorEncoder


class Phase1Trainer:
    def __init__(self, cfg: Phase1Cfg):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.env = ProxQuadEnv(cfg.env)
        E = self.env
        self.encoder = EnvFactorEncoder(E.priv_dim, cfg.z_dim).to(E.dev)
        self.policy = BasePolicy(E.tracking_dim, E.action_dim, E.top_dim, cfg.z_dim).to(E.dev)
        self.critic = Critic(E.priv_dim + E.tracking_dim + E.action_dim
                             + E.top_dim).to(E.dev)  # TBD
        # TODO: observation normalizers (utils.normalizer.RunningNorm) for x~_t, top, e_t

    def act(self, obs: dict[str, torch.Tensor]):
        z = self.encoder(obs["priv"])
        dist = self.policy(obs["tracking_err"], obs["prev_action"], obs["top"], z)
        return dist, z

    def collect_rollout(self):
        """Run `rollout_steps` env steps; store obs dicts, actions, rewards, dones and
        info["time_out"] (truncation: bootstrap the value, do not treat as terminal)."""
        raise NotImplementedError("depends on the chosen algorithm")

    def update(self, rollout) -> dict[str, float]:
        raise NotImplementedError("RL algorithm not decided yet")

    def save(self, path: str) -> None:
        torch.save({"encoder": self.encoder.state_dict(),
                    "policy": self.policy.state_dict(),
                    "critic": self.critic.state_dict(),
                    "cfg": self.cfg}, path)

    def train(self) -> None:
        for it in range(self.cfg.iterations):
            rollout = self.collect_rollout()
            stats = self.update(rollout)
            # TODO: tensorboard logging of stats, reward terms, crash reasons per scenario
