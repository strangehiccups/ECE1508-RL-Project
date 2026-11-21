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
from ..envs.observations import POLICY_INPUTS
from ..models import BasePolicy, Critic, EnvFactorEncoder


class Phase1Trainer:
    def __init__(self, cfg: Phase1Cfg):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.env = ProxQuadEnv(cfg.env)
        E = self.env
        dims = dict(POLICY_INPUTS)
        self.encoder = EnvFactorEncoder(E.priv_dim, cfg.z_dim, cfg.encoder).to(E.dev)
        self.policy = BasePolicy(dims, cfg.z_dim, E.action_dim, cfg.policy, E.a_hover).to(E.dev)
        self.critic = Critic(dims, E.priv_dim, cfg.critic).to(E.dev)

    @torch.no_grad()
    def update_norms(self, obs: dict[str, torch.Tensor]) -> None:
        """Running obs statistics of all three networks, from a rollout's obs (stacked
        over steps). Call AFTER the update on that rollout: its old log-probs and values
        were computed with the current statistics. Also once before the first update
        (warm-up rollout), so the input map does not jump after iteration 1."""
        self.encoder.update_norm(obs["priv"])
        self.policy.update_norm(obs)
        self.critic.update_norm(obs)

    def act(self, obs: dict[str, torch.Tensor]):
        z = self.encoder(obs["priv"])
        dist = self.policy(obs, z)
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
        # TODO: warm-up rollout -> update_norms, before the loop
        for it in range(self.cfg.iterations):
            rollout = self.collect_rollout()
            stats = self.update(rollout)
            # TODO: self.update_norms(rollout obs)
            # TODO: tensorboard logging of stats, reward terms, crash reasons per scenario
