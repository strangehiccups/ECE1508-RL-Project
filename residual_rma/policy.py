"""
Same three networks as v1 (code/policy.py), with two changes:
  * sizes follow the new env: obs 26, e_t 10, history entry 22
  * the actor's last layer starts near zero, so at the start of training the residual is ~0 and
    the drone flies on the nominal controller alone (it starts from a working system)
  * the adaptation module has an extra small head that predicts e_t itself (only used for the
    "can phi actually see the disturbance?" diagnostic, trained on detached features)
"""
import torch
import torch.nn as nn


class EnvEncoder(nn.Module):
    """mu: e_t -> z_t"""
    def __init__(self, env_dim=10, h1=256, h2=128, z_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(env_dim, h1), nn.ELU(),
                                 nn.Linear(h1, h2), nn.ELU(),
                                 nn.Linear(h2, z_dim))

    def forward(self, e):
        return self.net(e)


class ActorCritic(nn.Module):
    """pi: (x_t, a_{t-1}, z_t) -> a_t, plus the PPO critic. use_z=False gives a plain robust policy."""
    def __init__(self, obs_dim=26, h1=128, h2=128, z_dim=8, act_dim=4, use_z=True, init_std=-1.0):
        super().__init__()
        self.obs_dim, self.z_dim, self.act_dim, self.use_z = obs_dim, z_dim, act_dim, use_z
        inp = obs_dim + (z_dim if use_z else 0)

        def mlp(out):
            return nn.Sequential(nn.Linear(inp, h1), nn.ELU(), nn.Linear(h1, h2), nn.ELU(), nn.Linear(h2, out))
        self.actor, self.critic = mlp(act_dim), mlp(1)
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01)
            self.actor[-1].bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), init_std))

    def _in(self, obs, z):
        return torch.cat([obs, z], -1) if self.use_z else obs

    def dist(self, obs, z):
        return torch.distributions.Normal(self.actor(self._in(obs, z)), self.log_std.exp())

    def value(self, obs, z):
        return self.critic(self._in(obs, z)).squeeze(-1)


class AdaptationModule(nn.Module):
    """phi: last 50 (state, command) pairs -> z_hat. Same layout as v1 / the RMA paper."""
    def __init__(self, in_dim=22, fc1=32, fc2=32, ch=32, z_dim=8, hist_len=50, env_dim=10):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, fc1), nn.ELU(), nn.Linear(fc1, fc2), nn.ELU())
        self.conv = nn.Sequential(nn.Conv1d(fc2, ch, 8, stride=4), nn.ELU(),
                                  nn.Conv1d(ch, ch, 5, stride=1), nn.ELU(),
                                  nn.Conv1d(ch, ch, 5, stride=1), nn.ELU(),
                                  nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, fc2, hist_len)).shape[-1]
        self.out = nn.Linear(n_flat, z_dim)
        self.e_head = nn.Linear(n_flat, env_dim)      # diagnostic only

    def features(self, hist):
        return self.conv(self.embed(hist).transpose(1, 2))

    def forward(self, hist):
        return self.out(self.features(hist))

    def forward_both(self, hist):
        h = self.features(hist)
        return self.out(h), self.e_head(h.detach())
