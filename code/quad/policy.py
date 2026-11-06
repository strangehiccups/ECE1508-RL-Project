import torch
import torch.nn as nn


class RunningNorm:
    """Running mean/std observation normalizer on the GPU."""
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        b_mean, b_var, b_n = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=22, act_dim=4):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(obs_dim, 128), nn.ELU(),
                                 nn.Linear(128, 128), nn.ELU(),
                                 nn.Linear(128, out))
        self.actor, self.critic = mlp(act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)