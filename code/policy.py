import torch
import torch.nn as nn


# ---- (1) environment factor encoder:  e_t -> z_t ----------------------
class EnvEncoder(nn.Module):
    """3-layer MLP. Only used in training as the 'expert'."""
    def __init__(self, env_dim=7,
                 hidden_dim1=256, hidden_dim2=128,
                 z_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(env_dim, hidden_dim1), nn.ELU(),
                                 nn.Linear(hidden_dim1, hidden_dim2), nn.ELU(),
                                 nn.Linear(hidden_dim2, z_dim))

    def forward(self, e):
        return self.net(e)


# ---- (2) base policy:  (x_t, a_{t-1}, z_t) -> a_t  --------------------
class ActorCritic(nn.Module):
    """Base Policy"""
    def __init__(self, obs_dim=22,
                 hidden_dim1=128, hidden_dim2=128, 
                 z_dim=8, act_dim=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.act_dim = act_dim
        def mlp(out):
            return nn.Sequential(nn.Linear(self.obs_dim + self.z_dim, hidden_dim1), nn.ELU(),
                                 nn.Linear(hidden_dim1, hidden_dim2), nn.ELU(),
                                 nn.Linear(hidden_dim2, out))
        self.actor, self.critic = mlp(self.act_dim), mlp(1)
        self.log_std = nn.Parameter(torch.full((self.act_dim,), -1.0))

    def dist(self, obs, z):
        return torch.distributions.Normal(self.actor(torch.cat([obs, z], -1)), self.log_std.exp())

    def value(self, obs, z):
        return self.critic(torch.cat([obs, z], -1)).squeeze(-1)

# ---- (3) adaptation module:  history -> zhat_t ----------------------
class AdaptationModule(nn.Module):
    """Per-step 2-layer MLP embedding (-> 32) followed by a 3-layer 1-D CNN, then a linear
    projection to z_hat. Conv (in, out, kernel, stride): (32,32,8,4), (32,32,5,1), (32,32,5,1).
    With k = 50 the flattened CNN output is 32 * 3 = 96."""
    def __init__(self, obs_dim=22,
                 fc1_dim=32, fc2_dim=32,
                 channels1=32, kernel1=8, stride1=4,
                 channels2=32, kernel2=5, stride2=1,
                 channels3=32, kernel3=5, stride3=1,
                 z_dim=8, hist_len=50):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(obs_dim, fc1_dim), nn.ELU(),
                                   nn.Linear(fc1_dim, fc2_dim), nn.ELU())
        self.conv = nn.Sequential(nn.Conv1d(in_channels=fc2_dim, out_channels=channels1, kernel_size=kernel1, stride=stride1), nn.ELU(),
                                  nn.Conv1d(in_channels=channels1, out_channels=channels2, kernel_size=kernel2, stride=stride2), nn.ELU(),
                                  nn.Conv1d(in_channels=channels2, out_channels=channels3, kernel_size=kernel3, stride=stride3), nn.ELU(),
                                  nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, channels3, hist_len)).shape[-1]
        self.out = nn.Linear(n_flat, z_dim)

    def forward(self, hist):                       # hist: (B, T, obs_dim), normalized
        h = self.embed(hist).transpose(1, 2)       # (B, 32, T)
        return self.out(self.conv(h))
