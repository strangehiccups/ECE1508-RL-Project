"""Rollout storage for Phase 1 PPO: T = rollout_steps x N = num_envs transitions, all on
the env's device. Reference: rsl_rl/storage/rollout_storage.py (code/.venv)."""

from collections.abc import Iterator

import torch


class RolloutStorage:
    def __init__(self, T: int, N: int, obs_dims: dict[str, int], action_dim: int,
                 device: torch.device):
        """obs_dims: every obs key the networks read (pi, critic, mu), name -> dim."""
        z = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
        self.T, self.N = T, N
        self.obs = {k: z(T, N, d) for k, d in obs_dims.items()}   # obs_t, before the step
        self.actions = z(T, N, action_dim)       # sampled, unclamped
        self.log_probs = z(T, N)                 # log pi_old(a_t | s_t)
        self.action_mean = z(T, N, action_dim)   # pi_old, for the KL in the update
        self.action_std = z(T, N, action_dim)
        self.values = z(T, N)                    # V_old(s_t)
        self.rewards = z(T, N)
        self.dones = z(T, N)                     # 1.0 if the episode ended at this step
        self.advantages = z(T, N)
        self.returns = z(T, N)                   # value targets
        self.t = 0

    def clear(self) -> None:
        self.t = 0

    def add(self, obs: dict[str, torch.Tensor], action: torch.Tensor, log_prob: torch.Tensor,
            mean: torch.Tensor, std: torch.Tensor, value: torch.Tensor, reward: torch.Tensor,
            done: torch.Tensor, time_out: torch.Tensor, gamma: float) -> None:
        """Copy step self.t into the buffers and advance self.t."""
        
        assert self.t < self.T

        t = self.t

        # Assign the keys of obs to each key of self.obs for row t
        for k, buf in self.obs.items():
            buf[t] = obs[k]

        # Save actions, log probabilities, mean, and stds
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.action_mean[t] = mean
        self.action_std[t] = std

        # Store values, rewards and done
        self.values[t] = value
        self.rewards[t] = reward + gamma * value * time_out.float()  # Bootstrap with V(obs_t) if time_out
        self.dones[t] = done.float()

        # Advance self.t
        self.t += 1


    def compute_returns(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        """GAE, backwards over t = T-1 .. 0; last_value = V(obs after the last step)"""
        
        next_adv = 0
        for t in range(self.T-1, -1, -1):
            next_value = last_value if t == self.T-1 else self.values[t+1]
            not_done = 1 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]  # The TD-0 advantage
            next_adv = delta + gamma * lam * not_done * next_adv  # The forward view TD-lambda advantage
            self.advantages[t] = next_adv
        
        self.returns = self.advantages + self.values   # For critic target
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)  # For actor loss

    def minibatches(self, num_minibatches: int, epochs: int) -> Iterator[dict]:
        """Yield epochs x num_minibatches minibatches; a fresh random permutation of the
        T*N samples each epoch. Each item is a dict of flattened (B, ...) tensors,
        B = T*N / num_minibatches:
            "obs" (dict key -> (B, dim)), "actions", "log_probs", "action_mean",
            "action_std", "values", "returns", "advantages"."""
        
        TN = self.T * self.N

        # Flatten the samples across T and N
        flat = {
            "obs": {k: v.view(TN, -1) for k, v in self.obs.items()},
            "actions": self.actions.view(TN, -1),
            "log_probs": self.log_probs.view(TN),
            "action_mean": self.action_mean.view(TN, -1),
            "action_std": self.action_std.view(TN, -1),
            "values": self.values.view(TN),
            "returns": self.returns.view(TN),
            "advantages": self.advantages.view(TN)
        }

        B = TN // num_minibatches  # Batch size (best if num_minibatches * B = TN)

        # Train with the stored data for "epoch" epochs
        for _ in range(epochs):
            perm = torch.randperm(TN, device=self.actions.device)  # Randomize the sequnece of sample drawing
            for i in range(num_minibatches):
                idx = perm[i*B: (i+1)*B]

                # Pause the for loop and yield one minibatch at a time, instead of computing all at once
                yield {"obs": {k: v[idx] for k, v in flat["obs"].items()},
                       "actions": flat["actions"][idx],
                       "log_probs": flat["log_probs"][idx],
                       "action_mean": flat["action_mean"][idx],
                       "action_std": flat["action_std"][idx],
                       "values": flat["values"][idx],
                       "returns": flat["returns"][idx],
                       "advantages": flat["advantages"][idx]}
