"""Phase 1: jointly train mu (env-factor encoder) and pi (base policy) with PPO on
ProxQuadEnv, using the privileged e_t.

    z_t = mu(e_t);   a_t ~ pi(x~_t, a_{t-1}, o_t^top, z_t);   gradients flow into mu through pi.

x~_t = bottom drone's tracking errors, o_t^top = top drone's state + previous command.
Reference: rsl_rl/algorithms/ppo.py (code/.venv), code/quad/train.py.
"""

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from ..configs import Phase1Cfg
from ..envs import ProxQuadEnv
from ..envs.observations import POLICY_INPUTS
from ..envs.scenarios import SCENARIOS
from ..models import BasePolicy, Critic, EnvFactorEncoder
from .storage import RolloutStorage


class Phase1Trainer:
    def __init__(self, cfg: Phase1Cfg):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.env = E = ProxQuadEnv(cfg.env)
        dims = dict(POLICY_INPUTS)  # Determine the inputs the policy sees
        self.encoder = EnvFactorEncoder(E.priv_dim, cfg.z_dim, cfg.encoder).to(E.dev)
        self.policy = BasePolicy(dims, cfg.z_dim, E.action_dim, cfg.policy, E.a_hover).to(E.dev)
        self.critic = Critic(dims, E.priv_dim, cfg.critic).to(E.dev)

        # Adam optimizer; gradients are clipped per network in update()
        self.actor_params = [*self.encoder.parameters(), *self.policy.parameters()]
        self.critic_params = list(self.critic.parameters())
        self.lr = cfg.ppo.lr
        self.opt = torch.optim.Adam(self.actor_params + self.critic_params, lr=self.lr)

        self.storage = RolloutStorage(cfg.rollout_steps, E.n, {**dims, "priv": E.priv_dim},
                                      E.action_dim, E.dev)
        self.writer = SummaryWriter(cfg.log_dir)

        # per-env running sums of the current episode, and records of finished ones
        self.ep_return = torch.zeros(E.n, device=E.dev)
        self.ep_length = torch.zeros(E.n, device=E.dev)
        self.ep_terms: dict[str, torch.Tensor] = {}   # reward term -> (N,) running sum
        self.finished: list[dict] = []                # cleared by log()
        self._timing_prev = dict(E.timing)            # env.timing is cumulative

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

    @torch.no_grad()
    def collect_rollout(self) -> None:
        """Fill self.storage with cfg.rollout_steps transitions, continuing from the env's
        current obs (self.env.get_obs()), then compute advantages and returns.
        """
        
        self.storage.clear()  # Clear the buffer before each rollout
        obs = self.env.get_obs()  # Get the env's current observation
        ppo = self.cfg.ppo

        for _ in range(self.cfg.rollout_steps):
            dist, _ = self.act(obs)     # Get the action distribution
            a = dist.sample()           # Sample from the action distribution
            value = self.critic(obs)    # Get the value from critic network
            
            # Step the environment
            next_obs, reward, done, info = self.env.step(a)

            # Add observations, actions, values, rewards at time t to the buffer
            self.storage.add(obs, a, dist.log_prob(a), dist.mean, dist.stddev, value, reward, done, info["time_out"], ppo.gamma)
            
            # Save episode information
            self._track_episodes(reward, done, info)

            # Iterate to next observation
            obs = next_obs
        
        self.storage.compute_returns(self.critic(obs), ppo.gamma, ppo.lam)


    def update(self) -> dict[str, float]:
        """PPO on self.storage; returns mean diagnostics for log().

        For each minibatch of storage.minibatches(ppo.minibatches, ppo.epochs):
          1. With grad: z = self.encoder(mb["obs"]["priv"]) (mu learns through pi's loss),
             dist = self.policy(mb["obs"], z), value = self.critic(mb["obs"]).
          2. KL(old || new) of the diagonal Gaussians, closed form from mb["action_mean"],
             mb["action_std"], dist.mean, dist.stddev; summed over actions, mean over the
             batch. schedule "adaptive": lr / 1.5 if kl > 2 desired_kl, lr x 1.5 if
             kl < desired_kl / 2, clamped to lr_bounds; write it into self.opt.param_groups.
          3. ratio = exp(dist.log_prob(mb["actions"]) - mb["log_probs"])
             surrogate = -min(ratio A, clamp(ratio, 1 - clip, 1 + clip) A).mean()
          4. value loss = (value - returns)^2; if clip_value_loss, also with
             value_clipped = values + clamp(value - values, -clip, clip), take the max.
          5. loss = surrogate + value_coef value_loss - entropy_coef dist.entropy().mean()
          6. zero_grad, backward, clip_grad_norm_ on self.actor_params and on
             self.critic_params separately (max_grad_norm), step.
        Return e.g. surrogate, value_loss, entropy, kl, lr, clip_fraction (share of
        |ratio - 1| > clip).
        """
        
        ppo = self.cfg.ppo
        sums, n = {"surrogate": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_fraction": 0.0}, 0  # Track results for each minibatch

        for mb in self.storage.minibatches(ppo.minibatches, ppo.epochs):
            
            # Forward propagation of the three networks
            z = self.encoder(mb["obs"]["priv"])
            dist = self.policy(mb["obs"], z)
            value = self.critic(mb["obs"])
            
            # Compute KL divergence without gradients
            with torch.no_grad():
                mu0, s0 = mb["action_mean"], mb["action_std"]  # Old action distribution
                mu1, s1 = dist.mean, dist.stddev               # New action distribution
                kl = (torch.log(s1 / s0) + (s0**2 + (mu0 - mu1)**2) / (2 * s1**2) - 0.5).sum(-1).mean().item()  # Closed-form KL for normal distribution

            # Use an adaptive learning rate if enabled
            if ppo.schedule == "adaptive":
                # Reduce learning rate if two policies are far apart (kl>2*kl_desired), and vice versa
                if kl > 2*ppo.desired_kl: 
                    self.lr /= 1.5
                if kl < 0.5*ppo.desired_kl: 
                    self.lr *= 1.5
                self.lr = min(max(self.lr, ppo.lr_bounds[0]), ppo.lr_bounds[1])
                for g in self.opt.param_groups:
                    g["lr"] = self.lr

            # Compute ratio of probabilities of current action given new policy / old policy
            # Exponential of log-likelihood
            ratio = torch.exp(dist.log_prob(mb["actions"]) - mb["log_probs"])
            
            # Compute clipped surrogate function (objective: maximize return)
            surrogate = -torch.min(ratio*mb["advantages"], torch.clamp(ratio, 1 - ppo.clip, 1 + ppo.clip)*mb["advantages"]).mean()

            # Compute loss for critic network (objective: minimize value prediction error)
            value_loss = (value - mb["returns"]) ** 2
            # Clip the value loss if enabled
            if ppo.clip_value_loss:
                # Same as clipped policy update (clip gradient (clipped_value_loss) when it is already improving, full gradient (value_loss) if value is very wrong compared to baseline)
                value_clipped = mb["values"] + torch.clamp(value - mb["values"], -ppo.clip, ppo.clip)  # Only allow new values to be ppo.clip away from old values
                value_loss = torch.max(value_loss, (value_clipped - mb["returns"]) ** 2)  # clipped_value_loss = (value_clipped - mb["returns"]) ** 2
            value_loss = value_loss.mean()

            # Compute mean entropy of actions
            entropy = dist.entropy().mean()

            # Total loss function
            loss = surrogate + ppo.value_coef * value_loss - ppo.entropy_coef * entropy

            # Backpropagation
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_params, ppo.max_grad_norm) 
            nn.utils.clip_grad_norm_(self.critic_params, ppo.max_grad_norm)
            self.opt.step()

            sums["surrogate"] += surrogate.item()
            sums["value_loss"] += value_loss.item()
            sums["entropy"] += entropy.item()
            sums["kl"] += kl
            sums["clip_fraction"] += ((ratio - 1).abs() > ppo.clip).float().mean().item()
            n += 1
        
        return {k: v / n for k, v in sums.items()} | {"lr": self.lr}


    def _track_episodes(self, reward: torch.Tensor, done: torch.Tensor, info: dict) -> None:
        """Add this step to the running sums. If any env is done, append ONE dict of
        (k,) tensors (k = envs done this step) to self.finished, then zero their sums.
        Stays on the GPU; log() concatenates the entries."""
        self.ep_return += reward
        self.ep_length += 1

        # Track the contribution of each reward term
        for name, v in info["reward_terms"].items():
            if name not in self.ep_terms:
                self.ep_terms[name] = torch.zeros_like(self.ep_return)
            self.ep_terms[name] += torch.nan_to_num(v, nan=0.0)  # "nan" crash step
        
        # Track completed episodes
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        # info is built before the env resets, so reasons and scenario are the finished episode's
        self.finished.append({
            "return": self.ep_return[ids],
            "length": self.ep_length[ids],
            "terms": {name: s[ids] for name, s in self.ep_terms.items()},
            "crashed": info["crashed"][ids],
            "reasons": {name: r[ids] for name, r in info["crash_reasons"].items()},
            "time_out": info["time_out"][ids],
            "scenario": info["scenario"][ids],
        })
        self.ep_return[ids] = 0.0
        self.ep_length[ids] = 0.0
        for s in self.ep_terms.values():
            s[ids] = 0.0

    def log(self, it: int, stats: dict[str, float], t_rollout: float, t_update: float) -> None:
        """TensorBoard (self.writer) and one console line; then clear self.finished.
          train/    stats from update(), action std, Lipschitz bound
          episode/  over episodes finished this iteration (skipped if none): return,
                    length, reward terms, crash rates (per reason, per scenario)
          perf/     env steps per second, rollout / update / physics / aero seconds
        """
        # Training: update() stats, policy std, Lipschitz bound
        P = self.policy
        scalars = {f"train/{k}": float(v) for k, v in stats.items()}
        std = P.log_std.clamp(*P.log_std_bounds).exp()
        scalars["train/action_std"] = std.mean().item()
        scalars["train/lipschitz_bound"] = P.lipschitz_bound()

        # For episodes finished in this iteration: compute averages, per reward term, crash reason, scenario
        if self.finished:
            cat = lambda key: torch.cat([f[key] for f in self.finished])  # noqa: E731
            ret, crashed, kind = cat("return"), cat("crashed").float(), cat("scenario")
            scalars["episode/count"] = len(ret)
            scalars["episode/return"] = ret.mean().item()
            scalars["episode/length_s"] = cat("length").mean().item() * self.env.dt
            scalars["episode/crash_rate"] = crashed.mean().item()
            scalars["episode/timeout_rate"] = cat("time_out").float().mean().item()
            for name in self.finished[0]["terms"]:
                vals = torch.cat([f["terms"][name] for f in self.finished])
                scalars[f"episode_terms/{name}"] = vals.mean().item()
            for name in self.finished[0]["reasons"]:
                vals = torch.cat([f["reasons"][name] for f in self.finished])
                scalars[f"crash_reason/{name}"] = vals.float().mean().item()
            for i, name in enumerate(SCENARIOS):
                mask = kind == i
                if mask.any():
                    scalars[f"scenario_return/{name}"] = ret[mask].mean().item()
                    scalars[f"scenario_crash/{name}"] = crashed[mask].mean().item()

        # Throughput and wall-clock times; env.timing is cumulative, so log its increase
        steps = self.cfg.rollout_steps * self.env.n
        timing = {k: v - self._timing_prev.get(k, 0.0) for k, v in self.env.timing.items()}
        self._timing_prev = dict(self.env.timing)
        scalars["perf/steps_per_s"] = steps / (t_rollout + t_update)
        scalars["perf/rollout_s"], scalars["perf/update_s"] = t_rollout, t_update
        for k, v in timing.items():
            scalars[f"perf/{k}_s"] = v

        # Write everything to TensorBoard, x-axis = iteration
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, it)

        # Print summary of one episode in console, then reset the finished-episode records
        ep = (f"ret {scalars['episode/return']:8.2f}  len {scalars['episode/length_s']:5.2f}s  "
              f"crash {scalars['episode/crash_rate']:.2f}" if self.finished else "no episodes ended")
        train = "  ".join(f"{k} {v:.3g}" for k, v in stats.items())
        print(f"it {it:5d} | {ep} | std {scalars['train/action_std']:.3f}  {train} | "
              f"{scalars['perf/steps_per_s']:,.0f} steps/s ({t_rollout:.2f}+{t_update:.2f}s)",
              flush=True)
        self.finished.clear()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(),
                    "policy": self.policy.state_dict(),
                    "critic": self.critic.state_dict(),
                    "cfg": self.cfg}, path)

    def train(self) -> None:
        """Warm-up, then per iteration: rollout -> PPO update -> normalizers -> log / save."""
        for _ in range(self.cfg.ppo.warmup_rollouts):
            self.collect_rollout()
            self.update_norms(self.storage.obs)
        self.finished.clear()
        self._timing_prev = dict(self.env.timing)
        for it in range(self.cfg.iterations):
            t0 = time.perf_counter()
            self.collect_rollout()
            t1 = time.perf_counter()
            stats = self.update()
            self.update_norms(self.storage.obs)
            self.log(it, stats, t1 - t0, time.perf_counter() - t1)
            if (it + 1) % self.cfg.save_interval == 0 or it + 1 == self.cfg.iterations:
                self.save(self.cfg.checkpoint)
