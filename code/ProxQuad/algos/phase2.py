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

Training (RMA Algorithm 1): phi runs with grad at every step, each step's loss is
backpropagated at once (gradients accumulate), and Adam steps once per rollout.
No histories are stored.
"""

import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from ..configs import Phase2Cfg
from ..envs import ProxQuadEnv
from ..envs.observations import POLICY_INPUTS
from ..envs.scenarios import SCENARIOS
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
        self.encoder = EnvFactorEncoder(E.priv_dim, p1.z_dim, p1.encoder)
        self.policy = BasePolicy(dict(POLICY_INPUTS), p1.z_dim, E.action_dim, p1.policy,
                                 E.a_hover)
        for name in ("encoder", "policy"):
            net = getattr(self, name)
            net.load_state_dict(ck[name])
            net.to(E.dev).eval().requires_grad_(False)

        # trainable adaptation module
        self.phi = AdaptationModule(E.bottom_hist_dim, oc.bottom_history_len, E.top_dim,
                                    oc.top_history_len, p1.z_dim, cfg.adaptation).to(E.dev)
        self.opt = torch.optim.Adam(self.phi.parameters(), lr=cfg.lr)

        self.z_hat = torch.zeros(E.n, p1.z_dim, device=E.dev)   # held between refreshes
        self.k = 0   # env steps since start, for the refresh rate
        self.p1_cfg, self.z_dim = p1, p1.z_dim
        self.just_reset = torch.ones(E.n, dtype=torch.bool, device=E.dev)   # env just reset; kept across rollouts

        self.writer = SummaryWriter(cfg.log_dir)
        # per-env running sums of the current episode, and records of finished ones
        self.ep_return = torch.zeros(E.n, device=E.dev)
        self.ep_length = torch.zeros(E.n, device=E.dev)
        self.ep_terms: dict[str, torch.Tensor] = {}
        self.finished: list[dict] = []                # cleared by log()
        self._timing_prev = dict(E.timing)            # env.timing is cumulative

    @torch.no_grad()
    def target(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Regression target z_t from the frozen encoder."""
        return self.encoder(obs["priv"])

    def collect_and_update(self, learn: bool = True) -> dict[str, float]:
        """Returns means over the rollout, logged under train/ (z_var enables the *_rel ratios):
            loss          ||phi(h_t) - z_t||^2  (summed over z dims, mean over envs)
            deployed_mse  ||z_hat_t - z_t||^2 for the held z_hat that pi acted on
            z_var         Var(z_t) summed over z dims
        """
        E, cfg = self.env, self.cfg
        T = cfg.rollout_steps
        sums = {key: torch.zeros((), device=E.dev) for key in ("loss", "deployed_mse", "z_var")}

        obs = E.get_obs()

        for t in range(T):
            h = E.get_history()        # history at t = t-L, ..., t-1;
            self._update_norms(h)

            z_t = self.target(obs)  # target z_t from frozen encoder
            
            # Only evalulate z_hat_t with gradients when learn = True
            with torch.set_grad_enabled(learn):
                z_hat_t = self.phi(h["bottom"], h["top"])
            
            loss = ((z_hat_t - z_t) ** 2).sum(-1).mean()
            if learn:
                (loss/T).backward()  # To backpropage gradients of loss at one timestep
            
            # Update z_hat estimation with adaptation module at each adaptation step
            z_det_t = z_hat_t.detach()
            if self.k % cfg.adaptation_decimation == 0:
                self.z_hat = z_det_t                                                     # refresh all envs
            else:
                self.z_hat = torch.where(self.just_reset[:, None], z_det_t, self.z_hat)  # only just-reset envs
            self.k += 1

            # Track loss, deployed error, and variance
            sums["loss"] += loss.detach()
            sums["deployed_mse"] += ((self.z_hat - z_t)**2).sum(-1).mean()
            sums["z_var"] += torch.var(z_t, dim=0).sum(-1)

            # Act with z_hat prediction from adaptation module
            with torch.no_grad():
                a = self.policy.act_mean(obs, self.z_hat)

            # Step the environment with actions coming from predicted z_hat
            obs, reward, done, info = E.step(a)

            self._track_episodes(reward, done, info)
            self.just_reset = done

        # Step the optimizer after each rollout
        if learn:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        return {key: (v / T).item() for key, v in sums.items()}

    @torch.no_grad()
    def _update_norms(self, h: dict[str, torch.Tensor]) -> None:
        """phi's normalizers from the newest history entry only (each step counted once);
        envs that just reset are skipped: their newest entry is zero padding."""
        keep = ~self.just_reset
        if keep.any():
            self.phi.update_norm(h["bottom"][keep, -1], h["top"][keep, -1])

    def _track_episodes(self, reward: torch.Tensor, done: torch.Tensor, info: dict) -> None:
        """Add this step to the running sums. If any env is done, append ONE dict of
        (k,) tensors (k = envs done this step) to self.finished, then zero their sums.
        Copy of Phase1Trainer._track_episodes."""
        self.ep_return += reward
        self.ep_length += 1
        for name, v in info["reward_terms"].items():
            if name not in self.ep_terms:
                self.ep_terms[name] = torch.zeros_like(self.ep_return)
            self.ep_terms[name] += torch.nan_to_num(v, nan=0.0)  # "nan" crash step

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

    def log(self, it: int, stats: dict[str, float], t_iter: float) -> None:
        """TensorBoard (self.writer) and one console line; then clear self.finished.
          train/    stats from collect_and_update(), *_rel = mse / z_var, lr
          episode/  over episodes finished this iteration (skipped if none): return,
                    length, reward terms, crash rates (per reason, per scenario)
          perf/     env steps per second, iteration / physics / aero seconds
        """
        scalars = {f"train/{k}": float(v) for k, v in stats.items()}
        if stats.get("z_var", 0.0) > 0:
            for k in ("loss", "deployed_mse"):
                if k in stats:
                    scalars[f"train/{k}_rel"] = stats[k] / stats["z_var"]
        scalars["train/lr"] = self.opt.param_groups[0]["lr"]

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

        timing = {k: v - self._timing_prev.get(k, 0.0) for k, v in self.env.timing.items()}
        self._timing_prev = dict(self.env.timing)
        scalars["perf/steps_per_s"] = self.cfg.rollout_steps * self.env.n / t_iter
        scalars["perf/iteration_s"] = t_iter
        for k, v in timing.items():
            scalars[f"perf/{k}_s"] = v

        for k, v in scalars.items():
            self.writer.add_scalar(k, v, it)

        ep = (f"ret {scalars['episode/return']:8.2f}  crash {scalars['episode/crash_rate']:.3f}"
              if self.finished else "no episodes ended")
        train = "  ".join(f"{k.removeprefix('train/')} {v:.3g}" for k, v in scalars.items()
                          if k.startswith("train/") and k != "train/lr")
        print(f"it {it:5d} | {ep} | {train} | {scalars['perf/steps_per_s']:,.0f} steps/s "
              f"({t_iter:.2f}s)", flush=True)
        self.finished.clear()

    def save(self, path: str) -> None:
        """Self-contained: phi plus the frozen mu and pi, with both phases' configs."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"phi": self.phi.state_dict(),
                    "encoder": self.encoder.state_dict(),
                    "policy": self.policy.state_dict(),
                    "cfg": self.cfg,
                    "phase1_cfg": self.p1_cfg}, path)

    def train(self) -> None:
        """Warm-up rollouts (normalizers only), then per iteration: collect_and_update ->
        log / save."""
        for _ in range(self.cfg.warmup_rollouts):
            self.collect_and_update(learn=False)
        self.finished.clear()
        self._timing_prev = dict(self.env.timing)
        for it in range(self.cfg.iterations):
            t0 = time.perf_counter()
            stats = self.collect_and_update()
            self.log(it, stats, time.perf_counter() - t0)
            if (it + 1) % self.cfg.save_interval == 0 or it + 1 == self.cfg.iterations:
                self.save(self.cfg.checkpoint)
