"""Who flies the bottom drone during evaluation. Same interface for every phase:
act(env, obs) -> action (N, 4), called once per control step before env.step();
extras() -> per-step (N,) signals the rollout records next to its own."""

import torch

from ..configs import RobustControllerCfg
from ..envs import ProxQuadEnv
from ..envs.controller import GeometricController, RobustGeometricController
from ..envs.observations import POLICY_INPUTS
from ..models import AdaptationModule, BasePolicy, EnvFactorEncoder


class Controller:
    name = "controller"

    def reset(self, ids: torch.Tensor | None = None) -> None:
        """Envs ids (all if None) start a new episode."""

    def act(self, env: ProxQuadEnv, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def extras(self) -> dict[str, torch.Tensor]:
        return {}


class GeometricBaseline(Controller):
    """Lee controller on the nominal model (the top drone's gains), tracking the same
    reference as the policy. Knows nothing of the randomization or the downwash."""

    def __init__(self, env: ProxQuadEnv, name: str = "geometric"):
        self.name = name
        self.ctrl = GeometricController(env.cfg.top_controller, env.bottom, env.dt, env.n, env.dev)
        self.n, self.dev = env.n, env.dev

    def reset(self, ids=None):
        self.ctrl.reset(torch.arange(self.n, device=self.dev) if ids is None else ids)

    @torch.no_grad()
    def act(self, env, obs):
        ref = env.scenario.bottom_reference(env.steps.float() * env.dt)
        return env.omega_to_action(self.ctrl(env._bot_state, ref))


class RobustGeometricBaseline(GeometricBaseline):
    """Geometric controller + UDE (env.cfg.robust_controller): estimates the lumped
    disturbance force and torque from its own state and commands and compensates them.
    The non-learned counterpart of phi. cfg: default env.cfg.robust_controller."""

    def __init__(self, env: ProxQuadEnv, name: str = "geometric+UDE",
                 cfg: RobustControllerCfg | None = None):
        self.name = name
        self.ctrl = RobustGeometricController(cfg or env.cfg.robust_controller, env.bottom,
                                              env.dt, env.n, env.dev)
        self.n, self.dev = env.n, env.dev


class Phase1Policy(Controller):
    """pi with z = mu(e_t). Ablations: z_mode "zero" (z = 0) or "fixed" (z = z_fixed for
    every env and step). Deterministic act_mean unless stochastic. Accumulates mu's z
    over all calls, for mean_z()."""

    def __init__(self, encoder: EnvFactorEncoder, policy: BasePolicy, z_mode: str = "mu",
                 z_fixed: torch.Tensor | None = None, stochastic: bool = False,
                 name: str = "policy"):
        assert z_mode in ("mu", "zero", "fixed")
        assert z_mode != "fixed" or z_fixed is not None
        self.encoder, self.policy, self.name = encoder, policy, name
        self.z_mode, self.z_fixed, self.stochastic = z_mode, z_fixed, stochastic
        self.z_sum, self.z_count = 0.0, 0

    @torch.no_grad()
    def act(self, env, obs):
        z = self.encoder(obs["priv"])
        self.z_sum = self.z_sum + z.sum(0)
        self.z_count += z.shape[0]
        if self.z_mode == "zero":
            z = torch.zeros_like(z)
        elif self.z_mode == "fixed":
            z = self.z_fixed.expand_as(z)
        if self.stochastic:
            return self.policy(obs, z).sample()
        return self.policy.act_mean(obs, z)

    def mean_z(self) -> torch.Tensor:
        return self.z_sum / max(self.z_count, 1)


class Phase2Policy(Controller):
    """pi with z_hat = phi(bottom history, top history) at their own rates: pi acts every
    step (act_mean); phi refreshes z_hat every `decimation` steps and at the first step of
    an episode, pi holds it in between. extras(): z_mse = ||z_hat - mu(e_t)||^2 per step
    (mu only measures the error, it does not act)."""

    def __init__(self, encoder: EnvFactorEncoder, policy: BasePolicy, phi: AdaptationModule,
                 decimation: int, name: str = "policy"):
        self.encoder, self.policy, self.phi = encoder, policy, phi
        self.decimation, self.name = decimation, name
        self.k, self.z_hat, self.z_mse = 0, None, None

    def reset(self, ids=None):
        if ids is None:
            self.k, self.z_hat = 0, None

    @torch.no_grad()
    def act(self, env, obs):
        h = env.get_history()
        if self.z_hat is None or self.k % self.decimation == 0:
            self.z_hat = self.phi(h["bottom"], h["top"])
        else:
            new = env.steps == 0                      # episode just (auto-)reset
            if new.any():
                self.z_hat[new] = self.phi(h["bottom"][new], h["top"][new])
        self.k += 1
        self.z_mse = (self.z_hat - self.encoder(obs["priv"])).pow(2).sum(-1)
        return self.policy.act_mean(obs, self.z_hat)

    def extras(self):
        return {"z_mse": self.z_mse}


def _mu_pi(ck: dict, p1, env: ProxQuadEnv) -> tuple[EnvFactorEncoder, BasePolicy]:
    encoder = EnvFactorEncoder(env.priv_dim, p1.z_dim, p1.encoder).to(env.dev)
    policy = BasePolicy(dict(POLICY_INPUTS), p1.z_dim, env.action_dim, p1.policy,
                        env.a_hover).to(env.dev)
    encoder.load_state_dict(ck["encoder"])
    policy.load_state_dict(ck["policy"])
    return encoder.eval(), policy.eval()


def load_phase1(path: str, env: ProxQuadEnv) -> tuple[EnvFactorEncoder, BasePolicy]:
    """mu and pi from a Phase 1 checkpoint, built with the checkpoint's network configs
    and the env's dimensions; eval mode (normalizers frozen)."""
    ck = torch.load(path, map_location=env.dev, weights_only=False)
    return _mu_pi(ck, ck["cfg"], env)


def load_phase2(path: str, env: ProxQuadEnv):
    """(mu, pi, phi, decimation) from a Phase 2 checkpoint. The env needs the history
    lengths phi was trained with (Phase2Cfg.env.obs)."""
    ck = torch.load(path, map_location=env.dev, weights_only=False)
    cfg, p1 = ck["cfg"], ck["phase1_cfg"]
    Lb, Lt = cfg.env.obs.bottom_history_len, cfg.env.obs.top_history_len
    assert (env.cfg.obs.bottom_history_len, env.cfg.obs.top_history_len) == (Lb, Lt), \
        f"env histories must be ({Lb}, {Lt}) steps"
    encoder, policy = _mu_pi(ck, p1, env)
    phi = AdaptationModule(env.bottom_hist_dim, Lb, env.top_dim, Lt, p1.z_dim,
                           cfg.adaptation).to(env.dev)
    phi.load_state_dict(ck["phi"])
    return encoder, policy, phi.eval(), cfg.adaptation_decimation
