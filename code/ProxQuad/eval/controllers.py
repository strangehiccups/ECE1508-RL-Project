"""Who flies the bottom drone during evaluation. Same interface for every phase:
act(env, obs) -> action (N, 4), called once per control step before env.step();
extras() -> per-step (N,) signals the rollout records next to its own."""

import torch

from ..envs import ProxQuadEnv
from ..envs.controller import GeometricController
from ..envs.observations import POLICY_INPUTS
from ..models import BasePolicy, EnvFactorEncoder


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
    """pi with z_hat = phi(bottom history, top history) from env.get_history(). extras():
    |z_hat - mu(e_t)| per step, the adaptation error (how fast z_hat follows z when the
    wake arrives). To implement once Phase 2 training exists."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 2 evaluation: implement with Phase 2 training")


def load_phase1(path: str, env: ProxQuadEnv) -> tuple[EnvFactorEncoder, BasePolicy]:
    """mu and pi from a Phase 1 checkpoint, built with the checkpoint's network configs
    and the env's dimensions; eval mode (normalizers frozen)."""
    ck = torch.load(path, map_location=env.dev, weights_only=False)
    cfg = ck["cfg"]
    encoder = EnvFactorEncoder(env.priv_dim, cfg.z_dim, cfg.encoder).to(env.dev)
    policy = BasePolicy(dict(POLICY_INPUTS), cfg.z_dim, env.action_dim, cfg.policy,
                        env.a_hover).to(env.dev)
    encoder.load_state_dict(ck["encoder"])
    policy.load_state_dict(ck["policy"])
    return encoder.eval(), policy.eval()
