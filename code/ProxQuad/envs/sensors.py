"""State estimates seen by the bottom drone's controller (SensorCfg), for both drones:
each drone's true state delayed by d control steps (d drawn per episode, per drone), plus
white noise per axis on position, velocity, attitude (body-frame rotation) and body rate.
"""

import torch

from ..configs import SensorCfg
from ..utils.math import quat_from_rotvec, quat_mul, quat_to_rotmat
from .drone import DroneState

DRONES = ("bottom", "top")
FIELDS = (("pos", 3), ("vel", 3), ("quat", 4), ("omega", 3))


class SensorModel:
    def __init__(self, cfg: SensorCfg, num_envs: int, device: torch.device):
        self.cfg, self.n, self.dev = cfg, num_envs, device
        self.ranges = {"bottom": cfg.bottom_delay_steps, "top": cfg.top_delay_steps}
        assert all(0 <= lo <= hi for lo, hi in self.ranges.values())
        self.L = max(hi for _, hi in self.ranges.values()) + 1
        self.gen = torch.Generator(device=device)
        self.buf = {k: {f: torch.zeros(self.L, num_envs, d, device=device) for f, d in FIELDS}
                    for k in DRONES}
        self.delay = {k: torch.zeros(num_envs, dtype=torch.long, device=device) for k in DRONES}
        self.head = 0                                  # slot of the newest sample
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
        self.std = {"pos": t(cfg.pos_std), "vel": t(cfg.vel_std),
                    "att": torch.deg2rad(t(cfg.att_std_deg)), "omega": t(cfg.ang_vel_std)}
        self.envs = torch.arange(num_envs, device=device)

    def seed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

    def reset(self, ids: torch.Tensor) -> None:
        """New episode: draw delays; the next measure() fills the buffer with that state."""
        for k, (lo, hi) in self.ranges.items():
            self.delay[k][ids] = torch.randint(lo, hi + 1, (len(ids),), device=self.dev,
                                               generator=self.gen)
        self.fresh[ids] = True

    def measure(self, bot: DroneState, top: DroneState) -> tuple[DroneState, DroneState]:
        """Push the true states of this step, return the measured ones."""
        self.head = (self.head + 1) % self.L
        new = self.fresh[None, :, None]
        out = []
        for k, s in zip(DRONES, (bot, top)):
            b = self.buf[k]
            for f, _ in FIELDS:
                x = getattr(s, f)
                b[f][self.head] = x
                b[f].copy_(torch.where(new, x[None], b[f]))
            slot = (self.head - self.delay[k]) % self.L
            out.append(self._noisy({f: b[f][slot, self.envs] for f, _ in FIELDS}))
        self.fresh[:] = False
        return out[0], out[1]

    def _noisy(self, x: dict[str, torch.Tensor]) -> DroneState:
        s = self.cfg.noise_scale
        g = lambda v: torch.randn(v.shape, device=self.dev, generator=self.gen)  # noqa: E731
        pos = x["pos"] + s * self.std["pos"] * g(x["pos"])
        vel = x["vel"] + s * self.std["vel"] * g(x["vel"])
        dq = quat_from_rotvec(s * self.std["att"] * g(x["omega"]))
        quat = quat_mul(x["quat"], dq)                  # body-frame perturbation
        quat = quat / quat.norm(dim=-1, keepdim=True)
        omega = x["omega"] + s * self.std["omega"] * g(x["omega"])
        return DroneState(pos=pos, quat=quat, R=quat_to_rotmat(quat), vel=vel, omega=omega)
