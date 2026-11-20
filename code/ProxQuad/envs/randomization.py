"""Per-episode domain randomization of the BOTTOM drone. The top drone stays nominal.

mass, inertia, CoM offset -> per-world MuJoCo-Warp model fields
motor factor, rotor lag lambda, gust -> torch (read by the motor model / env)
"""

import torch
import warp as wp

import mujoco_warp as mjw

from ..configs import RandomizationCfg
from ..utils.math import uniform
from .drone import DroneIndex, DroneParams

# Fields written per world. body_subtreemass is the only derived constant the dynamics
# of a single free body read; it is updated directly in commit() (see there).
# body_ipos (CoM) has no derived constants: subtree_com is recomputed every step.
FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_subtreemass")


def expand_model_fields(model: mjw.Model, nworld: int, fields) -> dict[str, torch.Tensor]:
    """Tile model fields from shape (1, ...) to (nworld, ...). Returns torch views that
    share memory with the new Warp arrays (the dict also keeps them alive)."""
    views = {}
    for name in fields:
        arr = getattr(model, name)
        src = wp.to_torch(arr)
        if src.shape[0] == 1 and nworld > 1:
            dst = src.repeat(nworld, *([1] * (src.dim() - 1))).contiguous()
            setattr(model, name, wp.from_torch(dst, dtype=arr.dtype))
            src = dst
        views[name] = src
    return views


class DomainRandomizer:
    def __init__(self, cfg: RandomizationCfg, model: mjw.Model, bottom: DroneParams,
                 bottom_idx: DroneIndex, num_envs: int, device: torch.device):
        self.cfg, self.p, self.idx, self.dev = cfg, bottom, bottom_idx, device
        self.fields = expand_model_fields(model, num_envs, FIELDS)
        self.gust = torch.zeros(num_envs, 3, device=device)   # world frame [N]
        self.com_half_range = torch.tensor(cfg.com_offset, device=device)   # (3,) [m]
        self._dirty = False
        self._check_inertia_ranges()

    def _check_inertia_ranges(self) -> None:
        """Every sampled inertia must be physically valid: each principal moment no
        larger than the sum of the other two. The inequalities are linear in the two
        factors, so the four corners of the ranges cover every sample."""
        cfg, I0 = self.cfg, self.p.inertia_nom.tolist()
        for f_xy in cfg.inertia_factor_xy:
            for f_z in cfg.inertia_factor_z:
                I = (I0[0] * f_xy, I0[1] * f_xy, I0[2] * f_z)
                for a in range(3):
                    assert I[a] <= I[(a + 1) % 3] + I[(a + 2) % 3], (
                        f"inertia ranges allow invalid inertia {I} at factors "
                        f"xy = {f_xy}, z = {f_z} (triangle inequality)")

    def resample(self, ids: torch.Tensor) -> None:
        cfg, p, k, dev = self.cfg, self.p, len(ids), self.dev
        p.mass_factor[ids] = uniform(cfg.mass_factor, k, dev)
        p.inertia_factor[ids, 0] = uniform(cfg.inertia_factor_xy, k, dev)
        p.inertia_factor[ids, 1] = uniform(cfg.inertia_factor_z, k, dev)
        p.motor_factor[ids] = uniform(cfg.motor_factor, (k, 4), dev)
        if p.motor.rotor_lag:
            p.rotor_lambda[ids] = uniform(cfg.rotor_lambda, k, dev)
        self.gust[ids] = cfg.gust_force * p.weight_nom * (2 * torch.rand(k, 3, device=dev) - 1)
        self.fields["body_mass"][ids, self.idx.body] = p.mass[ids]
        self.fields["body_inertia"][ids, self.idx.body] = p.inertia[ids]
        p.com_offset[ids] = self.com_half_range * (2 * torch.rand(k, 3, device=dev) - 1)
        self.fields["body_ipos"][ids, self.idx.body] = p.com_offset[ids]
        self._dirty = True

    def commit(self) -> None:
        """Update derived constants after resample(); call once per step after resets.

        Writes body_subtreemass directly instead of calling mjw.set_const: each drone is
        one body under the world, so that is the only derived constant the dynamics read
        (invweight0 only scales floor-contact softness). set_const costs ~4 ms on all
        worlds at 4096 envs, more than the physics step, and resets happen every step.
        """
        if self._dirty:
            mass, sub = self.fields["body_mass"], self.fields["body_subtreemass"]
            sub[:, 1:] = mass[:, 1:]          # every non-world body is a leaf under world
            sub[:, 0] = mass.sum(-1)
            self._dirty = False
