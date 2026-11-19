"""
Everything the nominal controller does not know about (the "residual dynamics"):

  1. motor gains      each motor produces gain_i x the commanded thrust      (per episode)
  2. steady wind      constant force                                          (per episode)
  3. gusts            Ornstein-Uhlenbeck force, time constant 0.5 s           (every step)
  4. overhead pass    a scripted drone flies over the controlled one; its downwash pushes
                      down on each rotor and twists the drone when it is off-centre

All ranges are multiplied by `scale` (the curriculum / test multiplier, 1 = training range).

Downwash model (simplified, documented so it can be checked or replaced):
  the jet below the intruder is a Gaussian in horizontal distance d, with
      peak force  F_peak(dz) = 0.30 * W * (0.25 / dz)       W = weight of the controlled drone
      width       sigma(dz)  = 0.04 + 0.15 * dz            [m]
  i.e. about 30% / 15% / 10% of the drone's weight at 0.25 / 0.5 / 0.75 m below the intruder.
  The force is split over the 4 rotor discs, so an off-centre jet also produces roll/pitch torque.
  These numbers are chosen to be plausible, not fitted to data.
"""
import math
import torch

WEIGHT = 0.027 * 9.81          # [N]
PEAK_FRAC, DZ_REF = 0.30, 0.25
SIGMA0, SIGMA_SLOPE = 0.04, 0.15
DZ_MIN = 0.10                  # clamp to avoid blow-up if the intruder gets very close

GAIN_DEV = 0.20                # motor gains in [1-0.2, 1+0.2]
WIND_MAX = 0.05                # N per axis
GUST_STD, GUST_TAU = 0.02, 0.5 # N, s
PASS_DZ = (0.25, 0.75)         # m above the controlled drone's target
PASS_SPEED = (0.2, 0.5)        # m/s
PASS_OFFSET = 0.15             # m, lateral miss distance
PASS_T0 = (0.0, 3.0)           # s, when the intruder starts its 2 m straight line
PASS_HALF_LEN = 1.0            # m, path goes from -1 m to +1 m relative to the target

ROTORS = torch.tensor([[0.028, 0.028, 0.0], [-0.028, 0.028, 0.0],
                       [-0.028, -0.028, 0.0], [0.028, -0.028, 0.0]])


class Disturbances:
    def __init__(self, n, device, target):
        self.n, self.device = n, device
        self.target = target                                       # (3,)
        z = lambda *s: torch.zeros(n, *s, device=device)
        self.gain, self.wind, self.gust = torch.ones(n, 4, device=device), z(3), z(3)
        self.pass_on = torch.zeros(n, dtype=torch.bool, device=device)
        self.pass_dz, self.pass_speed, self.pass_off, self.pass_t0 = z(), z(), z(), z()
        self.pass_dir = torch.zeros(n, 2, device=device); self.pass_dir[:, 0] = 1.0
        self.gust_on = torch.ones(n, 1, device=device)     # 0 switches gusts off (evaluation scenarios)
        self.rotors = ROTORS.to(device)

    # ---- sampling ------------------------------------------------------------
    def _u(self, k, lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(k, *shape, device=self.device)

    def sample_slow(self, ids, scale):
        """motor gains + steady wind (also used for mid-episode re-sampling)."""
        k = len(ids)
        self.gain[ids] = 1.0 + self._u(k, -GAIN_DEV, GAIN_DEV, 4) * scale
        self.wind[ids] = self._u(k, -WIND_MAX, WIND_MAX, 3) * scale

    def sample(self, ids, scale, pass_prob):
        k = len(ids)
        self.sample_slow(ids, scale)
        self.gust_on[ids] = 1.0
        self.gust[ids] = GUST_STD * scale * torch.randn(k, 3, device=self.device)   # stationary start
        self.pass_on[ids] = torch.rand(k, device=self.device) < pass_prob
        self.pass_dz[ids] = self._u(k, *PASS_DZ)
        self.pass_speed[ids] = self._u(k, *PASS_SPEED)
        self.pass_off[ids] = self._u(k, -PASS_OFFSET, PASS_OFFSET)
        self.pass_t0[ids] = self._u(k, *PASS_T0)
        ang = self._u(k, 0.0, 2 * math.pi)
        self.pass_dir[ids] = torch.stack([torch.cos(ang), torch.sin(ang)], -1)

    def clear(self, ids):
        """no disturbance at all (used by evaluation scenarios)."""
        self.gain[ids] = 1.0; self.wind[ids] = 0.0; self.gust[ids] = 0.0; self.pass_on[ids] = False
        self.gust_on[ids] = 0.0

    def force_pass(self, ids, dz, speed=0.3, offset=0.0, t0=0.5, heading=0.0):
        self.pass_on[ids] = True
        self.pass_dz[ids], self.pass_speed[ids] = dz, speed
        self.pass_off[ids], self.pass_t0[ids] = offset, t0
        self.pass_dir[ids] = torch.tensor([math.cos(heading), math.sin(heading)], device=self.device)

    # ---- time evolution ------------------------------------------------------
    def step_gust(self, dt, scale):
        a = math.exp(-dt / GUST_TAU)
        noise = math.sqrt(1 - a * a) * GUST_STD * scale * torch.randn_like(self.gust)
        self.gust = (a * self.gust + noise) * self.gust_on

    def intruder_pos(self, t):
        """(N,3) world position of the intruder at time t (N,). Far away when inactive."""
        s = self.pass_speed * (t - self.pass_t0) - PASS_HALF_LEN
        perp = torch.stack([-self.pass_dir[:, 1], self.pass_dir[:, 0]], -1)
        xy = self.target[:2] + self.pass_dir * s.unsqueeze(-1) + perp * self.pass_off.unsqueeze(-1)
        zc = (self.target[2] + self.pass_dz).unsqueeze(-1)
        p = torch.cat([xy, zc], -1)
        active = self.pass_on & (t >= self.pass_t0) & (s <= PASS_HALF_LEN)
        far = torch.tensor([0.0, 0.0, 101.0], device=self.device)
        return torch.where(active.unsqueeze(-1), p, far)

    def downwash(self, pos, R, t, scale):
        """force (N,3) and torque (N,3), world frame, from the intruder's downwash."""
        p_int = self.intruder_pos(t)                                   # (N,3)
        r_world = torch.einsum('nij,kj->nki', R, self.rotors)          # (N,4,3) rotor offsets
        rotor_pos = pos.unsqueeze(1) + r_world
        dz = (p_int[:, 2:3] - rotor_pos[..., 2])                       # (N,4) intruder height above each rotor
        d2 = ((p_int[:, None, :2] - rotor_pos[..., :2]) ** 2).sum(-1)  # (N,4) horizontal distance^2
        dz_c = dz.clamp(min=DZ_MIN)
        sigma = SIGMA0 + SIGMA_SLOPE * dz_c
        peak = PEAK_FRAC * WEIGHT * (DZ_REF / dz_c) * scale
        f = (peak / 4.0) * torch.exp(-d2 / (2 * sigma ** 2)) * (dz > 0.05)   # (N,4), downward
        fz = -f
        force = torch.zeros_like(pos); force[:, 2] = fz.sum(-1)
        rotor_f = torch.zeros_like(r_world); rotor_f[..., 2] = fz
        torque = torch.linalg.cross(r_world, rotor_f).sum(1)
        return force, torque
