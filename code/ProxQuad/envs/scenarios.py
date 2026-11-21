"""Per-episode scenarios: reference paths and start states of both drones, batched over envs.

Bottom drone: hover at `target`, except in the circle scenarios.

Top drone: start point A, hold point B, end point C,
    t < t0          hold at A
    t0 <= t < t1    fly A -> B at constant speed
    t1 <= t < t2    hold at B (dwell)
    t2 <= t < t3    fly B -> C at constant speed
    t >= t3         hold at C
plus a small horizontal sinusoidal sway, so a "hover" still varies the overlap.

    hover_over            A = B = C, B above the target with a random lateral offset
    pass_over             A -> C straight through B, no dwell
    approach_hover_leave  A -> B, dwell, B -> C (C on a new random heading)
    far                   A = B = C, laterally far from the target (no downwash)

Circle scenarios (ScenarioCfg.circle): both drones circle the vertical axis through the
target, the bottom drone at the target's height, the top drone dz higher (+ sway):
    circle_same           same angular rate, the top drone at a fixed arc offset
    circle_opposite       opposite angular rate, random phase: crossings every pi r / v
When no circle scenario is enabled, bottom_reference stays the plain set-point (acc,
jerk = None), so the env keeps its set-point error path.
"""

import math

import torch

from ..configs import ScenarioCfg
from ..utils.math import uniform
from .controller import FlatRef

SCENARIOS = ("hover_over", "pass_over", "approach_hover_leave", "far",
             "circle_same", "circle_opposite")
CIRCLES = ("circle_same", "circle_opposite")
NEVER = 1e6  # "t = infinity" that stays finite in arithmetic


def circle(t, r, w, th0, center, z):
    """Horizontal circle, batched: angle th0 + w t (w signed) -> pos, vel, acc, jerk (N, 3)."""
    th = th0 + w * t
    c, s, zero = th.cos(), th.sin(), torch.zeros_like(th)
    u = torch.stack([c, s, zero], dim=-1)          # radial
    v = torch.stack([-s, c, zero], dim=-1)         # tangential
    pos = torch.stack([center[0] + r * c, center[1] + r * s, z.expand_as(th)], dim=-1)
    return (pos, (r * w)[:, None] * v, (-r * w**2)[:, None] * u, (-r * w**3)[:, None] * v)


class Scenarios:
    def __init__(self, cfg: ScenarioCfg, collision_radii: tuple[float, float], num_envs: int,
                 device: torch.device):
        unknown = set(cfg.top.scenario_probs) - set(SCENARIOS)
        assert not unknown, f"unknown scenarios {unknown}"
        # starting on the reference must be clear of the other drone (start_state fallback)
        assert cfg.top.dz[0] > cfg.start_clearance * collision_radii[1]
        self.cfg, self.radii, self.n, self.dev = cfg, collision_radii, num_envs, device
        self.target = torch.tensor(cfg.bottom.target, dtype=torch.float32, device=device)
        self.probs = torch.tensor([cfg.top.scenario_probs.get(k, 0.0) for k in SCENARIOS],
                                  device=device)
        z3 = lambda: torch.zeros(num_envs, 3, device=device)  # noqa: E731
        z1 = lambda: torch.zeros(num_envs, device=device)     # noqa: E731
        self.kind = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.A, self.B, self.C = z3(), z3(), z3()   # start, hold, end points of the top drone's reference path
        self.t0, self.t1, self.t2, self.t3 = z1(), z1(), z1(), z1()    # start, hold, dwell, end times of the top drone's reference path
        self.sway_dir, self.sway_amp, self.sway_w, self.sway_phase = z3(), z1(), z1(), z1()
        # circles: radius, signed angular rate, start angle; bottom (b) and top (t)
        self.circles_on = any(cfg.top.scenario_probs.get(k, 0.0) > 0 for k in CIRCLES)
        self.is_circle = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.r_b, self.w_b, self.th_b = z1(), z1(), z1()
        self.r_t, self.w_t, self.th_t, self.z_t = z1(), z1(), z1(), z1()

    def resample(self, ids: torch.Tensor) -> None:
        cfg, k, dev = self.cfg.top, len(ids), self.dev
        kind = torch.multinomial(self.probs, k, replacement=True)

        def heading():
            psi = 2 * math.pi * torch.rand(k, device=dev)
            return torch.stack([psi.cos(), psi.sin(), torch.zeros_like(psi)], dim=-1)

        dz = uniform(cfg.dz, k, dev)
        r = uniform(cfg.lateral_offset, k, dev)
        u_off, u_in, u_out = heading(), heading(), heading()
        up = torch.stack([torch.zeros_like(dz), torch.zeros_like(dz), dz], dim=-1)
        B = self.target + r[:, None] * u_off + up
        D = cfg.travel_distance
        v = uniform(cfg.speed, k, dev)
        t_start = uniform(cfg.start_delay, k, dev)
        dwell = uniform(cfg.dwell, k, dev)
        leg = D / v

        is_ = lambda name: kind == SCENARIOS.index(name)  # noqa: E731
        moving = is_("pass_over") | is_("approach_hover_leave")
        # far: hold point moved laterally away from the target
        B = torch.where(is_("far")[:, None], self.target + cfg.far_distance * u_off + up, B)
        # pass_over leaves along the approach direction; approach_hover_leave on a new one
        u_out = torch.where(is_("pass_over")[:, None], u_in, u_out)
        A = torch.where(moving[:, None], B - D * u_in, B)
        C = torch.where(moving[:, None], B + D * u_out, B)
        t0 = torch.where(moving, t_start, torch.zeros_like(t_start))
        t1 = torch.where(moving, t0 + leg, torch.zeros_like(t0))
        t2 = torch.where(is_("pass_over"), t1,
                         torch.where(is_("approach_hover_leave"), t1 + dwell,
                                     torch.full_like(t1, NEVER)))
        t3 = torch.where(moving, t2 + leg, torch.full_like(t2, NEVER))

        self.kind[ids] = kind
        self.A[ids], self.B[ids], self.C[ids] = A, B, C
        self.t0[ids], self.t1[ids], self.t2[ids], self.t3[ids] = t0, t1, t2, t3
        self.sway_dir[ids] = heading()
        self.sway_amp[ids] = uniform(cfg.sway_amp, k, dev)
        self.sway_w[ids] = 2 * math.pi * uniform(cfg.sway_freq, k, dev)
        self.sway_phase[ids] = 2 * math.pi * torch.rand(k, device=dev)

        # circles (drawn for every env, used where is_circle)
        c = self.cfg.circle
        same = is_("circle_same")
        r_b = uniform(c.radius, k, dev)
        sign = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
        w_b = sign * uniform(c.speed, k, dev) / r_b
        th_b = 2 * math.pi * torch.rand(k, device=dev)
        lead = th_b + sign * uniform(c.arc_offset, k, dev) / r_b
        self.is_circle[ids] = same | is_("circle_opposite")
        self.r_b[ids], self.w_b[ids], self.th_b[ids] = r_b, w_b, th_b
        self.r_t[ids] = r_b + uniform(c.radius_offset, k, dev)
        self.w_t[ids] = torch.where(same, w_b, -w_b)
        self.th_t[ids] = torch.where(same, lead, 2 * math.pi * torch.rand(k, device=dev))
        self.z_t[ids] = self.target[2] + dz

    def bottom_reference(self, t: torch.Tensor) -> FlatRef:
        """Hover at the target (acc, jerk, yaw = None) unless a circle scenario is
        enabled; then pos, vel, acc, jerk for all envs, the circle where is_circle."""
        pos = self.target.expand(self.n, 3)
        if not self.circles_on:
            return FlatRef(pos=pos, vel=torch.zeros_like(pos))
        cp, cv, ca, cj = circle(t, self.r_b, self.w_b, self.th_b, self.target, self.target[2])
        m, zero = self.is_circle[:, None], torch.zeros_like(pos)
        return FlatRef(pos=torch.where(m, cp, pos), vel=torch.where(m, cv, zero),
                       acc=torch.where(m, ca, zero), jerk=torch.where(m, cj, zero))

    def top_reference(self, t: torch.Tensor) -> FlatRef:
        """t: (N,) seconds since episode start -> FlatRef with pos, vel, acc, jerk (N, 3);
        yaw = 0. Off the circles, acc and jerk come from the sway only: the straight legs
        are constant velocity, and the impulses at their corners are left out."""
        d_in = (self.t1 - self.t0).clamp_min(1e-6)
        d_out = (self.t3 - self.t2).clamp_min(1e-6)
        a_in = ((t - self.t0) / d_in).clamp(0.0, 1.0)[:, None]
        a_out = ((t - self.t2) / d_out).clamp(0.0, 1.0)[:, None]
        pos = torch.where((t < self.t1)[:, None],
                          self.A + a_in * (self.B - self.A),
                          self.B + a_out * (self.C - self.B))
        in_leg = ((t >= self.t0) & (t < self.t1)).float()[:, None]
        out_leg = ((t >= self.t2) & (t < self.t3)).float()[:, None]
        vel = in_leg * (self.B - self.A) / d_in[:, None] + out_leg * (self.C - self.B) / d_out[:, None]
        acc, jerk = torch.zeros_like(pos), torch.zeros_like(pos)
        if self.circles_on:
            cp, cv, ca, cj = circle(t, self.r_t, self.w_t, self.th_t, self.target, self.z_t)
            m = self.is_circle[:, None]
            pos, vel = torch.where(m, cp, pos), torch.where(m, cv, vel)
            acc, jerk = torch.where(m, ca, acc), torch.where(m, cj, jerk)

        # sway: A sin(ph) along sway_dir, ph = w t + phase, differentiated exactly
        A, w, d = self.sway_amp, self.sway_w, self.sway_dir
        ph = w * t + self.sway_phase
        s, c = ph.sin(), ph.cos()
        pos = pos + (A * s)[:, None] * d
        vel = vel + (A * w * c)[:, None] * d
        acc = acc + (-A * w**2 * s)[:, None] * d
        jerk = jerk + (-A * w**3 * c)[:, None] * d
        return FlatRef(pos=pos, vel=vel, acc=acc, jerk=jerk)

    def start_state(self, ids: torch.Tensor):
        """(bottom pos, bottom vel, top pos, top vel) at t = 0 for envs ids. The top drone
        starts on its reference, the bottom drone around its reference with uniform noise,
        redrawn where it would start too close to the top drone."""
        b, k, dev = self.cfg.bottom, len(ids), self.dev
        t = torch.zeros(self.n, device=dev)
        bot, top = self.bottom_reference(t), self.top_reference(t)
        bot_p, top_p = bot.pos[ids], top.pos[ids]
        noise = lambda s, m: (2 * torch.rand(m, 3, device=dev) - 1) * s  # noqa: E731
        off = noise(b.init_pos_noise, k)
        for _ in range(10):
            bad = self._too_close(top_p - bot_p - off)
            if not bad.any():
                break
            off[bad] = noise(b.init_pos_noise, int(bad.sum()))
        off[self._too_close(top_p - bot_p - off)] = 0.0
        return bot_p + off, bot.vel[ids] + noise(b.init_vel_noise, k), top_p, top.vel[ids]

    def _too_close(self, d: torch.Tensor) -> torch.Tensor:
        """d = p_top - p_bottom (k, 3): inside the collision ellipsoid x start_clearance."""
        c = self.cfg.start_clearance
        r_xy, r_z = c * self.radii[0], c * self.radii[1]
        return (d[:, :2] / r_xy).pow(2).sum(-1) + (d[:, 2] / r_z).pow(2) < 1.0
