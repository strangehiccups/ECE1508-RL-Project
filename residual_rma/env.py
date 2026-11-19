"""
ResidualQuadEnv: hover at a target while the world pushes back.

  mode="residual" : motor command = nominal geometric controller + 0.5*hover*a   (a in [-1,1]^4)
  mode="e2e"      : motor command = hover*(1 + a)                                (like v1, no controller)

Physics at 500 Hz, policy at 100 Hz (decimation 5). The nominal controller runs at 500 Hz
(every physics substep) and the policy's residual is held constant for the 5 substeps.
The motors then apply gain_i * command, which neither the controller nor the policy knows.

Observation x_t (26):  pos error 3 | R 9 | lin vel 3 (world) | ang vel 3 (body) | f_nom/hover - 1 (4) | a_{t-1} (4)
Privileged e_t  (10):  (gain - 1)/0.2 (4) | external force / 0.1 N (3) | external torque / 1e-3 N m (3)
History entry   (22):  state before the step (18) | mean total motor command / hover - 1 (4)
"""
import math
from pathlib import Path
import torch
import mujoco
import mujoco_warp as mjw
import warp as wp

from utils import quat_to_rotmat, RunningNorm
from controller import GeometricController
from disturbances import Disturbances, GAIN_DEV

XML = str(Path(__file__).with_name("quadrotor.xml"))
NCONMAX, NJMAX = 8, 32

RES_SCALE = 0.5          # residual authority: +-50% of hover thrust per motor
F_NORM, T_NORM = 0.1, 1e-3
EPISODE_S = 10.0
RESET_SPREAD = 0.2       # +- m per axis around the target


class ResidualQuadEnv:
    def __init__(self, num_envs=1024, mode="residual", hist_len=50, rand_scale=1.0,
                 pass_prob=0.6, resample_prob=0.004, control_dt=0.01, episode_s=EPISODE_S,
                 device=None):
        assert mode in ("residual", "e2e")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.n, self.mode = device, num_envs, mode
        self.hist_len, self.rand_scale = hist_len, rand_scale
        self.pass_prob, self.resample_prob = pass_prob, resample_prob

        mjm = mujoco.MjModel.from_xml_path(XML)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self.mjm = mjm
        self.drone_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.phys_dt = mjm.opt.timestep
        self.control_dt = control_dt
        self.decimation = int(round(control_dt / self.phys_dt))
        self.max_steps = int(round(episode_s / control_dt))

        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=NCONMAX, njmax=NJMAX)
        self.qpos = wp.to_torch(self.d.qpos)
        self.qvel = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)
        self.mocap = wp.to_torch(self.d.mocap_pos)       # (N, 1, 3) intruder, visual only

        self.n_actions = mjm.nu
        self.mass = mujoco.mj_getTotalmass(mjm)
        self.f_max = float(mjm.actuator_ctrlrange[0, 1])
        self.hover = self.mass * -mjm.opt.gravity[2] / self.n_actions
        self.target = torch.tensor([0.0, 0.0, 1.0], device=device)

        self.ctrl_nom = GeometricController(device, f_max=self.f_max)
        self.dist = Disturbances(num_envs, device, self.target)

        z = lambda *s: torch.zeros(num_envs, *s, device=device)
        self.prev_action = z(4)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.manual_force = z(3)
        self.ext_force, self.ext_torque = z(3), z(3)     # what e_t reports for the coming step
        self.f_nom_now = z(4)
        self.last_cmd, self.last_thrust = z(4), z(4)     # for logging control effort
        self.last_crashed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.last_timeout = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.use_residual = True                         # play.py can switch the residual off
        self.auto_reset = True                           # evaluate.py turns this off

        self.obs_dim, self.env_dim, self.state_dim = 26, 10, 18
        self.hist_dim = self.state_dim + 4
        self.hist = z(hist_len, self.hist_dim)
        self.reset(torch.arange(num_envs, device=device))

    # ---- state helpers ---------------------------------------------------------
    def _kin(self):
        pos, quat = self.qpos[:, 0:3], self.qpos[:, 3:7]
        return pos, self.qvel[:, 0:3], quat_to_rotmat(quat), self.qvel[:, 3:6]

    def _nominal(self):
        pos, vel, R, om = self._kin()
        return self.ctrl_nom(pos, vel, R, om, self.target.expand(self.n, 3))

    def _state18(self):
        pos, vel, R, om = self._kin()
        return torch.cat([pos - self.target, R.reshape(self.n, 9), vel, om], -1)

    def _time(self, sub=0):
        return (self.steps * self.decimation + sub).float() * self.phys_dt

    def _refresh_ext(self):
        """external force/torque the drone will feel at the start of the next step (for e_t)."""
        pos, _, R, _ = self._kin()
        f_dw, t_dw = self.dist.downwash(pos, R, self._time(), self.rand_scale)
        self.ext_force = self.dist.wind + self.dist.gust + f_dw
        self.ext_torque = t_dw

    def reset(self, ids):
        k = len(ids)
        self.qpos[ids, 0:3] = self.target + (torch.rand(k, 3, device=self.device) * 2 - 1) * RESET_SPREAD
        self.qpos[ids, 3] = 1.0
        self.qpos[ids, 4:7] = 0.0
        self.qvel[ids] = 0.0
        self.ctrl[ids] = self.hover
        self.prev_action[ids] = 0.0
        self.steps[ids] = 0
        self.hist[ids] = 0.0
        self.manual_force[ids] = 0.0
        self.dist.sample(ids, self.rand_scale, self.pass_prob)
        self._after_change()

    def _after_change(self):
        """call after editing qpos / disturbances by hand (reset, evaluation scenarios, play.py)."""
        self.f_nom_now = self._nominal()
        self._refresh_ext()

    def obs(self):
        s = self._state18()
        fn = self.f_nom_now / self.hover - 1.0 if self.mode == "residual" else torch.zeros_like(self.f_nom_now)
        return torch.cat([s, fn, self.prev_action], -1)

    def privileged(self):
        return torch.cat([(self.dist.gain - 1.0) / GAIN_DEV,
                          self.ext_force / F_NORM,
                          self.ext_torque / T_NORM], -1)

    def norm_hist(self, norm: RunningNorm):
        sd = self.state_dim
        x = ((self.hist[..., :sd] - norm.mean[:sd]) / torch.sqrt(norm.var[:sd] + 1e-8)).clamp(-10, 10)
        return torch.cat([x, self.hist[..., sd:]], -1)

    def apply_disturbance(self, force_world):
        self.manual_force[:] = force_world

    def intruder_pos(self):
        return self.dist.intruder_pos(self._time())

    # ---- step --------------------------------------------------------------------
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        state_pre = self._state18()
        cmd_sum = torch.zeros(self.n, 4, device=self.device)
        thrust_sum = torch.zeros(self.n, 4, device=self.device)

        for k in range(self.decimation):
            pos, vel, R, om = self._kin()
            if self.mode == "residual":
                f_nom = self.ctrl_nom(pos, vel, R, om, self.target.expand(self.n, 3))
                res = RES_SCALE * self.hover * a if self.use_residual else 0.0
                cmd = (f_nom + res).clamp(0.0, self.f_max)
            else:
                cmd = (self.hover * (1.0 + a)).clamp(0.0, self.f_max)
            thrust = (self.dist.gain * cmd).clamp(0.0, self.f_max)
            f_dw, t_dw = self.dist.downwash(pos, R, self._time(k), self.rand_scale)
            self.ctrl[:] = thrust
            self.xfrc[:, self.drone_id, 0:3] = self.dist.wind + self.dist.gust + f_dw + self.manual_force
            self.xfrc[:, self.drone_id, 3:6] = t_dw
            if self.device == "cuda":
                torch.cuda.synchronize()
            mjw.step(self.m, self.d)
            wp.synchronize()
            cmd_sum += cmd
            thrust_sum += thrust

        self.steps += 1
        cmd_mean = cmd_sum / self.decimation
        self.last_cmd, self.last_thrust = cmd_mean, thrust_sum / self.decimation
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = torch.cat([state_pre, cmd_mean / self.hover - 1.0], -1)

        # ---- reward
        pos, vel, R, om = self._kin()
        e = (pos - self.target).norm(dim=-1)
        upright = R[:, 2, 2]
        crashed = (pos[:, 2] < 0.05) | (upright < 0.2) | (e > 3.0) | ~torch.isfinite(pos).all(-1)
        timeout = self.steps >= self.max_steps
        reward = (torch.exp(-e / 0.2)
                  - 0.02 * om.norm(dim=-1)
                  - 0.05 * (a ** 2).sum(-1)
                  - 0.01 * (a - self.prev_action).norm(dim=-1)
                  - 10.0 * crashed.float())
        reward = torch.nan_to_num(reward, nan=-10.0)
        done = crashed | timeout
        self.last_crashed, self.last_timeout = crashed, timeout & ~crashed
        self.prev_action = a.clone()

        # ---- disturbances evolve (gust at 100 Hz, occasional re-draw of gains + wind)
        self.dist.step_gust(self.control_dt, self.rand_scale)
        if self.resample_prob > 0:
            rs = (torch.rand(self.n, device=self.device) < self.resample_prob).nonzero().squeeze(-1)
            if len(rs) > 0:
                self.dist.sample_slow(rs, self.rand_scale)

        ids = done.nonzero().squeeze(-1)
        if len(ids) > 0 and self.auto_reset:
            self.reset(ids)          # also refreshes f_nom and e_t
        else:
            self._after_change()
        self.mocap[:, 0] = self.intruder_pos()
        return self.obs(), reward, done
