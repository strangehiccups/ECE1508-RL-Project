import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from utils import RunningNorm
from pathlib import Path

# ---- mujoco sim parameters ----------------------------------------- 
XML = str(Path(__file__).with_name("quadrotor.xml"))
NCONMAX = 8 # max average (over all worlds) collision contacts (size to your VRAM limits)
NJMAX = 32  # max per-world constraints (size to your VRAM limits)

# ---- environment parameters ----------------------------------------
GUST_MAX = 0.02     # N per axis, half-range of the training distribution
THRUST_DEV = 0.10   # +-10% per-motor gain, half-range of the training distribution
GUST_RANGE = 2.0*GUST_MAX
THRUST_DEV_RANGE = 2.0*THRUST_DEV

def quat_to_rotmat(q):
    """q: (N,4) in (w,x,y,z) -> (N,3,3)"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


class QuadHoverEnv:
    def __init__(self,
                 num_envs=1024,
                 hist_len=50,
                 rand_scale=1.0,
                 resample_prob=0.004,
                 control_dt=0.01,
                 simulation_horizon=5,
                 device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        mjm = mujoco.MjModel.from_xml_path(XML)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self.drone_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "drone")

        self.n = num_envs
        self.hist_len = hist_len
        self.rand_scale = rand_scale
        self.resample_prob = resample_prob
        self.control_dt = control_dt
        self.decimation = int(self.control_dt // mjm.opt.timestep)  # default 0.01/0.002=5
        self.max_steps = int(simulation_horizon // self.control_dt) # default 5/0.01=500
        self.device = device
        self.target = torch.tensor([0.0, 0.0, 1.0], device=self.device)        

        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(mjm)
            # small contact/constraint buffers: only the floor can touch the drone
            self.d = mjw.put_data(mjm, mjd, nworld=self.n, nconmax=NCONMAX, njmax=NJMAX)

        # torch views that share GPU memory with the Warp arrays
        self.qpos = wp.to_torch(self.d.qpos)            # (N, 7): [x,y,z,qw,qx,qy,qz]
        self.qvel = wp.to_torch(self.d.qvel)            # (N, 6): [vx,vy,vz,wx,wy,wz]
        self.ctrl = wp.to_torch(self.d.ctrl)            # (N, 4): [F1,F2,F3,F4] motor thrusts
        self.xfrc = wp.to_torch(self.d.xfrc_applied)    # (N, nbody, 6): [fx,fy,fz,tx,ty,tz] external forces(+torques) on rigid body, nbody = no. of rigid bodies in env (world: 0, drone1: 1, etc)

        self.n_actions = mjm.nu                         # no. of control variables: 4 thrust values for 4 props
        self.mass = mujoco.mj_getTotalmass(mjm)         # [kg]
        self.min_thrust = torch.tensor(mjm.actuator_ctrlrange[:, 0]).to(device)    # [N]
        self.max_thrust = torch.tensor(mjm.actuator_ctrlrange[:, 1]).to(device)    # [N]
        self.hover = self.mass * -mjm.opt.gravity[2] / self.n_actions   # [N] average force per prop required to keep quadrotor hovering 
        self.prev_action = torch.zeros(num_envs, self.n_actions, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # for randomizing initial positions and orientations in reset()
        self.gust = torch.zeros(num_envs, 3, device=self.device)                       # disturbance force
        self.thrust_scale = torch.ones(num_envs, self.n_actions, device=self.device)   # per-motor gain

        # dimensions
        self.force_dim = self.n_actions + self.gust.shape[1]
        self.obs_dim = self.obs().shape[1]
        self.state_dim = self.obs_dim - self.n_actions

        # state/action history for the adaptation module
        self.hist = torch.zeros(num_envs, hist_len, self.obs_dim, device=self.device)
        self.last_crashed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self.reset(torch.arange(num_envs, device=self.device))

    # ---- state helpers -------------------------------------------------
    def _sample_disturbance(self, ids):
        k = len(ids)
        self.gust[ids] = (torch.rand(k, 3, device=self.device) - 0.5) * GUST_RANGE * self.rand_scale
        self.thrust_scale[ids] = 1.0 + (torch.rand(k, 4, device=self.device) - 0.5) * THRUST_DEV_RANGE * self.rand_scale

    def reset(self, ids):
        k = len(ids)
        self.qpos[ids, 0:3] = torch.tensor([0., 0., 1.], device=self.device) \
            + (torch.rand(k, 3, device=self.device) - 0.5) * 0.4
        self.qpos[ids, 3] = 1.0
        self.qpos[ids, 4:7] = 0.0
        self.qvel[ids] = 0.0
        self.ctrl[ids] = self.hover
        self.prev_action[ids] = 0.0
        self.steps[ids] = 0
        self.hist[ids] = 0.0
        self._sample_disturbance(ids)

    def obs(self):
        pos_err = self.qpos[:, 0:3] - self.target
        R = quat_to_rotmat(self.qpos[:, 3:7]).reshape(self.n, 9)    # flatten rotation matrices
        lin_vel = self.qvel[:, 0:3]          # world frame
        ang_vel = self.qvel[:, 3:6]          # body frame (free-joint convention)
        return torch.cat([pos_err, R, lin_vel, ang_vel, self.prev_action], dim=-1)  # 3+9+3+3+4=22

    def privileged(self):
        """e_t: (N, 7), normalized by the *training* half-ranges (so OOD values can exceed +-1)."""
        return torch.cat([(self.thrust_scale - 1.0) / THRUST_DEV, self.gust / GUST_MAX], dim=-1)

    def norm_hist(self, norm: RunningNorm):
        """normalised state part of the history (B, T, obs_dim) with the (frozen) obs normalizer"""
        x = ((self.hist[..., :self.state_dim] - norm.mean[:self.state_dim])
            / torch.sqrt(norm.var[:self.state_dim] + 1e-8)).clamp(-10, 10)
        return torch.cat([x, self.hist[..., self.state_dim:]], dim=-1)

    # ---- disturbances --------------------------------------------------
    def apply_disturbance(self, force_world):
        """force_world: (N,3) tensor in newtons, applied to the drone body."""
        self.xfrc[:, self.drone_id, 0:3] = force_world

    # ---- step ----------------------------------------------------------
    def step(self, action):
        a = action.clamp(-1.0, 1.0) # action expected to be normalised to [-1,1]

        # save (x_t, a_t) for training adaptation module
        entry = torch.cat([self.obs()[:, :18], a], dim=-1)
        self.hist = torch.roll(self.hist, -1, dims=1)
        self.hist[:, -1] = entry

        self.xfrc[:, self.drone_id, 0:3] = self.gust
        self.ctrl[:] = (self.hover * (1.0 + a) * self.thrust_scale).clamp(self.min_thrust, self.max_thrust)
        if self.device == "cuda":
            torch.cuda.synchronize()
        for _ in range(self.decimation):
            mjw.step(self.m, self.d)
        wp.synchronize()
        self.steps += 1

        pos_err = self.qpos[:, 0:3] - self.target
        ang_vel = self.qvel[:, 3:6]
        R = quat_to_rotmat(self.qpos[:, 3:7])
        upright = R[:, 2, 2]                 # 1 = level, -1 = upside down

        reward = (1.0
                  - pos_err.norm(dim=-1)
                  - 0.05 * ang_vel.norm(dim=-1)
                  - 0.01 * (a - self.prev_action).norm(dim=-1))
        crashed = (self.qpos[:, 2] < 0.05) | (upright < 0.2) | (pos_err.norm(dim=-1) > 3.0)
        timeout = self.steps >= self.max_steps
        done = crashed | timeout
        self.last_crashed = crashed
        self.prev_action = a.clone()

        # mid-episode re-sampling of the hidden parameters (non-stationary environment)
        if self.resample_prob > 0:
            rs = (torch.rand(self.n, device=self.device) < self.resample_prob).nonzero(as_tuple=False).squeeze(-1)
            if len(rs) > 0:
                self._sample_disturbance(rs)

        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self.reset(ids)
        return self.obs(), reward, done