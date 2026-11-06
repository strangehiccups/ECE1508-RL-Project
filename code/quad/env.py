import torch
import mujoco
import mujoco_warp as mjw
import warp as wp
from pathlib import Path

XML = str(Path(__file__).with_name("quadrotor.xml"))


def quat_to_rotmat(q):
    """q: (N,4) in (w,x,y,z) -> (N,3,3)"""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
        2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)


class QuadHoverEnv:
    def __init__(self, num_envs=1024, device="cuda:0", decimation=5):
        self.n = num_envs
        self.dev = torch.device(device)
        self.decimation = decimation          # control dt = 5 * 0.002 = 0.01 s
        self.target = torch.tensor([0.0, 0.0, 1.0], device=self.dev)

        mjm = mujoco.MjModel.from_xml_path(XML)
        mjd = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, mjd)
        self.drone_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "drone")

        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            # small contact/constraint buffers: only the floor can touch the drone
            self.d = mjw.put_data(mjm, mjd, nworld=num_envs, nconmax=8, njmax=32)

        # torch views that share GPU memory with the Warp arrays
        self.qpos = wp.to_torch(self.d.qpos)            # (N, 7)
        self.qvel = wp.to_torch(self.d.qvel)            # (N, 6)
        self.ctrl = wp.to_torch(self.d.ctrl)            # (N, 4)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)    # (N, nbody, 6)

        self.hover = 0.027 * 9.81 / 4
        self.max_thrust = 0.15
        self.prev_action = torch.zeros(num_envs, 4, device=self.dev)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.dev)
        self.max_steps = 500                             # 5 s episodes

        # for randomizing initial positions and orientations in reset()
        self.gust = torch.zeros(num_envs, 3, device=self.dev)
        self.thrust_scale = torch.ones(num_envs, 1, device=self.dev)

        self.reset(torch.arange(num_envs, device=self.dev))

    # ---- state helpers -------------------------------------------------
    def reset(self, ids):
        k = len(ids)
        self.qpos[ids, 0:3] = torch.tensor([0., 0., 1.], device=self.dev) \
            + (torch.rand(k, 3, device=self.dev) - 0.5) * 0.4
        self.qpos[ids, 3] = 1.0
        self.qpos[ids, 4:7] = 0.0
        self.qvel[ids] = 0.0
        self.ctrl[ids] = self.hover
        self.prev_action[ids] = 0.0
        self.steps[ids] = 0
        # randomize thrust scale and gust force
        self.gust[ids] = (torch.rand(k, 3, device=self.dev) - 0.5) * 0.04       # up to ±0.02 N
        self.thrust_scale[ids] = 0.9 + 0.2 * torch.rand(k, 1, device=self.dev)  # ±10% motor gain

    def obs(self):
        pos_err = self.qpos[:, 0:3] - self.target
        R = quat_to_rotmat(self.qpos[:, 3:7]).reshape(self.n, 9)
        lin_vel = self.qvel[:, 0:3]          # world frame
        ang_vel = self.qvel[:, 3:6]          # body frame (free-joint convention)
        return torch.cat([pos_err, R, lin_vel, ang_vel, self.prev_action], dim=-1)  # 22

    # ---- disturbances --------------------------------------------------
    def apply_disturbance(self, force_world):
        """force_world: (N,3) tensor in newtons, applied to the drone body."""
        self.xfrc[:, self.drone_id, 0:3] = force_world

    # ---- step ----------------------------------------------------------
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        self.xfrc[:, self.drone_id, 0:3] = self.gust
        self.ctrl[:] = (self.hover * (1.0 + a) * self.thrust_scale).clamp(0.0, self.max_thrust)
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
        self.prev_action = a.clone()

        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self.reset(ids)
        return self.obs(), reward, done