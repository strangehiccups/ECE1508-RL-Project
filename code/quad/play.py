import ctypes
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer
from env import QuadHoverEnv, XML
from policy import ActorCritic, RunningNorm

env = QuadHoverEnv(num_envs=1)
env.max_steps = 10**9                      # no 5 s timeout reset
ckpt = torch.load("policy.pt", map_location=env.dev)
model = ActorCritic().to(env.dev)
model.load_state_dict(ckpt["model"])
norm = RunningNorm(22, env.dev)
norm.mean, norm.var = ckpt["mean"], ckpt["var"]

mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)

# Windows virtual-key codes
VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 0x25, 0x26, 0x27, 0x28
VK_PGUP, VK_PGDN = 0x21, 0x22
VK_ENTER, VK_HOME = 0x0D, 0x24

def held(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)

RATE = 0.10          # N per second while a key is held
MAX_PUSH = 0.15      # clamp on each force component, newtons
DECAY_ON_RELEASE = True   # True: force returns to 0 when you let go
dt = env.decimation * 0.002

push = np.zeros(3)   # world-frame external force on the drone

with mujoco.viewer.launch_passive(mjm, mjd) as v:
    obs = env.obs()
    print("hold arrow keys to apply a force on the drone, releasing the keys will stop the force. press 'r' to reset the drone to the origin.")
    while v.is_running():
        t0 = time.time()

        # ramp the force while keys are held
        inc = RATE * dt
        any_held = False
        for vk, axis, sign in [(VK_UP, 0, +1), (VK_DOWN, 0, -1),
                               (VK_LEFT, 1, +1), (VK_RIGHT, 1, -1),
                               (VK_PGUP, 2, +1), (VK_PGDN, 2, -1)]:
            if held(vk):
                push[axis] += sign * inc
                any_held = True
        if DECAY_ON_RELEASE and not any_held:
            push[:] = 0.0
        push = np.clip(push, -MAX_PUSH, MAX_PUSH)
        if held(VK_ENTER):
            push[:] = 0.0
        if held(VK_HOME):
            env.reset(torch.tensor([0], device=env.dev))

        force = torch.tensor([push], dtype=torch.float32, device=env.dev)
        if hasattr(env, "gust"):            # if you applied the 6.3 randomization edit
            env.gust[:] = force
        else:
            env.apply_disturbance(force)

        with torch.no_grad():
            a = model.dist(norm(obs)).mean
        obs, _, _ = env.step(a)
        mjd.qpos[:] = env.qpos[0].cpu().numpy()
        mjd.qvel[:] = env.qvel[0].cpu().numpy()
        mujoco.mj_forward(mjm, mjd)
        v.sync()
        time.sleep(max(0.0, dt - (time.time() - t0)))