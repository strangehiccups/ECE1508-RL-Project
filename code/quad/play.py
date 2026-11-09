import ctypes
import sys
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

# ---- reset: in play mode always respawn exactly at the hover target, level, with nominal motors.
# (env.step() also calls env.reset() after a crash, so this covers crashes too.)
_env_reset = env.reset
def reset_to_origin(ids):
    _env_reset(ids)                       # clears velocities, step counters, previous action
    env.qpos[ids, 0:3] = env.target       # no random position offset
    env.thrust_scale[ids] = 1.0           # no random motor gain
env.reset = reset_to_origin
env.reset(torch.arange(env.n, device=env.dev))   # initial spawn at the target too

# ---- keyboard input ------------------------------------------------------------
# Both platforms expose the same held(key) function; everything below it is shared.
if sys.platform == "win32":
    # Windows virtual-key codes, polled directly from the OS.
    VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 0x25, 0x26, 0x27, 0x28
    VK_PGUP, VK_PGDN = 0x21, 0x22
    VK_ENTER, VK_HOME, VK_BACKSPACE = 0x0D, 0x24, 0x08
    key_callback = None

    def held(vk):
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
else:
    # GLFW key codes, delivered by the MuJoCo viewer. The viewer reports presses and
    # OS auto-repeats but never releases, so "held" means "an event arrived recently".
    # Before auto-repeat starts we wait FIRST_WINDOW (covers the OS repeat delay); once
    # repeats are flowing we only wait REPEAT_WINDOW, so letting go stops the push quickly.
    VK_RIGHT, VK_LEFT, VK_DOWN, VK_UP = 262, 263, 264, 265
    VK_PGUP, VK_PGDN = 266, 267           # Mac laptop: fn+Up / fn+Down
    VK_ENTER, VK_HOME = 257, 268          # Mac laptop: fn+Left = Home
    VK_BACKSPACE = 259
    FIRST_WINDOW, REPEAT_WINDOW = 0.6, 0.15
    _last = {}                            # keycode -> time of latest event
    _repeating = {}                       # keycode -> True once auto-repeat has started

    def key_callback(keycode):
        now = time.time()
        _repeating[keycode] = now - _last.get(keycode, -1e9) < FIRST_WINDOW
        _last[keycode] = now

    def held(vk):
        window = REPEAT_WINDOW if _repeating.get(vk) else FIRST_WINDOW
        return time.time() - _last.get(vk, -1e9) < window

RESET_KEYS = (VK_HOME, VK_BACKSPACE)
_was_down = {k: False for k in RESET_KEYS + (VK_ENTER,)}

def pressed(vk):
    """True only on the frame a key goes down (one action per press, however long it's held)."""
    down = held(vk)
    fired = down and not _was_down[vk]
    _was_down[vk] = down
    return fired

# Training randomizes a constant gust of up to 0.02 N per axis. Tested: the policy rides
# out a steady push of up to ~0.08 N and crashes from ~0.10 N (the drone weighs ~0.26 N).
RATE = 0.06          # N per second while a key is held (reaches MAX_PUSH in 1 s)
MAX_PUSH = 0.06      # clamp on each force component, newtons
DECAY_ON_RELEASE = True   # True: force returns to 0 when you let go
dt = env.decimation * 0.002

push = np.zeros(3)   # world-frame external force on the drone

with mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback) as v:
    obs = env.obs()
    print("hold arrow keys to push the drone horizontally, PgUp/PgDn vertically (Mac: fn+Up/fn+Down). "
          "releasing the keys stops the force, Enter zeroes it. "
          "Home or Backspace resets the drone to the target (Mac: fn+Left = Home).")
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
        if pressed(VK_ENTER):
            push[:] = 0.0
        if any([pressed(k) for k in RESET_KEYS]):   # list, so every key's state is updated
            env.reset(torch.tensor([0], device=env.dev))
            push[:] = 0.0

        force = torch.tensor(push[None], dtype=torch.float32, device=env.dev)
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