"""
Fly the residual-RMA drone in the MuJoCo viewer and poke it.

  mjpython play.py                              # Mac (the viewer needs mjpython)
  python play.py                                # Linux / Windows
  mjpython play.py --mode expert                # z from the true e_t instead of the adaptation module
  mjpython play.py --ckpt checkpoints/res_noz_phase1.pt --mode robust

Keys
  P            send a second drone over the top (0.3 m above, 0.3 m/s)
  G            draw new random motor gains (+-20%), printed in the terminal
  M            switch the learned residual off / on (off = nominal controller alone)
  arrows       push the drone sideways, PgUp / PgDn up / down (Mac: fn+Up / fn+Down)
  Enter        stop pushing
  Backspace    reset to the target with healthy motors
"""
import argparse
import ctypes
import math
import sys
import time
import numpy as np
import torch
import mujoco
import mujoco.viewer

from env import ResidualQuadEnv, XML
from train import load

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/res_z.pt")
ap.add_argument("--mode", choices=["rma", "expert", "z0", "robust"], default="rma")
args = ap.parse_args()

device = "cpu"
enc, policy, adaptor, norm, cfg, adapt_every = load(args.ckpt, device)
assert args.mode != "rma" or adaptor is not None, "this checkpoint has no adaptation module (run train.py --phase 2)"
env = ResidualQuadEnv(num_envs=1, mode=cfg["mode"], rand_scale=1.0, pass_prob=0.0, resample_prob=0.0, device=device)
env.max_steps = 10 ** 9
ids = torch.tensor([0])


def calm_reset():
    env.reset(ids)
    env.dist.clear(ids)
    env.qpos[:, 0:3] = env.target
    env._after_change()


calm_reset()

# ---- keyboard (same scheme as code/play.py) ----------------------------------------------------
if sys.platform == "win32":
    K = dict(LEFT=0x25, UP=0x26, RIGHT=0x27, DOWN=0x28, PGUP=0x21, PGDN=0x22, ENTER=0x0D, BACK=0x08,
             P=0x50, G=0x47, M=0x4D)
    key_callback = None

    def held(vk):
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
else:
    K = dict(RIGHT=262, LEFT=263, DOWN=264, UP=265, PGUP=266, PGDN=267, ENTER=257, BACK=259, P=80, G=71, M=77)
    FIRST_WINDOW, REPEAT_WINDOW = 0.6, 0.15
    _last, _rep = {}, {}

    def key_callback(code):
        now = time.time()
        _rep[code] = now - _last.get(code, -1e9) < FIRST_WINDOW
        _last[code] = now

    def held(vk):
        return time.time() - _last.get(vk, -1e9) < (REPEAT_WINDOW if _rep.get(vk) else FIRST_WINDOW)

_was = {}


def pressed(vk):
    down = held(vk)
    fired = down and not _was.get(vk, False)
    _was[vk] = down
    return fired


RATE, MAX_PUSH = 0.06, 0.06          # N/s while held, N max
push = np.zeros(3)
mjm = mujoco.MjModel.from_xml_path(XML)
mjd = mujoco.MjData(mjm)
z = torch.zeros(1, 8)

print(__doc__)
with mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback) as v:
    v.cam.lookat[:] = [0, 0, 1.1]; v.cam.distance = 0.9; v.cam.elevation = -10
    obs = env.obs()
    t = 0
    while v.is_running():
        t0 = time.time()
        any_held = False
        for key, axis, sign in [("UP", 0, 1), ("DOWN", 0, -1), ("LEFT", 1, 1), ("RIGHT", 1, -1),
                                ("PGUP", 2, 1), ("PGDN", 2, -1)]:
            if held(K[key]):
                push[axis] += sign * RATE * env.control_dt
                any_held = True
        if not any_held or pressed(K["ENTER"]):
            push[:] = 0.0
        push = np.clip(push, -MAX_PUSH, MAX_PUSH)
        if pressed(K["BACK"]):
            calm_reset(); push[:] = 0.0
            print("reset")
        if pressed(K["P"]):
            env.dist.force_pass(ids, 0.3, speed=0.3, offset=0.0, t0=env._time()[0].item(),
                                heading=float(np.random.uniform(0, 2 * math.pi)))
            print("overhead pass at 0.3 m")
        if pressed(K["G"]):
            env.dist.sample_slow(ids, 1.0)
            env.dist.wind[:] = 0.0
            print("motor gains:", np.round(env.dist.gain[0].numpy(), 2))
        if pressed(K["M"]):
            env.use_residual = not env.use_residual
            print("learned residual", "ON" if env.use_residual else "OFF (nominal controller only)")
        env.apply_disturbance(torch.tensor(push[None], dtype=torch.float32))

        with torch.no_grad():
            z_true = enc(env.privileged())
            if args.mode == "expert":
                z = z_true
            elif args.mode == "z0":
                z = enc(torch.zeros(1, env.env_dim))
            elif args.mode == "rma" and t % adapt_every == 0:
                z = adaptor(env.norm_hist(norm))
            a = policy.dist(norm(obs), z).mean
        obs, _, _ = env.step(a)
        if env.last_crashed[0]:
            calm_reset(); obs = env.obs()
            print("crashed -> reset")
        if t % 100 == 0:
            e = obs[0, :3]
            print(f"distance {100 * e.norm():5.1f} cm  height {100 * e[2]:+5.1f} cm  |residual| {a.abs().mean():.2f}"
                  + (f"  |z_hat - z| {(z - z_true).norm():.2f}" if args.mode == "rma" else ""))
        t += 1
        mjd.qpos[:] = env.qpos[0].numpy()
        mjd.qvel[:] = env.qvel[0].numpy()
        mjd.mocap_pos[0] = env.mocap[0, 0].numpy()
        mujoco.mj_forward(mjm, mjd)
        v.sync()
        time.sleep(max(0.0, env.control_dt - (time.time() - t0)))
