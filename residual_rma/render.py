"""
Side-by-side video of several controllers flying the SAME scenario (no viewer window needed).

Scenario "demo": uneven motors (gains 0.85 / 1.10 / 0.90 / 1.15), a steady side wind of 0.04 N,
and a second drone flying straight over at 0.3 m (0.3 m/s), starting at t = 1 s.

  python render.py --methods nominal v1_rma res_rma                 # Mac / any machine with a display
  xvfb-run -a python render.py --methods nominal v1_rma res_rma     # headless Linux
Writes results/demo.mp4 and results/demo.gif
"""
import argparse
import os
import types
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
import cv2

from env import ResidualQuadEnv, XML
from evaluate import make_agent

LABEL = {"nominal": "nominal controller only", "res_robust": "residual, no z", "res_z0": "residual, z = mu(0)",
         "res_expert": "residual + true z", "res_rma": "residual RMA (v2)", "e2e_expert": "end-to-end + true z",
         "e2e_rma": "end-to-end RMA (new env)", "v1_rma": "v1 RMA (Edwin's rma.pt)"}


@torch.no_grad()
def simulate(agent, T, dz, device):
    env = ResidualQuadEnv(num_envs=1, mode=agent.env_mode, rand_scale=1.0, pass_prob=0.0, resample_prob=0.0)
    ids = torch.tensor([0], device=device)
    env.dist.clear(ids)
    env.dist.gain[:] = torch.tensor([0.85, 1.10, 0.90, 1.15], device=device)
    env.dist.wind[:] = torch.tensor([0.0, 0.04, 0.0], device=device)
    env.dist.force_pass(ids, dz, speed=0.3, offset=0.0, t0=1.0, heading=0.0)
    env.qpos[:, 0:3] = env.target; env.qvel[:] = 0.0
    env.steps[:] = 0; env._after_change()
    env.max_steps, env.auto_reset = T + 1, False
    agent.reset(1, device)
    obs = env.obs()
    qpos, mocap, err = [], [], []
    for t in range(T):
        obs, _, _ = env.step(agent.act(env, obs, t))
        qpos.append(env.qpos[0].cpu().numpy().copy())
        mocap.append(env.mocap[0, 0].cpu().numpy().copy())
        err.append(obs[0, :3].cpu().numpy().copy())
    return np.array(qpos), np.array(mocap), np.array(err)


def frames(qpos, mocap, err, label, every, w, h):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 1.12]; cam.distance = 0.6; cam.azimuth = 90.0; cam.elevation = 5.0
    out = []
    for i in range(0, len(qpos), every):
        d.qpos[:] = qpos[i]; d.mocap_pos[0] = mocap[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        img = np.ascontiguousarray(r.render())
        e = err[i]
        cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, f"t {(i + 1) * 0.01:4.1f} s   height error {100 * e[2]:+5.1f} cm   distance {100 * np.linalg.norm(e):4.1f} cm",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        out.append(img)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["nominal", "v1_rma", "res_rma"])
    ap.add_argument("--dz", type=float, default=0.3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--v1-ckpt", default="../code/rma.pt")
    ap.add_argument("--v1-code", default="../code")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    T = int(args.seconds / 0.01)
    panels = []
    for name in args.methods:
        agent = make_agent(name, types.SimpleNamespace(**vars(args)), device)
        q, mc, e = simulate(agent, T, args.dz, device)
        print(f"{name:>11}: mean distance {100 * np.linalg.norm(e[100:], axis=1).mean():.1f} cm, "
              f"lowest {100 * e[:, 2].min():+.1f} cm")
        panels.append(frames(q, mc, e, LABEL.get(name, name), every=2, w=480, h=360))
    video = [np.concatenate(fs, axis=1) for fs in zip(*panels)]
    imageio.mimsave(args.out + ".mp4", video, fps=50, macro_block_size=8)
    small = [cv2.resize(f, (f.shape[1] * 2 // 3, f.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA) for f in video[::2]]
    imageio.mimsave(args.out + ".gif", small, duration=40, loop=0)
    print("wrote", args.out + ".mp4", args.out + ".gif")
