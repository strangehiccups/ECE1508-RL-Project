"""Watch one ProxQuad episode in the MuJoCo viewer, or record it to MP4.

The env runs on the GPU as usual (one env); its state is mirrored into a CPU MjData for
drawing. The bottom drone is flown by the geometric controller, or by a trained Phase 1
policy with --checkpoint (deterministic, z = mu(e_t)). Downwash drawing (from the env's
wake model):
    field     arrows in the vertical plane through the top drone, facing the camera:
              wake velocity, length and brightness by speed
    inflow    cyan arrows: axial inflow through each bottom rotor; blue arrow: wake
              velocity at the bottom drone's center
    rotors    bottom rotor disks turn cyan as their thrust ratio drops

Run from code/:
    python -m ProxQuad.scripts.view_episode --scenario hover_over
    python -m ProxQuad.scripts.view_episode --video runs/videos/pass.mp4 --scenario pass_over
    python -m ProxQuad.scripts.view_episode --checkpoint runs/phase1_pilot/model.pt
"""

import argparse
import os
import sys
import time

if "--video" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering, also over SSH

import mujoco
import numpy as np
import torch

from ..configs import EnvCfg, SimCfg
from ..envs import ProxQuadEnv
from ..envs.prox_quad_env import ASSETS
from ..envs.scenarios import SCENARIOS
from ..eval import GeometricBaseline, Phase1Policy, load_phase1

FIELD_RGB = np.array([0.25, 0.6, 1.0])
CYAN = np.array([0.0, 0.9, 1.0, 0.95])
FIELD_SCALE = 0.07    # m of arrow per m/s, wake field
INFLOW_SCALE = 0.1    # m of arrow per m/s, at the bottom drone


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", choices=SCENARIOS, default="approach_hover_leave")
    p.add_argument("--no-downwash", action="store_true", help="disable the aero model")
    p.add_argument("--video", metavar="PATH", help="record an MP4 instead of opening a window")
    p.add_argument("--seconds", type=float, default=15.0, help="video length [s]")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--distance", type=float, default=6.0, help="camera distance [m]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", help="Phase 1 model.pt; default: geometric controller")
    return p.parse_args()


class DownwashDrawer:
    """Colors the bottom rotor disks and appends downwash geoms for env 0 to a scene."""

    def __init__(self, env: ProxQuadEnv, m: mujoco.MjModel):
        self.env = env
        self.u = np.arange(-1.2, 1.21, 0.2)              # across the plane [m]
        self.h = np.arange(0.15, 3.0, 0.2)               # below the top drone [m]
        self.v_h = env.bottom.hover_induced_velocity(env.cfg.aero.air_density)
        bot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bottom")
        self.disks = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bot
                      and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        self.disk_rgba = m.geom_rgba[self.disks].copy()

    def color_rotors(self, m: mujoco.MjModel) -> None:
        ratio = self.env._aero.thrust_ratio[0].cpu().numpy()
        loss = np.clip((1.0 - ratio) / 0.5, 0.0, 1.0)[:, None]
        m.geom_rgba[self.disks] = (1 - loss) * self.disk_rgba + loss * CYAN

    def draw(self, scn: mujoco.MjvScene, azimuth_deg: float) -> None:
        env = self.env
        if not env.cfg.aero.enabled:
            return
        top, bot = env._top_state, env._bot_state
        az = np.radians(azimuth_deg)
        across = np.array([-np.sin(az), np.cos(az), 0.0])
        p0 = top.pos[0].cpu().numpy()
        pts = (p0 + self.u[:, None, None] * across
               - self.h[None, :, None] * np.array([0.0, 0.0, 1.0])).reshape(-1, 3)
        pts = pts[pts[:, 2] > 0.05]
        thrust = env.top_motors.thrust(env.top_motors.omega)
        v = env.aero.wake.velocity(torch.tensor(pts[None], dtype=torch.float32, device=env.dev),
                                   top, env.top, thrust)[0].cpu().numpy()
        for p, vel in zip(pts, v):
            a = min(np.linalg.norm(vel) / self.v_h, 1.0)
            if a > 0.04:
                _arrow(scn, p, p + FIELD_SCALE * vel, [*FIELD_RGB, 0.35 + 0.65 * a], 0.012)

        R_b = bot.R[0].cpu().numpy()
        hubs = bot.pos[0].cpu().numpy() + env.bottom.rotor_pos.cpu().numpy() @ R_b.T
        for hub, w in zip(hubs, env._aero.rotor_inflow[0].cpu().numpy()):
            if w > 0.05:
                _arrow(scn, hub, hub - INFLOW_SCALE * w * R_b[:, 2], CYAN, 0.012)
        c, vc = bot.pos[0].cpu().numpy(), env._aero.center_wake[0].cpu().numpy()
        if np.linalg.norm(vc) > 0.05:
            _arrow(scn, c, c + INFLOW_SCALE * vc, [*FIELD_RGB, 0.95], 0.012)


def _arrow(scn, a, b, rgba, width):
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                             np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1


def make_env(args):
    torch.manual_seed(args.seed)
    cfg = EnvCfg(sim=SimCfg(num_envs=1, stagger_first_episode=False))
    cfg.scenario.top.scenario_probs = {args.scenario: 1.0}
    cfg.aero.enabled = not args.no_downwash
    env = ProxQuadEnv(cfg)
    if args.checkpoint:
        ctrl = Phase1Policy(*load_phase1(args.checkpoint, env))
    else:
        ctrl = GeometricBaseline(env)
    return env, ctrl


def advance(env, ctrl, steps: int) -> None:
    for _ in range(steps):
        _, _, done, info = env.step(ctrl.act(env, env.get_obs()))
        if bool(done[0]):
            ctrl.reset()
            why = [k for k, x in info["crash_reasons"].items() if bool(x[0])]
            print("episode end:", ", ".join(why) or "time-out", "-> new episode")


def mirror(env, m, d, drawer) -> None:
    d.qpos[:] = env.qpos[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    drawer.color_rotors(m)


def set_camera(cam, env, d, distance) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, 135.0, -8.0
    cam.lookat[:] = _midpoint(env, d)


def follow(cam, env, d, smooth: float = 0.9) -> None:
    """Keep both drones in view: move the look-at point toward their midpoint."""
    cam.lookat[:] = smooth * cam.lookat + (1.0 - smooth) * _midpoint(env, d)


def _midpoint(env, d) -> np.ndarray:
    b, t = env.idx["bottom"].qpos_adr, env.idx["top"].qpos_adr
    return 0.5 * (d.qpos[b:b + 3] + d.qpos[t:t + 3])


def run_live(args, env, ctrl, m, d, drawer) -> None:
    import mujoco.viewer
    frame_dt = 1.0 / args.fps
    steps = max(1, round(frame_dt / env.dt))
    mirror(env, m, d, drawer)
    with mujoco.viewer.launch_passive(m, d) as v:
        set_camera(v.cam, env, d, args.distance)
        while v.is_running():
            t0 = time.time()
            advance(env, ctrl, steps)
            with v.lock():
                mirror(env, m, d, drawer)
                follow(v.cam, env, d)
                v.user_scn.ngeom = 0
                drawer.draw(v.user_scn, v.cam.azimuth)
                ratio = env._aero.thrust_ratio[0].cpu().numpy()
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                             "scenario\nt\nthrust ratio",
                             f"{args.scenario}\n{float(env.steps[0]) * env.dt:5.2f} s\n"
                             + "  ".join(f"{r:.2f}" for r in ratio)))
            v.sync()
            time.sleep(max(0.0, frame_dt - (time.time() - t0)))


def run_video(args, env, ctrl, m, d, drawer) -> None:
    import mediapy
    W, H = args.size
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    r = mujoco.Renderer(m, H, W, max_geom=10000)
    cam = mujoco.MjvCamera()
    mirror(env, m, d, drawer)
    set_camera(cam, env, d, args.distance)
    steps = max(1, round(1.0 / args.fps / env.dt))
    frames = []
    for _ in range(round(args.seconds * args.fps)):
        advance(env, ctrl, steps)
        mirror(env, m, d, drawer)
        follow(cam, env, d)
        r.update_scene(d, cam)
        drawer.draw(r.scene, cam.azimuth)
        frames.append(r.render())
    r.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    mediapy.write_video(args.video, frames, fps=args.fps)
    print(f"wrote {args.video}: {len(frames)} frames, {args.seconds:.1f} s")


def main() -> None:
    args = parse_args()
    env, ctrl = make_env(args)
    m = mujoco.MjModel.from_xml_path(str(ASSETS / env.cfg.sim.xml))
    d = mujoco.MjData(m)
    drawer = DownwashDrawer(env, m)
    print(f"scenario {args.scenario}, downwash {'off' if args.no_downwash else 'on'}, "
          f"bottom drone: {args.checkpoint or ctrl.name}")
    (run_video if args.video else run_live)(args, env, ctrl, m, d, drawer)


if __name__ == "__main__":
    main()
