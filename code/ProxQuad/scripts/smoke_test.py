"""End-to-end check of the environment.

Run from code/:  python -m ProxQuad.scripts.smoke_test [--num-envs 4096] [--seconds 15]

The env has no base controller, so this script flies the bottom drone with a stand-in
policy: the same geometric controller the top drone uses (nominal model, no knowledge
of the randomization or the downwash), converted to actions. It reports observation
layouts and shapes, throughput, per-scenario tracking of both drones with downwash on vs
off, crash reasons, a check that mass randomization reaches MuJoCo, a check of the
sensor model (delay exact, noise std as configured), a check of the wake transport
delay (the peak thrust loss of a pass arrives later by the modelled delay), and a check of
the buffers pi and phi read (history filled with the reset step, command window order).
"""

import argparse
import time

import torch

from ..configs import EnvCfg, ObsCfg, SensorCfg, SimCfg
from ..envs import ProxQuadEnv
from ..envs.controller import GeometricController
from ..envs.observations import PROPRIO_LAYOUT, TOP_LAYOUT, TRACKING_LAYOUT, describe
from ..envs.scenarios import SCENARIOS


def run(num_envs: int, seconds: float, aero: bool, history: int) -> dict:
    cfg = EnvCfg(sim=SimCfg(num_envs=num_envs),
                 obs=ObsCfg(bottom_history_len=history, top_history_len=history))
    cfg.aero.enabled = aero
    env = ProxQuadEnv(cfg)
    steps = round(seconds / env.dt)
    ctrl = GeometricController(cfg.top_controller, env.bottom, env.dt, env.n, env.dev)

    S, dev = len(SCENARIOS), env.dev
    z = lambda: torch.zeros(S, device=dev)  # noqa: E731
    err_sum, top_err_sum, cnt, crashes, episodes, sag_sum = z(), z(), z(), z(), z(), z()
    min_ratio = torch.ones(S, device=dev)
    sag = torch.zeros(env.n, device=dev)       # worst altitude drop below the reference, per env
    reasons_total: dict[str, int] = {}
    env.timing = {k: 0.0 for k in env.timing}

    t0 = time.perf_counter()
    for _ in range(steps):
        kind = env.scenario.kind.clone()
        ratio = env._aero.thrust_ratio.min(-1).values
        bot_ref = env.scenario.bottom_reference(env.steps.float() * env.dt)
        a = env.omega_to_action(ctrl(env._bot_state, bot_ref))
        top_ref = env.scenario.top_reference(env.steps.float() * env.dt).pos
        top_err = (env._top_state.pos - top_ref).norm(dim=-1)
        _, _, done, info = env.step(a)
        if done.any():
            ctrl.reset(done.nonzero().squeeze(-1))
        pos = env.qpos[:, 0:3]
        ref = env.scenario.bottom_reference(env.steps.float() * env.dt).pos   # after the step
        err = (pos - ref).norm(dim=-1)
        settled = env.steps > round(2.0 / env.dt)
        sag = torch.where(done, sag, torch.maximum(sag, ref[:, 2] - pos[:, 2]))  # done: already reset
        err_sum.index_add_(0, kind, err * settled)
        top_err_sum.index_add_(0, kind, top_err * settled)
        cnt.index_add_(0, kind, settled.float())
        min_ratio = torch.minimum(min_ratio, torch.ones(S, device=dev)
                                  .scatter_reduce(0, kind, ratio, "amin"))
        if done.any():
            crashes.index_add_(0, kind[done], info["crashed"][done].float())
            episodes.index_add_(0, kind[done], torch.ones_like(err[done]))
            sag_sum.index_add_(0, kind[done], sag[done])
            sag[done] = 0.0
            for k, v in info["crash_reasons"].items():
                reasons_total[k] = reasons_total.get(k, 0) + int(v.sum())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    kind = env.scenario.kind                   # count unfinished episodes too
    episodes.index_add_(0, kind, torch.ones_like(sag))
    sag_sum.index_add_(0, kind, sag)

    rows = []
    for s, name in enumerate(SCENARIOS):
        c, e = cnt[s].clamp_min(1), episodes[s].clamp_min(1)
        rows.append((name, (err_sum[s] / c).item(), (top_err_sum[s] / c).item(),
                     (sag_sum[s] / e).item(), min_ratio[s].item(),
                     (crashes[s] / e).item(), int(episodes[s].item())))
    return {"env": env, "wall": wall, "steps": steps, "reasons": reasons_total,
            "rows": rows}


def report(title: str, r: dict) -> None:
    env = r["env"]
    sps = env.n * r["steps"] / r["wall"]
    print(f"\n== {title}: {env.n} envs x {r['steps']} steps, {sps:,.0f} env-steps/s "
          f"(= {sps * env.dt:,.0f} s of flight per wall-clock s)")
    print(f"   time split: physics {env.timing['physics']:.2f}s, "
          f"aero {env.timing['aero']:.2f}s, total {r['wall']:.2f}s")
    print(f"   {'scenario':22s} {'bottom |e_p|':>12s} {'top |e_p|':>10s} {'max sag':>8s} "
          f"{'min ratio':>10s} {'crash rate':>10s} {'episodes':>9s}")
    for name, e, te, sag, mr, cr, ep in r["rows"]:
        print(f"   {name:22s} {e:12.3f} {te:10.3f} {sag:8.3f} {mr:10.3f} {cr:10.2%} {ep:9d}")
    print(f"   crash reasons: {r['reasons']}")


def mass_check(env: ProxQuadEnv) -> None:
    """With ki = 0 a heavier / weaker-motored drone sags lower: z error should
    anti-correlate with the true/nominal load ratio."""
    settled = env.steps > round(3.0 / env.dt)
    far = (env.scenario.kind == SCENARIOS.index("far")) & settled
    if far.sum() < 10:
        print("   (too few settled 'far' envs for the mass check)")
        return
    z_err = (env.qpos[:, 2] - env.scenario.target[2])[far]
    load = (env.bottom.mass_factor / env.bottom.motor_factor.mean(-1))[far]
    c = torch.corrcoef(torch.stack([z_err, load]))[0, 1].item()
    print(f"   mass randomization check (settled 'far' envs, no downwash): "
          f"corr(z error, load ratio) = {c:+.2f}  (expect strongly negative)")


def com_check(env: ProxQuadEnv) -> None:
    """In steady hover the rotor thrusts have no net torque about the CoM, so their
    centroid sum(f_i r_i) / T lies on the CoM in xy. Recover the xy CoM offset from
    the actual rotor thrusts and compare with the sampled one."""
    settled = env.steps > round(3.0 / env.dt)
    far = (env.scenario.kind == SCENARIOS.index("far")) & settled
    if far.sum() < 10:
        print("   (too few settled 'far' envs for the CoM check)")
        return
    f = env.bot_motors.thrust(env.bot_motors.omega)[far]          # (k, 4) true thrusts
    r = env.bottom.rotor_pos                                       # (4, 3) body frame
    T = f.sum(-1)
    est = (f @ r[:, :2]) / T[:, None]                              # thrust centroid xy
    true = env.bottom.com_offset[far, :2]
    c = [torch.corrcoef(torch.stack([est[:, i], true[:, i]]))[0, 1].item() for i in range(2)]
    rms = (est - true).pow(2).mean().sqrt().item()
    print(f"   CoM randomization check (settled 'far' envs): offset recovered from rotor "
          f"thrusts, corr x {c[0]:+.3f}, y {c[1]:+.3f}, rms error {1000 * rms:.2f} mm "
          f"(expect ~ +1 and << {1000 * env.cfg.randomization.com_offset[0]:.0f} mm)")


def sensor_check(num_envs: int = 512) -> None:
    """Noise off, delay 3: the measured state equals the true state 3 steps earlier.
    Noise on, delay 0: std of measured - true per axis, against SensorCfg."""
    cfg = EnvCfg(sim=SimCfg(num_envs=num_envs, stagger_first_episode=False),
                 sensors=SensorCfg(enabled=True, noise_scale=0.0, bottom_delay_steps=(3, 3),
                                   top_delay_steps=(3, 3)))
    env = ProxQuadEnv(cfg)
    hover = torch.full((env.n, 4), env.a_hover, device=env.dev)
    true = []
    worst = 0.0
    for k in range(12):
        true.append((env._bot_state.pos.clone(), env._top_state.omega.clone()))
        if k >= 3:
            bp, tw = true[k - 3]
            worst = max(worst, (env.measured_state("bottom").pos - bp).abs().max().item(),
                        (env.measured_state("top").omega - tw).abs().max().item())
        env.step(hover)
    print(f"   sensor delay check (3 steps, no noise): max |measured - true(t - 3)| = "
          f"{worst:.1e}  (expect 0)")
    del env

    cfg.sensors = SensorCfg(enabled=True, bottom_delay_steps=(0, 0), top_delay_steps=(0, 0))
    env = ProxQuadEnv(cfg)
    d = {"pos": [], "vel": [], "att": [], "omega": []}
    for _ in range(20):
        for which, true_s in (("bottom", env._bot_state), ("top", env._top_state)):
            m = env.measured_state(which)
            d["pos"].append(m.pos - true_s.pos)
            d["vel"].append(m.vel - true_s.vel)
            E = true_s.R.transpose(1, 2) @ m.R                     # small rotation, body
            d["att"].append(torch.rad2deg(torch.stack([E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]], -1)))
            d["omega"].append(m.omega - true_s.omega)
        env.step(hover)
    want = {"pos": cfg.sensors.pos_std, "vel": cfg.sensors.vel_std,
            "att": cfg.sensors.att_std_deg, "omega": cfg.sensors.ang_vel_std}
    for k, v in d.items():
        got = torch.cat(v).std(0).tolist()
        print(f"   sensor noise {k:5s}: std {[f'{x:.4g}' for x in got]}  "
              f"(SensorCfg {[f'{x:.4g}' for x in want[k]]})")


def delay_check(num_envs: int = 512, seconds: float = 10.0) -> None:
    """pass_over with the transport delay on (s_tau = 1) and off, same episodes, no
    turbulence: per env, the shift of the peak mean rotor thrust loss against the delay
    the model used at that peak (mean over the rotor points and jets)."""
    peak = {}
    for on in (False, True):
        torch.manual_seed(0)
        cfg = EnvCfg(sim=SimCfg(num_envs=num_envs, stagger_first_episode=False))
        cfg.scenario.top.scenario_probs = {"pass_over": 1.0}
        cfg.aero.transport_delay, cfg.aero.transport_delay_scale = on, (1.0, 1.0)
        cfg.aero.turb_force_std = cfg.aero.turb_torque_std = 0.0
        env = ProxQuadEnv(cfg)
        ctrl = GeometricController(cfg.top_controller, env.bottom, env.dt, env.n, env.dev)
        T = round(seconds / env.dt)
        loss = torch.zeros(T, env.n, device=env.dev)
        delay = torch.zeros(T, env.n, device=env.dev)
        for t in range(T):
            loss[t] = (1.0 - env._aero.thrust_ratio).mean(-1)
            if on:
                delay[t] = env.aero.transport.steps[:, :4].float().mean((1, 2)) * env.dt
            ref = env.scenario.bottom_reference(env.steps.float() * env.dt)
            env.step(env.omega_to_action(ctrl(env._bot_state, ref)))
        i = loss.argmax(0)
        peak[on] = (i.float() * env.dt, loss.max(0).values, delay[i, torch.arange(env.n)])
        del env
    ok = (peak[False][1] > 0.05) & (peak[True][1] > 0.05)
    shift = (peak[True][0] - peak[False][0])[ok]
    model = peak[True][2][ok]
    print(f"   transport delay check (pass_over, {int(ok.sum())} envs with > 5 % peak loss): "
          f"peak loss later by {1000 * shift.median():.0f} ms (median; p10-p90 "
          f"{1000 * shift.quantile(0.1):.0f}-{1000 * shift.quantile(0.9):.0f}), modelled "
          f"delay at the peak {1000 * model.median():.0f} ms (p10-p90 "
          f"{1000 * model.quantile(0.1):.0f}-{1000 * model.quantile(0.9):.0f}); "
          f"peak loss {peak[False][1][ok].mean():.3f} without, {peak[True][1][ok].mean():.3f} with")


def buffer_check(num_envs: int = 64, steps: int = 20) -> None:
    """After a reset every history slot holds the first step's entry (hover command as the
    action); the command window is a_{t-1} .. a_{t-K}, newest first, hover-filled at reset."""
    torch.manual_seed(0)
    cfg = EnvCfg(sim=SimCfg(num_envs=num_envs, stagger_first_episode=False),
                 obs=ObsCfg(bottom_history_len=125, top_history_len=125, action_history_len=8))
    env = ProxQuadEnv(cfg)
    obs, h = env.get_obs(), env.get_history()
    first = {"bottom": torch.cat([obs["proprio"], obs["prev_action"]], -1), "top": obs["top"]}
    for k, v in h.items():
        assert torch.equal(v, first[k][:, None].expand_as(v)), f"{k} history not reset-filled"
    assert (obs["action_hist"] == env.a_hover).all(), "command window not hover-filled"
    sent = []
    for _ in range(steps):
        a = env.a_hover + 0.05 * torch.randn(env.n, 4, device=env.dev)
        sent.append(a.clamp(-1, 1))
        obs, _, done, _ = env.step(a)
    assert not done.any(), "an env finished during the buffer check"
    want = torch.stack(sent[::-1][:8], 1).flatten(1)
    assert torch.allclose(obs["action_hist"], want), "command window order"
    print(f"   history reset-filled ({', '.join(h)}), command window K = 8 newest first: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seconds", type=float, default=15.0, help="simulated time per run")
    args = ap.parse_args()

    on = run(args.num_envs, args.seconds, aero=True, history=125)
    env = on["env"]
    print(f"rotor lag: {env.cfg.motor.rotor_lag}, wake transport delay: {env.cfg.aero.transport_delay}")
    print("tracking errors x~_t (pi):\n" + describe(TRACKING_LAYOUT))
    print("true state x_t (bottom history):\n" + describe(PROPRIO_LAYOUT))
    print("privileged e_t:\n" + describe(env.priv_layout))
    print("top-drone obs:\n" + describe(TOP_LAYOUT))
    obs = env.get_obs()
    print(f"\nhover action a_hover = {env.a_hover:+.3f}, "
          f"hover rotor speed = {env.bottom.omega_hover:.1f} rad/s")
    print("obs shapes:", {k: tuple(v.shape) for k, v in obs.items()})
    print("history shapes:", {k: tuple(v.shape) for k, v in env.get_history().items()})
    for k, v in obs.items():
        assert torch.isfinite(v).all(), f"non-finite values in obs[{k}]"
    report("downwash ON, bottom flown by the geometric controller", on)
    mass_check(env)
    com_check(env)
    del on, env, obs

    off = run(args.num_envs, args.seconds, aero=False, history=0)
    report("downwash OFF, bottom flown by the geometric controller", off)
    del off

    print("\n== sensor model")
    sensor_check()

    print("\n== wake transport delay")
    delay_check()

    print("\n== policy buffers")
    buffer_check()


if __name__ == "__main__":
    main()
