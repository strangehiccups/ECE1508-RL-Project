"""Figure comparing the two rotor models (thrust ratio of a bottom rotor under inflow),
computed with the env's own classes.

Run from code/:  python -m ProxQuad.scripts.plot_rotor_models [--out ProxQuad/docs/rotor_models.png]

  (a) thrust ratio vs inflow at hover rotor speed: efficiency law (tier A) vs BEMT
      (tier B), shaded over each model's per-episode random range
  (b) BEMT depends on rotor speed; the efficiency law does not
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..configs import AeroCfg, EnvCfg, ObsCfg, SimCfg  # noqa: E402
from ..envs.aero.rotor import BEMTRotor, EfficiencyRotor  # noqa: E402
from ..envs.prox_quad_env import ProxQuadEnv  # noqa: E402
from .inspect_downwash import build, nominal, state  # noqa: E402
from .plot_downwash import GRID, INK, INK_2, MUTED, SURFACE, style  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def ratio(model_cls, param_name, value, env, u, omega_scale=1.0):
    """Thrust ratio for inflow u [m/s] (1-D), one model parameter fixed to value."""
    cfg = AeroCfg()
    k = u.shape[0]
    m = model_cls(cfg.rotor, cfg.air_density, k, env.dev)
    getattr(m, param_name).fill_(value)
    p = nominal(env.bottom, k)
    inflow = torch.tensor(u, dtype=torch.float32, device=env.dev)[:, None].expand(k, 4)
    omega = torch.full((k, 4), omega_scale * p.omega_hover, device=env.dev)
    return m.thrust_ratio(inflow, p, omega)[:, 0].cpu().numpy()


def stacked_inflow(env, dz: float) -> float:
    """Mean axial inflow / v_h at the bottom rotors, drones stacked dz apart."""
    model = build(1, env.dev, "rotors", "efficiency")
    bp, tp = nominal(env.bottom, 1), nominal(env.top, 1)
    out = model.compute(state(torch.tensor([[0.0, 0.0, -dz]], device=env.dev)), bp,
                        torch.full((1, 4), bp.omega_hover, device=env.dev),
                        state(torch.zeros(1, 3, device=env.dev)), tp,
                        torch.full((1, 4), tp.f_hover, device=env.dev), env.dt)
    return out.rotor_inflow.mean().item() / bp.hover_induced_velocity(1.225)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ProxQuad/docs/rotor_models.png")
    args = ap.parse_args()

    env = ProxQuadEnv(EnvCfg(sim=SimCfg(num_envs=8), obs=ObsCfg()))
    rc = AeroCfg().rotor
    v_h = env.bottom.hover_induced_velocity(1.225)
    x = np.linspace(0.0, 1.5, 151)                  # inflow / v_h
    u = x * v_h

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.32))

    # (a) the two models over their random ranges, at hover rotor speed
    ax = axes[0]
    style(ax, "(a) Thrust ratio vs inflow, at hover rotor speed")
    lo, hi = rc.efficiency_gain
    e_lo, e_mid, e_hi = (ratio(EfficiencyRotor, "k", v, env, u) for v in (hi, (lo + hi) / 2, lo))
    s_lo, s_hi = rc.solidity
    b_lo, b_mid, b_hi = (ratio(BEMTRotor, "sigma", v, env, u)
                         for v in (s_lo, (s_lo + s_hi) / 2, s_hi))
    ax.fill_between(x, e_lo, e_hi, color=BLUE, alpha=0.15, lw=0)
    ax.fill_between(x, np.minimum(b_lo, b_hi), np.maximum(b_lo, b_hi), color=ORANGE,
                    alpha=0.18, lw=0)
    ax.plot(x, e_mid, color=BLUE, lw=2, label=f"efficiency law, k_eta {lo}-{hi}")
    ax.plot(x, b_mid, color=ORANGE, lw=2, label=f"BEMT, solidity {s_lo}-{s_hi}")
    x_far, x_near = stacked_inflow(env, 2.0), stacked_inflow(env, 0.5)
    ax.axvspan(x_far, x_near, color=GRID, alpha=0.8, lw=0, zorder=0)
    ax.annotate("drones stacked\n2.0 m to 0.5 m apart\n(current wake model)",
                (x_near, 1.0), xytext=(6, -2), textcoords="offset points", fontsize=8,
                color=MUTED, va="top")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"axial inflow through rotor / hover induced velocity ({v_h:.2f} m/s)")
    ax.set_ylabel("thrust in wake / thrust in still air")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")

    # (b) rotor-speed dependence of BEMT
    ax = axes[1]
    style(ax, "(b) BEMT depends on rotor speed (solidity 0.10)")
    s_mid = (s_lo + s_hi) / 2
    ax.plot(x, e_mid, color=BLUE, lw=2, label="efficiency law (any speed)")
    for c, sc in [(AQUA, 0.7), (ORANGE, 1.0), (YELLOW, 1.3)]:
        y = ratio(BEMTRotor, "sigma", s_mid, env, u, omega_scale=sc)
        ax.plot(x, y, color=c, lw=2, label=f"BEMT at {sc:g} x hover rpm")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"axial inflow through rotor / hover induced velocity ({v_h:.2f} m/s)")
    ax.set_ylabel("thrust in wake / thrust in still air")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower left")

    fig.suptitle("Rotor models: how much thrust a bottom rotor keeps in the top drone's wake "
                 "(X650, 15x5\" prop)", x=0.06, ha="left", color=INK, fontsize=12)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {out}")

    # numbers for the write-up
    R, rho, k_f = env.bottom.prop_radius, 1.225, env.bottom.k_f
    ct0 = k_f / (rho * np.pi * R**4)
    lam_h = np.sqrt(ct0 / 2)
    sa = s_mid * rc.lift_slope
    theta = 4 * ct0 / sa + lam_h
    b0 = sa / 16
    dlam = 0.5 - b0 / (2 * np.sqrt(b0**2 + sa * theta / 8))
    print(f"C_T0 {ct0:.5f}, lambda_h {lam_h:.4f}, theta_tip {theta:.4f} rad "
          f"({np.degrees(theta):.1f} deg), tip speed at hover {env.bottom.omega_hover * R:.1f} m/s")
    print(f"BEMT slope at hover -> equivalent k_eta {lam_h * dlam / (theta - lam_h):.3f}")
    for dz in (0.5, 1.0, 2.0):
        print(f"stacked {dz} m: inflow/v_h {stacked_inflow(env, dz):.3f}")


if __name__ == "__main__":
    main()
