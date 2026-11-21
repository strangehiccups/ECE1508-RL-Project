"""Figure of the current downwash model, computed with the env's own model code.

Run from code/:  python -m ProxQuad.scripts.plot_downwash [--rotor bemt|efficiency] [--out PATH]

Both drones nominal X650s, level; top drone hovers at the origin at hover thrust.
Random model parameters at the middle of their ranges; turbulence (zero-mean, random)
is not shown.
  (a) wake speed in the vertical slice through two of the top drone's rotors
  (b) stacked drones: lift lost to rotor efficiency vs to the downward drag force
  (c) lift loss vs horizontal offset      (d) tilt torque from uneven rotor loss
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..configs import AeroCfg, EnvCfg, ObsCfg, SimCfg  # noqa: E402
from ..envs.aero.wake import GaussianJetWake  # noqa: E402
from ..envs.prox_quad_env import ProxQuadEnv  # noqa: E402
from .inspect_downwash import build, nominal, state  # noqa: E402

# reference palette (light mode)
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
         "#184f95", "#0d366b"]
DZ_SET = [0.5, 1.0, 2.0]


def effects(env, bot_pos: torch.Tensor, rotor: str = AeroCfg().rotor.model):
    """Lift loss split [% weight] and tilt torque [N m] for bottom drones at bot_pos."""
    k = bot_pos.shape[0]
    model = build(k, env.dev, "rotors", rotor)
    bp, tp = nominal(env.bottom, k), nominal(env.top, k)
    out = model.compute(state(bot_pos), bp, torch.full((k, 4), bp.omega_hover, device=env.dev),
                        state(torch.zeros(k, 3, device=env.dev)), tp,
                        torch.full((k, 4), tp.f_hover, device=env.dev), env.dt)
    f = out.thrust_ratio * bp.f_hover
    W = bp.weight_nom
    eff = 100 * (bp.f_hover - f).sum(-1) / W
    drag = -100 * out.drag[:, 2] / W
    tilt = (f @ (bp.alloc / bp.k_f).T)[:, 1:3].norm(dim=-1)
    return eff.cpu().numpy(), drag.cpu().numpy(), tilt.cpu().numpy()


def wake_slice(env, xs, zs, y):
    cfg = AeroCfg()
    wake = GaussianJetWake(cfg.wake, cfg.air_density, 1, env.dev)
    X, Z = np.meshgrid(xs, zs)
    pts = torch.tensor(np.stack([X, np.full_like(X, y), Z], -1).reshape(1, -1, 3),
                       dtype=torch.float32, device=env.dev)
    tp = env.top
    v = wake.velocity(pts, state(torch.zeros(1, 3, device=env.dev)), tp,
                      torch.full((1, 4), tp.f_hover, device=env.dev))
    v_h = env.bottom.hover_induced_velocity(cfg.air_density)
    return (v.norm(dim=-1).reshape(X.shape) / v_h).cpu().numpy(), v_h


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def end_label(ax, x, y, text):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK_2)


def drone(ax, x, z, label, left=False, rotor_x=0.229907, r=0.1905, plane=0.1313):
    for sx in (-1, 1):
        cx = x + sx * rotor_x
        ax.plot([cx - r, cx + r], [z + plane] * 2, color=INK, lw=2.5, solid_capstyle="round")
    ax.plot([x - rotor_x, x + rotor_x], [z + plane] * 2, color=INK, lw=1)
    ax.plot([x], [z], "s", color=INK, ms=5)
    edge = x - rotor_x - r if left else x + rotor_x + r
    ax.annotate(label, (edge, z + plane), xytext=(-6 if left else 6, 0),
                textcoords="offset points", va="center", ha="right" if left else "left",
                fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotor", default=AeroCfg().rotor.model, choices=["efficiency", "bemt"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = "" if args.rotor == "efficiency" else f"_{args.rotor}"
    args.out = args.out or f"ProxQuad/docs/downwash_model{suffix}.png"
    rotor_name = {"efficiency": "efficiency law", "bemt": "BEMT"}[args.rotor]

    env = ProxQuadEnv(EnvCfg(sim=SimCfg(num_envs=8), obs=ObsCfg()))
    dev = env.dev
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=SURFACE,
                             gridspec_kw=dict(hspace=0.32, wspace=0.3))

    # (a) wake field
    ax = axes[0, 0]
    xs, zs = np.linspace(-1.5, 1.5, 301), np.linspace(-3.0, 0.5, 351)
    speed, v_h = wake_slice(env, xs, zs, y=0.229907)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(speed, extent=(xs[0], xs[-1], zs[0], zs[-1]), origin="lower",
                   cmap=cmap, vmin=0, vmax=float(np.percentile(speed, 99.5)), aspect="auto")
    style(ax, "(a) Wake speed below the top drone (slice through two rotors)")
    ax.grid(False)
    drone(ax, 0.0, 0.0, "top")
    drone(ax, 0.0, -1.0, "bottom, stacked")
    drone(ax, 0.6, -2.0, "bottom, partial overlap", left=True)
    ax.set_xlabel("horizontal position x [m]")
    ax.set_ylabel("height relative to top drone [m]")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(f"wake speed / hover induced velocity ({v_h:.2f} m/s)", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)
    cb.outline.set_edgecolor(AXIS)

    # (b) stacked drones: efficiency vs force
    ax = axes[0, 1]
    dz = np.linspace(0.25, 3.0, 111)
    pos = torch.tensor([[0.0, 0.0, -d] for d in dz], dtype=torch.float32, device=dev)
    eff, drag, _ = effects(env, pos, args.rotor)
    style(ax, "(b) Drones stacked: where the lift goes")
    for y, c, name in [(eff, SERIES[0], "rotor efficiency loss"),
                       (drag, SERIES[1], "downward drag force")]:
        ax.plot(dz, y, color=c, lw=2, label=name)
        end_label(ax, dz, y, name)
    ax.set_xlim(0, 4.1)
    ax.set_ylim(0, None)
    ax.set_xlabel("vertical separation [m]")
    ax.set_ylabel("lift lost [% of weight]")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")

    # (c), (d) vs horizontal offset
    dx = np.linspace(0.0, 1.5, 151)
    res = {}
    for d in DZ_SET:
        pos = torch.tensor([[x, 0.0, -d] for x in dx], dtype=torch.float32, device=dev)
        res[d] = effects(env, pos, args.rotor)
    for ax, title, ylabel, get in [
        (axes[1, 0], "(c) Lift loss vs horizontal offset", "lift lost [% of weight]",
         lambda r: r[0] + r[1]),
        (axes[1, 1], "(d) Tilt torque from uneven rotor loss", "roll/pitch torque [N m]",
         lambda r: r[2]),
    ]:
        style(ax, title)
        for d, c in zip(DZ_SET, SERIES):
            y = get(res[d])
            ax.plot(dx, y, color=c, lw=2, label=f"{d:.1f} m below")
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, None)
        ax.set_xlabel("horizontal offset of bottom drone [m]")
        ax.set_ylabel(ylabel)
        ax.legend(title="vertical separation", title_fontsize=9, frameon=False,
                  fontsize=9, labelcolor=INK_2, loc="upper right")
        ax.axvline(0.42, color=AXIS, lw=1, ls="--")
        ax.annotate("arm + prop radius", (0.42, ax.get_ylim()[1]), xytext=(4, -10),
                    textcoords="offset points", fontsize=8, color=MUTED)

    fig.suptitle("Downwash model: X650 under X650, wake = 4 Gaussian rotor jets, "
                 f"rotor model = {rotor_name} (mid-range parameters, no turbulence)",
                 x=0.06, ha="left", color=INK, fontsize=12)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved {out}")
    i = int(np.argmin(np.abs(dz - 1.0)))
    print(f"stacked at 1.0 m: efficiency {eff[i]:.1f} %, drag {drag[i]:.1f} %, "
          f"total {eff[i] + drag[i]:.1f} % of weight")


if __name__ == "__main__":
    main()
