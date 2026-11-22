"""Figures for docs/controller_comparison_phase2_circles_3000.md.

Run from code/:
    PYTHONPATH=$PWD uv run python ProxQuad/docs/figures/phase2_circles_3000/make_figures.py [--onset]
Reads the evaluation outputs in runs/phase2_circles_3000/ (eval_report, stress). --onset first
reruns shuttle_under (4096 envs, seed 0; pi on z_hat, pi on true z, geometric+UDE) and caches
the onset-aligned averages in onset.npz (GPU, ~4 min).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[3] / "runs" / "phase2_circles_3000"
CK = RUNS / "model.pt"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = {"policy": ("π on ẑ (RMA)", "#2a78d6"),
          "z=true": ("π on true z", "#eb6834"),
          "geometric+UDE": ("geometric + UDE", "#1baf7a")}
SCEN = ["hover_over", "pass_over", "approach_hover_leave", "far", "circle_same",
        "circle_opposite", "figure8_same", "figure8_opposite", "shuttle_under", "all"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig(w, h):
    f = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return f


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, **kw)


def pos_rms(summary):
    f = fig(8, 6.2)
    ax = f.add_subplot()
    style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(SCEN))[::-1]
    h = 0.24
    for i, (k, (label, c)) in enumerate(SERIES.items()):
        v = [summary[k][s]["pos_rms_cm_mean"] for s in SCEN]
        ax.barh(y + (1 - i) * (h + 0.03), v, height=h, color=c, label=label)
        ax.text(v[-1] + 0.04, y[-1] + (1 - i) * (h + 0.03), f"{v[-1]:.2f}", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(y, [s.replace("_", " ") for s in SCEN], color=INK)
    ax.set_xlabel("position RMS [cm], mean over episodes", color=INK2, fontsize=9)
    ax.set_title("Tracking error per scenario (nominal conditions)", loc="left", color=INK,
                 fontsize=11)
    legend(ax, loc="lower right")
    f.tight_layout()
    f.savefig(HERE / "pos_rms_by_scenario.png", dpi=150, facecolor=SURFACE)


def wake_bins(summary):
    bins = ["none", "0.9-1", "0.8-0.9", "0.7-0.8", "<0.7"]
    f = fig(7.5, 4.2)
    ax = f.add_subplot()
    style(ax)
    x = np.arange(len(bins))
    for k, (label, c) in SERIES.items():
        v = [summary[k]["all"]["wake"][b]["pos_rms_cm"] for b in bins]
        ax.plot(x, v, color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=label)
        ax.text(x[-1] + 0.08, v[-1], f"{v[-1]:.2f}", va="center", fontsize=8.5, color=INK)
    share = [summary["policy"]["all"]["wake"][b]["share"] for b in bins]
    ax.set_xticks(x, [f"{b}\n{100 * s:.0f} % of steps" for b, s in zip(bins, share)], color=INK)
    ax.set_xlim(-0.3, len(bins) - 0.4)
    ax.set_ylim(0, None)
    ax.set_xlabel("minimum rotor thrust ratio (stronger wake to the right)", color=INK2,
                  fontsize=9)
    ax.set_ylabel("position RMS [cm]", color=INK2, fontsize=9)
    ax.set_title("Tracking error by wake strength (all scenarios)", loc="left", color=INK,
                 fontsize=11)
    legend(ax, loc="upper left")
    f.tight_layout()
    f.savefig(HERE / "pos_rms_by_wake.png", dpi=150, facecolor=SURFACE)


def stress(res):
    conds = list(res)
    names = {"nominal": "nominal", "rand_x2": "randomization ×2",
             "wake_gain_2.5-3": "wake gain 2.5–3", "gap_0.38-0.45": "gap 0.38–0.45 m",
             "combined": "combined"}
    f = fig(8, 4.2)
    ax = f.add_subplot()
    style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(conds))
    w = 0.24
    for i, k in enumerate(SERIES):
        v = []
        for cnd in conds:
            key = k
            if k == "geometric+UDE":
                udes = [u for u in res[cnd] if u.startswith("UDE ")]
                key = min(udes, key=lambda u: res[cnd][u]["all"]["pos_rms_cm_mean"])
            v.append(res[cnd][key]["all"]["pos_rms_cm_mean"])
        label = SERIES[k][0] + (" (best filter)" if k == "geometric+UDE" else "")
        ax.bar(x + (i - 1) * (w + 0.03), v, width=w, color=SERIES[k][1], label=label)
    ax.set_xticks(x, [names.get(c, c) for c in conds], color=INK)
    ax.set_ylabel("position RMS [cm], all scenarios", color=INK2, fontsize=9)
    ax.set_title("Tracking error under stress conditions", loc="left", color=INK, fontsize=11)
    legend(ax, loc="upper left")
    f.tight_layout()
    f.savefig(HERE / "pos_rms_stress.png", dpi=150, facecolor=SURFACE)


def onset_data():
    import torch

    from ProxQuad.configs import EnvCfg
    from ProxQuad.envs import ProxQuadEnv
    from ProxQuad.eval import (Phase1Policy, Phase2Policy, RobustGeometricBaseline,
                               load_phase2, onset_average, rollout)
    from ProxQuad.eval.metrics import onset_lead_ms

    cfg = EnvCfg()
    cfg.sim.stagger_first_episode = False
    cfg.scenario.top.scenario_probs = {"shuttle_under": 1.0}
    obs = torch.load(CK, map_location="cpu", weights_only=False)["cfg"].env.obs
    cfg.obs.bottom_history_len, cfg.obs.top_history_len = (obs.bottom_history_len,
                                                            obs.top_history_len)
    env = ProxQuadEnv(cfg)
    enc, pi, phi, dec = load_phase2(str(CK), env)
    out = {}
    for ctrl in (Phase2Policy(enc, pi, phi, dec), Phase1Policy(enc, pi, name="z=true"),
                 RobustGeometricBaseline(env)):
        rec = rollout(env, ctrl, 0)
        a = onset_average(rec, torch.ones(env.n, dtype=torch.bool))
        for key in ("t", "thrust_loss", "collective", "z_err"):
            out[f"{ctrl.name}/{key}"] = a[key]
        out[f"{ctrl.name}/count"] = a["count"]
        out[f"{ctrl.name}/lead_ms"] = onset_lead_ms(a)
        print(ctrl.name, a["count"], "onsets, lead", out[f"{ctrl.name}/lead_ms"], "ms", flush=True)
    np.savez(HERE / "onset.npz", **out)


def onset():
    d = np.load(HERE / "onset.npz")
    f = fig(7.5, 5.6)
    ax1, ax2 = f.subplots(2, 1, sharex=True)
    for ax in (ax1, ax2):
        style(ax)
        ax.axvline(0, color=INK2, lw=0.8)
    t = d["policy/t"]
    ax1.plot(t, d["policy/thrust_loss"], color=INK2, lw=2, ls=(0, (1, 1.5)),
             label="mean rotor thrust loss (the wake)")
    for k, (label, c) in SERIES.items():
        lead = float(d[f"{k}/lead_ms"])
        ax1.plot(t, d[f"{k}/collective"] - 1.0, color=c, lw=2,
                 label=f"{label}, command lead {lead:+.0f} ms".replace("-", "−"))
        ax2.plot(t, d[f"{k}/z_err"], color=c, lw=2)
    ax1.set_ylabel("thrust loss / extra command\n[fraction of hover thrust]", color=INK2,
                   fontsize=9)
    ax2.set_ylabel("height error [cm]", color=INK2, fontsize=9)
    ax2.set_xlabel("time from wake onset [s]", color=INK2, fontsize=9)
    n = int(d["policy/count"])
    f.suptitle(f"shuttle_under: average over {n} wake onsets", x=0.02, y=0.985, ha="left",
               color=INK, fontsize=11)
    f.legend(*ax1.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.95),
             ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    f.tight_layout(rect=(0, 0, 1, 0.88))
    f.savefig(HERE / "onset_shuttle.png", dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onset", action="store_true", help="rerun shuttle_under for onset.npz")
    args = ap.parse_args()
    if args.onset:
        onset_data()
    summary = json.loads((RUNS / "eval_report" / "summary.json").read_text())
    pos_rms(summary)
    wake_bins(summary)
    stress(json.loads((RUNS / "stress" / "stress.json").read_text()))
    if (HERE / "onset.npz").exists():
        onset()


if __name__ == "__main__":
    main()
