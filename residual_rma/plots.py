"""
Figures for the README (reads logs/*.csv and results/results.json).
  python plots.py   ->  results/training_curves.png, results/pass_trace.png, results/summary.png
"""
import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# color follows the method everywhere
COLOR = {"res_rma": "#2a78d6", "res_z": "#2a78d6", "v1_rma": "#eb6834", "e2e_rma": "#1baf7a", "e2e_z": "#1baf7a",
         "res_robust": "#eda100", "res_noz": "#eda100", "nominal": MUTED}
NAME = {"res_rma": "residual RMA (v2)", "res_z": "residual + z (v2)", "v1_rma": "v1 RMA (Edwin)",
        "e2e_rma": "end-to-end RMA, new env", "e2e_z": "end-to-end + z, new env",
        "res_robust": "residual, no z", "res_noz": "residual, no z", "nominal": "nominal controller only",
        "res_expert": "residual + true z", "res_z0": "residual, z = mu(0)", "e2e_expert": "end-to-end + true z"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "text.color": INK})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def smooth(y, k=10):
    out, acc = [], []
    for v in y:
        acc.append(v); acc = acc[-k:]
        out.append(sum(acc) / len(acc))
    return out


def end_label(ax, x, y, text):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)


def training_curves(out):
    runs = [r for r in ["res_z", "res_noz", "e2e_z"] if os.path.exists(f"logs/{r}_phase1.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for r in runs:
        rows = read_csv(f"logs/{r}_phase1.csv")
        x = [(int(row["it"]) + 1) * 2048 * 24 / 1e6 for row in rows]
        for ax, key, scale in [(axes[0], "step_rew", 1.0), (axes[1], "pos_err", 100.0)]:
            y = smooth([float(row[key]) * scale for row in rows])
            ax.plot(x, y, color=COLOR[r], lw=2, label=NAME[r], solid_capstyle="round")
    axes[0].set_title("Reward per step", loc="left", color=INK, fontsize=11)
    axes[1].set_title("Mean distance to target (cm)", loc="left", color=INK, fontsize=11)
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
    axes[1].set_ylim(bottom=0, top=min(axes[1].get_ylim()[1], 40))
    axes[0].legend(loc="lower right", fontsize=9)
    fig.text(0.01, 0.01, "Disturbance curriculum ramps from 0.3x to 1x of the training ranges over the first third of each run.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=160); plt.close(fig)


def pass_trace(results, out, scn="pass_0.25"):
    methods = [m for m in ["nominal", "v1_rma", "e2e_rma", "res_rma"] if scn in results.get(m, {})]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for m in methods:
        z = results[m][scn]["trace_z"]
        t = [(i + 1) * 0.01 for i in range(len(z))]
        y = [100 * (v - 1.0) for v in z]
        ax.plot(t, y, color=COLOR[m], lw=2, label=NAME[m], solid_capstyle="round")
    ax.axhline(0, color=AXIS, lw=1)
    dz = scn.split("_")[1]
    ax.set_title(f"Height error while a second drone passes {dz} m overhead (0.3 m/s, straight over the target)",
                 loc="left", color=INK, fontsize=11)
    ax.set_xlabel("time (s)"); ax.set_ylabel("height error (cm)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def summary(results, out):
    methods = [m for m in ["nominal", "v1_rma", "e2e_rma", "res_robust", "res_rma"] if m in results]
    panels = [("train", "rmse", 100, "Tracking error, training ranges (RMSE, cm)"),
              ("strong", "success", 1, "Survival at 2x the training ranges (%)"),
              ("pass_0.25", "max_drop", 100, "Height lost under a 0.25 m pass (cm)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, (scn, key, scale, title) in zip(axes, panels):
        vals = [results[m].get(scn, {}).get(key, float("nan")) * scale for m in methods]
        y = list(range(len(methods)))[::-1]
        ax.barh(y, vals, height=0.55, color=[COLOR[m] for m in methods])
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.0f}" if key == "success" else f" {v:.1f}", va="center", color=INK2, fontsize=9)
        ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in methods] if ax is axes[0] else [""] * len(methods))
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v for v in vals if v == v) * 1.25)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    training_curves("results/training_curves.png")
    if os.path.exists("results/results.json"):
        res = json.load(open("results/results.json"))
        pass_trace(res, "results/pass_trace.png")
        summary(res, "results/summary.png")
    print("figures written to results/")
