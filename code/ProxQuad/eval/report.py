"""Console table, summary.json / summary.csv, and PNG plots (matplotlib, Agg: no window)."""

import csv
import json
import math
from pathlib import Path

import torch

from ..envs.scenarios import SCENARIOS
from .metrics import WAKE_BINS, per_env
from .rollout import Recording

COLUMNS = (   # header, summary key, format
    ("crash%", "crash_pct", "6.1f"),
    ("return", "return_mean", "7.0f"),
    ("posRMS", "pos_rms_cm_mean", "6.2f"),
    ("p95", "pos_rms_cm_p95", "6.2f"),
    ("posMax95", "pos_max_cm_p95", "8.1f"),
    ("tilt°", "tilt_rms_deg_mean", "6.2f"),
    ("yaw°", "yaw_rms_deg_mean", "6.2f"),
    ("velRMS", "vel_rms_mean", "6.3f"),
    ("|da|", "act_rate_mean", "7.4f"),
    ("sat%", "sat_pct_mean", "5.2f"),
)
PANELS = (    # signal, scale, y label
    ("pos_err", 100.0, "pos err [cm]"),
    ("tilt_deg", 1.0, "tilt err [deg]"),
    ("yaw_deg", 1.0, "|yaw err| [deg]"),
    ("thrust_ratio", 1.0, "min rotor thrust ratio"),
)


def print_summary(results: dict[str, dict]) -> None:
    """results: controller name -> summarize() output. Errors in cm and degrees (RMS over
    each episode, then mean / p95 over episodes); posMax95 = p95 of each episode's max."""
    names = list(results)
    first = results[names[0]]
    w = max(len(n) for n in names) + 2
    for g in [*(s for s in SCENARIOS if s in first), "all"]:
        print(f"\n{g}  ({first[g]['episodes']} episodes)")
        print(" " * (w + 2) + " ".join(f"{h:>{len(format(0.0, f))}s}" for h, _, f in COLUMNS))
        for n in names:
            d = results[n][g]
            print(f"  {n:<{w}s}" + " ".join(format(d[k], f) for _, k, f in COLUMNS))
            why = [f"{k[6:-4]} {v:.1f}%" for k, v in d.items()
                   if k.startswith("crash_") and k != "crash_pct" and v > 0]
            if why:
                print(f"  {'':<{w}s}crashes: " + ", ".join(why))
        wake = first[g]["wake"]
        if sum(wake[b]["share"] for b, _, _ in WAKE_BINS[1:]) > 0:
            print(f"  wake bin  share   pos RMS cm ({' / '.join(names)})   "
                  f"tilt RMS° ({' / '.join(names)})")
            for b, _, _ in WAKE_BINS:
                if wake[b]["share"] == 0:
                    continue
                pos = " / ".join(f"{results[n][g]['wake'][b]['pos_rms_cm']:.2f}" for n in names)
                tilt = " / ".join(f"{results[n][g]['wake'][b]['tilt_rms_deg']:.2f}" for n in names)
                print(f"  {b:<8s} {100 * wake[b]['share']:5.1f}%   {pos:<24s} {tilt}")


def save_summary(results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["controller", "scenario", "metric", "value"])
        for n, res in results.items():
            for g, d in res.items():
                for k, v in d.items():
                    if k == "wake":
                        for b, bd in v.items():
                            for kk, vv in bd.items():
                                wr.writerow([n, g, f"wake[{b}].{kk}", vv])
                    else:
                        wr.writerow([n, g, k, v])


def _masked(rec: Recording, key: str, envs, stride: int) -> torch.Tensor:
    x = rec.signals[key][::stride][:, envs]
    return torch.where(rec.alive[::stride][:, envs], x, math.nan)


def save_plots(recs: list[Recording], scenarios, out: Path, stride: int = 5,
               settle_s: float = 2.0) -> None:
    """Per scenario: <s>_timeseries.png (mean and 10-90 % band over episodes, per
    controller) and <s>_examples.png (the median and worst episode by the first
    controller's position RMS, all controllers on that same episode)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    T = recs[0].alive.shape[0]
    t = torch.arange(0, T, stride) * recs[0].dt
    colors = [f"C{i}" for i in range(len(recs))]
    for g in scenarios:
        envs = recs[0].kind == SCENARIOS.index(g)
        if not envs.any():
            continue
        fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=(10, 9))
        for rec, c in zip(recs, colors):
            for ax, (key, scale, label) in zip(axes, PANELS):
                x = _masked(rec, key, envs, stride) * scale
                x = x.abs() if key == "yaw_deg" else x
                lo, hi = torch.nanquantile(x, torch.tensor([0.1, 0.9]), dim=1)
                ax.plot(t, torch.nanmean(x, 1), c, lw=1.2, label=rec.name)
                ax.fill_between(t, lo, hi, color=c, alpha=0.15, lw=0)
                ax.set_ylabel(label)
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"{g}: mean and 10-90 % over {int(envs.sum())} episodes")
        axes[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(out / f"{g}_timeseries.png", dpi=110)
        plt.close(fig)

        ids = envs.nonzero().squeeze(-1)
        score = per_env(recs[0], settle_s)["pos_rms_cm"][ids]
        order = torch.nan_to_num(score, nan=math.inf).argsort()   # ended before settle_s: worst
        picks = (("median", ids[order[len(order) // 2]]), ("worst", ids[order[-1]]))
        rows = (*PANELS[:2], PANELS[3], ("dxy", 1.0, "top drone: dxy, dz [m]"))
        fig, axes = plt.subplots(len(rows), 2, sharex=True, figsize=(13, 9))
        for col, (title, e) in enumerate(picks):
            for rec, c in zip(recs, colors):
                for r, (key, scale, label) in enumerate(rows):
                    y = _masked(rec, key, e.view(1), stride).squeeze(1) * scale
                    axes[r, col].plot(t, y, c, lw=1.0, label=rec.name)
                    if key == "dxy":
                        axes[r, col].plot(t, _masked(rec, "dz", e.view(1), stride).squeeze(1),
                                          c, lw=1.0, ls="--")
                    axes[r, 0].set_ylabel(label)
            crashed = " (crash: " + recs[0].name + ")" if recs[0].crashed[e] else ""
            axes[0, col].set_title(f"{g}, {title} episode #{int(e)}{crashed}")
            axes[-1, col].set_xlabel("t [s]")
        axes[0, 0].legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out / f"{g}_examples.png", dpi=110)
        plt.close(fig)
