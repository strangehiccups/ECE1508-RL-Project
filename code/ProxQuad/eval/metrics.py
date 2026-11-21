"""Metrics from a Recording: per env over its first episode, then per scenario (and "all"):
mean over envs, plus p95 and worst for the tracking errors. Wake bins split the alive
steps by the minimum rotor thrust ratio (1.0 = no wake)."""

import math

import torch

from ..envs.scenarios import SCENARIOS
from .rollout import SIGNALS, Recording

WAKE_BINS = (           # label, lower, upper (min rotor thrust ratio in [lower, upper))
    ("none", 0.999, 2.0),
    ("0.9-1", 0.9, 0.999),
    ("0.8-0.9", 0.8, 0.9),
    ("0.7-0.8", 0.7, 0.8),
    ("<0.7", -1.0, 0.7),
)
TAILS = ("pos_rms_cm", "pos_max_cm", "tilt_rms_deg", "yaw_rms_deg")   # also p95 and max


def tracked(rec: Recording, settle_s: float) -> torch.Tensor:
    """(T, N) steps that count for tracking metrics: first episode and t >= settle_s
    (skips the transient from the randomized start state)."""
    t = torch.arange(rec.alive.shape[0]) * rec.dt
    return rec.alive & (t >= settle_s)[:, None]


def per_env(rec: Recording, settle_s: float = 2.0) -> dict[str, torch.Tensor]:
    """(N,) per metric. Return and length over each env's whole first episode; tracking
    metrics over t >= settle_s (NaN if the episode ended before)."""
    s, a = rec.signals, rec.alive.float()
    track = tracked(rec, settle_s)
    m = track.float()
    n = m.sum(0)
    ok = lambda x: torch.where(n > 0, x, math.nan)  # noqa: E731
    rms = lambda x: ok(((x**2 * m).sum(0) / n.clamp_min(1.0)).sqrt())  # noqa: E731
    mean = lambda x: ok((x * m).sum(0) / n.clamp_min(1.0))  # noqa: E731
    peak = lambda x: ok(torch.where(track, x, -math.inf).max(0).values)  # noqa: E731
    out = {
        "return": (s["reward"] * a).sum(0),
        "length_s": a.sum(0) * rec.dt,
        "pos_rms_cm": 100 * rms(s["pos_err"]),
        "pos_max_cm": 100 * peak(s["pos_err"]),
        "vel_rms": rms(s["vel_err"]),
        "ang_vel_rms": rms(s["ang_vel_err"]),
        "tilt_rms_deg": rms(s["tilt_deg"]),
        "yaw_rms_deg": rms(s["yaw_deg"]),
        "act_rate": mean(s["act_rate"]),
        "sat_pct": 100 * mean(s["sat"]),
    }
    for k in s.keys() - set(SIGNALS):            # controller extras
        out[k] = mean(torch.nan_to_num(s[k]))
    return out


def wake_bins(rec: Recording, envs: torch.Tensor, settle_s: float) -> dict[str, dict[str, float]]:
    alive = tracked(rec, settle_s)[:, envs]
    tr = rec.signals["thrust_ratio"][:, envs][alive]
    pe = rec.signals["pos_err"][:, envs][alive]
    tilt = rec.signals["tilt_deg"][:, envs][alive]
    out = {}
    for label, lo, hi in WAKE_BINS:
        sel = (tr >= lo) & (tr < hi)
        k = int(sel.sum())
        out[label] = {
            "share": k / max(len(tr), 1),
            "pos_rms_cm": 100 * pe[sel].pow(2).mean().sqrt().item() if k else math.nan,
            "tilt_rms_deg": tilt[sel].pow(2).mean().sqrt().item() if k else math.nan,
        }
    return out


def summarize(rec: Recording, scenarios=SCENARIOS, settle_s: float = 2.0) -> dict[str, dict]:
    """{scenario | "all": {metric: float, "wake": {bin: {...}}}}; scenarios without
    episodes are left out. Tracking metrics over t >= settle_s."""
    pe = per_env(rec, settle_s)
    groups = {g: rec.kind == SCENARIOS.index(g) for g in scenarios}
    groups["all"] = torch.stack(list(groups.values())).any(0)
    out = {}
    for g, envs in groups.items():
        if not envs.any():
            continue
        d = {"episodes": int(envs.sum()),
             "crash_pct": 100 * rec.crashed[envs].float().mean().item()}
        for r, v in rec.reasons.items():
            d[f"crash_{r}_pct"] = 100 * v[envs].float().mean().item()
        for k, v in pe.items():
            x = v[envs]
            x = x[~x.isnan()]
            d[f"{k}_mean"] = x.mean().item() if len(x) else math.nan
            if k in TAILS:
                d[f"{k}_p95"] = torch.quantile(x, 0.95).item() if len(x) else math.nan
                d[f"{k}_max"] = x.max().item() if len(x) else math.nan
        d["wake"] = wake_bins(rec, envs, settle_s)
        out[g] = d
    return out
