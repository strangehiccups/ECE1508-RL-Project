"""Metrics from a Recording: per env over its first episode, then per scenario (and "all"):
mean over envs, plus p95 and worst for the tracking errors. Wake bins split the alive
steps by the minimum rotor thrust ratio (1.0 = no wake). Encounters (ENC): spans where the
mean rotor thrust loss exceeds `off`, gaps < gap_s merged, kept if the loss exceeds `on`;
the onset is the first step above `on`. Per encounter: peak position error over the window
after the onset and after the end (the wake leaving), and the time from the end until the
error stays below recover_cm for recover_hold_s. Onset-aligned averages of thrust loss,
collective command and z error give the lead of the command over the loss."""

import math

import numpy as np
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
LOW_TAILS = ("margin_min",)                                             # also p5 and min
ENC = dict(on=0.05, off=0.02, gap_s=0.2, window_s=1.0, pre_s=0.5, recover_cm=1.0,
           recover_hold_s=0.25)


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
    rx, rz = rec.collision_radii
    margin = ((s["dxy"] / rx) ** 2 + (s["dz"] / rz) ** 2).sqrt() - 1.0
    out = {
        "return": (s["reward"] * a).sum(0),
        "margin_min": torch.where(rec.alive, margin, math.inf).min(0).values,
        "length_s": a.sum(0) * rec.dt,
        "pos_rms_cm": 100 * rms(s["pos_err"]),
        "pos_max_cm": 100 * peak(s["pos_err"]),
        "top_rms_cm": 100 * rms(s["top_err"]),
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


def encounters(rec: Recording, settle_s: float = 2.0) -> dict[str, np.ndarray]:
    """Per encounter (onset at t >= settle_s): env, onset / end step, ended (the wake left
    within the episode), onset and exit peak position error [cm] (exit: NaN without a full
    window), recovery time [s] (NaN: not ended / not recovered). Cached on the Recording."""
    key = ("encounters", settle_s)
    if key in rec.cache:
        return rec.cache[key]
    p, dt = ENC, rec.dt
    loss = rec.signals["thrust_loss"].numpy()
    err = 100 * rec.signals["pos_err"].numpy()
    n_ep = rec.alive.numpy().sum(0)                       # first episode = a prefix
    gap, win = round(p["gap_s"] / dt), round(p["window_s"] / dt)
    hold = round(p["recover_hold_s"] / dt)
    rows = []
    for e in range(loss.shape[1]):
        n = int(n_ep[e])
        x, ep = loss[:n, e], err[:n, e]
        d = np.diff(np.r_[0, (x > p["off"]).astype(np.int8), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)   # ends exclusive
        if not len(starts):
            continue
        cut = np.flatnonzero(starts[1:] - ends[:-1] >= gap)              # gaps kept
        starts, ends = starts[np.r_[0, cut + 1]], ends[np.r_[cut, len(ends) - 1]]
        for s0, s1 in zip(starts, ends):
            above = x[s0:s1] > p["on"]
            if not above.any():
                continue
            on = s0 + int(np.argmax(above))
            if on * dt < settle_s:
                continue
            exit_peak = recover = math.nan
            if s1 + win <= n:
                exit_peak = ep[s1:s1 + win].max()
            if s1 < n:
                ok = np.convolve(ep[s1:] < p["recover_cm"], np.ones(hold), "valid") >= hold
                if ok.any():
                    recover = int(np.argmax(ok)) * dt
            rows.append((e, on, s1, s1 < n, ep[on:on + win].max(), exit_peak, recover))
    a = np.array(rows, dtype=float).reshape(-1, 7)
    out = {"env": a[:, 0].astype(int), "onset": a[:, 1].astype(int), "end": a[:, 2].astype(int),
           "ended": a[:, 3].astype(bool), "onset_peak_cm": a[:, 4], "exit_peak_cm": a[:, 5],
           "recover_s": a[:, 6]}
    rec.cache[key] = out
    return out


def onset_average(rec: Recording, envs: torch.Tensor, settle_s: float = 2.0
                  ) -> dict[str, np.ndarray] | None:
    """Mean over the encounters of `envs` (bool (N,)) of thrust_loss, collective and z_err
    [cm] from pre_s before to window_s after the onset; "t" [s]. None if no full window."""
    enc, dt = encounters(rec, settle_s), rec.dt
    pre, win = round(ENC["pre_s"] / dt), round(ENC["window_s"] / dt)
    n_ep = rec.alive.numpy().sum(0)
    sel = envs.numpy()[enc["env"]] & (enc["onset"] >= pre) & (enc["onset"] + win <= n_ep[enc["env"]])
    if not sel.any():
        return None
    e, on = enc["env"][sel], enc["onset"][sel]
    idx = on[:, None] + np.arange(-pre, win)[None]
    out = {"t": np.arange(-pre, win) * dt, "count": int(sel.sum())}
    for k, scale in (("thrust_loss", 1.0), ("collective", 1.0), ("z_err", 100.0)):
        x = rec.signals[k].numpy()
        out[k] = scale * x[idx, e[:, None]].mean(0)
    return out


def onset_lead_ms(avg: dict[str, np.ndarray] | None) -> float:
    """Time by which the collective's rise reaches half its size before the thrust loss
    does [ms] (> 0: the command leads). Rise = mean over the last half of the window minus
    mean over the first half of the pre-onset span."""
    if avg is None:
        return math.nan
    pre = int(np.sum(avg["t"] < 0))
    t50 = []
    for k in ("thrust_loss", "collective"):
        x = avg[k]
        lo, hi = x[:pre // 2].mean(), x[-(len(x) - pre) // 2:].mean()
        if hi <= lo:
            return math.nan
        t50.append(avg["t"][int(np.argmax(x >= lo + 0.5 * (hi - lo)))])
    return float(1000 * (t50[0] - t50[1]))


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
            if k in LOW_TAILS:
                d[f"{k}_p5"] = torch.quantile(x, 0.05).item() if len(x) else math.nan
                d[f"{k}_min"] = x.min().item() if len(x) else math.nan
        d.update(_encounter_summary(rec, envs, settle_s))
        d["wake"] = wake_bins(rec, envs, settle_s)
        out[g] = d
    return out


def _encounter_summary(rec: Recording, envs: torch.Tensor, settle_s: float) -> dict[str, float]:
    enc = encounters(rec, settle_s)
    m = envs.numpy()[enc["env"]]
    nanmean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else math.nan  # noqa: E731
    p95 = lambda x: float(np.nanquantile(x, 0.95)) if np.isfinite(x).any() else math.nan  # noqa: E731
    rec_s = enc["recover_s"][m]
    ended = enc["ended"][m]
    return {
        "enc_count": int(m.sum()),
        "enc_onset_peak_cm_mean": nanmean(enc["onset_peak_cm"][m]),
        "enc_onset_peak_cm_p95": p95(enc["onset_peak_cm"][m]),
        "enc_exit_peak_cm_mean": nanmean(enc["exit_peak_cm"][m]),
        "enc_exit_peak_cm_p95": p95(enc["exit_peak_cm"][m]),
        "enc_recover_s_mean": nanmean(rec_s),
        "enc_recovered_pct": (100 * float(np.isfinite(rec_s).sum() / ended.sum())
                              if ended.any() else math.nan),
        "onset_lead_ms": onset_lead_ms(onset_average(rec, envs, settle_s)),
    }
