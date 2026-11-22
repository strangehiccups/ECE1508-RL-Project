"""Noise and latency of the state the X650 controller flies on, from CCM_quad flight logs
(npz files of CCM_quad/tools/flight_bag_extract.py). Source of the SensorCfg defaults.

Stream: ccm_rotor_actuation/state ("ccm_state": EKF2 fused with OptiTrack, as the
controller received it). Airborne part of each flight only (3 s trimmed at both ends).
    position / velocity / attitude  estimate minus OptiTrack (velocity: its derivative,
                                    15 Hz low-pass), at the lag that minimizes the residual;
                                    mean (attitude: the constant tilt bias, reported) and
                                    < 0.2 Hz content removed; std per axis
    body rate                       content above 10 Hz of the stream itself
    latency                         rate vs PX4 gyro (cross-correlation, 0.5-8 Hz); attitude
                                    lag vs OptiTrack arrival; state age at the command
                                    (node debug [23], CCM-node flights only)
Each stream's unique samples only (position / velocity are a 100 Hz estimate held at 200 Hz).
Prints one block per flight and the medians over flights.

Run from code/:
    python -m ProxQuad.scripts.measure_sensor_noise ../../CCM_quad/bags/plots/x650_*_2026-11-2*.npz
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation as Rot

FG = 1000.0   # Hz, grid for the cross-correlation


def hp(x, fs, fc):
    return sosfiltfilt(butter(2, fc, "high", fs=fs, output="sos"), x, axis=0)


def lp(x, fs, fc):
    return sosfiltfilt(butter(2, fc, fs=fs, output="sos"), x, axis=0)


def unique_rows(t, x):
    keep = np.r_[True, np.any(np.diff(x, axis=0) != 0, axis=1)]
    return t[keep], x[keep]


def interp(t, x, tq):
    return np.stack([np.interp(tq, t, x[:, k]) for k in range(x.shape[1])], 1)


def rot(q_wxyz):
    return Rot.from_quat(np.c_[q_wxyz[:, 1:], q_wxyz[:, :1]])


def best_lag(residual, lags):
    """Lag [ms] minimizing residual(lag) (sum of per-axis std)."""
    return min(lags, key=residual)


def flight(path: Path) -> dict:
    d = np.load(path)
    s, m, imu = d["ccm_state"], d["mocap"], d["imu"]
    z = s[:, 4]
    t0, t1 = s[z > z.min() + 0.3, 0][[0, -1]] + (3.0, -3.0)
    win = lambda a: a[(a[:, 0] > t0) & (a[:, 0] < t1)]  # noqa: E731
    s, m, imu = win(s), win(m), win(imu)
    out = {}

    # position / velocity against OptiTrack
    tp, p = unique_rows(s[:, 0], s[:, 2:5])
    fs = 1 / np.median(np.diff(tp))
    res_p = lambda L: hp(p - interp(m[:, 0], m[:, 1:4], tp - L / 1e3), fs, 0.2).std(0)  # noqa: E731
    lag_p = best_lag(lambda L: res_p(L).sum(), range(-60, 61, 2))
    out["pos_std"] = res_p(lag_p)
    tv, v = unique_rows(s[:, 0], s[:, 5:8])
    fs_m = 1 / np.median(np.diff(m[:, 0]))
    v_m = np.gradient(lp(m[:, 1:4], fs_m, 15.0), m[:, 0], axis=0)
    out["vel_std"] = hp(v - interp(m[:, 0], v_m, tv - lag_p / 1e3), fs, 0.2).std(0)

    # attitude against OptiTrack (body-frame rotation vector of R_mocap^T R)
    tq, q = unique_rows(s[:, 0], s[:, 8:12])
    fs_q = 1 / np.median(np.diff(tq))

    def att_err(L):
        qm = interp(m[:, 0], m[:, 4:8], tq - L / 1e3)
        return (rot(qm / np.linalg.norm(qm, axis=1, keepdims=True)).inv() * rot(q)).as_rotvec()

    lag_att = best_lag(lambda L: hp(att_err(L), fs_q, 0.2).std(0).sum(), range(-40, 41, 2))
    e = att_err(lag_att)
    out["att_bias_deg"] = np.rad2deg(e.mean(0))
    out["att_std_deg"] = np.rad2deg(hp(e, fs_q, 0.2).std(0))

    # body rate
    tw, w = unique_rows(s[:, 0], s[:, 12:15])
    out["ang_vel_std"] = hp(w, 1 / np.median(np.diff(tw)), 10.0).std(0)

    # latency
    tg = np.arange(s[0, 0] + 0.5, s[-1, 0] - 0.5, 1 / FG)
    held = s[np.clip(np.searchsorted(s[:, 0], tg, side="right") - 1, 0, len(s) - 1), 12:14]
    gyro = interp(imu[:, 0], imu[:, 2:4] * np.array([1.0, -1.0]), tg)     # FRD -> FLU
    band = lambda x: sosfiltfilt(butter(2, [0.5, 8.0], "band", fs=FG, output="sos"), x, axis=0)  # noqa: E731
    a, b, K = band(gyro), band(held), 60
    corr = [np.sum(a[K:-K] * b[K + L:len(b) - K + L]) for L in range(-K, K + 1)]
    out["rate_lag_vs_gyro_ms"] = float(np.argmax(corr) - K)
    out["att_lag_vs_mocap_ms"] = float(lag_att)
    age = d["node_dbg"][:, 23 + 1] if "node_dbg" in d.files else np.zeros(1)
    out["state_age_ms"] = (1e3 * np.percentile(age, [50, 95]) if np.any(age > 0)
                           else np.array([np.nan, np.nan]))
    out["rates_hz"] = np.array([fs, fs_q, 1 / np.median(np.diff(tw))]).round()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz", nargs="+", type=Path)
    args = ap.parse_args()
    np.set_printoptions(precision=4, suppress=True)
    rows = []
    for path in args.npz:
        r = flight(path)
        rows.append(r)
        print(path.stem)
        for k, v in r.items():
            print(f"  {k:22s} {v}")
    print(f"\nmedian over {len(rows)} flights")
    for k in rows[0]:
        print(f"  {k:22s} {np.nanmedian(np.stack([np.atleast_1d(r[k]) for r in rows]), 0)}")


if __name__ == "__main__":
    main()
