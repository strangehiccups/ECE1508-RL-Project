# ProxQuad: notes for Claude

Design, decisions and open [CONFIRM] items: `README.md`. This file: how to run things and
the results log. `runs/` is gitignored, so results live only on this machine and here.

Branch `dev_RMA`: the hardware-realism version (rotor lag, wake transport delay and the
sensor model on by default, e_t 27 dims; 31 with the rotor speeds before 2026-11-23,
those checkpoints run at commit `f4c609c`). Training draws the bottom drone's sensor delay
from 3–8 steps, evaluation from the measured 3–6 (`TRAIN_BOTTOM_DELAY_STEPS`). Checkpoints trained before 2026-11-22
(24-dim e_t, e.g. `runs/phase2_circles_3000`) do not load here; they belong to `dev_enoch`,
which keeps both off. Results below up to the stress tests were produced with those
checkpoints.

## Running (from `code/`)

The venv is `code/.venv`; run through `uv` with `PYTHONPATH=$PWD`.

- Train: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.train_phase1 --log-dir runs/<name> [--iterations N] [--with-circles | --with-figure8] [--sanity]`
- Train Phase 2 (φ): `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.train_phase2 --phase1 runs/<phase1>/model.pt --log-dir runs/<name> [--with-circles | --with-figure8] [--iterations N]`
  (use the teacher's scenario mix; `collect_and_update` still to be written)
- Evaluate: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.evaluate --checkpoint runs/<name>/model.pt [--phase 2] [--ablate-z] [--scenarios ...] [--sensors | --noise-scale S] [--no-downwash]`
  writes `runs/<name>/eval/` (summary.json, summary.csv, PNGs incl. `<scenario>_onset.png`)
- Stress tests (Phase 2 checkpoint vs geometric+UDE at several filter pairs): `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.stress --checkpoint runs/<phase2>/model.pt [--conditions ...] [--envs N]`
  writes `runs/<phase2>/stress/` (stress.json, stress.csv)
- Crash diagnosis (Phase 2 checkpoint; π on ẑ and on the true z, same episodes, sensors on): `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.crash_diag --checkpoint runs/<phase2>/model.pt`
  prints crash % per bottom delay, timing, factor quintiles and the flips' last second; writes `runs/<phase2>/crash_diag/crash_diag.npz`
- Sensor noise from flight logs: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.measure_sensor_noise ../../CCM_quad/bags/plots/x650_*_2026-11-*.npz`
  (CPU only; source of the `SensorCfg` defaults)
- Video: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.view_episode --checkpoint runs/<name>/model.pt --scenario <s> --video out.mp4`
  (the live viewer without `--video` opens a window: give the user the command, don't leave it running)
- TensorBoard: `uv run tensorboard --logdir runs/<name>`

Scenarios (`envs/scenarios.py`, in `SCENARIOS` order): hover_over, pass_over,
approach_hover_leave, far, circle_same, circle_opposite, figure8_same, figure8_opposite,
shuttle_under. The default training mix uses the first 4; `--with-circles` adds the circles
and `--with-figure8` adds circles and figure-8s (mixes in `scripts/train_phase1.py`);
shuttle_under is in no mix. Evaluation and `view_episode --scenario` cover all 9. Results
below that list 6 or 8 scenarios predate the figure-8s / the shuttle; re-evaluating a
checkpoint adds them (unseen). Adding the shuttle (2026-11-22) changed the random draws
per episode, so seed-0 episodes from then on differ from those in earlier tables.

## Results log

Evaluation conditions unless stated otherwise: default `EnvCfg` (training randomization,
downwash on), 4096 envs, all 6 scenarios equally likely, one 15 s episode each, seed 0,
deterministic policy (`act_mean`, z = μ(e_t)). Tracking errors over t ≥ 2 s (skips the
start transient from up to ±0.3 m initial offset); crash % and return over the whole
episode. Every controller flies the same episodes. `geometric` = Lee controller on the
nominal model (`top_controller` gains, no integral term); from 2026-11-22 `evaluate.py`
uses `geometric+UDE` instead (README, "Bottom baseline"), so older tables show the plain one. `z=0` / `z=mean`: π with z
replaced, to test whether π uses z.

### 2026-11-21 · phase1_pilot (`runs/phase1_pilot`)

Training: default `Phase1Cfg`, 4 scenarios (hover_over, pass_over, approach_hover_leave
0.3 each, far 0.1), 1000 iterations × 4096 envs × 60 steps, about 15 min on the RTX 4070.
Crashes near 0 from iteration ~100. A transient wave of flips in approach_hover_leave
at iterations 115–160 cleared by 175. Return 1429 at the end (survival alone gives 1500).
σ fell from 0.20 to 0.018 and was still falling. Critic check: V(s₀) 92.8 against a
Monte Carlo return of 91.8.

Evaluation (position RMS in cm, mean over episodes, p95 in brackets; yaw RMS in degrees):

| scenario | policy | z=0 | z=mean | geometric | policy yaw | crashes |
|---|---|---|---|---|---|---|
| hover_over | 2.36 (4.38) | 5.37 | 4.99 | 35.3 | 7.5 | 0 (geometric 0.1 %) |
| pass_over | 1.09 (1.90) | 5.53 | 4.80 | 22.6 | 7.2 | 0 |
| approach_hover_leave | 1.78 (3.19) | 5.61 | 5.00 | 27.4 | 7.2 | 0 (geometric 0.2 %) |
| far | 0.42 (0.76) | 5.19 | 4.37 | 17.5 | 7.2 | 0 |
| circle_same (unseen) | 3.21 (5.19) | 6.40 | 5.88 | 34.8 | 7.2 | 0 |
| circle_opposite (unseen) | 2.43 (4.13) | 6.30 | 5.48 | 20.3 | 7.1 | 0 (geometric 0.1 %) |
| all | 1.89 (4.21) | 5.73 | 5.09 | 26.3 | 7.2 | 0 (geometric 0.1 %) |

Policy position RMS by wake strength (min rotor thrust ratio), all scenarios: no wake 1.24,
0.9–1 1.80, 0.8–0.9 2.30, 0.7–0.8 3.18, < 0.7 4.34 cm.

Findings:
- No crashes in any scenario, including the circles it never trained on. The reference
  there moves and is tilted, and the error is only 2.4–3.2 cm.
- The error grows with wake strength, from 1.2 cm out of the wake to 4.3 cm at more
  than 30 % thrust loss on one rotor.
- π relies on z. With z = 0 or z = mean z the error roughly triples (1.9 → 5.7 / 5.1 cm),
  even out of the wake (far: 0.4 → 4.4–5.2 cm). So μ encodes the drone's own randomized
  parameters and the gust, which π compensates without an integrator. This is the
  motivation for Phase 2's φ.
- Yaw: the policy settles at a heading offset of about 7° within about 4 s (geometric:
  0.1°). The attitude penalty 50·(3 − tr R) costs only about 0.75/s at 7° against 100/s of
  survival. It's a candidate for a reward change (a stronger or separate yaw term).
- The ~2° tilt seen with every controller is leaning against the constant gust (up to 4 %
  of weight, about 2.3°). It's physics, not a policy error.
- The geometric baseline is weak (17–35 cm). With no integral term, the gust and the mass
  mismatch leave a steady offset. A fairer baseline would turn `ki_pos` on.

### 2026-11-21 · phase1_circles (`runs/phase1_circles`)

Training: `--with-circles` (hover_over, pass_over, approach_hover_leave 0.2 each, far 0.1,
circle_same and circle_opposite 0.15 each), 2000 iterations, about 33 min, seed 0.
Crashes near 0 from iteration ~100, with only a few isolated crashes up to iteration ~250
and no wave like the pilot's. Return plateaued at about 1440–1450 from iteration ~1200.
σ ended at 0.014. The attitude penalty per episode was −2.4, against −12.3 for the pilot.

Evaluation (same conditions and units as above):

| scenario | policy | z=0 | z=mean | geometric | policy yaw | crashes | pilot (same episodes) |
|---|---|---|---|---|---|---|---|
| hover_over | 1.75 (3.51) | 4.13 | 3.68 | 35.3 | 1.9 | 0 | 2.36 |
| pass_over | 0.79 (1.47) | 4.44 | 3.55 | 22.6 | 1.9 | 0 | 1.09 |
| approach_hover_leave | 1.29 (2.50) | 4.27 | 3.57 | 27.4 | 1.9 | 0 | 1.78 |
| far | 0.21 (0.37) | 4.30 | 3.30 | 17.5 | 1.9 | 0 | 0.42 |
| circle_same | 2.13 (3.87) | 4.55 | 4.00 | 34.8 | 1.9 | 0 | 3.21 |
| circle_opposite | 1.16 (1.82) | 4.48 | 3.60 | 20.3 | 2.1 | 0 | 2.43 |
| all | 1.22 (2.85) | 4.36 | 3.61 | 26.3 | 1.9 | 0 | 1.89 |

Policy position RMS by wake strength, all scenarios: no wake 0.56, 0.9–1 0.88, 0.8–0.9
1.46, 0.7–0.8 2.30, < 0.7 3.37 cm (pilot: 1.24 / 1.80 / 2.30 / 3.18 / 4.34).

Findings:
- Better than the pilot in every scenario: −35 % position error overall, and −52 % on
  circle_opposite. Yaw is 1.9° instead of 7.2°. No crashes anywhere.
- Confound: 2000 against 1000 iterations, with only the final checkpoint kept. The control
  run `phase1_pilot2000` (below) separates the two: the circles help beyond longer training.
- π still relies on z: z=0 and z=mean give 4.4 and 3.6 cm against 1.2 cm.
- Crossings in circle_opposite (worst episode: a small fast circle, a crossing every
  ~2.7 s, thrust ratio down to 0.6) stay within 1–5 cm. With z=0 it's 10–18 cm, and
  ~20 cm for the geometric controller.
- To check (env, not policy): at t ≈ 0 the minimum rotor thrust ratio dips to ~0.07 for a
  few steps, with every controller. It looks like an aero-model initialization artifact at
  reset. Measured 2026-11-22 (1024 envs, geometric controller): only in scenarios that
  start in the wake (far: ratio 1.000 throughout). hover_over: mean min-rotor ratio 0.77
  at t = 0, 0.66 at step 1, back to 0.77 by step ~7 (28 ms); the worst env reaches 0 at
  steps 1–4 while its mean rotor speed drops from 419 to ~355 rad/s. Likely cause: the
  wake is instantaneous, so the bottom drone starts inside a fully developed wake; a
  lower rotor speed in the same inflow gives less thrust ratio (BEMT). Not fixed.

### 2026-11-21 · phase1_pilot2000 (`runs/phase1_pilot2000`): control for the circle run

Training: the pilot's config (original 4-scenario mix, no circles), but 2000 iterations
like the circle run, seed 0, about 33 min. No crashes after iteration 200. Return 1441 at
the end. Attitude penalty per episode −8.2 (circle run −2.4).

Policy position RMS in cm, all three runs on the same evaluation episodes:

| scenario | pilot (1000) | pilot2000 | circles (2000) |
|---|---|---|---|
| hover_over | 2.36 | 1.91 | 1.75 |
| pass_over | 1.09 | 0.88 | 0.79 |
| approach_hover_leave | 1.78 | 1.46 | 1.29 |
| far | 0.42 | 0.26 | 0.21 |
| circle_same | 3.21 | 2.78 | 2.13 |
| circle_opposite | 2.43 | 2.14 | 1.16 |
| all | 1.89 (p95 4.21) | 1.58 (p95 3.73) | 1.22 (p95 2.85) |
| yaw RMS, all [deg] | 7.2 | 5.9 | 1.9 |
| z=0 / z=mean, all | 5.73 / 5.09 | 4.95 / 4.52 | 4.36 / 3.61 |

Wake bins (all scenarios; none / 0.9–1 / 0.8–0.9 / 0.7–0.8 / < 0.7): pilot2000 1.06 / 1.56 /
1.92 / 2.75 / 3.98 cm; circles 0.56 / 0.88 / 1.46 / 2.30 / 3.37 cm.

Conclusions (one seed per run, so differences of ~10 % may be seed noise):
- Longer training alone, 1000 → 2000 iterations: −16 % position error overall (1.89 →
  1.58 cm), and yaw 7.2° → 5.9°.
- Adding circles at equal iterations: another −23 % (1.58 → 1.22 cm). It's largest on the
  circles themselves (circle_opposite −46 %, circle_same −24 %), but the original
  scenarios improve too (−8 to −20 %). Circle training costs no hover performance.
- Yaw: only the circle run holds the heading (1.9° against 5.9°). Longer training barely
  helps. The cause is not established. A yaw offset may hurt tracking on a moving, tilted
  reference more than it does in hover, which would make the reward penalize it indirectly.
- Every run uses z: the ablations stay at 3.6–5.7 cm.
- No crashes for any trained policy in any scenario.

### 2026-11-21 · figure-8 scenarios, no figure-8 training (`runs/<name>/eval_figure8`)

Existing checkpoints evaluated on the 4 moving-path scenarios only (`--scenarios circle_same
circle_opposite figure8_same figure8_opposite`, ~1000 episodes each, otherwise the usual
conditions). Neither model saw a figure-8 in training; the pilot never saw a moving path.

| scenario | circles (2000) | pilot (1000) | geometric | crashes |
|---|---|---|---|---|
| circle_same | 2.10 (3.88) | 3.24 (5.60) | 34.4 | 0 |
| circle_opposite | 1.16 (1.88) | 2.47 (4.43) | 20.7 | 0 (geometric 0.7 %) |
| figure8_same | 2.19 (4.00) | 3.13 (5.31) | 35.3 | 0 |
| figure8_opposite | 1.02 (1.60) | 2.04 (3.11) | 20.9 | 0 (geometric 0.3 %) |

Findings:
- Both models fly the figure-8s as well as the circles without training on them: same-
  direction ≈ circle_same, opposite slightly better than circle_opposite (cause not
  established; it is not less wake exposure, 43 % of steps in a wake vs 39 %). The varying curvature and the faster centre
  crossing (1.41 v) cost nothing measurable.
- Same wake-strength dependence as before (circles model, figure8_same: 1.2 cm at 0.9–1
  up to 3.8 cm below 0.7).
- Videos (`runs/videos/`, 30 s = two episodes): `policy_figure8_same.mp4`,
  `policy_figure8_opposite.mp4` (circles model), `geometric_figure8_*.mp4`.

### 2026-11-22 · phase2_circles (`runs/phase2_circles`): first φ training

Training: `train_phase2 --phase1 runs/phase1_circles/model.pt --with-circles` (μ, π
frozen from the circle run; same scenario mix), RMA method, 1000 iterations × 4096 envs ×
20 steps (82 M env steps), one Adam step per iteration, about 14 min. Log:
`runs/phase2_circles.log`.

| iteration | 0 | 100 | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|---|---|
| MSE / Var(z) | 1.19 | 0.48 | 0.44 | 0.30 | 0.25 | 0.22 | 0.18 |

- Var(z) ≈ 71 (summed over the 8 dims). Deployed error (the held ẑ π flew on) equals the
  training MSE to 3 digits: the 2-step hold costs nothing measurable.
- No crashes at any point, also with the random initial φ. Return under ẑ ≈ 1440 once
  full episodes finish (iteration ~190), the same as Phase 1 with the true z. Return
  alone does not show tracking quality: that needs the evaluation.
- Not converged: MSE / Var(z) still falling (0.22 → 0.18 over the last 200 iterations).

Evaluation (`evaluate --phase 2 --ablate-z`, usual conditions, all 8 scenarios, ~510
episodes each; `policy` = π on ẑ at the deployment rates, `z=true` = the same π on
μ(e_t), i.e. phase1_circles; position RMS in cm, p95 in brackets; ‖ẑ − z‖² per step):

| scenario | policy (ẑ) | z=true | z=mean | z=0 | geometric | ‖ẑ − z‖² |
|---|---|---|---|---|---|---|
| hover_over | 1.81 (3.37) | 1.77 | 3.70 | 4.14 | 36.0 | 17.5 |
| pass_over | 0.85 (1.48) | 0.77 | 3.50 | 4.34 | 22.4 | 4.2 |
| approach_hover_leave | 1.38 (2.56) | 1.33 | 3.72 | 4.47 | 27.4 | 9.8 |
| far | 0.36 (0.64) | 0.21 | 3.28 | 4.20 | 17.5 | 1.2 |
| circle_same | 2.36 (3.99) | 2.10 | 3.98 | 4.53 | 34.5 | 20.6 |
| circle_opposite | 1.51 (2.55) | 1.15 | 3.69 | 4.60 | 20.7 | 11.7 |
| figure8_same (unseen) | 2.51 (3.92) | 2.14 | 3.98 | 4.46 | 36.1 | 26.7 |
| figure8_opposite (unseen) | 1.87 (2.95) | 1.02 | 3.51 | 4.38 | 20.6 | 23.5 |
| all | 1.57 (3.28) | 1.30 | 3.67 | 4.39 | 26.8 | 14.4 |

Wake bins, all scenarios (none / 0.9–1 / 0.8–0.9 / 0.7–0.8 / < 0.7): policy 1.11 / 1.44 /
1.81 / 2.49 / 3.39 cm; z=true 0.62 / 0.93 / 1.48 / 2.33 / 3.43 cm.

Findings:
- φ closes 89 % of the gap between z=mean and the true z (3.67 → 1.57 cm, against 1.30).
  No crashes for any learned controller; return 1443 against 1450.
- The remaining gap is largest OUT of the wake (none: 1.11 against 0.62 cm; far: 0.36
  against 0.21) and gone in the strongest wake (< 0.7: 3.39 against 3.43). Suggests φ
  recovers the downwash part of z but less of the episode constants (mass, motors, CoM,
  gust), which a 0.5 s window shows only weakly: the case for the GRU memory (README
  "Planned"). Not yet tested (e.g. per-component ẑ error).
- Largest gap and ẑ error on the unseen figure8_opposite (1.87 against 1.02 cm, 23.5).
- Training was not converged (MSE / Var(z) still falling), so a longer run may narrow
  the gap; the out-of-wake gap points at the window length, not only training time.
- Videos (`runs/videos/`, 30 s, seed 0): `RMA_circles_same/opposite.mp4`,
  `RMA_figure8_same/opposite.mp4` (this model), `policy_circles_same.mp4` (phase1_circles,
  true z), `geometric_circles_same.mp4`.

### 2026-11-22 · phase2_circles_3000 (`runs/phase2_circles_3000`): longer φ training

Same as phase2_circles but 3000 iterations (246 M env steps, ~45 min with other jobs on
the GPU). MSE / Var(z): 0.272 (500), 0.180 (1000), 0.153 (1500), 0.137 (2000), 0.138
(2500), 0.136 (3000): converged by ~2000. No crashes.

Evaluation (`evaluate --phase 2 --ablate-z`, 4096 envs, all 8 scenarios; baseline now
geometric+UDE, default T = 0.1 / 0.05 s; position RMS cm):

| scenario | ẑ, 3000 its | ẑ, 1000 its | z=true | geometric+UDE | ‖ẑ − z‖² (3000 / 1000) |
|---|---|---|---|---|---|
| hover_over | 1.79 | 1.81 | 1.77 | 2.45 | 13.9 / 17.5 |
| pass_over | 0.79 | 0.85 | 0.77 | 1.28 | 3.0 / 4.2 |
| approach_hover_leave | 1.34 | 1.38 | 1.33 | 1.84 | 7.4 / 9.8 |
| far | 0.25 | 0.36 | 0.21 | 0.15 | 0.7 / 1.2 |
| circle_same | 2.20 | 2.36 | 2.10 | 2.51 | 15.8 / 20.6 |
| circle_opposite | 1.20 | 1.51 | 1.15 | 1.76 | 5.3 / 11.7 |
| figure8_same (unseen) | 2.27 | 2.51 | 2.14 | 2.79 | 19.1 / 26.7 |
| figure8_opposite (unseen) | 1.24 | 1.87 | 1.02 | 1.60 | 8.1 / 23.5 |
| all (p95) | 1.38 (3.05) | 1.57 (3.28) | 1.30 (3.00) | 1.79 (3.90) | 9.1 / 14.4 |

Wake bins, all (none / 0.9–1 / 0.8–0.9 / 0.7–0.8 / < 0.7): ẑ 0.75 / 1.09 / 1.61 / 2.36 /
3.30; z=true 0.62 / 0.93 / 1.48 / 2.33 / 3.43; geometric+UDE 0.83 / 1.57 / 1.94 / 3.34 /
5.08 cm.

Findings:
- φ now closes 97 % of the gap between z=mean and the true z (1000 its: 89 %).
- The out-of-wake gap was mostly training time, not the 0.5 s window: 1.11 → 0.75 cm
  against 0.62 with the true z. The largest remaining gap is figure8_opposite (unseen).
- π on ẑ beats geometric+UDE in every scenario except far (0.25 against 0.15) and in
  every wake bin; the margin grows with wake strength (3.30 against 5.08 cm below 0.7).
- No crashes for any controller.

### 2026-11-22 · geometric + UDE baseline (no training)

`RobustGeometricController` on the bottom drone, nominal ranges, all 8 scenarios, 1024
envs, seed 0 (position RMS in cm):

| controller | crash | all (p95) | hover_over | far | circle_same | circle_opp | figure8_same | figure8_opp |
|---|---|---|---|---|---|---|---|---|
| geometric (plain) | 0.10 % | 26.3 (54.6) | 34.1 | 17.9 | 34.4 | 20.6 | 34.8 | 20.4 |
| UDE T = 0.1 / 0.05 s (default) | 0 | 1.74 (3.8) | 2.3 | 0.1 | 2.4 | 1.8 | 2.6 | 1.7 |
| UDE T = 0.05 / 0.03 s | 0 | 1.46 (3.3) | 1.9 | 0.1 | 2.2 | 1.5 | 2.3 | 1.4 |
| UDE T = 0.2 / 0.1 s | 0 | 2.38 (4.9) | 2.7 | 0.2 | 3.0 | 2.7 | 3.2 | 2.6 |

- The UDE baseline is as good as the learned policy: 1.74 / 1.46 cm against 1.57 (π on
  ẑ) and 1.30 (π on true z) in the phase2_circles evaluation (4096 envs; same scenarios
  and seed, different env count). Out of the wake it is far better (0.1 against 0.36 /
  0.21 cm): it cancels constant disturbances exactly, π has no integrator.
- Faster filters track better here, but the sim has no sensor noise; on hardware a fast
  UDE filter amplifies noise. φ also sees the true state, so the comparison is fair in
  sim, but the time constant should be chosen with noise in mind.

### 2026-11-22 · out-of-range survival (phase2_circles, geometric+UDE, plain geometric)

2048 envs, all 8 scenarios equally likely, seed 0, one 15 s episode each; one condition
changed at a time. Randomization ×s widens each range around its centre (mass, inertia
xy / z, motor factor) and scales CoM offset and gust by s. Wake gain = `WakeCfg.speed_gain`
(trained 0.8–2.0). Gap = `TopPathCfg.dz` (trained 0.5–2.0 m; the collision ellipsoid is
0.35 m tall; 0.38–0.45 needs `start_clearance` 1.05). Scripts: scratchpad `ood.py` /
`ood_ude.py`; logs `runs/phase2_circles_ood*.log`, JSON `runs/phase2_circles/ood*.json`.
Survival % (position RMS cm); 1 episode = 0.05 %:

| condition | π on ẑ | π on true z | geometric+UDE | geometric |
|---|---|---|---|---|
| nominal | 100 (1.56) | 100 (1.29) | 100 (1.75) | 99.85 (26.3) |
| randomization ×1.5 | 100 (2.17) | 100 (1.54) | 100 (1.91) | 99.41 (33.4) |
| randomization ×2 | 100 (3.42) | 99.90 (2.13) | 100 (2.57) | 98.83 (41.5) |
| wake gain 2.0–2.5 | 100 (2.23) | 100 (2.06) | 100 (3.73) | 99.95 (37.8) |
| wake gain 2.5–3.0 | 100 (2.70) | 100 (2.58) | 99.95 (6.77) | 99.56 (46.0) |
| gap 0.45–0.5 m | 99.95 (1.69) | 99.95 (1.44) | 99.90 (2.02) | 94.29 (28.2) |
| gap 0.38–0.45 m | 99.22 (1.70) | 99.17 (1.45) | 99.51 (2.06) | 87.16 (28.5) |

Findings:
- Survival holds for both learned controllers and geometric+UDE in every condition
  (≥ 99.2 %); all crashes except 2 episodes are collisions. Differences below ~0.2 %
  are a few episodes.
- Wider randomization hurts φ most: ×2 doubles its error (1.56 → 3.42 cm) against
  1.29 → 2.13 with the true z and 1.75 → 2.57 for the UDE. φ extrapolates parameter
  identification worse than a model-free estimator.
- A stronger wake hurts the UDE most: 1.75 → 6.77 cm at gain 2.5–3.0, against
  1.56 → 2.70 for π on ẑ. Likely because the UDE only reacts (0.1 s filter), while π
  sees the top drone's state directly and can act before the sag builds up (not tested).
- Smaller gaps: collisions in all controllers, mostly on the opposite-direction paths
  (crossings); the learned and UDE controllers lose 0.5–0.8 % at 0.38–0.45 m, the plain
  geometric controller (20–35 cm tracking error) 13 %.

### 2026-11-22 · out-of-range survival, phase2_circles_3000

Same setup as the table above (2048 envs, 8 scenarios, seed 0), with the 3000-iteration
φ. Randomization ×3 was dropped: independent xy / z inertia factors stay physically valid
(Izz ≤ Ixx + Iyy at every range corner) only up to ×2.08. Script: scratchpad `ood2.py`;
log `runs/phase2_circles_3000_ood.log`, JSON `runs/phase2_circles_3000/ood.json`.
Survival % (position RMS cm):

| condition | π on ẑ | π on true z | geometric+UDE | geometric |
|---|---|---|---|---|
| nominal | 100 (1.36) | 100 (1.29) | 100 (1.75) | 99.85 (26.3) |
| randomization ×2 | 100 (2.99) | 99.90 (2.13) | 100 (2.57) | 98.83 (41.5) |
| wake gain 2.5–3.0 | 100 (2.54) | 100 (2.58) | 99.95 (6.77) | 99.56 (46.0) |
| gap 0.38–0.45 m | 99.22 (1.51) | 99.17 (1.45) | 99.51 (2.06) | 87.16 (28.5) |

- Longer training helps out of range too (1000 → 3000 its: ×2 3.42 → 2.99, gain
  2.70 → 2.54, gap 1.70 → 1.51 cm); survival unchanged.
- Under wider randomization φ still trails the UDE (2.99 against 2.57 cm); under the
  strong wake π on ẑ matches the true z (2.54 / 2.58) and stays far ahead of the UDE (6.77).
- Small-gap survival is identical to the 1000-iteration φ (99.22 %) and close to the true
  z: suggests geometry (crossings within 3–10 cm of the ellipsoid) rather than estimation
  (episodes not compared one by one).
- Videos (`runs/videos/`, 15 s, one episode, seed 0): `RMA_circles_same/opposite.mp4`
  (this model), `robust_geometric_circles_same/opposite.mp4` (geometric+UDE).

### 2026-11-22 · evaluation cost vs env count

phase2_circles_3000 (π on ẑ) and geometric+UDE, one 15 s rollout each, all 8 scenarios:
512 envs 51 / 57 s, 1024 51 / 56 s, 2048 55 / 57 s, 4096 65 / 59 s. The loop is bound by
the 3750 sequential steps (kernel launches), not by the env count, so fewer envs save
almost nothing. Per-episode SD of the paired difference (UDE − π) in position RMS:
0.12 (far) to 1.39 cm (hover_over); the mean differences are 0.3–0.65 cm, so ~60 episodes
per scenario resolve means, but crash rates of 0.5–1 % need the large counts (2048 envs:
1 episode = 0.05 %). Stress default stays 2048.

### 2026-11-22 · new metrics and the shuttle scenario, phase2_circles_3000 (`eval_metrics`)

`evaluate --phase 2`, 4096 envs, all 9 scenarios (~455 episodes each), seed 0 (new episode
draws, see above). Output `runs/phase2_circles_3000/eval_metrics/`; shuttle without
downwash `eval_shuttle_nodw/`. Position RMS cm (crashes 0 everywhere); encounter metrics
over all scenarios: onset / exit peak = p95 over encounters of the max error within 1 s
after the wake's onset / exit; lead = collective's 50 %-rise time before the thrust
loss's (negative: after):

| scenario | π on ẑ | z=true | geometric+UDE |
|---|---|---|---|
| hover_over | 1.78 | 1.76 | 2.44 |
| pass_over | 0.77 | 0.76 | 1.25 |
| approach_hover_leave | 1.37 | 1.34 | 1.92 |
| far | 0.25 | 0.21 | 0.14 |
| circle_same | 2.18 | 2.10 | 2.56 |
| circle_opposite | 1.21 | 1.15 | 1.76 |
| figure8_same (unseen) | 2.29 | 2.20 | 2.75 |
| figure8_opposite (unseen) | 1.23 | 1.01 | 1.62 |
| shuttle_under (unseen) | 1.71 | 1.32 | 2.30 |
| shuttle_under, no downwash | 1.37 | 0.86 | 1.11 |
| all | 1.42 | 1.32 | 1.86 |
| onset peak p95 [cm] | 3.9 | 4.3 | 6.2 |
| exit peak p95 [cm] | 3.7 | 3.3 | 5.0 |
| recovery [s] | 0.75 | 0.29 | 1.36 |
| lead [ms] | −24 | −20 | −80 |
| margin p5 | 0.59 | 0.59 | 0.62 |

Findings:
- The shuttle separates the controllers most: the wake costs the UDE +1.19 cm (1.11 →
  2.30) against +0.46 for π on the true z (0.86 → 1.32). π on ẑ without downwash (1.37)
  is not a clean control: φ never saw a drone overhead without a wake.
- Onset-aligned averages (`shuttle_under_onset.png`): the UDE's collective rises ~60 ms
  later than π's and it sags 3 cm on average; π holds z within 0.3 cm. π does not
  anticipate (its collective also follows the loss, by ~20 ms) but responds faster: the
  "acts before the sag" hypothesis is not supported, "reacts faster" is.
- Leads are unreliable where onsets are rare (hover_over +352 ms, circle_same: the wake is
  present from the start, onsets come only from sway); trust pass_over, approach, the
  opposite-direction paths and the shuttle.
- margin p5 hardly separates the controllers (the UDE sags away from the top drone, which
  increases its margin): geometry dominates.

### 2026-11-22 · sensor noise and latency, phase2_circles_3000 (no retraining)

Sensor model with the measured magnitudes (README "Sensor model"); π was trained without
it, the UDE keeps its nominal-model gains. 512 envs, all 9 scenarios, UDE T = 0.1 / 0.05 s.
Survival % (position RMS cm), action rate |Δa|:

| condition | π on ẑ | z=true | geometric+UDE |
|---|---|---|---|
| nominal (no sensors; 4096 envs) | 100 (1.42), 0.0055 | 100 (1.32) | 100 (1.86), 0.0033 |
| noise only (×1, no delay) | 100 (1.45), 0.096 | 100 (1.35), 0.095 | 100 (1.90), 0.236 |
| latency 1 step (4 ms) | 100 (1.35), 0.017 | 100 (1.22) | 100 (1.87), 0.007 |
| latency 2 steps (8 ms) | 100 (1.43), 0.058 | 100 (1.25) | 100 (1.90), 0.008 |
| latency 3 steps (12 ms) | 99.61 (2.16), 0.146 | 99.80 (2.00) | 100 (1.96), 0.010 |
| latency 6 steps (24 ms) | 57.0 (14.6), 0.195 | 67.6 (13.3) | 93.8 (25.9), 0.245 |
| noise ×1 + latency 3–6 (256 envs) | 87.1 (6.6), 0.200 | 89.8 (6.2) | 98.4 (9.5), 0.270 |

Findings:
- Noise alone barely changes tracking (+0.03–0.04 cm) but multiplies the action jitter:
  ×17 for π, ×70 for the UDE (its d̂ passes velocity noise with gain m/T).
- Latency is the cliff: π degrades from 8–12 ms and collapses at 24 ms (57 % survival);
  the UDE holds to 12 ms and fails at 24 ms (94 %, 26 cm). The measured 12–24 ms straddles
  both cliffs. π needs retraining with the latency (user, 2026-11-22).
- The sim's bottom-controller gains (k_R = 20, k_ω = 4) are ~6–10× stiffer than the
  flight-validated X650 stack (`fsc_autopilot_ros2/config/params_single_drone_direct_
  actuation_x650.yaml`: angle P 3.25 1/s, rate P 0.045 normalized ≈ 0.6 N m s/rad), which
  is why the baseline breaks at a latency the real drone flies with. The user keeps the
  UDE baseline unaware of the delays.
- Script: scratchpad `isolate.py` (conditions as in `stress.py` plus fixed delays).

### 2026-11-22 · rotor lag + wake transport delay on (smoke test)

`smoke_test --num-envs 4096 --seconds 6` with the new defaults (rotor lag on, transport
delay on): e_t 30 dims. Stand-in geometric controller (plain, nominal gains): crashes
0–0.3 % per scenario (collisions, one ground). Transport delay check (pass_over, s_τ = 1,
on vs off): peak thrust loss 464 ms later (median), modelled delay at the peak 459 ms;
same peak loss (0.169 / 0.168). With the rotor lag the minimum rotor thrust ratio reaches
0 in pass_over / approach_hover_leave (was ≥ 0.16 without the lag; 0.02 % of steps below
0.3): the lagged stand-in controller drops its slowest rotor to ~175 rad/s (p1, against
265 without), where the BEMT rotor makes almost no thrust in a strong downwash. Caused by
the lag, not the delay (checked all four on/off combinations). Cost at 4096 envs, idle
GPU, hover actions: 11.35 ms per env step (no lag, no delay), 11.56 (lag), 12.49 (lag +
delay; aero 1.12 → 1.95 ms); the ring buffer adds ~225 MB of GPU memory.

### 2026-11-22 · stress tests, phase2_circles_3000 (`runs/phase2_circles_3000/stress/`)

`stress.py`, 2048 envs, all 9 scenarios, seed 0, no sensors; UDE at T = 0.03/0.02,
0.05/0.03, 0.1/0.05, 0.2/0.1 s, "UDE best" = lowest position RMS per condition (T in
brackets). Survival % (position RMS cm, p95); onset peak p95 over encounters [cm]:

| condition | π on ẑ | z=true | UDE best | onset p95 π / UDE best |
|---|---|---|---|---|
| nominal | 100 (1.43, 3.05) | 100 (1.33, 3.00) | 100 (1.59, 3.43) (0.05/0.03) | 4.0 / 4.5 |
| randomization ×2 | 99.95 (3.06, 6.78) | 100 (2.17, 5.25) | 100 (2.47, 7.18) (0.03/0.02) | 8.1 / 10.0 |
| wake gain 2.5–3 | 100 (2.59, 5.86) | 100 (2.64, 6.19) | 100 (6.12, 23.0) (0.03/0.02) | 6.6 / 17.5 |
| gap 0.38–0.45 m | 99.07 (1.57, 3.64) | 99.02 (1.49, 3.72) | 99.32 (1.89, 4.29) (0.03/0.02) | 4.5 / 5.2 |
| combined | 100 (3.28, 7.63) | 99.90 (2.97, 7.30) | 99.85 (7.70, 29.8) (0.03/0.02) | 8.1 / 23.9 |

Shuttle only (position RMS cm, π on ẑ / z=true / UDE best): nominal 1.71 / 1.34 / 1.99,
×2 3.30 / 2.11 / 2.62, wake gain 2.52 / 2.37 / 4.93, gap 1.80 / 1.41 / 2.06, combined
3.12 / 2.55 / 5.79.

Findings:
- The best UDE filter is the fastest (0.03/0.02) in every stressed condition; even so it
  trails π in all but randomization ×2 (2.47 against 3.06 cm, where π on the true z gets
  2.17: φ's identification extrapolates worst there, as before).
- Strong wakes separate them most: wake gain 2.5–3 2.4× the error (p95 3.9×), combined
  2.3× (p95 3.9×; onset peak p95 8.1 against 23.9 cm). Survival stays ≥ 99 % for all.
- The gap condition is collision-limited for every controller (margin p5 0.07–0.11,
  ~1 % lost).
- No sensor conditions here: π needs retraining with the latency first.

### 2026-11-22 · controller comparison report

`docs/controller_comparison_phase2_circles_3000.md` (figures and `make_figures.py` in
`docs/figures/phase2_circles_3000/`): π on ẑ, π on true z, z = 0 / z = mean, geometric+UDE
and plain geometric on all 9 scenarios (4096 envs, `runs/phase2_circles_3000/eval_report/`),
the shuttle without downwash (`eval_report_shuttle_nodw/`), the stress tests above, an
onset-aligned shuttle run (4096 envs, 11.8 k onsets: command lead −28 / −28 / −88 ms, mean
height sag 0.28 / 0.07 / 2.95 cm for π on ẑ / true z / UDE) and batch-1 CPU latency (π
0.04 ms, φ 0.22 ms median on the i5-10600K, TorchScript, one thread). Numbers identical to
the `eval_metrics` run (same code and seed).

### 2026-11-22 · dev_RMA: controller gains for latency + rotor lag, Phase 1 launch

Linear attitude loop (roll, J 0.0867, rotor lag 95 ms, 4 ms hold) phase margin at 0 / 12 /
24 ms latency: zero-delay sim gains (k_R 20, k_ω 4) 11° / −4° / −18°; flight X650 gains
converted (k_R 2.76, k_ω 0.85) 30° / 26° / 21° (worst training corner, λ 8.5, J ×0.85:
11°). Rollouts, 1024 envs, 9 scenarios, full dev_RMA physics, top drone on its measured
state with the flight gains (5.0–5.6 cm from its path in every run):

| bottom controller | sensors | survival | position RMS (p95) | tilt / yaw RMS |
|---|---|---|---|---|
| geometric + UDE, full flight tune (T 0.5 / 0.75) | on | 91.11 % (7.4 % ground) | 34.1 (66.4) cm | 5.2° / 6.3° |
| same | off | 91.89 % | 33.9 cm | 5.2° / 6.3° |
| plain geometric, flight tune | on / off | 30.1 / 31.3 % | 139 cm | |
| UDE, flight attitude + sim position gains, T 0.5 / 0.75 | on | 99.90 % | 15.4 (35.8) cm | 6.8° / 6.4° |
| same, T 0.1 / 0.75 | on | 100 % | 16.0 (40.8) cm | 7.6° / 6.7° |

The flight position loop (z: kp = kd = 0.86) cannot hold the bottom drone in the downwash;
the latency barely matters for it. Interim baseline: flight attitude + sim position + flight
UDE filters, confirmed by the user; the plain geometric baseline was removed. Scripts: scratchpad `check_flight_gains.py`,
`check_baseline.py`.

Phase 1 `runs/phase1_hifi_circles` launched 2026-11-22 (`train_phase1 --with-circles
--iterations 3000`, log `runs/phase1_hifi_circles.log`): sensors on (ramp 500 iterations),
e_t 31, z 12, critic on true states, action-rate 0.1. ~1.5 s per iteration (145–165 k
steps/s). Crash rate 1.0 for the first ~30 iterations (episodes ~1 s), 0.13–0.21 by
iteration 142 (episodes ~14 s, return ~800).

Next candidates: a yaw term in the reward (if yaw matters on hardware), the reset-time
thrust-ratio dip in the aero model, an integral term for the geometric baseline, and more
seeds before drawing finer conclusions. Then Phase 2 (φ), with the circle model as the
Phase 1 teacher.

### 2026-11-22 · phase1_hifi_circles → phase2_hifi_circles (dev_RMA: e_t 31, z 12)

Phase 1 `runs/phase1_hifi_circles` (3000 iterations, ~1.6 s each, 80 min): return 1395 at
500, 1402–1424 through 2500, 1381 at 2999 (crash 0.00–0.02). Check with sensors on (512
envs, shuttle_under + pass_over): π on the true z 2.0 % crashes (all flips; shuttle 3.6 %),
position RMS 1.24 cm, command lead +56 ms; geometric+UDE 0 %, 18.5 cm, −196 ms.

Phase 2 `runs/phase2_hifi_circles` (3000 iterations, 0.97 s each, 49 min): crash rate 1.0
for iterations ~100–300 (π falls over on a poor ẑ; the old model never did), 0.08 by
1000, 0.09 on average over the last 500. loss_rel 0.80 (100) → 0.57 (300) → 0.40 (1000)
→ 0.31 (2000) → 0.26 (2999), still falling; phase2_circles_3000 reached 0.136. z_var fell
from 2.6e3 to ~100 as the crashes stopped: part of z follows the drone's state.

The chain `runs/chain_phase2_hifi.sh` waited with `pgrep -f "<phase 1 command>"`, which
matched its own launcher shell's command line, so Phase 2 started by hand ~1 h late.
Chains now run their steps in sequence in one script (`runs/chain_hifi_z8.sh`).

Evaluation, all 9 scenarios, 4096 envs, geometric+UDE at the flight filters (T 0.5 / 0.75):

| controller | sensors on (`eval_sensors/`): crash % | position RMS (p95) cm | lead ms | sensors off (`eval_nosensors/`): crash % | position RMS (p95) cm |
|---|---|---|---|---|---|
| π on ẑ | 10.2 | 2.59 (5.28) | +20 | 1.0 | 2.52 (4.73) |
| π on true z | 2.3 | 1.47 (2.82) | +56 | 0.2 | 1.78 (3.24) |
| π, z = 0 / z = mean | 100 / 100 | | | | |
| geometric+UDE | 0.0 | 15.52 (35.14) | −188 | 0.0 | 15.95 (36.43) |

π on ẑ per scenario (crash %, position RMS cm): hover_over 11.3, 2.60; pass_over 5.0, 1.41;
approach_hover_leave 5.8, 2.08; far 5.0, 0.87; circle_same 14.4, 2.96; circle_opposite
12.6, 2.75; figure8_same 13.6, 3.02; figure8_opposite 9.4, 3.04; shuttle_under 14.9, 4.91.
Crashes are flips (collisions ≤ 0.2 %). Sensors off: `sensor_delay` = 0 in e_t, below the
3–6 step training range; the env now keeps that field without the sensor model so the
sensor-trained checkpoint loads.

Stress (`runs/phase2_hifi_circles/stress/`, 2048 envs, sensors on in every condition, UDE T
0.5/0.75, 0.2/0.4, 0.1/0.2; best UDE was 0.1/0.2, the fastest, in every condition).
Survival % (position RMS cm, p95), action rate |Δa|:

| condition | π on ẑ | z=true | UDE best |
|---|---|---|---|
| nominal (= noise ×1) | 89.79 (2.56, 5.09) 0.042 | 97.46 (1.42, 2.83) 0.036 | 99.95 (9.43, 27.8) 0.034 |
| randomization ×2 | 65.53 (6.17, 15.8) | 88.82 (2.62, 4.75) | 86.87 (27.5, 109) |
| wake gain 2.5–3 | 70.12 (4.95, 11.6) | 88.28 (3.09, 7.19) | 99.27 (23.5, 52.9) |
| gap 0.38–0.45 m | 86.18 (2.81, 5.50) | 95.65 (1.64, 3.48) | 99.41 (12.9, 34.0) |
| combined | 56.59 (6.77, 18.1) | 81.25 (3.89, 10.7) | 88.96 (37.5, 94.4) |
| latency only | 89.50 (2.58, 5.18) 0.019 | 97.66 (1.44, 2.88) 0.011 | 99.95 (9.41, 27.9) 0.007 |
| noise ×2 | 89.84 (2.52, 4.93) 0.072 | 97.31 (1.47, 2.86) 0.067 | 99.95 (9.46, 27.9) 0.066 |
| noise ×5 | 90.09 (2.95, 4.98) 0.179 | 97.27 (1.57, 2.95) 0.173 | 99.95 (9.64, 28.3) 0.163 |

Onset peak p95 (π on ẑ / UDE best): nominal 8.7 / 36.4 cm, wake gain 11.9 / 65.8, combined
14.3 / 98.4; command lead +20 / −24 ms nominal.

Findings:
- Retraining with the sensor model fixed the tracking (phase2_circles_3000 under the same
  noise and 3–6 step latency: 87.1 % survival, 6.6 cm) but not the survival: 10 % flips.
- π cannot fly without the per-step z: z = 0 and z = mean flip in every episode
  (phase2_circles_3000: error ×3, no crashes). The only new per-step factor in e_t is
  `rotor_speed`; π leans on it through z.
- The flips need the sensor model: without it π on ẑ crashes 1.0 % (true z 0.2 %), with
  it 10.2 % (true z 2.3 %), though φ's error is the same (‖ẑ − z‖² 25.6 against 24.7,
  z_var ~100). They occur without a wake too (far, sensors on: 5.0 % on ẑ, 0.9 % on the
  true z).
- π tracks ~6× tighter than geometric+UDE and leads the wake (+20 ms against −188 ms).
- Noise ×1 → ×5 leaves survival and tracking flat and raises the action rate ~4×; the
  latency alone gives the same survival.
- Wider randomization, strong wakes and the combined condition cost π far more survival
  than the UDE (65.5 / 70.1 / 56.6 against 86.9 / 99.3 / 89.0 %).

Decision (user, 2026-11-23): remove `rotor_speed` from e_t, z back to 8, rerun Phase 1,
Phase 2 and every evaluation.

### 2026-11-23 · phase1_hifi_z8_circles → phase2_hifi_z8_circles (e_t 27 without `rotor_speed`, z 8)

Same setup as the entry above, except e_t (no rotor speeds) and z. Chain
`runs/chain_hifi_z8.sh` (log `runs/chain_hifi_z8.log`): Phase 1 77 min, Phase 2 48 min,
evaluations 11 min, stress 77 min (4 UDE pairs).

Phase 1: return 1234 at 500, 1328–1371 from 1000 on (z 12 with rotor speeds: 1400–1424),
crash ≤ 0.02. Phase 2: no all-crash phase (crash 0.11–0.33 up to iteration 300, 0.02 on
average over the last 500); z_var flat at 62–71 throughout; loss_rel 0.87 (100) → 0.30
(500) → 0.25 (1000) → 0.20 (1500) → 0.17 (2999), against 0.26 with the rotor speeds and
0.136 for phase2_circles_3000.

Evaluation, all 9 scenarios, 4096 envs (`runs/phase2_hifi_z8_circles/eval_sensors/`,
`eval_nosensors/`); z 12 with rotor speeds in brackets:

| controller | sensors on: crash % | position RMS (p95) cm | lead ms | sensors off: crash % | position RMS (p95) cm |
|---|---|---|---|---|---|
| π on ẑ | 2.1 [10.2] | 3.46 (5.84) [2.59] | +24 [+20] | 0.0 [1.0] | 2.80 (4.75) [2.52] |
| π on true z | 0.8 [2.3] | 2.27 (4.21) [1.47] | +52 [+56] | 0.0 [0.2] | 2.28 (4.03) [1.78] |
| π, z = 0 | 54.5 [100] | 25.9 (71.4) | | | |
| π, z = mean | 15.3 [100] | 12.2 (27.7) | | | |
| geometric+UDE (T 0.5 / 0.75) | 0.0 | 15.52 (35.14) | −188 | 0.0 | 15.95 (36.43) |

π on ẑ with sensors, per scenario (crash %, position RMS cm): hover_over 3.6, 4.14; pass_over
1.5, 2.18; approach_hover_leave 2.5, 3.00; far 1.8, 1.51; circle_same 1.0, 3.40;
circle_opposite 1.9, 3.74; figure8_same 2.7, 4.56; figure8_opposite 2.2, 3.65;
shuttle_under 2.0, 4.88. Crashes are still nearly all flips (1.9 of 2.1 %). ‖ẑ − z‖²
11.4 with sensors, 34.5 without (sensor_delay = 0 lies outside μ's training range, so
the true z moves away from what φ predicts; π on ẑ still never crashes).

Stress (`runs/phase2_hifi_z8_circles/stress/`, 2048 envs, sensors on, UDE T 0.5/0.75,
0.2/0.4, 0.1/0.2, 0.05/0.1; best by position RMS was 0.1/0.2 everywhere). Survival %
(position RMS cm, p95); z 12 with rotor speeds: π on ẑ survival in brackets:

| condition | π on ẑ | z=true | UDE best |
|---|---|---|---|
| nominal (= noise ×1) | 98.19 (3.33, 5.74) [89.79] | 99.22 (2.35, 4.26) | 99.95 (9.43, 27.8) |
| randomization ×2 | 73.93 (13.3, 58.8) [65.53] | 83.84 (8.86, 35.7) | 86.87 (27.5, 109) |
| wake gain 2.5–3 | 84.23 (8.00, 25.3) [70.12] | 93.41 (5.15, 13.6) | 99.27 (23.5, 52.9) |
| gap 0.38–0.45 m | 95.21 (4.07, 6.89) [86.18] | 96.53 (2.62, 5.33) | 99.41 (12.9, 34.0) |
| combined | 66.55 (14.0, 56.2) [56.59] | 82.96 (8.68, 30.4) | 88.96 (37.5, 94.4) |
| latency only | 98.19 (3.23, 5.67) [89.50] | 99.37 (2.22, 4.26) | 99.95 (9.41, 27.9) |
| noise ×2 | 98.68 (3.26, 5.82) [89.84] | 99.17 (2.36, 4.25) | 99.95 (9.46, 27.9) |
| noise ×5 | 98.14 (3.66, 6.24) [90.09] | 99.07 (2.48, 4.49) | 99.95 (9.64, 28.3) |

Onset peak p95 (π on ẑ / UDE best): nominal 10.3 / 36.4 cm, wake gain 16.0 / 65.8,
combined 20.7 / 98.4; command lead +20 / −24 ms nominal. UDE 0.05/0.1: nominal 99.90 %,
17.0 cm (worse than 0.1/0.2), but the most survival when stressed (randomization ×2 89.2 %,
combined 93.4 %).

Findings:
- Dropping `rotor_speed` fixed the flips: π on ẑ 10.2 → 2.1 % with sensors, 1.0 → 0 %
  without. z is a parameter again: z = mean costs 12 cm and 15 % crashes instead of every
  episode, z_var no longer follows the state, and φ fits it better (loss_rel 0.17).
- The price is tracking: π on ẑ 2.59 → 3.46 cm, true z 1.47 → 2.27 cm (the rotor state
  is now hidden from π). Still 4.5× tighter than geometric+UDE, and π still leads the
  wake (+24 against −188 ms).
- Noise up to ×5 changes neither survival nor tracking much; latency alone gives the
  same survival as latency + noise.
- Outside the training ranges π still loses on survival to the UDE (randomization ×2
  73.9 against 86.9 %, wake gain 84.2 against 99.3 %, combined 66.6 against 89.0 %) while
  keeping 2–3× lower position error; even π on the true z trails the UDE there.

### 2026-11-23 · crash diagnosis of phase2_hifi_z8_circles, three changes

`scripts/crash_diag.py` (4096 envs, sensors on, seed 0): π on ẑ 2.15 % crashes (flipped
1.93, lost 0.12, ground 0.05, collision 0.05), true z 0.78 %; of 88 ẑ crashes the true z
survives 63. Crash % by bottom delay 3 / 4 / 5 / 6 steps: 0.0 / 0.3 / 1.4 / 6.8 (true z
0.0 / 0.2 / 0.5 / 2.4); at 6 steps the slowest third of rotor λ 13.7 % against 3.3 %. Top
delay, motor imbalance, gusts, inertia: no trend; wake gain and CoM offset raise ẑ crashes
only (0.7 → 3.5 %, 0.9 → 4.0 % by quintile). Flips: a ~2 Hz roll/pitch oscillation grows
(×2.7 median over the last second), motors saturate (36 % → 68 % at the end), tilt 20° →
78° in 0.23 s; ‖ẑ − z‖² ~57 in the second before (survivors 8.9). Two thirds come 1–3 s
into the episode (mostly outside the wake, ẑ from a zero-padded history); later ones are
wake encounters (89 % in the wake) that the true z survives (96 %).

Changes (user, 2026-11-23): π and the critic see the last K = 8 commands; the bottom drone's
delay is drawn from 3–8 steps in training (evaluation 3–6, `sensor_delay` / 4.5 fixed); φ's
histories start filled with the reset step. Results in the next entry.

### 2026-11-23 · phase1_hifi_ahist_circles → phase2_hifi_ahist_circles (K = 8 commands, training delay 3–8, reset-filled φ histories)

Chain `runs/chain_hifi_ahist.sh` (log `runs/chain_hifi_ahist.log`, 14:31–17:37): Phase 1
78 min, Phase 2 56 min, evaluations 11 min, stress 38 min (UDE 0.1/0.2 only; the
baseline's other pairs are in the z8 entry, same env and seed), crash diagnosis 3 min.
The three changes were trained together, so the gains below cannot be split among them.

Phase 1: return 1317 at 500, 1384–1411 from 1000 on (z8: 1234, 1328–1371); crash 0.004 on
average over the last 500 iterations, single spikes up to 0.43. Phase 2: loss_rel 3.59 (0)
→ 0.96 (100) → 0.36 (500) → 0.28 (1000) → 0.24 (1500) → 0.18 (2999) (z8: 0.17); z_var 80–87
(z8: 62–71); crash up to 0.42 in the first 300 iterations, 0.008 on average over the last 500.

Evaluation, all 9 scenarios, 4096 envs (`runs/phase2_hifi_ahist_circles/eval_sensors/`,
`eval_nosensors/`); z8 in brackets:

| controller | sensors on: crash % | position RMS (p95) cm | lead ms | sensors off: crash % | position RMS (p95) cm |
|---|---|---|---|---|---|
| π on ẑ | 0.1 [2.1] | 2.84 (5.63) [3.46] | +32 [+24] | 0.0 [0.0] | 2.81 (5.03) [2.80] |
| π on true z | 0.0 [0.8] | 1.65 (3.50) [2.27] | +52 [+52] | 0.0 [0.0] | 2.14 (3.92) [2.28] |
| π, z = 0 | 1.4 [54.5] | 17.0 (26.0) [25.9] | | | |
| π, z = mean | 0.0 [15.3] | 7.70 (12.7) [12.2] | | | |
| geometric+UDE (T 0.5 / 0.75) | 0.0 | 15.52 (35.14) | −188 | 0.0 | 15.95 (36.43) |

π on ẑ with sensors, per scenario (crash %, position RMS cm): hover_over 0.2, 2.80;
pass_over 0.0, 1.72; approach_hover_leave 0.0, 2.35; far 0.0, 1.02; circle_same 0.2, 2.99;
circle_opposite 0.4, 2.79; figure8_same 0.0, 3.43; figure8_opposite 0.0, 3.24;
shuttle_under 0.0, 5.21. Motors at the limit 0.10 % of steps (z8 0.66), action rate 0.039
(0.043). ‖ẑ − z‖² 14.8 with sensors, 58.6 without (z8 11.4 / 34.5).

Stress (`runs/phase2_hifi_ahist_circles/stress/`, 2048 envs, sensors on). Survival %
(position RMS cm, p95); z8 survival in brackets:

| condition | π on ẑ | z=true | UDE 0.1/0.2 |
|---|---|---|---|
| nominal (= noise ×1) | 99.90 (2.81, 5.45) [98.19] | 100.00 (1.66, 3.51) [99.22] | 99.95 (9.43, 27.8) |
| randomization ×2 | 97.17 (5.50, 9.78) [73.93] | 98.88 (2.81, 5.38) [83.84] | 86.87 (27.5, 109) |
| wake gain 2.5–3 | 96.34 (4.95, 9.71) [84.23] | 99.27 (3.37, 7.87) [93.41] | 99.27 (23.5, 52.9) |
| gap 0.38–0.45 m | 98.63 (3.09, 5.89) [95.21] | 99.12 (1.89, 4.19) [96.53] | 99.41 (12.9, 34.0) |
| combined | 90.23 (6.80, 15.9) [66.55] | 97.07 (4.19, 9.97) [82.96] | 88.96 (37.5, 94.4) |
| latency only | 99.90 (2.81, 5.47) [98.19] | 100.00 (1.67, 3.50) [99.37] | 99.95 (9.41, 27.9) |
| noise ×2 | 99.90 (2.82, 5.51) [98.68] | 100.00 (1.66, 3.50) [99.17] | 99.95 (9.46, 27.9) |
| noise ×5 | 99.85 (3.02, 5.36) [98.14] | 100.00 (1.87, 3.64) [99.07] | 99.95 (9.64, 28.3) |

Onset peak p95 (π on ẑ / UDE): nominal 9.7 / 36.4 cm, wake gain 14.6 / 65.8, combined
17.1 / 98.4; command lead +28 / −24 ms nominal.

Crash diagnosis (`crash_diag/`, 4096 envs, seed 0): π on ẑ 4 crashes (0.10 %, all flips),
true z none. By bottom delay 3 / 4 / 5 / 6 steps: 0.0 / 0.0 / 0.2 / 0.2 % (z8 0.0 / 0.3 /
1.4 / 6.8). All four come in the first 2 s, in the wake (hover_over 1, circle_same 1,
circle_opposite 2), with ‖ẑ − z‖² ~125 in the second before (survivors 13.5); the same ~2 Hz
growing oscillation as before (×2.2).

Findings:
- Crashes with sensors drop from 2.1 to 0.1 % (π on ẑ) and from 0.8 to 0 % (true z); at a
  6-step delay from 6.8 to 0.2 %. The early flips outside the wake are gone.
- Tracking improves as well: 3.46 → 2.84 cm on ẑ, 2.27 → 1.65 cm on the true z, and the
  motors saturate 6× less.
- π no longer needs ẑ to stay upright: z = mean crashes 0.0 % (z8 15.3 %) at 7.7 cm, z = 0
  1.4 % (54.5 %). z still carries the tracking (2.8 against 7.7 cm). Likely the command
  window gives π the delay and rotor-lag information it used to take from z.
- Stress: π on ẑ now outlasts the UDE 0.1/0.2 under randomization ×2 (97.2 against
  86.9 %) and combined (90.2 against 89.0 %; UDE 0.05/0.1 93.4 %), and still trails it under
  wake gain (96.3 against 99.3 %) and the small gap (98.6 against 99.4 %), at 3–5× lower
  position error.
- φ fits slightly worse (‖ẑ − z‖² 14.8 against 11.4, with z_var 82 against 62); without
  sensors 58.6, since sensor_delay = 0 is further outside the 3–8-step training range. π
  on ẑ still never crashes there (2.81 cm).

