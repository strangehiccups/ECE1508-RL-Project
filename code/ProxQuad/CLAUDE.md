# ProxQuad: notes for Claude

Design, decisions and open [CONFIRM] items: `README.md`. This file: how to run things and
the results log. `runs/` is gitignored, so results live only on this machine and here.

## Running (from `code/`)

The venv is `code/.venv`; run through `uv` with `PYTHONPATH=$PWD`.

- Train: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.train_phase1 --log-dir runs/<name> [--iterations N] [--with-circles | --with-figure8] [--sanity]`
- Train Phase 2 (φ): `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.train_phase2 --phase1 runs/<phase1>/model.pt --log-dir runs/<name> [--with-circles | --with-figure8] [--iterations N]`
  (use the teacher's scenario mix; `collect_and_update` still to be written)
- Evaluate: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.evaluate --checkpoint runs/<name>/model.pt [--ablate-z] [--scenarios ...]`
  writes `runs/<name>/eval/` (summary.json, summary.csv, PNGs)
- Video: `PYTHONPATH=$PWD uv run python -m ProxQuad.scripts.view_episode --checkpoint runs/<name>/model.pt --scenario <s> --video out.mp4`
  (the live viewer without `--video` opens a window: give the user the command, don't leave it running)
- TensorBoard: `uv run tensorboard --logdir runs/<name>`

Scenarios (`envs/scenarios.py`, in `SCENARIOS` order): hover_over, pass_over,
approach_hover_leave, far, circle_same, circle_opposite, figure8_same, figure8_opposite.
The default training mix uses the first 4; `--with-circles` adds the circles and
`--with-figure8` adds circles and figure-8s (mixes in `scripts/train_phase1.py`). Evaluation
and `view_episode --scenario` cover all 8. Results below that list 6 scenarios predate the
figure-8s; re-evaluating a checkpoint adds them (unseen).

## Results log

Evaluation conditions unless stated otherwise: default `EnvCfg` (training randomization,
downwash on), 4096 envs, all 6 scenarios equally likely, one 15 s episode each, seed 0,
deterministic policy (`act_mean`, z = μ(e_t)). Tracking errors over t ≥ 2 s (skips the
start transient from up to ±0.3 m initial offset); crash % and return over the whole
episode. Every controller flies the same episodes. `geometric` = Lee controller on the
nominal model (`top_controller` gains, no integral term). `z=0` / `z=mean`: π with z
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
  reset.

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

Next candidates: a yaw term in the reward (if yaw matters on hardware), the reset-time
thrust-ratio dip in the aero model, an integral term for the geometric baseline, and more
seeds before drawing finer conclusions. Then Phase 2 (φ), with the circle model as the
Phase 1 teacher.
