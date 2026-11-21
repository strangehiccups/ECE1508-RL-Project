# ProxQuad

RMA-style ([Kumar et al. 2021](../../Literature/RMA%20-%20Rapid%20Motor%20Adaptation%20for%20Legged%20Robots.pdf))
two-phase training of a policy for the **bottom** of two FSC X650 quadrotors flying in
close proximity, where the top drone's downwash costs the bottom drone thrust and tilts it.

| Phase | Trains | Inputs | Signal |
|---|---|---|---|
| 1 | env-factor encoder μ + base policy π | e_t (privileged), x̃_t, a_{t-1}, o_t^top | RL (algorithm TBD) |
| 2 | adaptation module φ (π, μ frozen) | bottom (x, a) history + top (state, command) history | regress ẑ_t onto z_t = μ(e_t) |

    Phase 1:  z_t = μ(e_t)                         a_t = π(x̃_t, a_{t-1}, o_t^top, z_t)
    Phase 2:  ẑ_t = φ(bottom hist, top hist)       a_t = π(x̃_t, a_{t-1}, o_t^top, ẑ_t)

x_t is the bottom drone's true state (v, R, ω; `obs["proprio"]`), stored in φ's history;
x̃_t is its tracking error against the flat reference (`obs["tracking_err"]`), π's input.
π sees both drones: (x̃_t, a_{t-1}) of the bottom drone and
o_t^top, the top drone's current state + previous command. o_t^top is measured at
deployment (anticipation, collision avoidance), so it is not privileged. e_t (24 dims; 29
with rotor lag) holds what cannot be measured: the bottom drone's parameters, the wake's
settings and the downwash over the next step. φ reads both drones' histories: the top one
lets it tell "heavy" from "in the wake" and predict the downwash.

**Status:** environment built and smoke-tested; networks μ, π, critic, φ implemented
(`models/`). Phase 1 PPO (`algos/phase1.py` + `algos/storage.py`) trains; pilot and
circle runs evaluated (results log in `CLAUDE.md`). Phase 2 loop: stub.

## Layout

```
ProxQuad/
├── assets/two_x650.xml      MuJoCo model: floor + bottom (ego) + top (scripted) X650
├── configs/default.py       every tunable; [CONFIRM] = open, [dev_CCM] = sourced
├── envs/
│   ├── prox_quad_env.py     ProxQuadEnv: reset/step, auto-reset, obs dict, history
│   ├── drone.py             nominal X650 params + per-env random factors, state
│   ├── motor.py             rotor lag + quadratic thrust (port of dev_CCM)
│   ├── controller.py        geometric controller flying the top drone
│   ├── scenarios.py         per-episode reference paths + start states of both drones
│   ├── randomization.py     per-episode factors -> per-world MuJoCo fields
│   ├── observations.py      layouts of x_t, x̃_t, e_t, top obs (the spec)
│   ├── rewards.py           reward terms + terminations
│   ├── history.py           Phase 2 history buffers
│   └── aero/                downwash: wake model + rotor model + drag + turbulence
├── models/                  μ, π (+ optional spectral norm) + critic, φ
├── algos/                   phase1.py + storage.py (PPO), phase2.py (stub)
├── eval/                    controllers, seeded rollouts, metrics, report (any phase)
├── utils/                   rotation math, running normalizer
└── scripts/                 smoke_test, inspect_downwash, view_episode, train_phase1/2, evaluate
```

Run from `code/`: `python -m ProxQuad.scripts.smoke_test`, `python -m ProxQuad.scripts.inspect_downwash`.
Watch an episode with the downwash drawn: `python -m ProxQuad.scripts.view_episode --scenario
hover_over` (live window), or add `--video runs/videos/hover.mp4` to record an MP4 (works
headless over SSH). The bottom drone is flown by the geometric controller, or by a trained
policy with `--checkpoint runs/<name>/model.pt`. Evaluate a checkpoint on all scenarios:
`python -m ProxQuad.scripts.evaluate --checkpoint runs/<name>/model.pt`.

## Decisions

Confirmed 2026-11-18 (simulation first; hardware questions deferred):
- MuJoCo-Warp on GPU (same stack as `code/quad`).
- Both drones are the FSC X650 from `fsc_PegasusSimulator` branch `dev_CCM`.
- Motors: plain thrust curve, **no rotor delay** for now. The dev_CCM rotor-lag model is
  implemented behind `MotorCfg.rotor_lag` (off); turning it on also adds rotor speeds and
  λ to e_t, since the lag makes (x_t, a_{t-1}) non-Markov.
- Rates (2026-11-19, was 100 Hz / 10 Hz as in RMA): policy **250 Hz** (2 × 2 ms physics),
  adaptation module **125 Hz** (every 2 policy steps; 100 Hz is not a whole number of
  4 ms steps) with the policy reusing the latest ẑ in between, histories 0.5 s = 125
  steps. Rewards are per second × dt, so returns do not depend on the rate; rollouts 60
  steps; discount for the chosen algorithm: γ_250 = γ_100^0.4. Physics cost per simulated
  second is unchanged; the per-control-step work (aero, controller, obs, policy) runs 2.5×
  as often.
- Bottom drone: **pure RL**, no base controller. A residual mode on a geometric
  controller is planned (below).
- Bottom drone's task (2026-11-20): track a **flat-output reference** (`FlatRef`: pos,
  vel, acc, jerk, yaw, yaw_dot), for now **hover** at the target with yaw 0. Flatness
  gives the reference attitude and body rate (`controller.flat_attitude`: thrust along
  acc + g e₃, mass cancels). π's input x̃_t (`tracking_err`) holds the errors:
  p − p_ref, v − v_ref, R_refᵀR, ω − RᵀR_ref ω_ref (18 dims, the controller's e_R / e_ω
  pair); the reward scores the same errors. acc reaches π through R_ref, jerk and yaw
  rate through ω_ref, so π gets no separate feedforward inputs. In hover R_ref = I and
  ω_ref = 0: x̃_t and the reward equal the raw-state versions. A hover `FlatRef` leaves
  acc / jerk / yaw as None and the env skips the flatness computation (~50 small kernels,
  12 % of a step at 4096 envs).
- The reference enters only π's input and the reward. φ's bottom history stores the
  **true state** x_t (`proprio`: v, R, ω; 15 dims, entry [x_k, a_k] = 19): φ identifies
  the dynamics from the actual motion against thrust, which errors would hide once the
  reference moves. Absolute position is left out: the dynamics do not depend on it, the
  world origin is arbitrary on hardware, and the position relative to the wake is in
  the top history.
- Top drone: geometric controller on scripted paths, no disturbances.
- Scenarios (2026-11-20): `envs/scenarios.py` (`Scenarios`, `ScenarioCfg`) samples both
  drones' reference paths per episode. Both paths are **absolute** (placed around the
  fixed target). Start states: the top drone on its reference at t = 0; the bottom drone
  around its reference (±0.3 m, ±0.2 m/s per axis), level, zero body rate, rotors at
  hover speed. A bottom start inside the collision ellipsoid × `start_clearance` (1.25)
  is redrawn (fallback: start on the reference, which `dz ≥ 0.5 m` keeps clear). Before
  this, ~0.2 % of episodes started in collision and ended on their first step.
- Circle scenarios (2026-11-21, `CircleCfg`): `circle_same` and `circle_opposite`, the
  first moving bottom reference. Both drones circle the vertical axis through the target,
  the bottom drone at r ∈ 1.0–2.5 m and v ∈ 0.5–1.5 m/s (≤ 2.25 m/s² centripetal, ~13° tilt);
  the top drone dz higher on r + δr (δr ∈ ±0.4 m), with the same angular rate at an arc
  offset of ±0.8 m (continuous or partial wake: a rotating hover_over), or the opposite
  rate from a random phase (crossings every πr/v at a closing speed of 2v). Exact vel, acc and
  jerk, so π gets R_ref and ω_ref. No thrust-magnitude input (horizontal circles change
  |acc + g|/g by < 3 %), so the obs layout and the networks are unchanged. The bottom drone
  starts level (brief transient up to the reference tilt). Off by default (the default mix
  stays the 4 original scenarios); `train_phase1 --with-circles` uses hover_over /
  pass_over / approach_hover_leave 0.2, far 0.1, circle_same / circle_opposite 0.15. With
  no circle enabled, `bottom_reference` keeps the set-point path (acc = None).
- Evaluation (2026-11-21, `eval/` + `scripts/evaluate.py`): every controller (policy,
  geometric baseline, z ablations; Phase 2's φ policy later) flies the same seeded episodes,
  one per env. Metrics per scenario: crashes, return, position / velocity / rate RMS,
  tilt and yaw error, action rate, saturation, and position / tilt error binned by the
  minimum rotor thrust ratio (wake strength, no single threshold). Tracking metrics skip
  the first 2 s (start transient). Results log: `CLAUDE.md`.
- Downwash: analytic model.
- Rotor model (2026-11-20): **BEMT** (option B), not the linear efficiency law (option A,
  its linearization at hover). Same cost; BEMT gets the larger loss in a strong wake
  (8 points more thrust lost at inflow = v_h, directly under a top rotor) and the
  rotor-speed dependence (a rotor spinning up loses relatively less), and its random
  parameter (solidity) is measurable. Its linearized loss slope is only
  k_η = σa / (σa + 16 λ_h) = 0.30–0.39 over σ ∈ [0.08, 0.12], vs [0.3, 0.7] for option A,
  so the robustness margin moved to the wake: `WakeCfg.speed_gain` [0.8, 1.2] → [0.8, 2.0].
  Gain 2 is roughly a jet whose momentum flux equals the rotor thrust (σ0 = 0.7 R with
  w0 = v_h carries only ~25 % of it). The wake gain also scales drag, turbulence and,
  once added, the transport speed consistently, which a margin on the rotor side would
  not. At the midpoint (1.4) the lift loss is close to the old defaults'; over the range
  it spans ~0.6–1.8× that.
- Inertia randomization: two independent factors, roll/pitch (xx = yy) ×[0.85, 1.15] and
  yaw (zz) ×[0.85, 1.15]; both are in e_t.
- Nominal inertia = 1.5 × CAD on all three axes. The 2026-11-10 hardware ID measured
  roll/pitch ≈ 1.5× CAD, so the CAD values are not treated as ground truth. Yaw was not
  measured; scaling it by 1.5 too is an assumption.
- CoM offset randomization: uniform ±2 cm in body x, y and ±1 cm in z from the
  geometric center (body origin); in e_t as offset / arm. Inertia about the shifted CoM
  is unchanged. Air forces (drag, gust, wake) act at the geometric center: MuJoCo
  applies `xfrc_applied` at the CoM, so the env adds the moment (p_origin − p_com) × F.
- Wake turbulence (Ornstein–Uhlenbeck force / torque) acts on the drone but is **not in
  e_t**: no history can predict it, so φ could not reproduce it and π would learn to rely
  on information it loses in Phase 2. e_t keeps the deterministic wake drag; the downwash
  tilt torque is carried by the per-rotor thrust ratios.

Planned (not yet): **wake transport delay** (`AeroCfg.transport_delay`). The wake is now
instantaneous. In the jet model w ∝ 1/s far downstream, so the travel time to distance s
is t(s) = s/w₀ + k s² / (2 w₀ σ₀) (w₀ ≈ 5.3 m/s at hover, σ₀ = 0.13 m): about 0.13 s at a
0.5 m gap, 0.35 s at 1 m and 1.0 s at 2 m (0.1–1.5 s over the randomization).
The delay is **randomized**: if the modelled timing were trusted and wrong, π (which sees
the top drone directly) would compensate for downwash that has not arrived yet. Design:
- Per-episode random scale τ = s_τ · t(s), s_τ ∈ `AeroCfg.transport_delay_scale`
  = [0.5, 2.0], so the policy cannot rely on one modelled delay. s_τ goes into e_t as
  `wake_delay_scale`; φ estimates it online by pairing wake emissions (top history)
  with their arrival (bottom history). Center the range on a delay measured in flight
  (top drone passes over a hovering bottom drone; time the onset of thrust loss).
- Ring buffer of the top drone's wake-source state per control step (rotor hubs, wake
  axis, rotor thrusts), covering the longest scaled delay: 2.0 × 1.5 s ≈ 3 s = 750 steps
  at 250 Hz (~125 MB at 4096 envs). Filled with the initial state at reset (steady wake, as if the
  top drone had always been there).
- Per evaluation point (4 bottom rotors + CoM): τ from the closed form above with the
  episode's spread / gain and s_τ, then evaluate the Gaussian jet with the top state at
  t − τ; optionally 1–2 fixed-point passes recomputing s from that past state.
- Frozen wake in the ground frame: parcels keep their emission axis and do not follow
  the top drone after emission. (Wake skew behind a moving top drone is the other big
  timing error; candidate for randomization too.)
- e_t: the downwash entries become the delayed values.
- Consequence for φ: the histories must cover the longest delay (~3 s = 750 steps at
  250 Hz, or a downsampled window, or the GRU below); the 0.5 s RMA window is not enough.
- Validation: thrust-ratio step response as the top drone moves over, against s_τ · t(s);
  and the Phase 2 policy with s_τ outside the training range (e.g. 2.5): measure early
  compensation (climb before the wake arrives).

Networks (2026-11-20, `models/`, sizes in `PolicyCfg` / `CriticCfg` / `EncoderCfg` /
`AdaptationCfg`), following RMA; activations ELU (RMA does not state them):
- μ: e_t (24) → 256 → 128 → z (8), linear output.
- π: [x̃_t, a_{t-1}, o_t^top] (44) + z (8) → 128 → 128 → 4. Diagonal Gaussian: the network
  gives the mean, log σ is a learned vector independent of the state (no variance
  network; PPO-style, as in `code/quad`), σ₀ = 0.2 throttle, clamped to e^[−5, 0.5].
  Phase 2 and deployment use the mean.
- Spectral normalization of π is a **switch**, `PolicyCfg.spectral_norm` (2026-11-20),
  **off** by default, kept for an ablation (tracking, action smoothness, measured slope
  with vs without). Any MLP with 1-Lipschitz activations is Lipschitz continuous; the
  switch only fixes the constant before training. Off: plain MLP, obs centered and
  scaled by the running mean / std, last-layer weights × `init_out_gain` (0.05) and zero
  bias so π starts at hover. Check after training: `BasePolicy.lipschitz_bound()`
  (product of the exact layer norms; certified but loose) and the largest
  |∂ mean / ∂ [obs, z]|₂ over visited states (what sets the noise sensitivity in flight;
  LipSDP if a tight certified number is needed). Without the switch the bound is in
  normalized units; in SI units it is × max 1/std, dominated by near-constant inputs: at
  init (normalizer fed reset states only, R = I exactly) max 1/std = 3000 and the
  measured slope in SI units is 26, vs 0.11 with the switch on.
- With the switch on, π is **Lipschitz** by construction: every layer is spectrally normalized
  (`torch.nn.utils.parametrizations.spectral_norm`, one power iteration per training
  forward) and scaled by K^(1/3), activations are 1-Lipschitz, so
  |mean(x) − mean(x')|₂ ≤ K |x − x'|₂ with K = `PolicyCfg.lipschitz` = 10. The bound also
  covers ẑ, so φ's estimation error moves the action by at most K |ẑ − z|.
  - Units: π's obs are only centered by a running mean, not divided by a running std, so
    the bound holds in the obs layout's SI units (m, m/s, rad/s, R entries) and does not
    move with the statistics. A std scaling would multiply it by 1/std of the quietest
    feature (R's diagonal in hover: std ~1e-3).
  - Choice of K: the geometric controller's Jacobian at hover in these units has
    singular values 1.4 (collective), 2.8 (roll / pitch), 28 (yaw: the drag-torque arm
    is 1.8 cm, so yaw needs large differential commands). The product-of-norms bound is
    loose (a random-init network reaches ~1/5 of it), so K = 10 leaves room for
    roll / pitch / thrust and softens yaw.
  - Output gain: spectral normalization undoes the usual small init of the last layer;
    with it the untrained mean was 0.6 (std per motor) off hover and 90 % of episodes
    crashed within 0.4 s. The output is multiplied by a learned g = tanh(ρ), |g| ≤ 1 (so
    the bound stays K), starting at 0.05: the untrained π then flies like hover + the
    exploration noise. `BasePolicy.lipschitz_bound()` reports |g| × the exact product of
    the layer norms (to log during training).
- Critic: separate MLP 256 → 256, asymmetric: [x̃_t, a_{t-1}, o_t^top, e_t] with the raw
  e_t, not μ, so the value loss does not shape z.
- φ: per-step 2-layer MLP embedding (32) **per drone**, the two stacked as 64 channels,
  then RMA's 3 × 1-D conv → flatten → linear. Every conv layer mixes both drones at each
  time step (telling mass from wake needs their correlation). First conv kernel / stride
  8 / 4 → 20 / 10 keeps RMA's 80 ms / 40 ms at 250 Hz; 125 steps give RMA's lengths
  11 → 7 → 3.
- Normalizers (`utils.normalizer.RunningNorm`, an nn.Module) live inside each network, so
  checkpoints carry them. Phase 1 updates μ's, π's and the critic's from each rollout
  (`Phase1Trainer.update_norms`); Phase 2 freezes them and updates only φ's.
  - Order: update the statistics **after** the PPO update on a rollout, not between
    collecting it and updating. The rollout's old log-probs and values were computed with
    the old statistics; new ones would make the ratio ≠ 1 before any parameter changed.
  - Warm-up: one rollout that only feeds the statistics before the first update. The
    initial statistics are the identity map, so without it π's input map jumps after
    iteration 1.
  - Statistics are cumulative: each rollout weighs 1/k at iteration k (245 k samples per
    rollout at 4096 envs × 60 steps), so the input map settles within tens of iterations;
    early, crash-heavy data stays in the average. Stopping the updates after N iterations
    would make it fully stationary if needed.
- Sizes: μ 40 k, π 24 k, critic 84 k, φ 56 k parameters; μ + π + critic forward and
  sample at 4096 envs ≈ 1.8 ms.

Phase 1 training (2026-11-20): **PPO**, own code (not SB3 / TorchRL / rsl_rl), written
from `code/quad/train.py` and checked against rsl_rl's `ppo.py` (installed via mjlab).
Synchronous on one GPU over the batched envs; distributed schemes such as Gorila (async
DQN over many machines) do not apply: discrete actions, off-policy, built for CPU-bound
simulators. Skeleton: `Phase1Trainer` (`algos/phase1.py`), `RolloutStorage`
(`algos/storage.py`), `PPOCfg`.
- Loop: warm-up rollout → normalizers; then rollout → update → normalizers → log / save.
- μ is in the actor's graph: the rollout stores e_t, the update recomputes z = μ(e_t)
  with gradients.
- Time-outs: the env resets inside `step()` and does not return the last obs, so the
  value is bootstrapped with V(s_t) (rsl_rl's approximation): r_t += γ V(s_t).
- One Adam over all parameters, gradients clipped per network: the critic's targets are
  ~100 at γ = 0.996 (0.4 per step in hover), so its gradients would dominate a shared norm.
- Defaults from rsl_rl (5 epochs × 4 minibatches, clip 0.2, lr 1e-3 adaptive on
  KL = 0.01, entropy 0.005, clipped value loss); γ = 0.99^0.4, λ = 0.95^0.4 for 250 Hz.
- Sanity run first: `train_phase1 --sanity` (aero off, top drone always far: plain hover).

Planned (not yet): **residual mode** (2026-11-20, deferred by the user). `ActionCfg.mode`
= "direct" | "residual". Residual: a geometric controller on the bottom drone with
nominal parameters (own `ControllerCfg`, ki = 0 so it holds no state π cannot see) gives
a_base,t at s_t; π outputs d_t and the env applies a_t = clip(a_base,t + s·d_t).
- Composed in the env, at the per-motor level: the RL action stays d ∈ [−1, 1]⁴ in both
  modes (same Gaussian, σ, clamping); a_{t-1}, φ's history and the action-rate penalty
  keep the applied command. π gets a_base,t as an extra input; its output starts at 0
  (= the controller) instead of hover; σ₀ in throttle units is divided by s.
- s = 0.5 (±184 rad/s): the default wake takes 33 % of the lift at 0.5 m directly below;
  with +15 % mass and −10 % motors the residual alone needs Δa ≈ 0.43, so 0.3 is too small.
- Optional reward term |d| to keep π close to the controller.
- Evaluation baseline: the controller alone (d = 0).

Planned (not yet): **bottom trajectory tracking**. The flat-reference plumbing is in
place; `Scenarios.bottom_reference(t)` returns hover, or the horizontal circles of the
circle scenarios (which need none of the items below). Before general trajectories train:
- Bottom path kinds in `Scenarios`. Top paths are absolute (placed around the fixed
  target); once the bottom drone moves, encounters must be placed on its path (e.g. a
  sampled crossing time), or the wake will mostly miss it.
- Thrust magnitude: R_ref holds the direction of acc_ref + g e₃ but not its length, so a
  vertical acceleration of the reference reaches π only through a growing vel_err. Add
  |acc_ref + g e₃| / g (1 dim, constant 1 in hover) to x̃_t.
- Reset around the reference at t = 0 including its attitude R_ref(0) (now: level).
- Yaw: the reference keeps yaw = 0. A nonzero yaw needs either a yaw-relative x̃_t or all
  observations of both drones in the reference heading frame.

Planned (not yet): **adaptation-module memory (GRU)**. Episode constants (the drone's
parameters, the wake's settings, the delay scale) need evidence accumulated over the
episode: a sliding-window CNN forgets the wake's settings once an encounter leaves its
window, and measuring the delay means pairing events seconds apart. Design:
- Each step's [bottom step, top step] → small per-step MLP embedding → **GRU**, hidden state
  carried across steps and reset at episode start → ẑ_t; optionally a short 1-D CNN over
  the recent window alongside, for the fast-changing downwash part of z. The GRU ingests
  every policy step (250 Hz, one GRU step each); ẑ is read out at φ's rate (125 Hz).
- If z is later split (e.g. drone vs wake heads, for ablations), the drone part must still
  be able to discount the wake (a sag is mass or wake): it needs the top drone's data or
  the wake estimate as input.
- Env: the long (~300-step) history buffers become unnecessary; the GRU needs the current
  step + its hidden state (and a short window if the CNN branch is used).
- Training: DAgger as before; store GRU hidden states in the rollouts and use truncated
  backpropagation through time over rollout chunks.
- Ablations: GRU vs RMA's 1-D CNN over a fixed window; φ with the bottom history only vs
  both histories (does identifying the drone need knowledge of the wake?).

Open (tagged `[CONFIRM]` in `configs/default.py`):
1. Action = per-motor normalized throttle, mapped linearly to rotor speed (as in PX4 direct
   actuator control). Hover ≈ −0.08.
2. 15 s episodes.
3. Nominal mass 3.25 kg / inertia 1.5 × CAD; randomized mass ×[0.85, 1.15], motor k_f ×[0.9, 1.1]
   per rotor, gust ±4 % weight.
4. Top-drone scenario ranges: vertical gap 0.5–2.0 m, lateral offset 0–0.8 m, speed 0.4–1.0 m/s.
5. Top-drone position gains (dev_CCM has none for X650); attitude gain lowered from the
   dev_CCM k_R = 30 to 20, needed when the rotor lag is on (see `ControllerCfg`).
6. Downwash calibration (below), reward weights.
7. Spectral norm on π (off; ablation) and its bound K = 10; the critic's inputs
   (asymmetric, raw e_t).
8. PPO hyperparameters (`PPOCfg`): rsl_rl defaults, to tune once training runs.

## X650 model (from dev_CCM)

| | Value | Source |
|---|---|---|
| Mass, inertia | 3.25 kg, diag(0.0867, 0.0961, 0.0975) kg m² = 1.5 × CAD | mass: `05_px4_single_drone_x650.py`; CAD inertia: `x650_params.py`; ×1.5: hardware ID 2026-11-10 |
| Rotors | (±0.2299, ±0.2299) m, PX4 Quad-X order, 0.131 m above CoM | `x650_params.py`, isaacflight `x650.yaml` |
| Thrust, yaw torque | f = 4.536e-5 ω², τ_z = 8.366e-7 ω² · dir | MN4014 + 15×5" bench fit |
| Rotor speed | 81.8 (idle) – 817.6 rad/s | bench fit |
| Rotor lag (off for now) | ω̇ = λ(ω_cmd − ω), λ = 10.51 1/s, exact ZOH per physics step | `lagged_thrust_curve.py` |
| Body drag | F = −diag(0.5, 0.3, 0) v_body | Pegasus default `LinearDrag` |

## Downwash model

Two swappable parts (`envs/aero/`), evaluated once per control step:
- **Wake** (`wake.py`): momentum-conserving Gaussian round jet from each top rotor (or one
  merged jet), axis along the top drone's thrust axis; random spread rate and speed gain.
- **Rotor** (`rotor.py`): thrust ratio of each bottom rotor under axial inflow. Default:
  option B, closed-form BEMT at the actual (lagged) rotor speed. Option A, a linear
  efficiency law, is kept for comparison.
- Plus airframe drag in the wake and Ornstein–Uhlenbeck turbulence scaled by wake strength.

Cost at 4096 envs: 0.8–1.0 ms per control step for any variant, vs 2.1 ms for physics.
CFD-grade models cannot run in the loop. They (or flight data) are for calibrating this
model or fitting the learned-surrogate stub (`surrogate.py`).

**Calibration needed.** `inspect_downwash` prints lift loss and tilt torque on a
separation grid. With current defaults (rotor jets, BEMT, random parameters at their
midpoints) the lift loss is still 13–16 % of weight at 3 m below and within 0.5 m
laterally (8–23 % on-axis over the speed-gain range), which is probably too long-ranged.
Needs reference data (published measurements or our own flights); the data should set
the `speed_gain` range, which is wide on purpose until then.
