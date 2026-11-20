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

**Status:** environment built and smoke-tested. Networks (`models/`) and training loops
(`algos/`) are interface stubs until the RL algorithm and architectures are decided.

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
├── models/                  μ, π + critic, φ (stubs)
├── algos/                   phase1.py, phase2.py (stubs: data flow only)
├── utils/                   rotation math, running normalizer
└── scripts/                 smoke_test, inspect_downwash, train_phase1/2
```

Run from `code/`: `python -m ProxQuad.scripts.smoke_test`, `python -m ProxQuad.scripts.inspect_downwash`.

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
- Bottom drone: **pure RL**, no base controller, no residual.
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
- Downwash: analytic model.
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

Planned (not yet): **bottom trajectory tracking**. The flat-reference plumbing is in
place; `Scenarios.bottom_reference(t)` returns hover. Before trajectories train:
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
6. Downwash calibration (below), rotor model tier (efficiency vs BEMT), reward weights.

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
- **Rotor** (`rotor.py`): thrust ratio of each bottom rotor under axial inflow. Tier A is a
  linear efficiency law. Tier B is closed-form BEMT at the actual (lagged) rotor speed.
- Plus airframe drag in the wake and Ornstein–Uhlenbeck turbulence scaled by wake strength.

Cost at 4096 envs: 0.8–1.0 ms per control step for any variant, vs 2.1 ms for physics.
CFD-grade models cannot run in the loop. They (or flight data) are for calibrating this
model or fitting the learned-surrogate stub (`surrogate.py`).

**Calibration needed.** `inspect_downwash` prints lift loss and tilt torque on a
separation grid. With current defaults the lift loss is still 11–16 % of weight at 3 m,
which is probably too long-ranged. Needs reference data (published measurements or our
own flights).
