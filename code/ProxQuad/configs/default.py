"""Default configuration for ProxQuad.

Values tagged [CONFIRM] are proposals still awaiting team sign-off. Values tagged
[dev_CCM] are copied from fsc_PegasusSimulator, branch dev_CCM:
    extensions/fsc_aerial_manipulation/fsc_aerial_manipulation/rotorcraft/x650_params.py
    .../rotorcraft/lagged_thrust_curve.py, application/px4_base/05_px4_single_drone_x650.py
A `Range` is (low, high) for a uniform sample drawn once per episode, per env.
"""

from dataclasses import dataclass, field

Range = tuple[float, float]
Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class SimCfg:
    num_envs: int = 4096
    device: str | None = None        # None -> "cuda:0" if available, else "cpu"
    xml: str = "two_x650.xml"        # file in ProxQuad/assets/ (sets the 0.002 s physics step)
    decimation: int = 2              # physics steps per control step -> 250 Hz (2026-11-19)
    episode_length_s: float = 15.0   # [CONFIRM]
    # first episode of each env gets a random length, so time-outs do not all land
    # on the same step (decorrelates rollouts); later episodes use the full length
    stagger_first_episode: bool = True
    use_cuda_graph: bool = True      # capture the decimation loop once, replay every step
    nconmax: int = 16                # contacts per world (only drone-floor contacts exist)
    njmax: int = 64                  # constraint rows per world


# ---------------------------------------------------------------------------
# X650 airframe and motors (both drones)
# ---------------------------------------------------------------------------
@dataclass
class MotorCfg:
    """MN4014 + 15x5" prop, bench-fit. Thrust f = k_f w^2 along rotor +z, yaw reaction
    torque kappa * f * rot_dir (kappa = rolling-moment coeff / k_f, set as actuator gear in
    the MJCF).

    rotor_lag = False: rotor speed = command, no delay (the default until 2026-11-22).
    rotor_lag = True: first-order lag, updated every physics step with the exact
    zero-order-hold solution (dev_CCM LaggedQuadraticThrustCurve):
        w[k+1] = w_cmd + (w[k] - w_cmd) exp(-lambda dt)
    and the actual rotor speeds and lambda are added to e_t (envs/observations.py).
    """
    rotor_constant: float = 4.536223e-05     # k_f, N/(rad/s)^2              [dev_CCM]
    rolling_moment_coeff: float = 8.366e-07  # N m/(rad/s)^2 (in MJCF gear)  [dev_CCM]
    omega_idle: float = 81.8374            # rad/s at 0% throttle, armed     [dev_CCM]
    omega_max: float = 817.5911            # rad/s at 100% throttle          [dev_CCM]
    rotor_lag: bool = True                 # on since 2026-11-22 (was off)
    rotor_lambda: float = 10.51            # 1/s, tau ~ 95 ms (lag only)      [dev_CCM]
    prop_radius: float = 0.1905            # m, 15" prop


# ---------------------------------------------------------------------------
# Bottom drone: pure RL, no base controller (residual mode planned, README)
# ---------------------------------------------------------------------------
@dataclass
class ActionCfg:
    """a in [-1, 1]^4 -> per-motor normalized throttle u = (a + 1) / 2 -> rotor speed
    command w_cmd = w_idle + u (w_max - w_idle). Same interface as PX4 direct actuator
    control (actuator_motors), on which rpm is linear in throttle. Hover is a ~ -0.08.
    Motor order = PX4 Quad-X: front-right, rear-left, front-left, rear-right. [CONFIRM]
    """
    mapping: str = "throttle"   # only option implemented


# ---------------------------------------------------------------------------
# Top drone: nominal X650 on a geometric controller, no disturbances
# ---------------------------------------------------------------------------
@dataclass
class ControllerCfg:
    """Geometric (Lee) tracking controller -> rotor speed commands via the same
    normalize-then-saturate allocation Pegasus uses (force_and_torques_to_velocities).

    Attitude gains start from the dev_CCM X650 sweep (x650_attitude_sim/simulate.py:
    b_omega = 4, b_r = 15; that law's e_r ~ 2 x rotation error, so k_R = 2 b_r = 30).
    That sweep ran the controller every 2 ms physics step; at 100 Hz with a 10 ms hold
    (the control rate until 2026-11-19; now 250 Hz, 4 ms hold), k_R = 30 is marginal: it
    flips the drone whenever the rotor lag is
    slower than lambda ~ 10.7/s. k_R = 20 (b_r = 10) flew 0 flips over the full
    randomized inertia x [0.8, 1.6] and lambda [8.5, 12.5] range (2048 envs x 10 s).
    Position gains are proposed here (dev_CCM has none for X650). [CONFIRM]
    """
    kp_pos: Vec3 = (4.0, 4.0, 6.0)       # 1/s^2 (acceleration per m)
    kd_pos: Vec3 = (3.5, 3.5, 4.5)       # 1/s
    ki_pos: Vec3 = (0.0, 0.0, 0.0)
    int_limit: float = 0.5
    k_R: Vec3 = (20.0, 20.0, 20.0)       # N m/rad
    k_omega: Vec3 = (4.0, 4.0, 4.0)      # N m s/rad                      [dev_CCM]
    max_tilt_deg: float = 30.0


@dataclass
class RobustControllerCfg(ControllerCfg):
    """Bottom-drone baseline: geometric controller + UDE (uncertainty and disturbance
    estimator). The UDE low-pass filters the nominal model's residual into a lumped
    force (inertial frame) and torque (body frame) estimate, subtracted from the commanded
    force and torque. Own state and commands only, like phi. ki_pos stays 0. Filter time
    constants confirmed 2026-11-22."""
    ude_tau_force: float = 0.1      # s, low-pass time constant of the force estimate
    ude_tau_torque: float = 0.05    # s, of the torque estimate
    ude_force_limit: float = 0.5    # |d_F| per axis, fraction of nominal weight
    ude_torque_limit: float = 2.0   # |d_tau| per axis [N m]


# ---------------------------------------------------------------------------
# Scenarios: reference paths and start states of both drones (envs/scenarios.py)
# ---------------------------------------------------------------------------
@dataclass
class BottomPathCfg:
    target: Vec3 = (0.0, 0.0, 1.5)       # hover set-point of the bottom drone [CONFIRM]
    init_pos_noise: float = 0.3          # +-m per axis around the reference at reset
    init_vel_noise: float = 0.2          # +-m/s per axis around the reference at reset


@dataclass
class TopPathCfg:
    """Scripted top-drone paths, placed around the bottom drone's TARGET. Scaled for the
    X650: span (rotor tip to tip) ~1.03 m, rotor radius 0.19 m. [CONFIRM all ranges]
    """
    scenario_probs: dict[str, float] = field(default_factory=lambda: {
        "hover_over": 0.3,             # holds a point above the target (full/partial overlap)
        "pass_over": 0.3,              # flies straight across at constant speed
        "approach_hover_leave": 0.3,   # ProxFly-style: approach, dwell, leave
        "far": 0.1,                    # stays far away: no downwash (negative examples)
    })
    dz: Range = (0.5, 2.0)               # vertical separation of the hold point [m]
    lateral_offset: Range = (0.0, 0.8)   # horizontal offset of the hold point [m]
    speed: Range = (0.4, 1.0)            # approach/leave speed [m/s]
    dwell: Range = (2.0, 5.0)            # hold duration for approach_hover_leave [s]
    start_delay: Range = (0.0, 2.0)      # time before the top drone starts moving [s]
    travel_distance: float = 2.5         # distance of start/end points from hold point [m]
    far_distance: float = 4.0            # horizontal distance for the "far" scenario [m]
    sway_amp: Range = (0.0, 0.1)         # horizontal sinusoidal sway of the path [m]
    sway_freq: Range = (0.05, 0.3)       # [Hz]


@dataclass
class CircleCfg:
    """circle_same / circle_opposite: both drones circle the vertical axis through the
    bottom target, the top drone dz (TopPathCfg.dz) higher. Same angular rate (same
    direction) or opposite rate (crossings every pi r / v). [CONFIRM all ranges]"""
    radius: Range = (1.0, 2.5)           # bottom drone's circle [m]
    speed: Range = (0.5, 1.5)            # bottom drone's speed [m/s]; centripetal <= 2.25 m/s^2
    radius_offset: Range = (-0.4, 0.4)   # top radius - bottom radius [m]
    arc_offset: Range = (-0.8, 0.8)      # circle_same: top drone's lead along the circle [m]


@dataclass
class Figure8Cfg:
    """figure8_same / figure8_opposite: both drones on a lemniscate of Gerono around the
    bottom target, the top drone dz higher. Parameter rate w = speed / half_width: speed
    at the lobe tips, x1.41 at the centre crossing. [CONFIRM all ranges]"""
    half_width: Range = (1.5, 3.0)       # bottom drone's centre-to-lobe-tip distance a [m]
    speed: Range = (0.5, 1.2)            # bottom drone's speed at the lobe tips [m/s]; |acc| <= 2.1 v^2/a
    size_offset: Range = (-0.4, 0.4)     # top half-width - bottom half-width [m]
    arc_offset: Range = (-0.8, 0.8)      # figure8_same: top drone's lead, ~ along the path [m]


@dataclass
class ShuttleCfg:
    """shuttle_under: the top drone holds dz (TopPathCfg.dz) above a horizontal line through
    the target; the bottom drone shuttles along it, x = a sin(th0 + w t), w = speed / a,
    passing under the top drone at its peak speed. Peak acc = speed^2 / a."""
    amplitude: Range = (1.5, 2.5)        # a [m]
    speed: Range = (1.0, 2.0)            # peak speed, at the line's centre [m/s]
    along_offset: Range = (-0.3, 0.3)    # top hold point along the line, from its centre [m]
    cross_offset: Range = (0.0, 0.4)     # top hold point across the line [m]


@dataclass
class ScenarioCfg:
    bottom: BottomPathCfg = field(default_factory=BottomPathCfg)
    top: TopPathCfg = field(default_factory=TopPathCfg)
    circle: CircleCfg = field(default_factory=CircleCfg)
    figure8: Figure8Cfg = field(default_factory=Figure8Cfg)
    shuttle: ShuttleCfg = field(default_factory=ShuttleCfg)
    start_clearance: float = 1.25        # start outside the collision ellipsoid x this [CONFIRM]


# ---------------------------------------------------------------------------
# Body drag (plant, both drones): Pegasus default LinearDrag, F = -diag(c) v_body
# ---------------------------------------------------------------------------
@dataclass
class DragCfg:
    linear: Vec3 = (0.5, 0.3, 0.0)       # N/(m/s), body FLU            [dev_CCM default]


# ---------------------------------------------------------------------------
# Downwash of the top drone on the bottom drone (analytic). See envs/aero/.
# ---------------------------------------------------------------------------
@dataclass
class WakeCfg:
    """Velocity field produced by the top drone (Gaussian round-jet model)."""
    model: str = "gaussian_jet"
    sources: str = "rotors"          # "rotors": 4 superposed rotor jets | "body": one jet
    sigma0_rotor: float = 0.7        # initial width of a rotor jet, in prop radii
    sigma0_body: float = 1.0         # initial width of the merged jet, in arm lengths
    spread_rate: Range = (0.15, 0.30)  # d(sigma)/ds, jet growth with downstream distance
    speed_gain: Range = (0.8, 2.0)   # w0 = gain * momentum-theory induced velocity
    onset_ramp: float = 0.1          # m, smooth 0->1 gate just below the wake source


@dataclass
class RotorCfg:
    """Thrust of a bottom rotor under axial inflow w from the wake."""
    model: str = "bemt"                  # "efficiency" (option A) | "bemt" (option B)
    # tier A: ratio = clip(1 - efficiency_gain * w / v_hover, efficiency_min, 1)
    efficiency_gain: Range = (0.3, 0.7)
    efficiency_min: float = 0.3
    # tier B: closed-form BEMT, ideal twist, uniform inflow, at the ACTUAL rotor speed.
    # Hover thrust coefficient follows from k_f: C_T = k_f / (rho pi R^4) ~ 0.0090.
    solidity: Range = (0.08, 0.12)       # 2 blades x chord / (pi R), 15x5" prop
    lift_slope: float = 5.7              # per rad


@dataclass
class AeroCfg:
    enabled: bool = True
    air_density: float = 1.225
    wake: WakeCfg = field(default_factory=WakeCfg)
    rotor: RotorCfg = field(default_factory=RotorCfg)
    airframe_cda: Range = (0.02, 0.06)  # drag coeff x area of the airframe in the wake [m^2]
    # Turbulence: Ornstein-Uhlenbeck force/torque, std scaled by intensity |w_center| / v_h
    turb_force_std: float = 0.05        # fraction of weight at intensity 1      [CONFIRM]
    turb_torque_std: float = 0.015      # fraction of weight x arm at intensity 1 [CONFIRM]
    turb_corr_time: float = 0.1         # s
    # Wake transport delay (aero/transport.py, README "Wake transport delay"): the wake at a
    # point comes from the top drone's sources tau = s_tau * t(s) earlier (frozen wake)
    transport_delay: bool = True
    # per-episode random scale s_tau on the travel time, so the policy cannot rely on one
    # modelled delay; in e_t as wake_delay_scale                      [CONFIRM range]
    transport_delay_scale: Range = (0.5, 2.0)
    transport_delay_max_s: float = 3.0   # ring buffer length; longer delays are clamped
    transport_delay_passes: int = 1      # fixed-point passes recomputing s from the past source


# ---------------------------------------------------------------------------
# Domain randomization of the BOTTOM drone (per episode). The top drone is nominal.
# ---------------------------------------------------------------------------
@dataclass
class RandomizationCfg:
    # Nominal = MJCF: 3.25 kg, inertia 1.5 x CAD on all axes (hardware ID 2026-11-10:
    # 3.387 kg, inside the mass range; roll/pitch ~1.5x CAD; yaw not measured).
    mass_factor: Range = (0.85, 1.15)                                        # [CONFIRM]
    # inertia: one factor shared by roll/pitch (xx, yy), one for yaw (zz); independent of
    # mass and of each other. DomainRandomizer checks the triangle inequality on all
    # range corners.
    inertia_factor_xy: Range = (0.85, 1.15)
    inertia_factor_z: Range = (0.85, 1.15)
    motor_factor: Range = (0.9, 1.1)      # per rotor, scales k_f (thrust and yaw torque)
    # CoM offset from the geometric center (body origin, where the rotor geometry and
    # the air forces are referenced), uniform in +- this per body axis [m]
    com_offset: Vec3 = (0.02, 0.02, 0.01)
    rotor_lambda: Range = (8.5, 12.5)     # 1/s, around the bench 10.51 (rotor_lag only)
    gust_force: float = 0.04              # +- fraction of weight, per axis, constant


# ---------------------------------------------------------------------------
# Reward, termination, observation
# ---------------------------------------------------------------------------
@dataclass
class RewardCfg:
    """r = survival - sum(w_k * penalty_k); see envs/rewards.py. To revisit with the
    RL algorithm. [CONFIRM]

    Rate terms are weights PER SECOND, integrated over the control step (x dt), so returns
    do not depend on the control rate. The values equal the earlier per-step weights at
    100 Hz x 100, i.e. the same return scale as before (15 s hover ~ 1500).
    action_rate penalizes |da/dt| x dt = |a - a_prev| per step (total variation of the
    action, rate-independent for smooth actions); crash is a one-off event.
    """
    survival: float = 100.0   # per second
    pos: float = 100.0        # |p - p_ref| [m], per second
    vel: float = 10.0         # |v - v_ref| [m/s], per second
    att: float = 50.0         # 3 - trace(R_ref^T R), in [0, 4], per second
    ang_vel: float = 5.0      # |omega - R^T R_ref omega_ref| [rad/s], per second
    action: float = 0.0       # |a - a_hover|, per second
    action_rate: float = 0.05  # |a - a_prev| per step (see above)
    crash: float = 10.0       # one-off penalty on a crash


@dataclass
class TerminationCfg:
    min_height: float = 0.2       # m
    max_pos_err: float = 3.0      # m
    min_upright: float = 0.2      # R_zz (cos of tilt); 0.2 ~ 78 deg
    # drones collide if (d_xy / r_xy)^2 + (d_z / r_z)^2 < 1: r_xy ~ prop span, r_z ~
    # frame height + rotor plane offset [CONFIRM]
    collision_radii: tuple[float, float] = (1.03, 0.35)


@dataclass
class SensorCfg:
    """State estimate seen by the bottom drone's controller (envs/sensors.py), for both
    drones: the true state delayed by a per-episode number of control steps, plus white
    noise per axis. Medians over X650 flights (scripts/measure_sensor_noise.py, README).
    Rewards, terminations, e_t and eval metrics keep the true state."""
    enabled: bool = False
    pos_std: Vec3 = (0.0029, 0.0027, 0.0018)     # m, world
    vel_std: Vec3 = (0.012, 0.012, 0.0105)       # m/s, world
    att_std_deg: Vec3 = (0.15, 0.16, 0.11)       # small rotation, body axes
    ang_vel_std: Vec3 = (0.028, 0.011, 0.012)    # rad/s, body
    noise_scale: float = 1.0                     # multiplies all four
    bottom_delay_steps: tuple[int, int] = (3, 6)  # uniform integer per episode, inclusive
    top_delay_steps: tuple[int, int] = (3, 6)


@dataclass
class ObsCfg:
    # Phase 2 histories, one per drone, in env steps (0 disables; Phase 1 needs neither).
    # Both feed the adaptation module phi (RMA: 0.5 s = 125 steps at 250 Hz). Once the
    # randomized transport delay is on, both must cover the longest
    # delay (~3 s = 750 steps at 250 Hz, or a downsampled window / GRU; README "Planned").
    bottom_history_len: int = 0
    top_history_len: int = 0


@dataclass
class EnvCfg:
    sim: SimCfg = field(default_factory=SimCfg)
    motor: MotorCfg = field(default_factory=MotorCfg)
    action: ActionCfg = field(default_factory=ActionCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    top_controller: ControllerCfg = field(default_factory=ControllerCfg)
    robust_controller: RobustControllerCfg = field(default_factory=RobustControllerCfg)  # bottom baseline
    drag: DragCfg = field(default_factory=DragCfg)
    aero: AeroCfg = field(default_factory=AeroCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    obs: ObsCfg = field(default_factory=ObsCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)


# ---------------------------------------------------------------------------
# Networks (models/). Sizes from RMA; activations ELU (RMA does not state them).
# ---------------------------------------------------------------------------
@dataclass
class PolicyCfg:
    """pi: MLP -> mean of a diagonal Gaussian; log std is a learned state-independent
    vector. spectral_norm: Lipschitz bound by construction (ablation, README)."""
    hidden: tuple[int, ...] = (128, 128)
    activation: str = "elu"           # must be 1-Lipschitz (elu / relu / tanh)
    spectral_norm: bool = False
    # spectral_norm only: L2 Lipschitz bound of the mean w.r.t. [obs in SI units, z]. The
    # geometric controller at hover needs 1.4 (collective), 2.8 (roll / pitch), 28 (yaw)
    lipschitz: float = 10.0
    # shrinks the initial output (start near hover): spectral_norm: initial learned output
    # gain tanh(rho); plain MLP: scale of the last layer's initial weights
    init_out_gain: float = 0.05
    init_std: float = 0.2             # throttle units
    log_std_bounds: Range = (-5.0, 0.5)


@dataclass
class CriticCfg:
    """V: separate MLP, asymmetric (raw e_t instead of z_t)."""
    hidden: tuple[int, ...] = (256, 256)
    activation: str = "elu"


@dataclass
class EncoderCfg:
    """mu: e_t -> z_t (RMA: 256, 128)."""
    hidden: tuple[int, ...] = (256, 128)
    activation: str = "elu"


@dataclass
class AdaptationCfg:
    """phi: per-drone 2-layer MLP step embedding -> both drones stacked as channels ->
    1-D CNN over time -> linear -> z_hat (RMA)."""
    embed_dim: int = 32                 # per drone -> 64 input channels
    # (out_channels, kernel, stride). RMA: (32, 8, 4), (32, 5, 1), (32, 5, 1) over 50 steps
    # at 100 Hz. First layer x 2.5 keeps its 80 ms kernel / 40 ms stride at 250 Hz, so 125
    # steps give RMA's lengths 11 -> 7 -> 3.
    conv: tuple[tuple[int, int, int], ...] = ((32, 20, 10), (32, 5, 1), (32, 5, 1))
    activation: str = "elu"


# ---------------------------------------------------------------------------
# Training (algorithm details are still open)
# ---------------------------------------------------------------------------
@dataclass
class PPOCfg:
    """rsl_rl PPO defaults; gamma / lam keep their horizon in seconds at 250 Hz
    (x_250Hz = x_100Hz ** 0.4)."""
    gamma: float = 0.996          # 0.99 ** 0.4
    lam: float = 0.98             # 0.95 ** 0.4
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4          # 4096 envs x 60 steps / 4 = 61 440 samples each
    lr: float = 1e-3
    schedule: str = "adaptive"    # "adaptive": lr follows the KL | "fixed"
    desired_kl: float = 0.01
    lr_bounds: Range = (1e-5, 1e-2)
    entropy_coef: float = 0.005
    value_coef: float = 1.0
    clip_value_loss: bool = True
    max_grad_norm: float = 1.0    # per network (actor incl. mu, critic)
    warmup_rollouts: int = 1      # feed the normalizers before the first update


@dataclass
class Phase1Cfg:
    """Phase 1: jointly train env-factor encoder mu and base policy pi with PPO."""
    env: EnvCfg = field(default_factory=EnvCfg)
    z_dim: int = 8                # mu(e_t: 24 dims) -> z_t (RMA uses 8)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    iterations: int = 1000
    rollout_steps: int = 60       # 0.24 s at 250 Hz (24 steps at 100 Hz before)
    save_interval: int = 50       # iterations
    seed: int = 0
    log_dir: str = "runs/phase1"
    checkpoint: str = "runs/phase1/model.pt"


@dataclass
class Phase2Cfg:
    """Phase 2: train the adaptation module phi to regress z from both drones' histories.

        z_hat_t = phi(bottom history, top history)  ~  z_t = mu(e_t)   (mu frozen)

    Rates (2026-11-19): the policy acts every env step (250 Hz); phi refreshes z_hat every
    `adaptation_decimation` = 2 steps (125 Hz, fast enough for the downwash in e_t) and
    the policy reuses the latest z_hat in between, in training rollouts too. Histories:
    0.5 s = 125 steps, as in RMA.
    """
    env: EnvCfg = field(default_factory=lambda: EnvCfg(
        obs=ObsCfg(bottom_history_len=125, top_history_len=125)))
    adaptation_decimation: int = 2
    adaptation: AdaptationCfg = field(default_factory=AdaptationCfg)
    phase1_checkpoint: str = "runs/phase1_circles/model.pt"
    iterations: int = 1000
    rollout_steps: int = 20       # one Adam step per rollout: 20 x 4096 = 82k samples (RMA: 80k)
    lr: float = 5e-4
    warmup_rollouts: int = 1      # feed phi's normalizers before the first Adam step
    save_interval: int = 50       # iterations
    seed: int = 0
    log_dir: str = "runs/phase2"
    checkpoint: str = "runs/phase2/model.pt"
