"""Two-X650 proximity-flight environment, vectorized with MuJoCo-Warp on the GPU.

bottom drone: ego drone, flown directly by the RL policy (rotor speed commands, no base
              controller). Randomized mass / inertia / motors / gust (+ rotor lag if
              enabled), and
              disturbed by the top drone's downwash.
top drone:    nominal X650 on a geometric controller, tracking a scripted scenario.
              No disturbances act on it.
sensors:      on by default on dev_RMA (SensorCfg). pi's inputs, phi's histories and the
              top drone's own controller use both drones' measured states (delayed +
              noisy); rewards, terminations, e_t and the critic's inputs keep the true states.

One control step (dt = decimation x 0.002 s):
    obs_t was built from s_t and the downwash over [t, t+dt)
    step(a_t): a_t -> rotor speed commands (bottom); geometric controller (top)
               -> rotor speeds per physics substep (= command, or dev_CCM rotor lag)
               -> per-substep thrusts x downwash thrust ratio -> MuJoCo substeps
               -> reward, termination -> auto-reset -> downwash for [t+dt, t+2dt) -> obs
"""

import time
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from ..configs import EnvCfg
from ..utils.math import quat_to_rotmat
from .aero import AeroOutput, DownwashModel
from .controller import GeometricController, flat_attitude
from .drone import DroneIndex, DroneParams, DroneState
from .history import TransitionHistory
from .motor import LaggedQuadraticMotors
from .observations import (PROPRIO_LAYOUT, TOP_LAYOUT, TRACKING_LAYOUT, layout_dim, pack,
                           priv_layout)
from .randomization import DomainRandomizer
from .rewards import compute_reward, compute_terminations
from .scenarios import Scenarios
from .sensors import SensorModel

ASSETS = Path(__file__).resolve().parent.parent / "assets"


class ProxQuadEnv:
    proprio_dim = layout_dim(PROPRIO_LAYOUT)      # true state x_t, history only
    tracking_dim = layout_dim(TRACKING_LAYOUT)    # tracking errors, pi's input
    top_dim = layout_dim(TOP_LAYOUT)
    action_dim = 4
    bottom_hist_dim = proprio_dim + action_dim    # one step of the bottom history

    def __init__(self, cfg: EnvCfg | None = None):
        self.cfg = cfg = cfg or EnvCfg()
        device = cfg.sim.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dev = torch.device(device)                      # Device for all torch tensors
        self.n = n = cfg.sim.num_envs                        # Number of parallel environments
        
        # privileged environment factors e_t (envs/observations.py)
        self.priv_layout = priv_layout(cfg.motor.rotor_lag, cfg.aero.transport_delay)
        self.priv_dim = layout_dim(self.priv_layout)

        mjm = mujoco.MjModel.from_xml_path(str(ASSETS / cfg.sim.xml))  # MuJoCo model
        mjd = mujoco.MjData(mjm)                                       # MuJoCo data
        mujoco.mj_forward(mjm, mjd)
        self.sim_dt = float(mjm.opt.timestep)                          # Physics time step
        self.dt = self.sim_dt * cfg.sim.decimation                     # Environment time step
        self.max_steps = round(cfg.sim.episode_length_s / self.dt)     # Max steps per episode
        self.g = float(-mjm.opt.gravity[2])                            # Gravity
        self.idx = {k: DroneIndex.from_model(mjm, k) for k in ("bottom", "top")}    # Indices of the two drones in the MuJoCo model
        
        # Check that both drones (id: 1, 2) are one body under the world (id: 0).
        # Requirement from DomainRandomizer.commit()
        assert mjm.nbody == 3 and all(mjm.body_parentid[i.body] == 0
                                      for i in self.idx.values()), "unexpected body tree"
        
        # Warp model and data on the GPU
        with wp.ScopedDevice(device):
            self.m = mjw.put_model(mjm)
            # Maximum allowed contacts and constraints per world
            self.d = mjw.put_data(mjm, mjd, nworld=n, nconmax=cfg.sim.nconmax, 
                                  njmax=cfg.sim.njmax)

        # Set up the drone parameters
        mk = lambda name: DroneParams(mjm, self.idx[name], cfg.motor, self.g, n, self.dev)  # noqa: E731
        self.bottom, self.top = mk("bottom"), mk("top")
        
        # Setup the domain randomizer
        self.randomizer = DomainRandomizer(cfg.randomization, self.m, self.bottom,
                                           self.idx["bottom"], n, self.dev)
        
        # Setup the motor models
        self.bot_motors = LaggedQuadraticMotors(cfg.motor, self.bottom, n, self.dev)
        self.top_motors = LaggedQuadraticMotors(cfg.motor, self.top, n, self.dev)

        # torch views sharing memory with the Warp data arrays
        self.qpos = wp.to_torch(self.d.qpos)            # Position of both drones (N, 14)
        self.qvel = wp.to_torch(self.d.qvel)            # Velocity of both drones (N, 12)
        self.xfrc = wp.to_torch(self.d.xfrc_applied)    # Forces and torques applied (N, nbody, 6) (Inertial frame, applied at the CoM)
        
        # per-substep actuator inputs, copied into d.ctrl before each MuJoCo substep
        self._ctrl_seq = torch.zeros(cfg.sim.decimation, n, mjm.nu, device=self.dev)  # (substeps, N, 8)
        self._ctrl_seq_wp = [wp.from_torch(self._ctrl_seq[k])
                             for k in range(cfg.sim.decimation)]

        # Reference paths of both drones, and the top drone's controller
        self.scenario = Scenarios(cfg.scenario, cfg.termination.collision_radii, n, self.dev)
        self.top_ctrl = GeometricController(cfg.top_controller, self.top, self.dt, n, self.dev)
        
        # Setup the downwash and drag models
        self.aero = DownwashModel(cfg.aero, n, self.dev, self.dt)
        self.drag = torch.tensor(cfg.drag.linear, dtype=torch.float32, device=self.dev)
        
        # State and action histories, one per drone with its own length (0 disables a buffer)
        Lb, Lt = cfg.obs.bottom_history_len, cfg.obs.top_history_len
        self.history = (TransitionHistory(n, Lb, self.bottom_hist_dim, Lt, self.top_dim,
                                          self.dev) if Lb > 0 or Lt > 0 else None)

        # State estimates the bottom drone's controller sees (None: the true states)
        self.sensors = SensorModel(cfg.sensors, n, self.dev) if cfg.sensors.enabled else None

        # Set up the iniial actions
        m = cfg.motor
        self.a_hover = 2 * (self.bottom.omega_hover - m.omega_idle) / (m.omega_max - m.omega_idle) - 1     # normalized hover command for bottom drone
        self.prev_action = torch.full((n, 4), self.a_hover, device=self.dev)                               # previous action of bottom drone
        self.action_hist = torch.full((n, cfg.obs.action_history_len, 4), self.a_hover,
                                      device=self.dev)   # a_{t-1} .. a_{t-K}, newest first
        self.top_a_hover = 2 * (self.top.omega_hover - m.omega_idle) / (m.omega_max - m.omega_idle) - 1    # normalized hover command for top drone
        self.top_prev_action = torch.full((n, 4), self.top_a_hover, device=self.dev)                       # previous action of top drone
        
        # Environment steps
        self.steps = torch.zeros(n, dtype=torch.long, device=self.dev)                                     # number of steps since the last reset
        self.episode_len = torch.full((n,), self.max_steps, dtype=torch.long, device=self.dev)             # Episode length

        self.timing = {"physics": 0.0, "aero": 0.0}   # wall-clock seconds, for profiling
        self._graph = self._capture_graph() if cfg.sim.use_cuda_graph else None   # CUDA graph for the physics substeps
        
        # Reset all environments and initialize the first observation
        self.reset()   

        # Randomize the episode lengths for the first episode
        if cfg.sim.stagger_first_episode:
            self.episode_len[:] = torch.randint(1, self.max_steps + 1, (n,), device=self.dev)   

    # ------------------------------------------------------------------ public API
    def reset(self, ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Reset envs `ids` (all if None) and return the observation dict. A full reset
        reseeds the sensor noise from torch.initial_seed() without drawing from the global
        RNG, so episodes do not depend on the sensor settings."""
        if ids is None and self.sensors is not None:
            self.sensors.seed((torch.initial_seed() + 1) % 2**63)
        self._reset_idx(torch.arange(self.n, device=self.dev) if ids is None else ids)
        self._prepare()
        return self._obs

    def step(self, action: torch.Tensor):
        """action (N, 4) in [-1, 1]: per-motor normalized throttle.
        Returns obs, reward (N,), done (N,), info. 
        Finished envs are reset automatically, their obs is the new episode's first."""
        a = action.clamp(-1.0, 1.0)

        # Store current observation and action in the history.
        if self.history is not None:
            self.history.push(self._obs, a)

        # rotor speeds over the substeps rolled out, actions are held over the control step
        dec = self.cfg.sim.decimation
        w_bot = self.bot_motors.rollout(self.action_to_omega(a), self.sim_dt, dec)
        w_top = self.top_motors.rollout(self._top_cmd, self.sim_dt, dec)
        
        # Compute the thrust of bottom drone (might be modified by the downwash model if enabled)
        f_bot = self.bot_motors.thrust(w_bot)
        if self.cfg.aero.enabled:
            f_bot = f_bot * self.aero.rotor.thrust_ratio(self._aero.rotor_inflow, self.bottom,
                                                         w_bot)
        self._ctrl_seq[:, :, self.idx["bottom"].actuators] = f_bot
        
        # Compute the thrust of top drone
        self._ctrl_seq[:, :, self.idx["top"].actuators] = self.top_motors.thrust(w_top)

        # external wrenches, held over the control step
        bot, top = self._bot_state, self._top_state
        b, t = self.idx["bottom"].body, self.idx["top"].body
        
        # MuJoCo applies xfrc_applied at the CoM and in the inertial frame
        # Aerodynamic forces act at the geometric center (body origin)
        # Their moment arms are added: (p_origin - p_com) x F
        F_air = self._aero.force + self.randomizer.gust + self._body_drag(bot)
        com_w = torch.einsum("nij,nj->ni", bot.R, self.bottom.com_offset)  # p_com - p_origin, rotated to inertial frame
        self.xfrc[:, b, 0:3] = F_air                                                    # Apply body force to the bottom drone
        self.xfrc[:, b, 3:6] = self._aero.torque + torch.cross(-com_w, F_air, dim=-1)   # Apply body torque to the bottom drone
        self.xfrc[:, t, 0:3] = self._body_drag(top)                                     # Apply body force to the top drone
        
        # Step the physics and update the state, downwash, and observation for the next step
        self._simulate()
        self.steps += 1

        # Compute the reward, termination, and info dict.
        bot, top = self._state("bottom"), self._state("top")
        err = self._tracking_errors(bot)
        crashed, reasons = compute_terminations(bot, top, err["pos_err"], self.cfg.termination)
        time_out = (self.steps >= self.episode_len) & ~crashed
        reward, terms = compute_reward(err, a, self.prev_action, self.a_hover, crashed,
                                       self.cfg.reward, self.dt)
        done = crashed | time_out
                
        info = {
            "time_out": time_out,          # for bootstrapping: truncated, not terminal
            "crashed": crashed,
            "crash_reasons": reasons,
            "reward_terms": terms,
            "scenario": self.scenario.kind.clone(),
        }

        # Update the previous actions for the next step
        # The normalized actions that are used to fly the drones
        self.prev_action = a.clone()
        self.action_hist = torch.cat([a[:, None], self.action_hist[:, :-1]], dim=1)
        self.top_prev_action = self.omega_to_action(self._top_cmd).clamp(-1.0, 1.0)

        # Reset the finished envs and prepare the next observation.
        ids = done.nonzero(as_tuple=False).squeeze(-1)  # Track the finished envs
        if len(ids) > 0:
            self._reset_idx(ids)  # Reset the finished envs
        self._prepare()
        return self._obs, reward, done, info

    def action_to_omega(self, a: torch.Tensor) -> torch.Tensor:
        # Convert normalized action in [-1, 1] to rotor speeds in rad/s. 
        m = self.cfg.motor
        return m.omega_idle + 0.5 * (a + 1.0) * (m.omega_max - m.omega_idle)

    def omega_to_action(self, w: torch.Tensor) -> torch.Tensor:
        # Convert rotor speeds in rad/s to normalized action in [-1, 1].
        m = self.cfg.motor
        return 2.0 * (w - m.omega_idle) / (m.omega_max - m.omega_idle) - 1.0

    def get_obs(self) -> dict[str, torch.Tensor]:
        # Return the current observation dict. 
        return self._obs

    def measured_state(self, which: str) -> DroneState:
        """State of "bottom" / "top" as the bottom drone's controller sees it at s_t (the
        true state when sensors are off). Baseline controllers fly on this."""
        return self._bot_meas if which == "bottom" else self._top_meas

    def get_history(self) -> dict[str, torch.Tensor]:
        """Phase 2 input: {"bottom": (N, L_bottom, 22), "top": (N, L_top, 22)}, oldest first; 
        only the buffers with a length > 0 in cfg.obs. Views: clone() to keep."""
        assert self.history is not None, "set cfg.obs.bottom/top_history_len > 0"
        return self.history.get()

    # ------------------------------------------------------------------ internals
    def _state(self, which: str) -> DroneState:
        i = self.idx[which]
        q = self.qpos[:, i.qpos_adr:i.qpos_adr + 7].clone()
        v = self.qvel[:, i.dof_adr:i.dof_adr + 6].clone()
        return DroneState(pos=q[:, :3], quat=q[:, 3:], R=quat_to_rotmat(q[:, 3:]),
                          vel=v[:, :3], omega=v[:, 3:])

    def _body_drag(self, s: DroneState) -> torch.Tensor:
        """LinearDrag: F_drag_I = - R^T diag(c) R v_I, in the inertial frame."""
        v_body = torch.einsum("nji,nj->ni", s.R, s.vel)  # R^T v_I = v_body
        return torch.einsum("nij,nj->ni", s.R, -self.drag * v_body)   # RDRT drag

    def _tracking_errors(self, s: DroneState) -> dict[str, torch.Tensor]:
        """Errors w.r.t. the flat reference at the current episode time (obs and reward)."""
        ref = self.scenario.bottom_reference(self.steps.float() * self.dt)
        if all(v is None for v in (ref.acc, ref.jerk, ref.yaw, ref.yaw_dot)):
            # set-point: R_ref = I, w_ref = 0
            return {"pos_err": s.pos - ref.pos, "vel_err": s.vel - ref.vel,
                    "rot_err": s.R, "ang_vel_err": s.omega}
        R_ref, w_ref = flat_attitude(ref, self.g)
        RtRr = s.R.transpose(1, 2) @ R_ref                               # R^T R_ref
        return {
            "pos_err": s.pos - ref.pos,
            "vel_err": s.vel - ref.vel,
            "rot_err": RtRr.transpose(1, 2),                             # R_ref^T R
            "ang_vel_err": s.omega - (RtRr @ w_ref[..., None]).squeeze(-1),
        }

    def _reset_idx(self, ids: torch.Tensor) -> None:
        # Reset the envs ids to a random state, and resample the randomization, scenario, and downwash.
        self.randomizer.resample(ids)
        self.scenario.resample(ids)
        self.aero.resample(ids)
        self.aero.reset(ids)
        self.top_ctrl.reset(ids)
        self.bot_motors.reset(ids, self.bottom.omega_hover)   # nominal hover speed
        self.top_motors.reset(ids, self.top.omega_hover)

        b, t = self.idx["bottom"], self.idx["top"]
        bot_p, bot_v, top_p, top_v = self.scenario.start_state(ids)
        self.qpos[ids, b.qpos_adr:b.qpos_adr + 3] = bot_p
        self.qvel[ids, b.dof_adr:b.dof_adr + 3] = bot_v
        self.qpos[ids, t.qpos_adr:t.qpos_adr + 3] = top_p
        self.qvel[ids, t.dof_adr:t.dof_adr + 3] = top_v
        for i in (b, t):
            self.qpos[ids, i.qpos_adr + 3:i.qpos_adr + 7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.dev)
            self.qvel[ids, i.dof_adr + 3:i.dof_adr + 6] = 0.0
        self.xfrc[ids] = 0.0
        self.steps[ids] = 0
        self.episode_len[ids] = self.max_steps
        self.prev_action[ids] = self.a_hover
        self.action_hist[ids] = self.a_hover
        self.top_prev_action[ids] = self.top_a_hover
        if self.history is not None:
            self.history.reset(ids)
        if self.sensors is not None:
            self.sensors.reset(ids)
        self.randomizer.commit()

    def _prepare(self) -> None:
        """Everything that depends only on s_t: the top drone's command and the downwash
        over [t, t+dt). Then build obs_t."""

        # Current time in the episode
        t = self.steps.float() * self.dt
        # The top and the bottom drones' state, true and measured
        self._bot_state, self._top_state = bot, top = self._state("bottom"), self._state("top")
        self._bot_meas, self._top_meas = (self.sensors.measure(bot, top) if self.sensors
                                          else (bot, top))
        # The top drone's command, from its measured state
        self._top_cmd = self.top_ctrl(self._top_meas, self.scenario.top_reference(t))

        # Compute the downwash and drag forces on the bottom drone
        t0 = time.perf_counter()
        top_thrust = self.top_motors.thrust(self.top_motors.omega)  # thrust of the top drone's rotors
        self._aero: AeroOutput = self.aero.compute(bot, self.bottom, self.bot_motors.omega,
                                                   top, self.top, top_thrust, self.dt)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()  # For accurate timing

        self.timing["aero"] += time.perf_counter() - t0  # To time the downwash computation
        
        # Build the observation dict: measured states for pi / phi, true ones for the critic
        self._obs = self._build_obs(self._bot_meas, self._top_meas,
                                    self._tracking_errors(self._bot_meas))
        if self.sensors is None:
            self._obs["tracking_err_true"], self._obs["top_true"] = (self._obs["tracking_err"],
                                                                     self._obs["top"])
        else:
            self._obs["tracking_err_true"] = pack(TRACKING_LAYOUT, self._tracking_errors(bot))
            self._obs["top_true"] = self._top_obs(bot, top)
        if self.history is not None:
            self.history.fill_fresh(self._obs, self.prev_action)

    def _top_obs(self, bot: DroneState, top: DroneState) -> torch.Tensor:
        rel_pos, rel_vel = top.pos - bot.pos, top.vel - bot.vel
        return pack(TOP_LAYOUT, {
            "rel_pos": rel_pos, "rel_vel": rel_vel, "rotmat": top.R,
            "ang_vel": top.omega, "prev_action": self.top_prev_action,
        })

    def _build_obs(self, bot: DroneState, top: DroneState, err: dict[str, torch.Tensor]):
        # Build the observation dict, including the proprioception, the top drone's state, and the privileged factors.
        bp = self.bottom

        # Pack proprioception
        proprio = pack(PROPRIO_LAYOUT, {"vel": bot.vel, "rotmat": bot.R, "ang_vel": bot.omega})
        
        # Pack tracking errors
        tracking = pack(TRACKING_LAYOUT, err)
        
        # Pack privileged environment factors
        W = bp.weight_nom
        wake = self.aero.episode_params()       # (N, 4 / 5) this episode's wake settings
        factors = {
            "mass_ratio": bp.mass_factor, "inertia_ratio": bp.inertia_factor,
            "com_offset": bp.com_offset / bp.arm,
            "motor_factor": bp.motor_factor,
            "rotor_lambda": bp.rotor_lambda / self.cfg.motor.rotor_lambda,
            "gust_force": self.randomizer.gust / W,
            "wake_gain": wake[:, 0], "wake_spread": wake[:, 1],
            "rotor_wake_coeff": wake[:, 2], "airframe_cda": wake[:, 3],
            "wake_delay_scale": wake[:, 4:],
            "sensor_delay": (self.sensors.delay["bottom"].float() / self.cfg.sensors.delay_norm_steps
                             if self.sensors else torch.zeros(self.n, device=self.dev)),
            "rotor_thrust_ratio": self._aero.thrust_ratio,
            "wake_drag": self._aero.drag / W,
        }
        priv = pack(self.priv_layout, factors)
        
        return {"proprio": proprio, "tracking_err": tracking,
                "prev_action": self.prev_action.clone(),
                "action_hist": self.action_hist.flatten(1).clone(),
                "top": self._top_obs(bot, top), "priv": priv}

    def _substeps(self) -> None:
        """Copy the per-substep actuator inputs into d.ctrl, then step the physics"""
        for k in range(self.cfg.sim.decimation):
            wp.copy(self.d.ctrl, self._ctrl_seq_wp[k])
            mjw.step(self.m, self.d)

    def _capture_graph(self):
        # Capture the physics substeps in a CUDA graph once for faster stepping. 
        # Replayed with wp.capture_launch(graph) in _simulate().
        # Returns None if capture fails or if not on CUDA.
        if self.dev.type != "cuda":
            return None
        try:
            with wp.ScopedDevice(str(self.dev)):
                with wp.ScopedCapture() as cap:
                    self._substeps()
            return cap.graph
        except Exception as e:  # fall back to eager stepping
            print(f"[ProxQuadEnv] CUDA graph capture failed ({e}); stepping eagerly")
            return None

    def _simulate(self) -> None:
        if self.dev.type == "cuda":
            torch.cuda.synchronize()  # For accurate timing
        t0 = time.perf_counter()

        # Step the physics for one control step (decimation substeps)
        with wp.ScopedDevice(str(self.dev)):
            if self._graph is not None:
                wp.capture_launch(self._graph)  # Replay the captured CUDA graph
            else:
                self._substeps()  # Eager stepping
        
        wp.synchronize()
        self.timing["physics"] += time.perf_counter() - t0  # To time the physics stepping
