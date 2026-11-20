"""Per-env physical parameters and kinematic state of one drone.
Two views of each drone's parameters:
  * nominal: the MJCF / dev_CCM values (what a model-based controller would assume)
  * true:    what is simulated, i.e. nominal x the random per-episode factors
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from ..configs import MotorCfg


@dataclass
class DroneState:
    pos: torch.Tensor    # (N, 3) world
    quat: torch.Tensor   # (N, 4) wxyz, body -> world
    R: torch.Tensor      # (N, 3, 3) body -> world
    vel: torch.Tensor    # (N, 3) world-frame linear velocity
    omega: torch.Tensor  # (N, 3) body-frame angular velocity (MuJoCo free-joint convention)


@dataclass
class DroneIndex:
    """MuJoCo indices of one drone in the compiled model."""
    body: int
    qpos_adr: int       # free joint: qpos[adr:adr+7] = pos(3), quat(4)
    dof_adr: int        # free joint: qvel[adr:adr+6] = lin vel (world), ang vel (body)
    sites: list[int]    # rotor sites m1..m4
    actuators: list[int]  # thrust actuators u1..u4

    @staticmethod
    def from_model(mjm: mujoco.MjModel, name: str) -> "DroneIndex":
        def nid(obj, n):
            i = mujoco.mj_name2id(mjm, obj, n)
            assert i >= 0, f"{n} not found in MJCF"
            return i
        body = nid(mujoco.mjtObj.mjOBJ_BODY, name)
        jnt = nid(mujoco.mjtObj.mjOBJ_JOINT, f"{name}_root")
        return DroneIndex(
            body=body,
            qpos_adr=int(mjm.jnt_qposadr[jnt]),
            dof_adr=int(mjm.jnt_dofadr[jnt]),
            sites=[nid(mujoco.mjtObj.mjOBJ_SITE, f"{name}_m{k}") for k in range(1, 5)],
            actuators=[nid(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_u{k}") for k in range(1, 5)],
        )


class DroneParams:
    """Nominal X650 parameters (scalars / small fixed tensors) plus per-env random
    factors (first dim is N: num_envs). Random factors stay at 1 unless DomainRandomizer sets them."""

    def __init__(self, mjm: mujoco.MjModel, idx: DroneIndex, motor: MotorCfg, g: float,
                 num_envs: int, device: torch.device):
        
        # ---- Intilializations
        self.motor = motor  # MotorCfg
        self.g = g          # Gravitational acceleration from MJCF (m/s^2)
        self.n = num_envs   # Number of parallel envs
        self.dev = device   # Device for all tensors (CPU or CUDA)
        
        # For creating tensors on the right device and type
        f32 = dict(dtype=torch.float32, device=device)

        # Check that the nominal CoM locates at the body origin
        assert np.allclose(mjm.body_ipos[idx.body], 0.0), "expects nominal CoM at body origin"

        # Check that the principal axes of inertia are aligned with the body frame axes
        assert np.allclose(mjm.body_iquat[idx.body], [1, 0, 0, 0]), "expects diagonal inertia"  
        
        # Check that the rotor sites are aligned with the body frame axes
        assert np.allclose(mjm.site_quat[idx.sites], [1, 0, 0, 0]), "rotor sites must be aligned"
        
        # Check that the actuator gear is set up correctly for thrust and torque
        # The rotor speeds translate to wrenches via the gear matrix
        gear = mjm.actuator_gear[idx.actuators]

        # Check resultant thrust acts along the body +z axis
        assert np.allclose(gear[:, :5], [0, 0, 1, 0, 0]), "thrust must act along site +z"

        # Check resultant torque acts according to k_m/k_f
        kappa = motor.rolling_moment_coeff / motor.rotor_constant
        assert np.allclose(np.abs(gear[:, 5]), kappa, rtol=1e-4), "MJCF gear != k_m / k_f"


        # ---- nominal (MJCF + MotorCfg)
        self.mass_nom = float(mjm.body_mass[idx.body])                        # Mass
        self.inertia_nom = torch.tensor(mjm.body_inertia[idx.body], **f32)    # Diagonal component of inertia (3,)
        self.rotor_pos = torch.tensor(mjm.site_pos[idx.sites], **f32)         # Rotor pos in body frame (4, 3)
        self.yaw_coef = torch.tensor(gear[:, 5], **f32)                       # kappa * rot_dir (4, 1)
        self.k_f = motor.rotor_constant
        self.prop_radius = motor.prop_radius
        self.weight_nom = self.mass_nom * g
        self.f_hover = self.weight_nom / 4.0                                   # N per rotor
        self.omega_hover = float(np.sqrt(self.f_hover / self.k_f))             # rad/s
        self.arm = float(self.rotor_pos[:, :2].norm(dim=-1).mean())
        
        # allocation in rotor-speed-squared units
        # [T, tau_x, tau_y, tau_z] = A @ w^2
        r = self.rotor_pos
        self.alloc = self.k_f * torch.stack(
            [torch.ones(4, **f32), r[:, 1], -r[:, 0], self.yaw_coef], dim=0)  # (4, 4)
        self.alloc_pinv = torch.linalg.pinv(self.alloc)

        # ---- per-env random factors (set by DomainRandomizer; 1 = nominal)
        self.mass_factor = torch.ones(num_envs, **f32)                            # Mass factor
        self.inertia_factor = torch.ones(num_envs, 2, **f32)                      # [xx = yy, zz] factors
        self.motor_factor = torch.ones(num_envs, 4, **f32)                        # Rotor efficiency factor
        self.rotor_lambda = torch.full((num_envs,), motor.rotor_lambda, **f32)    # Rotor delay time constant
        self.com_offset = torch.zeros(num_envs, 3, **f32)                         # CoM - body origin, body frame [m]

    # ---- true values --------------------------------------------------------
    @property
    def mass(self) -> torch.Tensor:              # (N,)
        return self.mass_nom * self.mass_factor

    @property
    def inertia(self) -> torch.Tensor:           # (N, 3)
        f_xy, f_z = self.inertia_factor.unbind(-1)
        return self.inertia_nom * torch.stack([f_xy, f_xy, f_z], dim=-1)

    def hover_induced_velocity(self, rho: float) -> float:
        """Momentum-theory induced velocity of one rotor at nominal hover [m/s]."""
        return float(np.sqrt(self.f_hover / (2.0 * rho * np.pi * self.prop_radius**2)))
