"""Observation layouts. Each layout is an ordered list of (name, dim); the builders
concatenate in this order and check the sizes, so the tables below are the spec.

    proprio    x_t      bottom drone's true state          -> bottom history (phi)
    tracking_err x~_t   bottom drone's tracking errors     -> base policy pi
    prev_action a_{t-1} bottom drone's previous action    -> base policy pi
    top        o_t^top  top drone: current state + its previous command -> base policy pi,
                        and (as a history) the adaptation modules (Phase 2)
    priv       e_t      privileged environment factors     -> encoder mu (Phase 1 only)

Units: forces in nominal weights, torques in weight x arm, lengths in arm, rotor speeds as
w / w_hover - 1, vectors in the world frame unless stated otherwise.
"""

import torch

# True state of the bottom drone
# Absolute position is left out: the dynamics do not depend on it (no ground effect modelled)
PROPRIO_LAYOUT = [
    ("vel", 3),             # world frame
    ("rotmat", 9),          # body -> world, row-major
    ("ang_vel", 3),         # body frame
]

# Errors w.r.t. the bottom drone's flat reference (env._tracking_errors; 
# the reward uses the same quantities). R_ref, w_ref come from controller.flat_attitude
TRACKING_LAYOUT = [
    ("pos_err", 3),         # p - p_ref, world frame
    ("vel_err", 3),         # v - v_ref, world frame
    ("rot_err", 9),         # R_ref^T R (body -> reference body), row-major
    ("ang_vel_err", 3),     # w - R^T R_ref w_ref, body frame (the controller's e_w)
]

# Privileged information from environment
PRIV_LAYOUT = [
    # episode constants of the bottom drone
    ("mass_ratio", 1),          # true / nominal mass
    ("inertia_ratio", 2),       # true / nominal inertia: [xx = yy, zz]
    ("com_offset", 3),          # CoM - geometric center, body frame / arm
    ("motor_factor", 4),        # per-rotor k_f factor
    ("rotor_lambda", 1),        # lag bandwidth / nominal 10.51            (rotor_lag only)
    ("gust_force", 3),
    # episode constants of the wake (DownwashModel.episode_params), each / range midpoint
    ("wake_gain", 1),           # jet speed gain k_w
    ("wake_spread", 1),         # jet spread rate k_s
    ("rotor_wake_coeff", 1),    # rotor model's random coefficient: efficiency gain
                                # (tier A) or solidity (tier B)
    ("airframe_cda", 1),        # drag coefficient x area of the airframe in the wake
    ("wake_delay_scale", 1),    # transport delay scale s_tau          (transport_delay only)
    ("sensor_delay", 1),        # bottom drone's state latency [steps] / range midpoint; 0 without sensors
    # downwash over the next control interval
    ("rotor_thrust_ratio", 4),  # "efficiency degradation" per rotor
    ("wake_drag", 3),           # airframe drag in the wake (deterministic). The OU
                                # turbulence force / torque still act on the drone but
                                # are not in e_t: no history can predict them, so phi
                                # could not reproduce them. Downwash tilt torque comes
                                # from the per-rotor thrust ratios above.
]

# Top drone related state and action history for adaptation module and actor-critic
TOP_LAYOUT = [
    ("rel_pos", 3),             # p_top - p_bottom
    ("rel_vel", 3),
    ("rotmat", 9),              # top body -> world
    ("ang_vel", 3),             # top body frame
    ("prev_action", 4),         # top rotor command applied over the last step
]


LAG_ONLY = ("rotor_lambda",)
DELAY_ONLY = ("wake_delay_scale",)


def priv_layout(rotor_lag: bool, transport_delay: bool) -> list[tuple[str, int]]:
    """e_t layout: 25 dims, + 1 with rotor lag, + 1 with the wake transport delay."""
    return [(k, d) for k, d in PRIV_LAYOUT
            if (rotor_lag or k not in LAG_ONLY) and (transport_delay or k not in DELAY_ONLY)]


def layout_dim(layout) -> int:
    return sum(d for _, d in layout)


# observations for the actor and critic needed
POLICY_INPUTS = [
    ("tracking_err", layout_dim(TRACKING_LAYOUT)),
    ("prev_action", 4),
    ("top", layout_dim(TOP_LAYOUT)),
]
# critic (asymmetric): the same quantities from the true states, plus e_t ("priv")
CRITIC_INPUTS = [
    ("tracking_err_true", layout_dim(TRACKING_LAYOUT)),
    ("prev_action", 4),
    ("top_true", layout_dim(TOP_LAYOUT)),
]

# Pack different parts of tensors
def pack(layout, parts: dict[str, torch.Tensor]) -> torch.Tensor:
    out = []
    for name, dim in layout:
        x = parts[name]
        x = x.reshape(x.shape[0], -1)
        assert x.shape[1] == dim, f"{name}: expected {dim}, got {x.shape[1]}"
        out.append(x)
    return torch.cat(out, dim=-1)

# Describe the name and dimensions of the layout
def describe(layout) -> str:
    rows, i = [], 0
    for name, dim in layout:
        rows.append(f"  [{i:2d}:{i + dim:2d}] {name}")
        i += dim
    return "\n".join(rows)
