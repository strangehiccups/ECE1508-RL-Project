"""Tier C (placeholder): learned surrogate of the interaction.

Idea: an MLP fit offline to real flight data (e.g. Neural-Swarm-style logs of the
residual force on a Crazyflie under another one) or to CFD runs, mapping the relative
state of the two drones (+ top thrust, + sizes) to the per-rotor inflow or directly to
the force/torque. Inference is one small MLP forward per env per control step, so it is
cheap enough for training; the cost is collecting and fitting the data.

Not implemented: needs a decision on the data source first (README, open decisions).
"""

from .base import WakeModel


class LearnedWake(WakeModel):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("learned surrogate: data source not decided yet")

    def velocity(self, points, top, top_params, top_thrust):
        raise NotImplementedError
