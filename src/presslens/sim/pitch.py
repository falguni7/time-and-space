"""Pitch geometry and coordinate frames.

Global frame: origin at the centre spot, x along the pitch (-52.5 .. 52.5 m),
y across it (-34 .. 34 m). A team frame is the global frame rotated so that the team
attacks towards +x; converting is a multiplication by the team's direction (+1 / -1).
"""

from __future__ import annotations

import numpy as np

LENGTH = 105.0
WIDTH = 68.0
HALF_L = LENGTH / 2
HALF_W = WIDTH / 2
GOAL_WIDTH = 7.32
PENALTY_DEPTH = 16.5
PENALTY_WIDTH = 40.32
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_WIDTH = 18.32
CENTRE_CIRCLE_R = 9.15
PENALTY_SPOT = 11.0

OWN_GOAL = np.array([-HALF_L, 0.0])  # in team frame
OPP_GOAL = np.array([HALF_L, 0.0])   # in team frame


def to_team_frame(xy: np.ndarray, direction: int) -> np.ndarray:
    return np.asarray(xy, dtype=float) * direction


from_team_frame = to_team_frame  # the rotation is its own inverse


def clip_to_pitch(xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    out = np.array(xy, dtype=float, copy=True)
    out[..., 0] = np.clip(out[..., 0], -HALF_L - margin, HALF_L + margin)
    out[..., 1] = np.clip(out[..., 1], -HALF_W - margin, HALF_W + margin)
    return out


def third(x_team: float) -> str:
    """Pitch third of an x coordinate in a team frame."""
    if x_team < -LENGTH / 6:
        return "defensive"
    if x_team > LENGTH / 6:
        return "attacking"
    return "middle"


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)
