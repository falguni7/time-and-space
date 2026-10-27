"""Features of a passing option, measured from tracking at the moment of the pass.

The same function describes the pass a player actually played and every other teammate he
could have passed to, so the xPass model can value all options at a decision moment.
"""

from __future__ import annotations

import numpy as np

from .data import MatchArrays

PASS_SPEED = 15.0   # m/s, rough average used to project where a receiver will be
FEATURES = [
    "distance", "forward", "abs_angle", "lofted", "passer_pressure_m", "receiver_marking_m",
    "defenders_closer_to_target", "lane_min_m", "lane_within_2m", "lane_within_4m",
    "target_ax", "target_ay", "start_ax", "is_restart",
]


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def option_features(m: MatchArrays, frame: int, team: int, passer: int, receiver: int,
                    lofted: bool, is_restart: bool) -> np.ndarray:
    """Feature vector for passer -> receiver (object indices 0..21) at `frame`, attacking frame."""
    d = m.direction(frame, team)
    own = slice(0, 11) if team == 0 else slice(11, 22)
    opp = slice(11, 22) if team == 0 else slice(0, 11)
    p = m.xy[frame, passer] * d
    r = m.xy[frame, receiver] * d
    rv = m.vel[frame, receiver] * d
    t_est = np.linalg.norm(r - p) / PASS_SPEED
    target = r + rv * t_est * 0.7
    defenders = m.xy[frame, opp] * d
    outfield = defenders[1:]
    dist = float(np.linalg.norm(target - p))
    lane = _segment_distance(outfield, p, target) if dist > 1.0 else np.full(len(outfield), 99.0)
    d_target = np.linalg.norm(defenders - target, axis=1)
    d_rec = float(np.linalg.norm(r - target))
    del own
    return np.array([
        dist,
        target[0] - p[0],
        abs(np.arctan2(target[1] - p[1], target[0] - p[0] + 1e-9)),
        float(lofted),
        float(np.linalg.norm(defenders - p, axis=1).min()),
        float(d_target.min()),
        float((d_target < d_rec).sum()),
        float(lane.min()),
        float((lane < 2.0).sum()),
        float((lane < 4.0).sum()),
        target[0], target[1], p[0],
        float(is_restart),
    ])


def option_target(m: MatchArrays, frame: int, team: int, passer: int, receiver: int) -> np.ndarray:
    """Projected target point of the option (attacking frame), as used in option_features."""
    d = m.direction(frame, team)
    p, r, rv = m.xy[frame, passer] * d, m.xy[frame, receiver] * d, m.vel[frame, receiver] * d
    return r + rv * (np.linalg.norm(r - p) / PASS_SPEED) * 0.7
