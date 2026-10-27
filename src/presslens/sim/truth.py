"""Hidden ground-truth models used by the simulation to decide outcomes.

The engine never reads these. It has to estimate its own xT grid and pass-completion
model from the generated data, and validation compares those estimates with these
functions.

All coordinates are in the attacking team's frame (attacking towards +x).
"""

from __future__ import annotations

import math

import numpy as np

from .pitch import GOAL_WIDTH, HALF_L, LENGTH

XG_SCALE = 0.5       # how hard scoring is: xG = XG_SCALE * (goal-mouth angle)^2 * modifiers
XG_CONE_FACTOR = 0.8  # xG multiplier per defender in the shooting cone
PENALTY_XG = 0.76     # average penalty conversion
XG_LONG_DECAY_M = 20.0  # beyond 16 m xG falls by a factor e every this many metres
GROUND_DRAG = 0.45   # 1/s, rolling ball
AIR_DRAG = 0.10      # 1/s, lofted ball (horizontal)
REACTION_S = 0.5     # defender reaction time before moving to a pass lane
GK_CLAIM = 0.35      # chance the keeper claims a cross into the six-yard area
GK_CLAIM_WIDE = 0.15  # ...and one dropping between the six-yard line and the penalty spot
BOX_AERIAL_EDGE = 1.6  # defenders' advantage in aerial duels inside their own box (logit units)
BOX_GROUND_EDGE = 0.25 # extra chance a defender wins a ground ball played into his own box


def goal_angle(xy: np.ndarray) -> np.ndarray:
    """Angle (radians) of the goal mouth as seen from each point."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    dx = np.maximum(HALF_L - xy[:, 0], 0.05)
    a1 = np.arctan2(GOAL_WIDTH / 2 - xy[:, 1], dx)
    a2 = np.arctan2(-GOAL_WIDTH / 2 - xy[:, 1], dx)
    return np.abs(a1 - a2)


def true_value(xy: np.ndarray) -> np.ndarray:
    """Hidden danger map V: chance of the possession leading to a goal from this spot."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    progress = np.clip((xy[:, 0] + HALF_L) / LENGTH, 0.0, 1.0)
    return np.clip(0.003 + 0.04 * progress**3 + 0.5 * goal_angle(xy) ** 2, 0.0, 0.3)


def pressure(nearest_defender_m: float | np.ndarray) -> float | np.ndarray:
    """0 (no defender within 5 m) .. 1 (defender within 1 m)."""
    return np.clip((5.0 - np.asarray(nearest_defender_m)) / 4.0, 0.0, 1.0)


def true_xg(xy: np.ndarray, shot_pressure: float, finishing: float, header: bool = False,
            cone: int = 0) -> float:
    """Chance a shot from `xy` is scored against an average keeper.

    `cone` = outfield defenders between the ball and the posts: bodies in the way shrink the target.
    """
    base = XG_SCALE * float(goal_angle(xy)[0]) ** 2
    base *= (1.0 - 0.5 * shot_pressure) * finishing * XG_CONE_FACTOR**cone
    dist = float(np.hypot(HALF_L - xy[0], xy[1]))
    if dist > 16.0:
        base *= np.exp(-(dist - 16.0) / XG_LONG_DECAY_M)  # long shots: more time for the keeper, less power
    if header:
        base *= 0.5
    return float(np.clip(base, 0.005, 0.6))


def defenders_in_cone(xy: np.ndarray, defenders: np.ndarray, margin: float = 1.5) -> int:
    """Outfield defenders inside the triangle from the ball to the posts (widened by a body's reach)."""
    p = np.asarray(xy, dtype=float)
    half = GOAL_WIDTH / 2 + margin
    posts = np.array([[HALF_L, -half], [HALF_L, half]])
    a, b, c = p, posts[0], posts[1]
    v0, v1 = c - a, b - a
    v2 = defenders - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-9:
        return 0
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    return int(((u >= 0) & (v >= 0) & (u + v <= 1)).sum())


def arrival_speed(dist: float) -> float:
    """Ground passes are weighted to reach the receiver still moving (faster for longer passes)."""
    return float(np.clip(4.0 + 0.12 * dist, 5.0, 9.0))


def pass_speed(dist: float, lofted: bool) -> float:
    if lofted:
        return float(np.clip(12.0 + 0.25 * dist, 14.0, 22.0))
    # Under rolling drag speed falls linearly with distance: v(s) = v0 - k * s.
    return float(min(arrival_speed(dist) + GROUND_DRAG * dist, 30.0))


def travel_time(dist: float | np.ndarray, v0: float, lofted: bool) -> float | np.ndarray:
    """Time for a ball kicked at v0 to cover `dist` under drag."""
    k = AIR_DRAG if lofted else GROUND_DRAG
    frac = np.minimum(k * np.asarray(dist) / v0, 0.95)
    return -np.log(1.0 - frac) / k


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def pass_completion(
    passer: np.ndarray,
    target: np.ndarray,
    receiver: np.ndarray,
    receiver_vmax: float,
    receiver_aerial: float,
    defenders: np.ndarray,
    def_vmax: np.ndarray,
    def_reach: np.ndarray,
    def_aerial: np.ndarray,
    error_deg: float,
    lofted: bool,
) -> dict:
    """True completion probability of a pass and how it would fail.

    Returns p_complete plus the probabilities of each failure mode and the most likely
    interceptor (index into `defenders`) and interception point.
    """
    dist = float(np.linalg.norm(target - passer))
    v0 = pass_speed(dist, lofted)
    t_arrive = float(travel_time(dist, v0, lofted))

    # Execution: lateral miss of dist * tan(error) vs. what the receiver can adjust to.
    sigma = max(dist * math.tan(math.radians(error_deg)), 1e-3)
    tolerance = 2.0 + 0.08 * dist  # receivers adjust more to longer balls
    p_exec = math.erf(tolerance / (sigma * math.sqrt(2.0)))

    # Lane interception (ground passes only): can a defender reach the line before the ball?
    p_lane_each = np.zeros(len(defenders))
    lane_point = np.repeat(target[None, :], len(defenders), axis=0)
    if not lofted and dist > 5.0:
        # The last 3 m before the receiver are the receiver-vs-marker contest, not the lane.
        s = np.linspace(2.0, max(dist - 3.0, 2.5), 8)
        pts = passer + (target - passer)[None, :] * (s / dist)[:, None]          # (8, 2)
        t_ball = travel_time(s, v0, lofted)                                         # (8,)
        d = np.linalg.norm(defenders[:, None, :] - pts[None, :, :], axis=2)         # (n, 8)
        move = np.maximum(d - def_reach[:, None], 0.0)
        # Reaction time only matters if the defender has to move; one standing in the lane just blocks it.
        t_def = move / def_vmax[:, None] + REACTION_S * np.minimum(move, 1.0)
        in_lane = np.clip(1.0 - move, 0.0, 1.0)
        p = (0.6 + 0.3 * in_lane) * _sigmoid((t_ball[None, :] - t_def) / 0.3)
        best = p.argmax(axis=1)
        p_lane_each = p[np.arange(len(defenders)), best]
        lane_point = pts[best]
    p_lane = float(np.prod(1.0 - p_lane_each))

    # Contest at the target: does a defender get there before (or with) the receiver?
    t_rec = float(np.linalg.norm(target - receiver)) / receiver_vmax * 1.1 + 0.2
    d_t = np.linalg.norm(defenders - target, axis=1)
    t_def_t = np.maximum(d_t - 1.0, 0.0) / def_vmax + REACTION_S
    j = int(np.argmin(t_def_t))
    # Who gets to the ball when it arrives? Defender only: he wins. Both: a duel the
    # receiver usually wins (body between man and ball), or an aerial duel for high balls.
    rec_there = float(_sigmoid((t_arrive + 0.3 - t_rec) / 0.2))
    def_there = float(_sigmoid((t_arrive + 0.1 - t_def_t[j]) / 0.2))
    if lofted:
        # In their own box defenders attack the ball facing it; attackers often have their back to goal.
        in_box = target[0] > HALF_L - 16.5 and abs(target[1]) < 20.16
        duel = float(_sigmoid((def_aerial[j] - receiver_aerial) * 4.0 + (BOX_AERIAL_EDGE if in_box else 0.0)))
    else:
        # The player closer to the ball usually wins it; a clearly first receiver loses ~15%.
        d_rec = float(np.linalg.norm(target - receiver))
        duel = 0.08 + 0.4 * float(_sigmoid((d_rec - d_t[j]) / 1.0))
        if target[0] > HALF_L - 16.5 and abs(target[1]) < 20.16:
            duel += BOX_GROUND_EDGE  # crowded box: defenders get a foot in more often
    p_contest = def_there * ((1.0 - rec_there) * 0.85 + rec_there * duel)

    if lofted and target[0] > HALF_L - 12.0 and abs(target[1]) < 12.0:
        # The keeper comes for crosses: often into the six-yard area, sometimes as far as the penalty spot.
        claim = GK_CLAIM if target[0] > HALF_L - 6.0 and abs(target[1]) < 10.0 else GK_CLAIM_WIDE
        p_contest = 1.0 - (1.0 - p_contest) * (1.0 - claim)

    p_complete = p_exec * p_lane * (1.0 - p_contest)
    k = int(np.argmax(p_lane_each)) if len(defenders) else 0
    return {
        "p_complete": float(p_complete),
        "p_exec_fail": 1.0 - p_exec,
        "p_lane_fail": p_exec * (1.0 - p_lane),
        "p_contest_fail": p_exec * p_lane * p_contest,
        "lane_defender": k,
        "lane_point": lane_point[k],
        "contest_defender": j,
        "t_arrive": t_arrive,
        "v0": v0,
        "dist": dist,
    }
