"""Team shape, pressing and marking: where each player wants to be this frame.

All functions work in the team's own frame (the team attacks towards +x).
Returns target points plus a per-player speed cap (jog vs sprint).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PlayerParams, Tactics
from .formations import FORMATIONS
from .pitch import HALF_L, HALF_W, OWN_GOAL, clip_to_pitch, unit

# How far forward the defensive line may step when out of possession, per press height.
DEF_LINE_CAP = {"high": 10.0, "mid": -12.0, "low": -28.0}
# Ball x (defending team frame) beyond which a team stops pressing, per press height.
PRESS_ZONE_LIMIT = {"high": HALF_L, "mid": HALF_L / 3, "low": -HALF_L / 3}
MARK_RANGE_M = 18.0      # markers only pick up attackers this close to their zone
BOX_TIGHTEN = 0.9        # how much tighter marking gets when the ball is deep in our own third
OPEN_SPACE_RADIUS_M = 6.0


@dataclass
class Targets:
    xy: np.ndarray         # (11, 2) target points, team frame
    speed_cap: np.ndarray  # (11,) m/s
    pressing: np.ndarray   # (11,) bool, player is closing down the ball


def block(ball_x: float, in_possession: bool, tactics: Tactics) -> tuple[float, float, float]:
    """Defensive-line x, block length and block width for the team's shape."""
    c = tactics.compactness
    if in_possession:
        x_def = float(np.clip(ball_x - 25.0, -38.0, 10.0))
        # The front line pushes up towards the opponent's defence even when building from deep.
        front = float(np.clip(ball_x + 28.0, 10.0, 40.0))
        length = max(38.0 + 10.0 * (1.0 - c), front - x_def)
        width = 60.0
    else:
        length = 22.0 + 12.0 * (1.0 - c)
        width = 34.0 + 16.0 * (1.0 - c)
        x_def = float(np.clip(ball_x - 18.0, -45.0, DEF_LINE_CAP[tactics.press_height]))
        if ball_x < -HALF_L / 3:
            # Ball in our own third: the block drops and squeezes around the box.
            squeeze = min((-HALF_L / 3 - ball_x) / 20.0, 1.0)
            length *= 1.0 - 0.45 * squeeze
            width *= 1.0 - 0.3 * squeeze
    return x_def, length, width


def shape_slots(ball: np.ndarray, in_possession: bool, tactics: Tactics) -> np.ndarray:
    """Formation slot positions (11, 2) for the current ball position."""
    slots = FORMATIONS[tactics.formation]
    x_def, length, width = block(ball[0], in_possession, tactics)
    shift = float(np.clip(ball[1] * (0.25 if in_possession else 0.4), -12.0, 12.0))
    xy = np.array([[x_def + depth * length, shift + w * width / 2.0] for _, depth, w in slots])
    # Goalkeeper: on the line, edging out and across with the ball.
    gk_x = -HALF_L + (6.0 if in_possession else 1.5 + np.clip((ball[0] + HALF_L) * 0.06, 0.0, 6.0))
    xy[0] = [gk_x, np.clip(ball[1] * 0.12, -3.0, 3.0)]
    return clip_to_pitch(xy, margin=-1.0)


def offside_line(opponents: np.ndarray, ball: np.ndarray) -> float:
    """x of the offside line in the attacking team's frame: second-last opponent, ball or halfway."""
    second_last = float(np.sort(opponents[:, 0])[-2])
    return max(second_last, float(ball[0]), 0.0)


def _cruise_cap(params: list[PlayerParams], dist: np.ndarray) -> np.ndarray:
    vmax = np.array([p.vmax for p in params])
    return np.where(dist > 10.0, 0.6 * vmax, np.where(dist > 4.0, 0.4 * vmax, 0.18 * vmax))


def defending_targets(
    own: np.ndarray,
    opponents: np.ndarray,
    ball: np.ndarray,
    carrier: int | None,
    params: list[PlayerParams],
    tactics: Tactics,
    mark_error: np.ndarray | None = None,
    allow_press: bool = True,
) -> Targets:
    """Targets for the team without the ball. `own`, `opponents`, `ball` are in this team's frame.

    `mark_error` (11, 2) is each player's current positional mistake (bigger for low
    defensive awareness); `allow_press=False` keeps everyone in shape (dead balls).
    """
    base = shape_slots(ball, in_possession=False, tactics=tactics)
    targets = base.copy()
    vmax = np.array([p.vmax for p in params])
    cap = _cruise_cap(params, np.linalg.norm(base - own, axis=1))
    pressing = np.zeros(11, dtype=bool)

    # Pressers: the nearest outfield players close down the ball inside the team's press zone.
    zone_active = ball[0] < PRESS_ZONE_LIMIT[tactics.press_height]
    n_press = (2 if tactics.press_intensity >= 0.6 else 1) + (1 if ball[0] < -HALF_L / 3 else 0)
    n_press = n_press if allow_press else 0
    d_ball = np.linalg.norm(own - ball, axis=1)
    for i in np.argsort(d_ball):
        if i == 0 or pressing.sum() >= n_press:
            continue
        trigger = params[i].press_trigger_m if zone_active else 4.0
        if d_ball[i] <= trigger:
            standoff = 3.5 - 2.9 * params[i].press  # 3.5 m (passive) .. 0.6 m (aggressive)
            goal_side = unit(OWN_GOAL - ball)
            targets[i] = ball + goal_side * standoff
            cap[i] = vmax[i] * (0.8 + 0.2 * params[i].press)
            pressing[i] = True

    # Markers: remaining outfield players pick up nearby attackers (one each).
    markers = [i for i in range(1, 11) if not pressing[i]]
    marked = [j for j in range(1, 11) if j != carrier]
    # Defending our own third: every defender tightens up on his man, whatever the press setting.
    urgency = float(np.clip((-HALF_L / 3 - ball[0]) / 20.0, 0.0, 1.0))
    mark_range = MARK_RANGE_M * (1.0 + 0.4 * urgency)
    if markers and marked:
        cost = np.linalg.norm(base[markers][:, None, :] - opponents[marked][None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            i, j = markers[r], marked[c]
            dist_zone = cost[r, c]
            if dist_zone > mark_range:
                continue
            press = params[i].press
            mark_dist = (6.0 - 4.5 * press) * (1.0 - BOX_TIGHTEN * urgency)  # 6 m (loose) .. 1.5 m (tight)
            att = opponents[j]
            cover = unit(0.3 * unit(ball - att) + 0.7 * unit(OWN_GOAL - att))
            mark_point = att + cover * mark_dist
            weight = (1.0 - dist_zone / mark_range) * (0.4 + 0.6 * press)
            weight += (1.0 - weight) * BOX_TIGHTEN * urgency
            targets[i] = (1.0 - weight) * base[i] + weight * mark_point
            if np.linalg.norm(targets[i] - own[i]) > 5.0:
                cap[i] = max(cap[i], 0.7 * vmax[i])

    for i in range(1, 11):
        if pressing[i]:
            continue
        if mark_error is not None:
            targets[i] = targets[i] + mark_error[i]
        # Recovering runs back towards goal: speed depends on defensive work rate.
        if own[i][0] - targets[i][0] > 5.0:
            cap[i] = vmax[i] * (0.45 + 0.5 * params[i].track_back)

    return Targets(clip_to_pitch(targets), cap, pressing)


def attacking_targets(
    own: np.ndarray,
    opponents: np.ndarray,
    ball: np.ndarray,
    carrier: int | None,
    params: list[PlayerParams],
    tactics: Tactics,
    runs: dict[int, np.ndarray] | None = None,
) -> Targets:
    """Targets for the team in possession. `own`, `opponents`, `ball` are in this team's frame.

    `runs` maps a player slot to the target of a forward run he is making (team frame).
    """
    base = shape_slots(ball, in_possession=True, tactics=tactics)
    targets = base.copy()

    # Get open: drift away from the nearest opponent when he is close.
    for i in range(1, 11):
        if i == carrier:
            continue
        d = np.linalg.norm(opponents - targets[i], axis=1)
        k = int(np.argmin(d))
        if d[k] < OPEN_SPACE_RADIUS_M:
            targets[i] += unit(targets[i] - opponents[k]) * (OPEN_SPACE_RADIUS_M - d[k]) * 0.8

    # Stay onside unless making a run.
    line = offside_line(opponents, ball)
    targets[1:, 0] = np.minimum(targets[1:, 0], line - 0.5)

    if carrier is not None:
        targets[carrier] = ball

    cap = _cruise_cap(params, np.linalg.norm(targets - own, axis=1))
    for i, run_target in (runs or {}).items():
        if i != carrier:
            targets[i] = run_target
            cap[i] = params[i].vmax * 0.95
    if carrier is not None:
        cap[carrier] = params[carrier].vmax * 0.85
    return Targets(clip_to_pitch(targets), cap, np.zeros(11, dtype=bool))


def kickoff_positions(tactics: Tactics) -> np.ndarray:
    """Starting positions in the team frame: shape for a ball on the centre spot, own half only."""
    xy = shape_slots(np.zeros(2), in_possession=False, tactics=tactics)
    xy[:, 0] = np.minimum(xy[:, 0], -1.0)
    xy[:, 1] = np.clip(xy[:, 1], -HALF_W + 2.0, HALF_W - 2.0)
    return xy
