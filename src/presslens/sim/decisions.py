"""The ball carrier's options and choice.

Every option gets a true value = chance of success x hidden danger value of where the
ball ends up (`truth.true_value`), or the true xG for shots. The carrier then picks with
a softmax whose temperature grows with pressure. That pressure effect (scaled down by
composure) is the ground truth the engine's decision-gap metric has to rediscover.

Everything here is in the attacking team's frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PlayerParams
from .pitch import HALF_L, HALF_W, OPP_GOAL, unit
from .tactics import offside_line
from .truth import (
    PENALTY_XG,
    defenders_in_cone,
    pass_completion,
    pass_speed,
    pressure,
    travel_time,
    true_value,
    true_xg,
)

K_PRESS = 3.0  # how much full pressure multiplies decision noise for a zero-composure player
CARRY_M = 6.0
CARRY_DISCOUNT = 0.7  # carrying costs time and invites pressure
CARRY_CROWD_BOX = 0.3   # keep-probability factor per nearby defender for carries near goal (0.85 elsewhere)
HOLD_DISCOUNT = 0.7  # holding lets the defence organise
RISK_WEIGHT = 1.0    # weight of the opponent's value at the turnover spot
PROGRESS_PER_M = 0.0001  # utility bonus per metre the ball moves forward (scaled by tempo)
SHOT_BIAS_BOX = 1.7      # shot appetite inside 18 m
LONG_SHOT_BASE = 0.3     # shot appetite from distance (plus up to 0.6 for good long-range shooters)
CONE_PENALTY = 0.3       # shot appetite multiplier per defender in the shooting cone
MAX_HOLD_S = 2.5     # after this long on the ball the carrier must pass, carry or shoot
BOX_X = HALF_L - 16.5
BOX_HALF_W = 20.16


@dataclass
class Context:
    carrier: int
    own: np.ndarray            # (11, 2)
    own_vel: np.ndarray        # (11, 2)
    own_params: list[PlayerParams]
    opp: np.ndarray            # (11, 2)
    opp_params: list[PlayerParams]
    ball: np.ndarray           # (2,)
    decision_quality: float
    tempo: float = 0.5
    runs: dict[int, np.ndarray] = field(default_factory=dict)  # slot -> run target
    restart: str | None = None
    header: bool = False
    time_on_ball: float = 0.0
    lead: int = 0              # own goals minus opponent goals (score effects)


@dataclass
class Option:
    kind: str                  # pass | carry | shot | hold
    target: np.ndarray
    receiver: int | None = None
    lofted: bool = False
    through: bool = False
    p_success: float = 1.0
    value: float = 0.0
    info: dict = field(default_factory=dict)


def carrier_pressure(ctx: Context) -> float:
    return float(pressure(np.linalg.norm(ctx.opp - ctx.ball, axis=1).min()))


def temperature(decision_quality: float, press: float, composure: float) -> float:
    base = 0.008 - 0.0072 * decision_quality
    return base * (1.0 + K_PRESS * press * (1.0 - composure))


def _in_pitch(xy: np.ndarray, margin: float = 1.0) -> np.ndarray:
    return np.array([np.clip(xy[0], -HALF_L + margin, HALF_L - margin),
                     np.clip(xy[1], -HALF_W + margin, HALF_W - margin)])


def turnover_cost(xy: np.ndarray) -> float:
    """What losing the ball here is worth to the opponent (their danger map, mirrored), risk-weighted."""
    return RISK_WEIGHT * float(true_value(-np.asarray(xy, dtype=float))[0])


def game_state(lead: int) -> tuple[float, float, float]:
    """Score effects: (progress, shot appetite, hold-time) multipliers.

    Two goals up, a team manages the game; two goals down, it pushes.
    """
    if lead >= 2:
        k = min(lead - 1, 3)
        return 1.0 - 0.25 * k, 1.0 - 0.15 * k, 1.0 + 0.4 * k
    if lead <= -2:
        return 1.5, 1.1, 0.8
    return 1.0, 1.0, 1.0


def progress_bonus(ctx: Context, target: np.ndarray) -> float:
    return PROGRESS_PER_M * (0.5 + ctx.tempo) * game_state(ctx.lead)[0] * float(target[0] - ctx.ball[0])


def _in_box(xy: np.ndarray) -> bool:
    return xy[0] > BOX_X and abs(xy[1]) < BOX_HALF_W


def _pass_options(ctx: Context) -> list[Option]:
    me = ctx.own_params[ctx.carrier]
    opp_vmax = np.array([p.vmax for p in ctx.opp_params])
    opp_reach = np.array([p.intercept_reach_m for p in ctx.opp_params])
    opp_aerial = np.array([p.aerial for p in ctx.opp_params])
    max_dist = {"goal_kick": 65.0, "gk": 60.0, "throw_in": 30.0, "kickoff": 25.0}.get(ctx.restart, me.vision_m)
    error_scale = 1.6 if ctx.restart == "throw_in" else (1.4 if ctx.header else 1.0)
    line = offside_line(ctx.opp, ctx.ball)
    out = []
    for i in range(11):
        if i == ctx.carrier:
            continue
        rel = ctx.own[i] - ctx.ball
        dist = float(np.linalg.norm(rel))
        if dist > max(max_dist, 15.0) or dist < 2.0:
            continue
        if ctx.restart == "kickoff" and ctx.own[i][0] > ctx.ball[0] - 1.0:
            continue
        if i == 0 and ctx.ball[0] > -15.0:
            continue
        if ctx.own[i][0] > line + 0.3 and ctx.restart not in ("throw_in", "goal_kick", "corner"):
            continue  # offside
        if ctx.restart == "corner" and not (_in_box(ctx.own[i]) or dist < 15.0):
            continue
        # Play the ball into where the receiver will be when it arrives.
        t_travel = float(travel_time(dist, pass_speed(dist, False), False))
        through = i in ctx.runs
        if through:
            run_dir = ctx.runs[i] - ctx.own[i]
            lead = min(ctx.own_params[i].vmax * t_travel * 0.9, float(np.linalg.norm(run_dir)))
            meet = ctx.own[i] + unit(run_dir) * lead
        else:
            meet = ctx.own[i] + ctx.own_vel[i] * t_travel * 0.7
        meet = _in_pitch(meet)
        d_meet = float(np.linalg.norm(meet - ctx.ball))
        lofted_choices = [False]
        if ctx.restart == "corner" and _in_box(meet):
            lofted_choices = [True]
        elif ctx.restart != "throw_in" and d_meet > 18.0:
            lofted_choices = [False, True] if d_meet < 40.0 else [True]
        best = None
        for lofted in lofted_choices:
            err = (me.long_pass_error_deg if (lofted or d_meet > 30.0) else me.pass_error_deg) * error_scale
            info = pass_completion(
                ctx.ball, meet, ctx.own[i], ctx.own_params[i].vmax, ctx.own_params[i].aerial,
                ctx.opp, opp_vmax, opp_reach, opp_aerial, err, lofted,
            )
            p = info["p_complete"]
            lost_at = info["lane_point"] if info["p_lane_fail"] > info["p_contest_fail"] else meet
            value = p * (float(true_value(meet)[0]) + progress_bonus(ctx, meet)) - (1.0 - p) * turnover_cost(lost_at)
            if best is None or value > best.value:
                best = Option("pass", meet, i, lofted, through, info["p_complete"], value, info)
        out.append(best)
    return out


def _carry_options(ctx: Context) -> list[Option]:
    me = ctx.own_params[ctx.carrier]
    press_now = carrier_pressure(ctx)
    to_goal = unit(OPP_GOAL - ctx.ball)
    out = []
    for deg in (-40.0, 0.0, 40.0):
        a = np.radians(deg)
        d = np.array([to_goal[0] * np.cos(a) - to_goal[1] * np.sin(a),
                      to_goal[0] * np.sin(a) + to_goal[1] * np.cos(a)])
        # A defender standing in the path has to be beaten: the carry becomes a take-on.
        rel = ctx.opp - ctx.ball
        along = rel @ d
        across = np.abs(rel @ np.array([-d[1], d[0]]))
        in_path = np.where((along > 0.0) & (along < CARRY_M) & (across < 1.8))[0]
        if len(in_path):
            j = int(in_path[np.argmin(along[in_path])])
            target = _in_pitch(ctx.ball + d * (along[j] + 2.5))
            p_keep = float(np.clip(0.4 + 0.45 * me.dribble - 0.35 * ctx.opp_params[j].tackle, 0.15, 0.8))
            kind_info = {"take_on": True, "defender": j}
        else:
            target = _in_pitch(ctx.ball + d * CARRY_M)
            press_there = float(pressure(np.linalg.norm(ctx.opp - target, axis=1).min()))
            p_keep = (1.0 - 0.3 * press_there * (1.0 - 0.6 * me.dribble)) * (1.0 - 0.25 * press_now * (1.0 - me.dribble))
            kind_info = {"take_on": False}
        # Near goal there is cover behind every defender: dribbling into a crowded box rarely works.
        near_goal = float(np.clip((30.0 - np.linalg.norm(OPP_GOAL - target)) / 15.0, 0.0, 1.0))
        crowd = int((np.linalg.norm(ctx.opp[1:] - target, axis=1) < 6.0).sum())
        p_keep *= (0.85 - (0.85 - CARRY_CROWD_BOX) * near_goal) ** crowd
        if kind_info["take_on"]:
            p_keep *= 1.0 - 0.3 * near_goal
        gain = CARRY_DISCOUNT * float(true_value(target)[0]) + progress_bonus(ctx, target)
        value = p_keep * gain - (1.0 - p_keep) * turnover_cost(ctx.ball)
        out.append(Option("carry", target, p_success=p_keep, value=value, info=kind_info))
    return out


def _shot_option(ctx: Context) -> Option | None:
    if ctx.restart == "penalty":
        finishing = ctx.own_params[ctx.carrier].finishing
        xg = float(np.clip(PENALTY_XG * (0.85 + 0.15 * finishing), 0.6, 0.88))
        return Option("shot", OPP_GOAL.copy(), p_success=xg, value=xg, info={"xg": xg, "cone": 0})
    dist = float(np.linalg.norm(OPP_GOAL - ctx.ball))
    if dist > 28.0 or ctx.ball[0] > HALF_L - 0.5:
        return None
    me = ctx.own_params[ctx.carrier]
    cone = defenders_in_cone(ctx.ball, ctx.opp[1:])
    xg = true_xg(ctx.ball, carrier_pressure(ctx), me.finishing, ctx.header, cone)
    if ctx.restart == "free_kick":
        # Free-kick xG already assumes a wall (it acts through the block chance), so no cone penalty.
        technique = 0.5 * me.finishing + 0.5 * (0.75 + 0.5 * me.long_shot_bias)  # free-kick skill
        xg = true_xg(ctx.ball, 0.0, technique)
        value = xg * (0.3 + 0.8 * me.long_shot_bias)
        return Option("shot", OPP_GOAL.copy(), p_success=xg, value=value,
                      info={"xg": xg, "cone": cone, "free_kick": True})
    bias = (LONG_SHOT_BASE + 0.6 * me.long_shot_bias) if dist > 18.0 else SHOT_BIAS_BOX
    bias *= game_state(ctx.lead)[1]
    value = xg * bias * CONE_PENALTY**cone
    return Option("shot", OPP_GOAL.copy(), p_success=xg, value=value, info={"xg": xg, "cone": cone})


def options(ctx: Context) -> list[Option]:
    opts = [] if ctx.restart == "penalty" else _pass_options(ctx)
    if ctx.restart in (None, "free_kick", "penalty") or ctx.header:
        shot = _shot_option(ctx)
        if shot is not None:
            opts.append(shot)
    if ctx.restart is None and not ctx.header and ctx.carrier != 0:
        opts += _carry_options(ctx)
    if ctx.restart is None and not ctx.header and ctx.time_on_ball < MAX_HOLD_S * game_state(ctx.lead)[2]:
        me = ctx.own_params[ctx.carrier]
        p_keep = 1.0 - 0.6 * carrier_pressure(ctx) * (1.0 - 0.5 * me.strength)
        value = HOLD_DISCOUNT * p_keep * float(true_value(ctx.ball)[0]) - (1.0 - p_keep) * turnover_cost(ctx.ball)
        opts.append(Option("hold", ctx.ball.copy(), p_success=p_keep, value=value))
    if not opts:  # e.g. a throw-in with nobody near: hold until someone comes short
        opts.append(Option("hold", ctx.ball.copy(), value=0.0))
    return opts


def choose(opts: list[Option], temp: float, rng: np.random.Generator) -> int:
    v = np.array([o.value for o in opts])
    z = (v - v.max()) / max(temp, 1e-6)
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(opts), p=p))
