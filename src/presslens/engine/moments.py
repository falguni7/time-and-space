"""Moment cards: the moments worth explaining, picked by rules from the metrics cube.

Types: goal, big_chance, press_trap, press_broken, missed_option, momentum_swing.
A card holds only numbers and ids (the agents write the words) plus an importance score
that the relevance selector weighs per viewer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data
from .cube import connect_cube

MOMENTS_PATH = data.MODELS / "moments.parquet"

# Thresholds set from the season's distributions for roughly 3-6 moments of each kind per match.
HIGH_PRESSURE = 0.75
TRAP_PRESSURE = 0.9
TRAP_MAX_FREE_OPTIONS = 5
BROKEN_MIN_XT = 0.025
MISSED_MIN_GAP = 0.025
BIG_CHANCE_XG = 0.15
SWING_WINDOW_MIN = 5.0
SWING_MIN_XT = 0.10   # a side 'in control' has a 5-minute xT balance beyond this
BASE_IMPORTANCE = {"goal": 1.0, "big_chance": 0.7, "momentum_swing": 0.6, "press_trap": 0.5,
                   "press_broken": 0.45, "missed_option": 0.4}
FAILED = ("intercepted", "lost_duel", "misplaced", "out", "overhit")


def _card(row, kind: str, magnitude: float, **fields) -> dict:
    return {
        "match_id": row.match_id, "frame": int(row.frame), "minute": round(float(row.minute), 2),
        "period": int(row.period), "type": kind, "team": row.team, "opponent": row.opponent,
        "player_id": row.player_id, "importance": round(min(1.0, BASE_IMPORTANCE[kind] + magnitude), 3), **fields,
    }


def _pass_location(r) -> dict:
    return {"ax": round(float(r.start_ax), 1), "ay": round(float(r.start_ay), 1), "third": r.third, "lane": r.lane}


def match_moments(con, match_id: str) -> list[dict]:
    p = con.execute("select * from passes where match_id = ? and restart is null order by frame", [match_id]).df()
    s = con.execute("select * from shots where match_id = ? order by frame", [match_id]).df()
    cards: list[dict] = []

    for r in s.itertuples():
        located = {"location": {"ax": round(float(r.ax), 1), "ay": round(float(r.ay), 1), "third": r.third,
                                "lane": r.lane}}
        metrics = {"provider_xg": round(float(r.provider_xg), 3), "header": bool(r.header), "restart": r.restart,
                   "outcome": r.outcome}
        if r.outcome == "goal":
            cards.append(_card(r, "goal", 0.0, metrics=metrics, **located))
        elif r.provider_xg >= BIG_CHANCE_XG and r.restart != "penalty":
            cards.append(_card(r, "big_chance", 0.3 * float(r.provider_xg), metrics=metrics, **located))

    pressing = p[p.pressure >= HIGH_PRESSURE]
    for r in pressing.itertuples():
        metrics = {"pressure": round(float(r.pressure), 2), "carrier_proximity_m": round(float(r.carrier_proximity_m), 2),
                   "closing_speed": round(float(r.closing_speed), 2), "free_options": int(r.free_options),
                   "chosen_xpass": round(float(r.chosen_xpass), 3), "xt_added": round(float(r.xt_added), 4),
                   "outcome": r.outcome, "phase": r.phase, "opponent_phase": r.opponent_phase}
        if r.outcome in FAILED and r.pressure >= TRAP_PRESSURE and r.free_options <= TRAP_MAX_FREE_OPTIONS:
            # Credited to the pressing side: swap team / opponent for the card.
            card = _card(r, "press_trap", 0.5 * float(r.xt_start) * 10, metrics=metrics,
                         location=_pass_location(r), pressed_player_id=r.player_id)
            card["team"], card["opponent"] = r.opponent, r.team
            card["player_id"] = None
            cards.append(card)
        elif r.outcome == "complete" and r.xt_added >= BROKEN_MIN_XT:
            cards.append(_card(r, "press_broken", 5.0 * float(r.xt_added), metrics=metrics,
                               location=_pass_location(r), receiver_id=r.intended_id))

    for r in p[p.decision_gap >= MISSED_MIN_GAP].itertuples():
        cards.append(_card(r, "missed_option", 5.0 * float(r.decision_gap), location=_pass_location(r),
                           chosen_receiver_id=r.intended_id, best_receiver_id=r.best_receiver_id,
                           metrics={"chosen_value": round(float(r.chosen_value), 4),
                                    "best_value": round(float(r.best_value), 4),
                                    "decision_gap": round(float(r.decision_gap), 4),
                                    "chosen_xpass": round(float(r.chosen_xpass), 3),
                                    "pressure": round(float(r.pressure), 2), "outcome": r.outcome}))

    cards += _momentum_swings(con, match_id)
    cards.sort(key=lambda c: c["frame"])
    for k, c in enumerate(cards):
        c["moment_id"] = f"{match_id}-M{k:03d}"
    return cards


def _momentum_swings(con, match_id: str) -> list[dict]:
    acts = con.execute("""
        select team, opponent, minute, frame, period, xt_added from passes where match_id = ?
        union all
        select team, opponent, minute, frame, period, xt_added from carries where match_id = ?
        order by frame""", [match_id, match_id]).df()
    if acts.empty:
        return []
    home = acts.team.iloc[0]
    away = acts.opponent.iloc[0]
    acts["signed"] = np.where(acts.team == home, acts.xt_added, -acts.xt_added)
    minutes = np.arange(SWING_WINDOW_MIN, acts.minute.max() + 0.5, 0.5)
    bal = np.array([acts.signed[(acts.minute > t - SWING_WINDOW_MIN) & (acts.minute <= t)].sum() for t in minutes])
    cards, side, last = [], 0, -99.0
    for i, t in enumerate(minutes):
        now = 1 if bal[i] >= SWING_MIN_XT else (-1 if bal[i] <= -SWING_MIN_XT else 0)
        if now == 0:
            continue
        if side != 0 and now != side and t - last >= SWING_WINDOW_MIN:
            # Control passed from one side to the other.
            row = acts.iloc[(acts.minute - t).abs().argmin()]
            team, other = (home, away) if now > 0 else (away, home)
            cards.append({"match_id": match_id, "frame": int(row.frame), "minute": round(float(t), 2),
                          "period": int(row.period), "type": "momentum_swing", "team": team, "opponent": other,
                          "player_id": None,
                          "importance": round(min(1.0, BASE_IMPORTANCE["momentum_swing"] + abs(bal[i])), 3),
                          "metrics": {"xt_balance_5min": round(float(abs(bal[i])), 3)}})
            last = t
        side = now
    return cards


def build(match_ids: list[str] | None = None) -> pd.DataFrame:
    con = connect_cube()
    try:
        cards = [c for mid in (match_ids or data.match_ids()) for c in match_moments(con, mid)]
    finally:
        con.close()
    df = pd.DataFrame([{**{k: c.get(k) for k in ("moment_id", "match_id", "frame", "minute", "period", "type",
                                                 "team", "opponent", "player_id", "importance")},
                        "card": json.dumps(c)} for c in cards])
    df.to_parquet(MOMENTS_PATH, index=False)
    return df


def load_moments(match_id: str) -> list[dict]:
    df = pd.read_parquet(MOMENTS_PATH, filters=[("match_id", "=", match_id)])
    return [json.loads(c) for c in df.sort_values("frame").card]
