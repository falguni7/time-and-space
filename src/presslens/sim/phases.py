"""Phases of play for both teams, provider-style.

In possession:  set_piece (5 s after a corner / free kick / penalty), counter_attack (ball won in
                our half and in the final third within 10 s), transition (first 3 s after winning the ball),
                otherwise build_up / create / finish by the third the ball is in.
Out of possession: high_press / mid_block / low_block from where the defending team's outfield
                players stand: median x in their own frame above halfway / down to 22 m back / deeper.
Segments shorter than 2 s are merged into the neighbouring segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .match import DEAD, FPS, MatchResult
from .pitch import HALF_L

SET_PIECE_S = 5.0
TRANSITION_S = 3.0
MIN_REGAIN_S = 3.0   # a regain counts for transition / counter only if the ball is kept this long
COUNTER_WINDOW_S = 10.0
MIN_SEGMENT_S = 2.0
HIGH_PRESS_X = 0.0     # defending block's median outfield x (own frame) above this: high press
LOW_BLOCK_X = -22.0    # ...below this: low block
SET_PIECE_RESTARTS = ("corner", "free_kick", "penalty")


def _frame_labels(r: MatchResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame: team in possession (-1 dead), its phase, the defending team's phase."""
    T = len(r.period)
    inplay = r.ball_state != DEAD
    poss = np.where(inplay, r.possession, -1).astype(int)
    d = np.array([r.directions[p - 1][max(t, 0)] if t >= 0 else 1 for p, t in zip(r.period, poss)])
    x_att = r.ball[:, 0] * d

    attack = np.empty(T, dtype=object)
    attack[:] = None
    attack[inplay & (x_att < -HALF_L / 3)] = "build_up"
    attack[inplay & (x_att >= -HALF_L / 3) & (x_att <= HALF_L / 3)] = "create"
    attack[inplay & (x_att > HALF_L / 3)] = "finish"

    # Regains: open-play possession changes between consecutive in-play frames.
    regains = [t for t in range(1, T) if inplay[t] and inplay[t - 1] and poss[t] != poss[t - 1]
               and r.period[t] == r.period[t - 1]]
    for t0 in regains:
        team = poss[t0]
        end = t0
        while end + 1 < T and poss[end + 1] == team:
            end += 1
        if end - t0 + 1 < MIN_REGAIN_S * FPS:
            continue  # a scramble, not a real change of possession
        window = min(end, t0 + int(COUNTER_WINDOW_S * FPS))
        reached = np.where(x_att[t0:window + 1] > HALF_L / 3)[0]
        if x_att[t0] < 0.0 and len(reached):
            # Won in our own half and in the final third within 10 s: a counter-attack.
            attack[t0:t0 + reached[0] + 1] = "counter_attack"
        else:
            attack[t0:min(end, t0 + int(TRANSITION_S * FPS)) + 1] = "transition"

    for e in r.events:
        if e["type"] in SET_PIECE_RESTARTS and e.get("x") is not None:
            t0 = e["frame"]
            team = 0 if e["team"] == r.home else 1
            t1 = t0
            while t1 + 1 < T and t1 - t0 < SET_PIECE_S * FPS and poss[t1 + 1] == team:
                t1 += 1
            attack[t0:t1 + 1] = np.where(inplay[t0:t1 + 1], "set_piece", attack[t0:t1 + 1])

    defend = np.empty(T, dtype=object)
    defend[:] = None
    for t in np.where(inplay)[0]:
        dt = 1 - poss[t]
        dd = r.directions[r.period[t] - 1][dt]
        median_x = float(np.median(r.pos[t, dt, 1:, 0] * dd))
        defend[t] = "high_press" if median_x > HIGH_PRESS_X else ("mid_block" if median_x > LOW_BLOCK_X else "low_block")
    return poss, attack, defend


def _segments(team: np.ndarray, label: np.ndarray, period: np.ndarray) -> list[list]:
    """Run-length segments [team, label, start, end]; short ones merged into a neighbour."""
    segs: list[list] = []
    for t in range(len(label)):
        if label[t] is None:
            continue
        if segs and segs[-1][3] == t - 1 and segs[-1][0] == team[t] and segs[-1][1] == label[t] \
                and period[t] == period[segs[-1][2]]:
            segs[-1][3] = t
        else:
            segs.append([team[t], label[t], t, t])
    min_len = int(MIN_SEGMENT_S * FPS)
    merged: list[list] = []
    for s in segs:
        short = s[3] - s[2] + 1 < min_len
        if short and merged and merged[-1][3] == s[2] - 1 and merged[-1][0] == s[0] or merged and merged[-1][3] == s[2] - 1 and merged[-1][0] == s[0] and merged[-1][1] == s[1]:
            merged[-1][3] = s[3]
        else:
            merged.append(list(s))
    return merged


def build_phases(r: MatchResult) -> pd.DataFrame:
    poss, attack, defend = _frame_labels(r)
    clubs = (r.home, r.away)
    rows = []
    for team_arr, labels, in_poss in ((poss, attack, True), (np.where(poss >= 0, 1 - poss, -1), defend, False)):
        for team, label, start, end in _segments(team_arr, labels, r.period):
            rows.append({"team": clubs[int(team)], "in_possession": in_poss, "phase": label,
                         "start_frame": int(start), "end_frame": int(end), "period": int(r.period[start]),
                         "duration_s": round((end - start + 1) / FPS, 1)})
    df = pd.DataFrame(rows).sort_values(["start_frame", "in_possession"], ascending=[True, False])
    df.insert(0, "phase_id", np.arange(len(df), dtype=np.int32))
    return df.reset_index(drop=True)
