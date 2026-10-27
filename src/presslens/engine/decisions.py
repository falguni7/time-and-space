"""Pressure, option values and decision gap at every pass (the Proximity Score idea, built out).

At each pass the engine measures pressure on the passer, values every teammate as an option
    value = xPass * xT(target) - (1 - xPass) * xT_opponent(target)
and compares the chosen option with the best one (decision gap, efficiency).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import MatchArrays, connect, load_arrays, single_threaded
from .features import FEATURES, option_features, option_target
from .xpass import XPassModel
from .xt import NX, NY, XTModel, zone_index

FREE_OPTION_M = 5.0     # an option with no defender within this distance counts as free
PRESSURE_FAR_M = 5.0    # pressure index: 0 at this distance ...
PRESSURE_NEAR_M = 1.0   # ... 1 at this distance or closer


@lru_cache(maxsize=1)
def _models() -> tuple[XTModel, XPassModel]:
    return XTModel.load(), XPassModel.load()


def pressure_index(distance_m: np.ndarray | float) -> np.ndarray | float:
    return np.clip((PRESSURE_FAR_M - np.asarray(distance_m)) / (PRESSURE_FAR_M - PRESSURE_NEAR_M), 0.0, 1.0)


def carrier_pressure(m: MatchArrays, frame: int, team: int, carrier: int) -> tuple[float, float]:
    """Nearest-defender distance to the carrier and that defender's closing speed (m/s, + = closing)."""
    opp = slice(11, 22) if team == 0 else slice(0, 11)
    rel = m.xy[frame, opp] - m.xy[frame, carrier]
    dist = np.linalg.norm(rel, axis=1)
    j = int(np.argmin(dist))
    if dist[j] < 1e-6:
        return float(dist[j]), 0.0
    rel_vel = m.vel[frame, opp][j] - m.vel[frame, carrier]
    closing = -float(rel_vel @ (rel[j] / dist[j]))
    return float(dist[j]), closing


def zone_labels(ax: float, ay: float) -> dict:
    ix, iy = zone_index(np.array([ax]), np.array([ay]))
    third = "defensive" if ax < -17.5 else ("attacking" if ax > 17.5 else "middle")
    lane = ("left_wing" if ay > 20.16 else "left_half_space" if ay > 9.16 else
            "centre" if ay > -9.16 else "right_half_space" if ay > -20.16 else "right_wing")
    return {"zone_x": int(ix[0]), "zone_y": int(iy[0]), "zone": int(ix[0] * NY + iy[0]),
            "third": third, "lane": lane}


def evaluate_options(m: MatchArrays, frame: int, team: int, passer: int, lofted: bool, restart: bool,
                     xt_model: XTModel, xpass_model: XPassModel):
    """Every teammate as a pass option: (object indices, xPass, expected value, receiver marking m)."""
    own = range(11) if team == 0 else range(11, 22)
    opts = [i for i in own if i != passer]
    X = np.array([option_features(m, frame, team, passer, i, lofted, restart) for i in opts])
    xpass = xpass_model.predict(X)
    targets = np.array([option_target(m, frame, team, passer, i) for i in opts])
    gain = xt_model.value(targets[:, 0], targets[:, 1])
    loss = xt_model.opponent_value(targets[:, 0], targets[:, 1])
    value = xpass * gain - (1.0 - xpass) * loss
    return opts, xpass, value, X[:, FEATURES.index("receiver_marking_m")]


def match_decisions(match_id: str) -> pd.DataFrame:
    xt_model, xpass_model = _models()
    con = connect()
    passes = con.execute(f"""
        select e.event_id, e.frame, e.period, e.minute, e.team, e.player_id, e.intended_id, e.outcome,
               e.lofted, e.restart, e.ax, e.ay, e.end_ax, e.end_ay
        from events_a e where e.match_id = '{match_id}' and e.type = 'pass' and e.intended_id is not null
        order by frame""").df()
    phases = con.execute(f"""
        select team, in_possession, phase, start_frame, end_frame from phases where match_id = '{match_id}'""").df()
    m = load_arrays(match_id)
    rows = []
    for p in passes.itertuples():
        team = m.team_index(p.team)
        passer = m.player_index(p.frame, p.player_id)
        intended = m.player_index(p.frame, p.intended_id)
        if passer < 0 or intended < 0:
            continue
        opts, xpass, value, marking = evaluate_options(m, p.frame, team, passer, bool(p.lofted),
                                                       p.restart is not None, xt_model, xpass_model)
        k_chosen = opts.index(intended)
        k_best = int(np.argmax(value))

        dist, closing = carrier_pressure(m, p.frame, team, passer)
        xt_start = float(xt_model.value(p.ax, p.ay))
        xt_end = float(xt_model.value(p.end_ax, p.end_ay)) if p.outcome == "complete" else 0.0
        ph = phases[(phases.start_frame <= p.frame) & (phases.end_frame >= p.frame)]
        att = ph[ph.in_possession & (ph.team == p.team)]
        dfn = ph[~ph.in_possession & (ph.team != p.team)]
        best = float(value[k_best])
        chosen = float(value[k_chosen])
        rows.append({
            "match_id": match_id, "event_id": p.event_id, "frame": p.frame, "period": p.period,
            "minute": float(p.minute), "team": p.team, "opponent": m.away if team == 0 else m.home,
            "player_id": p.player_id, "intended_id": p.intended_id, "outcome": p.outcome,
            "restart": p.restart, "lofted": bool(p.lofted), "start_ax": p.ax, "start_ay": p.ay,
            **zone_labels(p.ax, p.ay),
            "phase": att.phase.iloc[0] if len(att) else None,
            "opponent_phase": dfn.phase.iloc[0] if len(dfn) else None,
            "carrier_proximity_m": dist, "pressure": float(pressure_index(dist)), "closing_speed": closing,
            "n_options": len(opts), "free_options": int((marking > FREE_OPTION_M).sum()),
            "mean_option_proximity_m": float(marking.mean()),
            "chosen_xpass": float(xpass[k_chosen]), "chosen_value": chosen,
            "best_value": best, "best_receiver_id": m.names[m.ids[p.frame, opts[k_best]]],
            "chose_best": k_best == k_chosen, "decision_gap": best - chosen,
            "xt_start": xt_start, "xt_end": xt_end,
            "xt_added": (xt_end - xt_start) if p.outcome == "complete" else -xt_start,
        })
    return pd.DataFrame(rows)


def all_decisions(match_ids: list[str], workers: int = 56) -> pd.DataFrame:
    with ProcessPoolExecutor(min(workers, len(match_ids)), initializer=single_threaded) as ex:
        return pd.concat(list(ex.map(match_decisions, match_ids)), ignore_index=True)


__all__ = ["NX", "all_decisions", "carrier_pressure", "match_decisions", "pressure_index"]
