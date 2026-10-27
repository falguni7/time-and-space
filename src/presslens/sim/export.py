"""Turn a simulated match into provider-style tables (public) and ground-truth tables (hidden).

Public: tracking, events (with `provider_xg` on shots), metadata.
Hidden (truth/): true xG and pass-completion per event, every decision with all options,
club tactics and players' simulation parameters.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pyarrow as pa

from . import provider
from .config import Club, Player, player_params
from .formations import FORMATIONS
from .match import DEAD, DT, FPS, MatchResult
from .pitch import LENGTH, WIDTH

GENERATOR_VERSION = "0.3.0"
PRESSURE_RADIUS_M = 3.0
PRESSURE_MIN_S = 0.5
CARRY_MIN_M = 3.0

# Raw event fields that are ground truth and never appear in the public events.
TRUTH_FIELDS = ("xg", "p_goal", "p_complete")

PUBLIC_EVENT_COLUMNS = [
    "event_id", "frame", "end_frame", "period", "clock_s", "minute", "type", "team", "player_id",
    "x", "y", "end_x", "end_y", "outcome", "receiver_id", "intended_id", "other_player_id",
    "lofted", "through", "header", "restart", "provider_xg", "min_distance_m", "score",
]
PASS_OUTCOMES = ("complete", "intercepted", "lost_duel", "misplaced", "out", "overhit")


# ---------------------------------------------------------------------------------------- helpers
def _dictionary(codes: np.ndarray, values: list[str | None]) -> pa.DictionaryArray:
    """Dictionary-encoded string column; code -1 becomes null."""
    codes = np.asarray(codes, dtype=np.int32)
    mask = codes < 0
    indices = pa.array(np.where(mask, 0, codes), type=pa.int32(), mask=mask)
    return pa.DictionaryArray.from_arrays(indices, pa.array(values, type=pa.string()))


def _velocity(xy: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Central-difference velocity along axis 0, zero across period boundaries."""
    v = np.gradient(xy, DT, axis=0)
    edges = np.where(np.diff(period) != 0)[0]
    for e in edges:
        v[e] = 0.0
        v[e + 1] = 0.0
    return v


def carrier_slots(r: MatchResult) -> np.ndarray:
    """(T,) slot of the ball carrier within the team in possession, -1 if nobody holds the ball."""
    slots = np.full(len(r.period), -1, dtype=np.int16)
    held = np.where(r.carrier >= 0)[0]
    for t in held:
        team = r.possession[t]
        match = np.where(r.ids[t, team] == r.carrier[t])[0]
        if len(match):
            slots[t] = match[0]
    return slots


def _attack_xy(r: MatchResult, x: float, y: float, team: int, period: int) -> tuple[float, float]:
    d = r.directions[period - 1][team]
    return x * d, y * d


# ---------------------------------------------------------------------------------------- tracking
def build_tracking(r: MatchResult) -> pa.Table:
    T = len(r.period)
    n_obj = 23
    codes = np.concatenate([r.ids.reshape(T, 22), np.full((T, 1), len(r.player_ids))], axis=1)
    team_idx = np.concatenate([np.zeros((T, 11)), np.ones((T, 11)), np.full((T, 1), -1)], axis=1)

    pos = np.concatenate([r.pos.reshape(T, 22, 2), r.ball[:, None, :2]], axis=1)       # (T, 23, 2)
    z = np.concatenate([np.zeros((T, 22)), r.ball[:, None, 2]], axis=1)
    vel = _velocity(pos, r.period)
    speed = np.linalg.norm(vel, axis=2)

    carrier_code = np.where(r.carrier >= 0, r.carrier, -2)
    is_carrier = codes == carrier_code[:, None]
    possession = np.where(r.ball_state == DEAD, -1, r.possession)

    rep = lambda a: np.repeat(a, n_obj)
    return pa.table({
        "frame": pa.array(rep(np.arange(T, dtype=np.int32))),
        "period": pa.array(rep(r.period.astype(np.int8))),
        "clock_s": pa.array(rep(r.clock.astype(np.float32))),
        "object_id": _dictionary(codes.ravel(), [*r.player_ids, "ball"]),
        "team": _dictionary(team_idx.ravel().astype(np.int32), [r.home, r.away]),
        "x": pa.array(pos[..., 0].ravel().astype(np.float32)),
        "y": pa.array(pos[..., 1].ravel().astype(np.float32)),
        "z": pa.array(z.ravel().astype(np.float32)),
        "vx": pa.array(vel[..., 0].ravel().astype(np.float32)),
        "vy": pa.array(vel[..., 1].ravel().astype(np.float32)),
        "speed": pa.array(speed.ravel().astype(np.float32)),
        "ball_state": _dictionary(rep((r.ball_state == DEAD).astype(np.int32)), ["in_play", "dead"]),
        "possession_team": _dictionary(rep(possession.astype(np.int32)), [r.home, r.away]),
        "is_carrier": pa.array(is_carrier.ravel()),
    })


# ---------------------------------------------------------------------------------------- events
def _arrival_frame(r: MatchResult, frame: int) -> int:
    """First frame after `frame` at which the ball is no longer in flight from that kick."""
    t = frame + 1
    while t < len(r.ball_state) and r.ball_state[t] == 1:  # FLIGHT
        t += 1
    return min(t, len(r.ball_state) - 1)


def _derived_events(r: MatchResult, slots: np.ndarray) -> list[dict]:
    """Possession spells, carries and pressure episodes, worked out from tracking like a provider."""
    out: list[dict] = []
    T = len(r.period)
    t = 0
    min_frames = int(PRESSURE_MIN_S * FPS)
    while t < T:
        if slots[t] < 0:
            t += 1
            continue
        team, code, start = int(r.possession[t]), int(r.carrier[t]), t
        while t + 1 < T and r.carrier[t + 1] == code and r.period[t + 1] == r.period[start]:
            t += 1
        end = t
        pid = r.player_ids[code]
        club = (r.home, r.away)[team]
        (x0, y0), (x1, y1) = r.ball[start, :2], r.ball[end, :2]
        base = {"period": int(r.period[start]), "team": club, "player_id": pid,
                "x": float(x0), "y": float(y0), "end_x": float(x1), "end_y": float(y1)}
        out.append({"frame": start, "end_frame": end, "type": "possession", **base})
        if np.hypot(x1 - x0, y1 - y0) >= CARRY_MIN_M:
            out.append({"frame": start, "end_frame": end, "type": "carry", **base})
        # Pressure on this carrier from each opponent within PRESSURE_RADIUS_M.
        sl = slice(start, end + 1)
        carrier_xy = r.pos[sl, team, slots[start]]
        d = np.linalg.norm(r.pos[sl, 1 - team] - carrier_xy[:, None, :], axis=2)       # (n, 11)
        for j in range(11):
            close = d[:, j] <= PRESSURE_RADIUS_M
            k = 0
            while k < len(close):
                if not close[k]:
                    k += 1
                    continue
                k0 = k
                while k + 1 < len(close) and close[k + 1]:
                    k += 1
                if k - k0 + 1 >= min_frames:
                    f0 = start + k0
                    out.append({
                        "frame": f0, "end_frame": start + k, "type": "pressure", "period": int(r.period[f0]),
                        "team": (r.home, r.away)[1 - team], "player_id": r.player_ids[int(r.ids[f0, 1 - team, j])],
                        "other_player_id": pid, "x": float(r.ball[f0, 0]), "y": float(r.ball[f0, 1]),
                        "min_distance_m": float(d[k0:k + 1, j].min()),
                    })
                k += 1
        t += 1
    return out


def build_events(r: MatchResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public events (with provider xG) and the matching truth rows, both keyed by event_id."""
    coef = provider.load_coefficients()
    rows, truth = [], []
    for e in r.events:
        row = {
            "frame": e["frame"], "end_frame": e["frame"], "period": e["period"], "type": e["type"],
            "team": e["team"], "player_id": e["player_id"], "x": e.get("x"), "y": e.get("y"),
            "end_x": e.get("end_x"), "end_y": e.get("end_y"), "outcome": e.get("outcome"),
            "receiver_id": e.get("receiver_id"), "intended_id": e.get("intended_id"),
            "other_player_id": e.get("tackled_id") or e.get("fouled_id") or e.get("pressured_by") or e.get("off_id"),
            "lofted": e.get("lofted"), "through": e.get("through"), "header": e.get("header"),
            "restart": e.get("restart"),
            "score": "-".join(map(str, e["score"])) if "score" in e else None,
        }
        if e["type"] == "save":
            row["outcome"] = "held" if e.get("held") else "parried"
        if e["type"] == "foul" and e.get("penalty"):
            row["outcome"] = "penalty"
        if e["type"] == "clearance" and e.get("challenge"):
            row["outcome"] = "block"
        if e["type"] in ("pass", "shot", "clearance"):
            row["end_frame"] = _arrival_frame(r, e["frame"])
        if e["type"] == "shot":
            team = 0 if e["team"] == r.home else 1
            ax, ay = _attack_xy(r, e["x"], e["y"], team, e["period"])
            row["provider_xg"] = provider.provider_xg(ax, ay, bool(e.get("header")), e.get("restart"), coef)
        rows.append(row)
        truth.append({k: e.get(k) for k in TRUTH_FIELDS} | {"type": e["type"]})
    events = pd.DataFrame(rows)
    events["_truth"] = range(len(events))
    derived = pd.DataFrame(_derived_events(r, carrier_slots(r)))
    events = pd.concat([events, derived], ignore_index=True)
    order = {"possession": 0, "carry": 1, "pressure": 2}
    events["_order"] = events["type"].map(order).fillna(3)
    events = events.sort_values(["frame", "_order"], kind="stable").reset_index(drop=True)
    last = len(r.period) - 1  # period_end is stamped one frame after the last recorded frame
    events["frame"] = events["frame"].clip(upper=last).astype(np.int32)
    events["end_frame"] = events["end_frame"].clip(upper=last).astype(np.int32)
    events["event_id"] = np.arange(len(events), dtype=np.int32)
    events["clock_s"] = r.clock[events["frame"].to_numpy()].astype(np.float32)
    events["minute"] = ((events["period"] - 1) * 45 + events["clock_s"] / 60.0).astype(np.float32)

    truth_df = pd.DataFrame(truth)
    linked = events.dropna(subset=["_truth"])
    truth_df = truth_df.iloc[linked["_truth"].astype(int).to_numpy()].reset_index(drop=True)
    truth_df.insert(0, "event_id", linked["event_id"].to_numpy())
    truth_df = truth_df[truth_df["type"].isin(["pass", "shot"])].rename(
        columns={"xg": "true_xg", "p_goal": "true_p_goal", "p_complete": "true_p_complete"})
    for col in PUBLIC_EVENT_COLUMNS:
        if col not in events:
            events[col] = None
    return events[PUBLIC_EVENT_COLUMNS], truth_df.reset_index(drop=True)


# ---------------------------------------------------------------------------------------- truth
def build_decisions(r: MatchResult) -> pd.DataFrame:
    """One row per option of every carrier decision (coordinates in the attacking team's frame)."""
    rows = []
    for k, d in enumerate(r.decisions):
        for i, o in enumerate(d["options"]):
            rows.append({
                "decision_id": k, "frame": d["frame"], "period": d["period"], "team": d["team"],
                "player_id": d["player_id"], "ball_x": d["x"], "ball_y": d["y"], "pressure": d["pressure"],
                "composure": d["composure"], "temperature": d["temperature"], "restart": d["restart"],
                "header": d["header"], "option": i, "kind": o["kind"], "receiver_id": o["receiver"],
                "target_x": o["tx"], "target_y": o["ty"], "lofted": o["lofted"], "through": o["through"],
                "true_p_success": o["p_success"], "true_value": o["value"], "chosen": i == d["chosen"],
            })
    return pd.DataFrame(rows)


def build_settings(r: MatchResult, clubs: dict[str, Club], players: dict[str, Player]) -> dict:
    out = {}
    for club_id in (r.home, r.away):
        club = clubs[club_id]
        out[club_id] = {
            "tactics": dataclasses.asdict(club.tactics),
            "players": {p.id: dataclasses.asdict(player_params(p, club.tactics))
                        for p in players.values() if p.club_id == club_id},
        }
    return out


# ---------------------------------------------------------------------------------------- meta
def build_meta(r: MatchResult, fixture, clubs: dict[str, Club], players: dict[str, Player]) -> dict:
    period_rows, start = [], 0
    for k, n in enumerate(r.period_frames):
        period_rows.append({"period": k + 1, "start_frame": start, "end_frame": start + n - 1,
                            "directions": {r.home: r.directions[k][0], r.away: r.directions[k][1]}})
        start += n

    def lineup(team: int, club_id: str) -> dict:
        formation = clubs[club_id].tactics.formation
        xi = [{"slot": s, "position": FORMATIONS[formation][s][0], "player_id": r.player_ids[int(r.ids[0, team, s])],
               "name": players[r.player_ids[int(r.ids[0, team, s])]].name,
               "number": players[r.player_ids[int(r.ids[0, team, s])]].number} for s in range(11)]
        starters = {p["player_id"] for p in xi}
        bench = [{"player_id": p.id, "name": p.name, "number": p.number, "position": p.position}
                 for p in players.values() if p.club_id == club_id and p.id not in starters]
        return {"formation": formation, "starting_xi": xi, "bench": bench}

    subs = [{"frame": e["frame"], "minute": round(e["minute"], 1), "team": e["team"],
             "on": e["player_id"], "off": e.get("off_id")} for e in r.events if e["type"] == "substitution"]
    return {
        "match_id": fixture.match_id, "round": fixture.round, "seed": r.seed,
        "generator_version": GENERATOR_VERSION, "fps": FPS,
        "pitch": {"length_m": LENGTH, "width_m": WIDTH, "origin": "centre spot"},
        "home": {"id": r.home, "name": clubs[r.home].name, "colors": clubs[r.home].colors, **lineup(0, r.home)},
        "away": {"id": r.away, "name": clubs[r.away].name, "colors": clubs[r.away].colors, **lineup(1, r.away)},
        "score": {r.home: r.score[0], r.away: r.score[1]},
        "periods": period_rows,
        "substitutions": subs,
        "total_frames": len(r.period),
    }
