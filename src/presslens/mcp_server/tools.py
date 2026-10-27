"""The agents' read-only view of the engine: plain functions returning small JSON-ready dicts.

server.py exposes each one as an MCP tool. Everything here comes from public provider data and
engine outputs (metrics cube, decisions, moment cards) plus the public roster: names, numbers and
positions only. Club tactics and player ratings are the simulation's hidden truth and are never
returned, so an agent cannot explain a moment with knowledge a real analyst would not have.

Every number is rounded, every aggregate carries its sample size n (and a 95% CI for averages),
and bad arguments raise ValueError with the allowed choices so the agent can correct itself.
"""

from __future__ import annotations

import json
import math
import threading
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml

from ..engine import cube, data, moments
from ..engine.decisions import (
    FREE_OPTION_M,
    _models,
    carrier_pressure,
    evaluate_options,
    pressure_index,
    zone_labels,
)
from ..engine.features import option_target
from ..paths import CONFIG_DIR

MAX_ROWS = 40
MOMENT_TYPES = sorted(moments.BASE_IMPORTANCE)
PLAYER_METRICS = ["xt_generated", "passes", "pass_completion_pct", "decision_gap", "chose_best_pct",
                  "pressure_faced", "carries", "shots", "provider_xg", "pressures_applied"]
TEAM_METRICS = ["shots", "provider_xg", "xt_generated", "passes", "pass_completion_pct", "pressures_applied",
                "pressure_faced", "decision_gap"]
RANK_MIN_PASSES = 150   # players need this many passes to be ranked against the league

_lock = threading.Lock()


# --------------------------------------------------------------------------- shared readers


def _r(x, nd: int = 3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _records(df: pd.DataFrame, nd: int = 3) -> list[dict]:
    return [{k: _r(v, nd) for k, v in row.items()} for row in df.to_dict("records")]


@lru_cache(maxsize=1)
def _roster() -> dict[str, dict]:
    players = yaml.safe_load((CONFIG_DIR / "players.yaml").read_text())["players"]
    return {p["id"]: {"player_id": p["id"], "name": p["name"], "club": p["club_id"], "number": p["number"],
                      "position": p["position"]} for p in players}


@lru_cache(maxsize=1)
def _clubs() -> dict[str, str]:
    clubs = yaml.safe_load((CONFIG_DIR / "clubs.yaml").read_text())["clubs"]
    return {c["id"]: c["name"] for c in clubs}


def _player(pid: str | None) -> dict | None:
    if pid is None:
        return None
    p = _roster().get(pid)
    return dict(p) if p else {"player_id": pid, "name": pid}


def _name(pid: str | None) -> str | None:
    return (_player(pid) or {}).get("name")


@lru_cache(maxsize=1)
def _index() -> pd.DataFrame:
    return pd.read_parquet(data._public(data.MATCHES / "index.parquet")).set_index("match_id")


@lru_cache(maxsize=1)
def _public_con():
    return data.connect()


def _events(sql: str, params: list) -> pd.DataFrame:
    with _lock:
        return _public_con().execute(sql, params).df()


@lru_cache(maxsize=1)
def _moments() -> pd.DataFrame:
    return pd.read_parquet(moments.MOMENTS_PATH)


@lru_cache(maxsize=1)
def _decisions() -> pd.DataFrame:
    return pd.read_parquet(data.MODELS / "decisions.parquet")


@lru_cache(maxsize=2)
def _arrays(match_id: str) -> data.MatchArrays:
    return data.load_arrays(match_id)


def _check_match(match_id: str) -> pd.Series:
    if match_id not in _index().index:
        raise ValueError(f"unknown match_id {match_id!r}; call list_matches for the ids")
    return _index().loc[match_id]


def _player_id(player: str) -> str:
    """A player id, or a player's exact name (case-insensitive), as an id."""
    if player in _roster():
        return player
    by_name = {p["name"].lower(): pid for pid, p in _roster().items()}
    if player.strip().lower() in by_name:
        return by_name[player.strip().lower()]
    raise ValueError(f"unknown player {player!r}; use an id like 'RIV-09' (club code, shirt slot) or a full name")


def _check_player(player_id: str) -> dict:
    return _player(_player_id(player_id))


def _metric(metric: str, group_by=None, filters=None, min_n: int = 1) -> pd.DataFrame:
    with _lock:
        con = cube.connect_cube()
        try:
            return cube.query_metrics(metric, group_by, filters, con=con, min_n=min_n)
        finally:
            con.close()


def clock(minute: float, period: int) -> str:
    """Broadcast clock: 29.9 -> "30'", 46.1 in the first half -> "45+2'", 92.4 -> "90+3'"."""
    end = 45 if period == 1 else 90
    m = max(1, math.ceil(minute))
    return f"{end}+{m - end}'" if m > end else f"{m}'"


def _score_at(match_id: str, frame: int) -> str:
    goals = _events("select frame, score from events where match_id = ? and type = 'goal' and frame <= ? "
                    "order by frame", [match_id, frame])
    return goals.score.iloc[-1] if len(goals) else "0-0"


# --------------------------------------------------------------------------- catalogue


def list_metrics() -> dict:
    """Metric catalogue: which metrics exist, how they aggregate, how they can be sliced, and the
    values each categorical dimension takes."""
    values = {
        "team": sorted(_clubs()), "third": ["defensive", "middle", "attacking"],
        "lane": ["left_wing", "left_half_space", "centre", "right_half_space", "right_wing"],
        "phase": ["build_up", "create", "finish", "counter_attack", "transition", "set_piece"],
        "opponent_phase": ["high_press", "mid_block", "low_block"], "pressure_band": ["none", "some", "high"],
        "game_state": ["winning", "drawing", "losing"],
        "time_bucket": ["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90+"],
        "restart": ["kickoff", "throw_in", "free_kick", "corner", "goal_kick", "gk"], "period": [1, 2],
        "action": ["pass", "carry"],
    }
    tables: dict[str, dict] = {}
    for m, d in cube.describe().items():
        t = tables.setdefault(d["table"], {"dimensions": d["dimensions"], "metrics": {}})
        t["metrics"][m] = d["kind"]
    return {
        "tables": tables,
        "dimension_values": values,
        "notes": ("Locations are from the perspective of the team in possession (attacking left to right). "
                  "pressure is 0 (nearest defender 5 m+) to 1 (1 m or closer). decision_gap is the expected-"
                  "value difference between the best available pass and the chosen one (0 = best choice). "
                  "xT values are probabilities of scoring in the next few actions. 'sum' metrics are season "
                  "or match totals; query_metrics adds per_match (value / matches) for them: compare teams on that."),
    }


def query_metrics(metric: str, group_by: list[str] | None = None,
                  filters: dict[str, str | int] | None = None,
                  min_n: int = 20) -> dict:
    """Aggregate one metric from the metrics cube, grouped by up to 3 dimensions, with equality
    filters (e.g. {"team": "RIV", "third": "attacking"}). Groups with fewer than min_n actions are
    dropped. Returns value, n (actions), matches and, for averages, a 95% CI; totals ('sum'/'count')
    also get per_match. Rows are sorted best first (by per_match for totals)."""
    # Common slips: "pressure" for the band column, "player" for player_id.
    alias = {"pressure": "pressure_band", "player": "player_id"}
    group_by = [alias.get(g, g) for g in (group_by or [])][:3]
    filters = {alias.get(k, k): (int(v) if k == "period" else v) for k, v in (filters or {}).items()}
    if "player_id" in filters:
        filters["player_id"] = _player_id(filters["player_id"])
    codes = {name.lower(): code for code, name in _clubs().items()}
    for k in ("team", "opponent"):
        if k in filters:
            filters[k] = codes.get(str(filters[k]).lower(), filters[k])
    df = _metric(metric, group_by, filters, min_n=min_n)
    table, _, kind = cube.METRICS[metric]
    if kind in ("sum", "count"):
        df["per_match"] = df["value"] / df["matches"].clip(lower=1)
        df = df.sort_values("per_match", ascending=False) if group_by else df
    if "team" in df:
        df.insert(df.columns.get_loc("team") + 1, "team_name", df["team"].map(_clubs()))
    if "player_id" in df:
        df.insert(df.columns.get_loc("player_id") + 1, "player", df["player_id"].map(_name))
    rows = _records(df.drop(columns="metric"), 4)
    return {"metric": metric, "kind": kind, "table": table, "group_by": group_by, "filters": filters,
            "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS}


# --------------------------------------------------------------------------- matches and moments


def list_matches(team: str | None = None) -> dict:
    """The season's matches (round, teams, score), optionally only those of one club code."""
    idx = _index().reset_index()
    if team:
        if team not in _clubs():
            raise ValueError(f"unknown team {team!r}; choose from {sorted(_clubs())}")
        idx = idx[(idx.home == team) | (idx.away == team)]
    return {"clubs": _clubs(), "matches": [
        {"match_id": r.match_id, "round": int(r["round"]), "home": r.home, "away": r.away,
         "score": f"{r.home_goals}-{r.away_goals}"} for _, r in idx.iterrows()]}


def match_summary(match_id: str) -> dict:
    """Score, goals (minute, scorer), per-team totals (shots, xG, xT, passing, pressing) and the
    match's top threat creators."""
    row = _check_match(match_id)
    goals = _events("""
        select g.minute, g.period, g.team, g.player_id, g.score, s.restart, s.header, s.provider_xg
        from events g left join events s on s.match_id = g.match_id and s.type = 'shot' and s.outcome = 'goal'
             and s.frame between g.frame - 50 and g.frame
        where g.match_id = ? and g.type = 'goal' order by g.frame""", [match_id])
    teams = {}
    for metric in TEAM_METRICS:
        for r in _metric(metric, ["team"], {"match_id": match_id}).itertuples():
            teams.setdefault(r.team, {"club": _clubs()[r.team]})[metric] = _r(r.value, 3)
    top = _metric("xt_generated", ["player_id"], {"match_id": match_id}).head(3)
    return {
        "match_id": match_id, "round": int(row["round"]),
        "home": {"club_id": row.home, "name": _clubs()[row.home]},
        "away": {"club_id": row.away, "name": _clubs()[row.away]},
        "score": f"{row.home_goals}-{row.away_goals}",
        "goals": [{"clock": clock(g.minute, g.period), "team": g.team, "scorer": _name(g.player_id),
                   "player_id": g.player_id, "score_after": g.score,
                   "type": ("penalty" if g.restart == "penalty" else "free_kick" if g.restart == "free_kick"
                            else "header" if g.header else "open_play"),
                   "provider_xg": _r(g.provider_xg)} for g in goals.itertuples()],
        "teams": teams,
        "top_threat_creators": [{"player_id": r.player_id, "name": _name(r.player_id),
                                 "xt_generated": _r(r.value, 3), "actions": int(r.n)} for r in top.itertuples()],
    }


def list_moments(match_id: str, types: list[str] | None = None, min_importance: float = 0.0,
                 limit: int = 30) -> dict:
    """The match's moment cards in time order (id, frame, clock, type, team, player, importance).
    Pass a row's frame to pass_options / pressure_at to see that situation.
    types filters by kind: goal, big_chance, momentum_swing, press_trap, press_broken, missed_option."""
    _check_match(match_id)
    bad = [t for t in types or [] if t not in MOMENT_TYPES]
    if bad:
        raise ValueError(f"unknown moment types {bad}; choose from {MOMENT_TYPES}")
    m = _moments()
    m = m[(m.match_id == match_id) & (m.importance >= min_importance)]
    if types:
        m = m[m.type.isin(types)]
    m = m.sort_values("frame")
    rows = []
    for r in m.head(max(1, min(int(limit), 60))).itertuples():
        row = {"moment_id": r.moment_id, "frame": int(r.frame), "clock": clock(r.minute, r.period),
               "type": r.type, "team": r.team, "opponent": r.opponent}
        if r.player_id:
            row["player"] = _name(r.player_id)
        pressed = json.loads(r.card).get("pressed_player_id")
        if pressed:
            # Press traps are credited to the pressing team; the pressed player belongs to the opponent.
            row["pressed_player"] = f"{_name(pressed)} ({r.opponent})"
        rows.append({**row, "importance": _r(r.importance, 2)})
    return {"match_id": match_id, "count": len(m), "moments": rows}


def get_moment(moment_id: str) -> dict:
    """One moment card in full: its metrics and location, every player resolved to a name, the
    score at that moment and the club names."""
    m = _moments()
    hit = m[m.moment_id == moment_id]
    if hit.empty:
        raise ValueError(f"unknown moment_id {moment_id!r}; call list_moments for the ids")
    card = json.loads(hit.card.iloc[0])
    for key in [k for k in list(card) if k.endswith(("player_id", "receiver_id"))]:
        card[key.replace("_id", "")] = _player(card[key])
    card["team_name"], card["opponent_name"] = _clubs()[card["team"]], _clubs()[card["opponent"]]
    row = _index().loc[card["match_id"]]
    home_goals, away_goals = _score_at(card["match_id"], card["frame"] - 1).split("-")
    card["clock"] = clock(card["minute"], card["period"])
    card["score_before"] = f"{row.home} {home_goals}-{away_goals} {row.away}"
    return card


# --------------------------------------------------------------------------- players


def player_profile(player_id: str, match_id: str | None = None) -> dict:
    """Public bio plus season (or one match's) metrics, each with n, and the player's league rank
    for threat created among players with enough passes. player_id may also be the full name."""
    player_id = _player_id(player_id)
    profile = _check_player(player_id)
    filters = {"player_id": player_id}
    if match_id:
        _check_match(match_id)
        filters["match_id"] = match_id
    stats = {}
    for metric in PLAYER_METRICS:
        df = _metric(metric, None, filters)
        if len(df) and df.n.iloc[0] > 0:
            r = df.iloc[0]
            stats[metric] = {"value": _r(r.value, 4), "n": int(r.n),
                             **({"ci_low": _r(r.ci_low, 4), "ci_high": _r(r.ci_high, 4)} if "ci_low" in df else {})}
    goals = _events("select count(*) as g from events where type = 'goal' and player_id = ?"
                    + (" and match_id = ?" if match_id else ""), [player_id] + ([match_id] if match_id else []))
    stats["goals"] = int(goals.g.iloc[0])
    out = {**profile, "club_name": _clubs()[profile["club"]], "scope": match_id or "season", "metrics": stats}
    if not match_id:
        passes = _metric("passes", ["player_id"])
        eligible = set(passes[passes.value >= RANK_MIN_PASSES].player_id)
        xt = _metric("xt_generated", ["player_id"])
        xt = xt[xt.player_id.isin(eligible)].sort_values("value", ascending=False).reset_index(drop=True)
        hit = xt.index[xt.player_id == player_id]
        out["league_rank_xt_generated"] = ({"rank": int(hit[0]) + 1, "of": len(xt)} if len(hit) else None)
        out["matches"] = int(_metric("passes", ["match_id"], {"player_id": player_id}).shape[0])
    return out


# --------------------------------------------------------------------------- frame level


def pressure_at(match_id: str, frame: int) -> dict:
    """Pressure on the ball carrier at one tracking frame (10 per second): nearest defender, distance,
    closing speed, pressure index (0-1), defenders within 5 m, and the carrier's pitch zone."""
    _check_match(match_id)
    m = _arrays(match_id)
    frame = int(frame)
    if not 0 <= frame < len(m.carrier):
        raise ValueError(f"frame must be in 0..{len(m.carrier) - 1}")
    # A pass's own frame is the release, when nobody holds the ball: use the last carrier within 1 s.
    held = [f for f in range(frame, max(-1, frame - 11), -1) if m.carrier[f] >= 0]
    if not held:
        return {"match_id": match_id, "frame": frame, "carrier": None,
                "note": "nobody controlled the ball in the second before this frame (ball in flight or dead)"}
    frame = held[0]
    c = int(m.carrier[frame])
    team = 0 if c < 11 else 1
    dist, closing = carrier_pressure(m, frame, team, c)
    opp = slice(11, 22) if team == 0 else slice(0, 11)
    d_all = np.linalg.norm(m.xy[frame, opp] - m.xy[frame, c], axis=1)
    nearest = (11 if team == 0 else 0) + int(np.argmin(d_all))
    ax, ay = data.attack_frame(m.xy[frame, c], m.direction(frame, team))
    loc = zone_labels(float(ax), float(ay))
    return {
        "match_id": match_id, "frame": frame, "period": int(m.period[frame]),
        "team": m.home if team == 0 else m.away,
        "carrier": _player(m.names[m.ids[frame, c]]),
        "nearest_defender": _player(m.names[m.ids[frame, nearest]]),
        "distance_m": _r(dist, 2), "closing_speed_ms": _r(closing, 2), "pressure": _r(pressure_index(dist), 2),
        "defenders_within_5m": int((d_all <= 5.0).sum()),
        "location": {"ax": _r(ax, 1), "ay": _r(ay, 1), "third": loc["third"], "lane": loc["lane"]},
    }


def pass_options(match_id: str, frame: int) -> dict:
    """Every passing option at the pass starting at (or within 1 s after) a frame - use the frame of a
    moment from list_moments / get_moment. Gives each option's receiver, success
    probability (xPass), expected value, nearest-marker distance and target zone, plus which one was
    chosen and which was best. This is how a missed option or a press trap is explained."""
    _check_match(match_id)
    d = _decisions()
    d = d[(d.match_id == match_id) & (d.frame >= int(frame)) & (d.frame <= int(frame) + 10)]
    if d.empty:
        raise ValueError(f"no pass starts within 1 s of frame {frame} in {match_id}")
    p = d.sort_values("frame").iloc[0]
    m = _arrays(match_id)
    f = int(p.frame)
    team = m.team_index(p.team)
    passer = m.player_index(f, p.player_id)
    xt_model, xpass_model = _models()
    opts, xpass, value, marking = evaluate_options(m, f, team, passer, bool(p.lofted), p.restart is not None,
                                                   xt_model, xpass_model)
    rows = []
    for i, k in enumerate(opts):
        pid = m.names[m.ids[f, k]]
        tx, ty = option_target(m, f, team, passer, k)
        loc = zone_labels(float(tx), float(ty))
        rows.append({"receiver": _name(pid), "player_id": pid, "position": (_player(pid) or {}).get("position"),
                     "xpass": _r(xpass[i], 3), "value": _r(value[i], 4), "marking_m": _r(marking[i], 1),
                     "free": bool(marking[i] > FREE_OPTION_M), "third": loc["third"], "lane": loc["lane"],
                     "chosen": pid == p.intended_id, "best": pid == p.best_receiver_id})
    rows.sort(key=lambda r: -r["value"])
    return {
        "match_id": match_id, "frame": f, "minute": _r(p.minute, 2), "team": p.team,
        "passer": _player(p.player_id), "outcome": p.outcome, "lofted": bool(p.lofted), "restart": p.restart,
        "pressure": _r(p.pressure, 2), "carrier_proximity_m": _r(p.carrier_proximity_m, 2),
        "free_options": int(p.free_options), "decision_gap": _r(p.decision_gap, 4),
        "xt_added": _r(p.xt_added, 4), "options": rows,
    }


TOOLS = [list_metrics, query_metrics, list_matches, match_summary, list_moments, get_moment, player_profile,
         pressure_at, pass_options]
