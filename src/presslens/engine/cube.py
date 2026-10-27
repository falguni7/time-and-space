"""The metrics cube: every xT and pressure metric, sliceable by zone, phase, time, team, player, ...

Stored as a DuckDB file (data/engine/metrics.duckdb) with tables passes, carries, pressures, shots
and a view actions (passes + carries). Locations (third / lane / zone) are always from the
perspective of the team in possession.

`query_metrics` is the single safe entry point (used by the Analyst agent via MCP): metric and
dimension names come from fixed lists, filter values are bound parameters, and every value comes
with its sample size and a 95% confidence interval.
"""

from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd

from . import data
from .decisions import zone_labels
from .xt import XTModel

CUBE_PATH = data.MODELS / "metrics.duckdb"

# metric -> (table, SQL aggregate, kind). kind "mean" metrics get a confidence interval.
METRICS: dict[str, tuple[str, str, str]] = {
    "xt_generated": ("actions", "sum(greatest(xt_added, 0))", "sum"),
    "xt_net": ("actions", "sum(xt_added)", "sum"),
    "xt_lost_on_turnovers": ("passes", "sum(case when outcome <> 'complete' then xt_start else 0 end)", "sum"),
    "potential_xt": ("passes", "sum(best_value)", "sum"),
    "passes": ("passes", "count(*)", "count"),
    "pass_completion_pct": ("passes", "100 * avg((outcome = 'complete')::int)", "mean"),
    "decision_gap": ("passes", "avg(decision_gap)", "mean"),
    "chose_best_pct": ("passes", "100 * avg(chose_best::int)", "mean"),
    "pressure_faced": ("passes", "avg(pressure)", "mean"),
    "carrier_proximity_m": ("passes", "avg(carrier_proximity_m)", "mean"),
    "free_options": ("passes", "avg(free_options)", "mean"),
    "mean_option_proximity_m": ("passes", "avg(mean_option_proximity_m)", "mean"),
    "xt_added_per_pass": ("passes", "avg(xt_added)", "mean"),
    "carries": ("carries", "count(*)", "count"),
    "shots": ("shots", "count(*)", "count"),
    "provider_xg": ("shots", "sum(provider_xg)", "sum"),
    "pressures_applied": ("pressures", "count(*)", "count"),
    "pressure_duration_s": ("pressures", "avg(duration_s)", "mean"),
    "pressure_min_distance_m": ("pressures", "avg(min_distance_m)", "mean"),
}
# The column a "mean" metric averages (for its confidence interval).
MEAN_COLUMN = {
    "pass_completion_pct": "100 * (outcome = 'complete')::int", "decision_gap": "decision_gap",
    "chose_best_pct": "100 * chose_best::int", "pressure_faced": "pressure",
    "carrier_proximity_m": "carrier_proximity_m", "free_options": "free_options",
    "mean_option_proximity_m": "mean_option_proximity_m", "xt_added_per_pass": "xt_added",
    "pressure_duration_s": "duration_s", "pressure_min_distance_m": "min_distance_m",
}
DIMENSIONS = {
    "passes": {"match_id", "team", "opponent", "player_id", "phase", "opponent_phase", "third", "lane",
               "zone", "period", "time_bucket", "game_state", "pressure_band", "restart"},
    "carries": {"match_id", "team", "opponent", "player_id", "third", "lane", "zone", "period", "time_bucket",
                "game_state"},
    "actions": {"match_id", "team", "opponent", "player_id", "third", "lane", "zone", "period", "time_bucket",
                "game_state", "action"},
    "shots": {"match_id", "team", "opponent", "player_id", "third", "lane", "zone", "period", "time_bucket",
              "game_state"},
    "pressures": {"match_id", "team", "opponent", "player_id", "pressed_player_id", "third", "lane", "zone",
                  "period", "time_bucket", "game_state"},
}


def _game_state(df: pd.DataFrame, goals: pd.DataFrame) -> pd.Series:
    """winning / drawing / losing for `team` at `frame` (goals strictly before the frame)."""
    out = []
    by_match = {m: g for m, g in goals.groupby("match_id")}
    for r in df[["match_id", "frame", "team"]].itertuples(index=False):
        g = by_match.get(r.match_id)
        if g is None:
            out.append("drawing")
            continue
        before = g[g.frame < r.frame]
        diff = int((before.team == r.team).sum() - (before.team != r.team).sum())
        out.append("winning" if diff > 0 else ("losing" if diff < 0 else "drawing"))
    return pd.Series(out, index=df.index)


def _time_bucket(minute: pd.Series) -> pd.Series:
    edges = [0, 15, 30, 45, 60, 75, 200]
    labels = ["0-15", "15-30", "30-45+", "45-60", "60-75", "75-90+"]
    return pd.cut(minute, edges, labels=labels, right=False).astype(str)


def _locate(df: pd.DataFrame, ax: str, ay: str) -> pd.DataFrame:
    labels = pd.DataFrame([zone_labels(x, y) for x, y in zip(df[ax], df[ay])], index=df.index)
    return pd.concat([df, labels[["zone", "third", "lane"]]], axis=1)


def build(decisions: pd.DataFrame | None = None) -> duckdb.DuckDBPyConnection:
    decisions = decisions if decisions is not None else pd.read_parquet(data.MODELS / "decisions.parquet")
    public = data.connect()
    xt_model = XTModel.load()
    goals = public.execute("select match_id, frame, team from events where type = 'goal'").df()
    opponents = public.execute("select match_id, home, away from idx").df().set_index("match_id")

    def opponent(df: pd.DataFrame) -> pd.Series:
        return pd.Series([opponents.loc[m, "away"] if t == opponents.loc[m, "home"] else opponents.loc[m, "home"]
                          for m, t in zip(df.match_id, df.team)], index=df.index)

    passes = decisions.copy()
    passes["game_state"] = _game_state(passes, goals)
    passes["time_bucket"] = _time_bucket(passes.minute)
    passes["pressure_band"] = pd.cut(passes.pressure, [-0.01, 0.0, 0.5, 1.0],
                                     labels=["none", "some", "high"]).astype(str)

    carries = public.execute("""
        select match_id, event_id, frame, end_frame, period, minute, team, player_id, ax, ay, end_ax, end_ay
        from events_a where type = 'carry'""").df()
    carries["xt_added"] = xt_model.value(carries.end_ax, carries.end_ay) - xt_model.value(carries.ax, carries.ay)
    carries = _locate(carries, "ax", "ay")
    carries["opponent"] = opponent(carries)
    carries["game_state"] = _game_state(carries, goals)
    carries["time_bucket"] = _time_bucket(carries.minute)

    shots = public.execute("""
        select match_id, event_id, frame, period, minute, team, player_id, ax, ay, outcome, header, restart,
               provider_xg
        from events_a where type = 'shot'""").df()
    shots = _locate(shots, "ax", "ay")
    shots["opponent"] = opponent(shots)
    shots["game_state"] = _game_state(shots, goals)
    shots["time_bucket"] = _time_bucket(shots.minute)

    # Pressures: located in the pressed team's attacking frame (the team in possession).
    pressures = public.execute("""
        select p.match_id, p.event_id, p.frame, p.end_frame, p.period, p.minute, p.team, p.player_id,
               p.other_player_id as pressed_player_id, p.min_distance_m,
               (p.end_frame - p.frame + 1) / 10.0 as duration_s,
               p.x * d.direction as ax, p.y * d.direction as ay
        from events p join dirs d on d.match_id = p.match_id and d.period = p.period
          and d.team = split_part(p.other_player_id, '-', 1)
        where p.type = 'pressure'""").df()
    pressures = _locate(pressures, "ax", "ay")
    pressures["opponent"] = opponent(pressures)
    pressures["game_state"] = _game_state(pressures, goals)
    pressures["time_bucket"] = _time_bucket(pressures.minute)

    if CUBE_PATH.exists():
        CUBE_PATH.unlink()
    CUBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CUBE_PATH))
    for name, df in (("passes", passes), ("carries", carries), ("shots", shots), ("pressures", pressures)):
        con.register(f"{name}_df", df)
        con.execute(f"create table {name} as select * from {name}_df")
    con.execute("""
        create view actions as
        select match_id, team, opponent, player_id, third, lane, zone, period, time_bucket, game_state,
               'pass' as action, xt_added from passes
        union all
        select match_id, team, opponent, player_id, third, lane, zone, period, time_bucket, game_state,
               'carry' as action, xt_added from carries""")
    return con


def connect_cube(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(CUBE_PATH), read_only=read_only)


def query_metrics(metric: str, group_by: list[str] | None = None, filters: dict | None = None,
                  con: duckdb.DuckDBPyConnection | None = None, min_n: int = 1) -> pd.DataFrame:
    """value, n and 95% CI of `metric`, grouped by allowed dimensions, with equality filters."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    table, agg, kind = METRICS[metric]
    group_by = list(group_by or [])
    filters = dict(filters or {})
    allowed = DIMENSIONS[table]
    bad = [d for d in [*group_by, *filters] if d not in allowed]
    if bad:
        raise ValueError(f"{metric} cannot be sliced by {bad}; allowed: {sorted(allowed)}")
    where = " and ".join(f"{k} = ?" for k in filters) or "true"
    select_dims = ", ".join(group_by)
    extra = ""
    if kind == "mean":
        extra = f", stddev_samp({MEAN_COLUMN[metric]}) as sd"
    sql = (f"select {select_dims + ', ' if group_by else ''}{agg} as value, count(*) as n, "
           f"count(distinct match_id) as matches{extra} "
           f"from {table} where {where} "
           f"{'group by ' + select_dims if group_by else ''} having count(*) >= {int(min_n)} "
           f"{'order by value desc' if group_by else ''}")
    own = con is None
    con = con or connect_cube()
    try:
        df = con.execute(sql, list(filters.values())).df()
    finally:
        if own:
            con.close()
    if kind == "mean":
        half = 1.96 * df["sd"].fillna(0.0) / np.sqrt(df["n"].clip(lower=1))
        df["ci_low"], df["ci_high"] = df["value"] - half, df["value"] + half
        df = df.drop(columns="sd")
    df.insert(0, "metric", metric)
    return df


def describe() -> dict:
    """Metric and dimension catalogue (for agents and the analyst page)."""
    return {m: {"table": t, "kind": k, "dimensions": sorted(DIMENSIONS[t])} for m, (t, _, k) in METRICS.items()}


__all__ = ["CUBE_PATH", "DIMENSIONS", "METRICS", "build", "connect_cube", "describe", "math", "query_metrics"]
