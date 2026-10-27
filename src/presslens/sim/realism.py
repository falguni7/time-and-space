"""Realism report for the generated season (reads data/matches, including truth/).

    bin/pl python -m presslens.sim.realism          # writes data/checks/realism_report.md

Every number is compared with typical values from professional football.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb
import numpy as np
import pandas as pd

from .. import paths
from .config import load_clubs
from .pitch import HALF_L

MATCHES = paths.DATA_DIR / "matches"
REPORT = paths.DATA_DIR / "checks" / "realism_report.md"
BOX_X, BOX_HALF_W = HALF_L - 16.5, 20.16

# Per team per match unless noted: (low, high) of typical professional values.
REAL = {
    "possession_pct": (35, 65), "passes": (350, 650), "pass_completion_pct": (72, 90),
    "shots": (7, 20), "shots_on_target": (2, 8), "goals": (0.9, 1.7), "xg": (0.9, 1.8),
    "tackles": (10, 25), "interceptions": (5, 15), "fouls": (7, 15), "corners": (2, 9),
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for name, path in (("events", "*/events.parquet"), ("truth_events", "*/truth/events.parquet"),
                       ("phases", "*/phases.parquet")):
        con.execute(f"""
            create view {name} as
            select *, regexp_extract(filename, '(R\\d+_[A-Z]+-[A-Z]+)', 1) as match_id
            from read_parquet('{MATCHES / path}', filename = true)""")
    con.execute(f"create view tracking as select *, regexp_extract(filename, '(R\\d+_[A-Z]+-[A-Z]+)', 1) "
                f"as match_id from read_parquet('{MATCHES / '*/tracking.parquet'}', filename = true)")
    con.execute(f"create view idx as select * from read_parquet('{MATCHES / 'index.parquet'}')")
    rows = []
    for meta_path in sorted(MATCHES.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        for p in meta["periods"]:
            for team, d in p["directions"].items():
                rows.append({"match_id": meta["match_id"], "period": p["period"], "team": team, "direction": d})
    con.register("dirs_df", pd.DataFrame(rows))
    con.execute("create table dirs as select * from dirs_df")
    return con


def _check(value: float, lo: float, hi: float) -> str:
    return "✓" if lo <= value <= hi else "✗"


def team_stats(con) -> pd.DataFrame:
    ev = con.execute("""
        select match_id, team,
          count(*) filter (where type = 'pass') as passes,
          100.0 * avg(case when type = 'pass' then (outcome = 'complete')::int end) as pass_completion_pct,
          count(*) filter (where type = 'shot') as shots,
          count(*) filter (where type = 'shot' and outcome in ('goal', 'saved')) as shots_on_target,
          count(*) filter (where type = 'goal') as goals,
          count(*) filter (where type = 'tackle') as tackles,
          count(*) filter (where type = 'interception') as interceptions,
          count(*) filter (where type = 'foul') as fouls,
          count(*) filter (where type = 'corner') as corners
        from events where team is not null group by 1, 2""").df()
    xg = con.execute("""
        select e.match_id, e.team, sum(t.true_xg) as xg, sum(e.provider_xg) as provider_xg
        from events e join truth_events t using (match_id, event_id)
        where e.type = 'shot' group by 1, 2""").df()
    poss = con.execute("""
        select match_id, possession_team as team,
          100.0 * count(*) / sum(count(*)) over (partition by match_id) as possession_pct
        from tracking where object_id = 'ball' and possession_team is not null group by 1, 2""").df()
    df = ev.merge(xg, on=["match_id", "team"], how="left").merge(poss, on=["match_id", "team"], how="left")
    return df.fillna({"xg": 0.0, "provider_xg": 0.0})


def physical(con) -> dict:
    players = con.execute("""
        select match_id, object_id, count(*) as frames, sum(speed) * 0.1 / 1000 as km, max(speed) as top
        from tracking where object_id <> 'ball' and team is not null group by 1, 2""").df()
    full = players[players.frames >= 80 * 60 * 10]
    in_play = con.execute("""
        select avg(n) from (select match_id, count(*) / 600.0 as n from tracking
        where object_id = 'ball' and ball_state = 'in_play' group by 1)""").fetchone()[0]
    return {"distance_km": float(full.km.mean()), "top_speed": float(np.percentile(players.top, 99)),
            "ball_in_play_min": float(in_play)}


def shots(con) -> pd.DataFrame:
    df = con.execute("""
        select e.match_id, e.team, e.x * d.direction as ax, e.y * d.direction as ay, e.outcome, e.header,
               e.restart, e.provider_xg, t.true_xg
        from events e join truth_events t using (match_id, event_id)
        join dirs d on d.match_id = e.match_id and d.period = e.period and d.team = e.team
        where e.type = 'shot'""").df()
    df["in_box"] = (df.ax > BOX_X) & (df.ay.abs() < BOX_HALF_W)
    df["goal"] = df.outcome == "goal"
    return df


def funnel(con) -> pd.DataFrame:
    """Possessions (kept through same-team restarts) and how far they get, per team per match."""
    ball = con.execute("""
        select t.match_id, t.frame, t.period, t.possession_team as team, t.x, t.y
        from tracking t where object_id = 'ball' order by match_id, frame""").df()
    dirs = con.execute("select * from dirs").df().set_index(["match_id", "period", "team"])["direction"]
    out = []
    for mid, b in ball.groupby("match_id", sort=False):
        team = b.team.ffill().to_numpy()
        valid = pd.notna(team)
        change = np.r_[True, team[1:] != team[:-1]]
        seg_id = np.cumsum(change)
        d = np.array([dirs.get((mid, p, t), 1) if isinstance(t, str) else 1 for p, t in zip(b.period, team)])
        ax, ay = b.x.to_numpy() * d, np.abs(b.y.to_numpy())
        seg = pd.DataFrame({"seg": seg_id, "team": team, "ax": ax,
                            "in_box": (ax > BOX_X) & (ay < BOX_HALF_W)})[valid]
        g = seg.groupby("seg").agg(team=("team", "first"), max_x=("ax", "max"), box=("in_box", "max"))
        g["final_third"] = g.max_x > HALF_L / 3
        for t, gt in g.groupby("team"):
            out.append({"match_id": mid, "team": t, "possessions": len(gt),
                        "final_third": int(gt.final_third.sum()), "box": int(gt.box.sum())})
    return pd.DataFrame(out)


def league_table(con) -> pd.DataFrame:
    idx = con.execute("select * from idx").df()
    clubs = load_clubs()
    rows = {c: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0} for c in clubs}
    for _, m in idx.iterrows():
        for c, gf, ga in ((m.home, m.home_goals, m.away_goals), (m.away, m.away_goals, m.home_goals)):
            r = rows[c]
            r["P"] += 1
            r["GF"] += gf
            r["GA"] += ga
            r["W"] += int(gf > ga)
            r["D"] += int(gf == ga)
            r["L"] += int(gf < ga)
            r["Pts"] += 3 * int(gf > ga) + int(gf == ga)
    tab = pd.DataFrame(rows).T
    tab["GD"] = tab.GF - tab.GA
    tab.insert(0, "Club", [clubs[c].name for c in tab.index])
    return tab.sort_values(["Pts", "GD", "GF"], ascending=False)


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report() -> str:
    con = _connect()
    n = con.execute("select count(*) from idx").fetchone()[0]
    ts = team_stats(con)
    sh = shots(con)
    ph = physical(con)
    fu = funnel(con)
    lines = ["# PressLens season realism report",
             f"_{n} matches, generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Reads `data/matches` (incl. hidden truth)._", ""]

    goals, txg, pxg = int(sh.goal.sum()), sh.true_xg.sum(), sh.provider_xg.sum()
    lines += ["## Headline", "",
              f"- Goals per match: **{goals / n:.2f}** (real ~2.65–2.8)",
              (f"- True xG per match: {txg / n:.2f}; goals ÷ true xG **{goals / txg:.2f}**; "
               f"goals ÷ provider xG {goals / pxg:.2f}; provider vs true xG correlation "
               f"{np.corrcoef(sh.provider_xg, sh.true_xg)[0, 1]:.2f}"), ""]

    rows = []
    for k, (lo, hi) in REAL.items():
        v = ts[k].mean()
        rows.append({"stat (per team per match)": k, "mean": round(float(v), 2),
                     "p10": round(float(ts[k].quantile(0.1)), 1), "p90": round(float(ts[k].quantile(0.9)), 1),
                     "real range": f"{lo}–{hi}", "": _check(v, lo, hi)})
    lines += ["## Match statistics", "", _md_table(pd.DataFrame(rows)), ""]
    lines += ["## Physical", "",
              (f"- Distance per full-match player: **{ph['distance_km']:.1f} km** (real 9–12) "
               f"{_check(ph['distance_km'], 9, 12)}"),
              (f"- Top speed (99th percentile of players' maxima): **{ph['top_speed']:.1f} m/s** (real ≤ ~10) "
               f"{_check(ph['top_speed'], 7, 10)}"),
              (f"- Ball in play: **{ph['ball_in_play_min']:.1f} min** (real 55–60) "
               f"{_check(ph['ball_in_play_min'], 55, 60)}"), ""]

    f = fu.groupby("team")[["possessions", "final_third", "box"]].mean().mean()
    shot_poss = len(sh) / (2 * n)
    lines += ["## Attack funnel (per team per match)", "",
              "| possessions | reach final third | reach box | shots |", "|---|---|---|---|",
              (f"| {f.possessions:.0f} (real 100–120) | {f.final_third:.0f} (real 45–55) | "
               f"{f.box:.0f} (real 20–25) | {shot_poss:.1f} (real 12–13) |"), ""]

    inbox = sh[sh.in_box]
    out = sh[~sh.in_box & (sh.restart != "penalty")]
    lines += ["## Shot locations", "",
              (f"- Inside the box: **{100 * len(inbox) / len(sh):.0f}%** of shots (real ~60–65%), "
               f"xG/shot {inbox.true_xg.mean():.3f} (real ~0.15), **{100 * inbox.goal.sum() / sh.goal.sum():.0f}%** "
               f"of goals (real ~80–85%)"),
              f"- Outside the box: xG/shot {out.true_xg.mean():.3f} (real ~0.03–0.04)",
              f"- Blocked: {100 * (sh.outcome == 'blocked').mean():.0f}% of shots (real ~25–30%)", ""]

    cal = []
    for name, sel in (("open play", sh.restart.isna() | ~sh.restart.isin(["penalty", "free_kick"])),
                      ("free kick", sh.restart == "free_kick"), ("penalty", sh.restart == "penalty"),
                      ("header", sh.header.fillna(False).astype(bool)), ("all", sh.index == sh.index)):
        s = sh[sel]
        if len(s):
            cal.append({"shots": name, "n": len(s), "true xG": round(s.true_xg.sum(), 1), "goals": int(s.goal.sum()),
                        "goals ÷ xG": round(s.goal.sum() / max(s.true_xg.sum(), 1e-9), 2)})
    lines += ["## xG calibration (goals ÷ true xG should be close to 1)", "", _md_table(pd.DataFrame(cal)), ""]

    box = con.execute(f"""
        select count(*) filter (where e.type = 'pass' and e.lofted and e.end_x * d.direction > {BOX_X}
                                and abs(e.end_y) < {BOX_HALF_W} and e.restart is null) as crosses,
               count(*) filter (where e.type = 'pass' and e.lofted and e.end_x * d.direction > {BOX_X}
                                and abs(e.end_y) < {BOX_HALF_W} and e.restart is null and e.outcome = 'complete') as crosses_ok,
               count(*) filter (where e.type = 'clearance') as clearances,
               count(*) filter (where e.type = 'clearance' and e.header) as headed,
               count(*) filter (where e.type = 'save') as saves,
               count(*) filter (where e.type = 'foul' and e.outcome = 'penalty') as penalties
        from events e left join dirs d on d.match_id = e.match_id and d.period = e.period and d.team = e.team
        """).df().iloc[0]
    fk = sh[sh.restart == "free_kick"]
    lines += ["## Defending and set pieces", "",
              (f"- Crosses into the box: {box.crosses / (2 * n):.1f} per team, "
               f"{100 * box.crosses_ok / max(box.crosses, 1):.0f}% completed (real ~25%)"),
              f"- Clearances: {box.clearances / (2 * n):.1f} per team ({box.headed / (2 * n):.1f} headed; real 15–25)",
              f"- Keeper saves: {box.saves / (2 * n):.1f} per team (real 3–4)",
              f"- Penalties: {box.penalties / n:.2f} per match (real 0.25–0.3)",
              f"- Direct free-kick shots: {len(fk) / n:.1f} per match, xG/shot {fk.true_xg.mean():.3f} (real ~0.05–0.07)",
              ""]

    phases = con.execute("""
        select in_possession, phase, sum(duration_s) as s from phases group by 1, 2""").df()
    for flag, title in ((True, "in possession"), (False, "out of possession")):
        p = phases[phases.in_possession == flag]
        share = (100 * p.set_index("phase").s / p.s.sum()).round(0).astype(int).to_dict()
        lines.append(f"- Phase time {title}: " + ", ".join(f"{k} {v}%" for k, v in sorted(share.items())))
    lines += ["", "## League table", "", _md_table(league_table(con).reset_index(drop=True), "{:.0f}"), ""]
    return "\n".join(lines)


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    text = build_report()
    REPORT.write_text(text + "\n")
    print(text)
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()
