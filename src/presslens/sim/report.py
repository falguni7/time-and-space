"""Match statistics from a simulated match, compared with typical real-football ranges."""

from __future__ import annotations

import numpy as np

from .match import DEAD, DT, MatchResult

# Typical per-team values in professional football (approximate, used as sanity ranges).
REAL_RANGES = {
    "possession_pct": (35, 65),
    "passes": (350, 650),
    "pass_completion_pct": (72, 90),
    "shots": (7, 20),
    "shots_on_target": (2, 8),
    "goals": (0, 4),
    "xg": (0.5, 2.8),
    "tackles": (10, 25),
    "interceptions": (5, 15),
    "fouls": (7, 15),
    "corners": (2, 9),
}


def team_stats(r: MatchResult) -> list[dict]:
    in_play = r.ball_state != DEAD
    stats = []
    for team_idx, club in enumerate((r.home, r.away)):
        ev = [e for e in r.events if e["team"] == club]
        passes = [e for e in ev if e["type"] == "pass"]
        shots = [e for e in ev if e["type"] == "shot"]
        stats.append({
            "club": club,
            "goals": r.score[team_idx],
            "possession_pct": 100.0 * np.mean(r.possession[in_play] == team_idx),
            "passes": len(passes),
            "pass_completion_pct": 100.0 * np.mean([p["outcome"] == "complete" for p in passes]) if passes else 0.0,
            "shots": len(shots),
            "shots_on_target": sum(s["outcome"] in ("goal", "saved") for s in shots),
            "xg": sum(s["xg"] for s in shots),
            "tackles": sum(e["type"] == "tackle" for e in ev),
            "interceptions": sum(e["type"] == "interception" for e in ev),
            "duels_won": sum(e["type"] == "duel_won" for e in ev),
            "clearances": sum(e["type"] == "clearance" for e in ev),
            "fouls": sum(e["type"] == "foul" for e in ev),
            "corners": sum(e["type"] == "corner" for e in ev),
        })
    return stats


def physical_stats(r: MatchResult) -> dict:
    """Distance covered (km) per player who played at least 80 minutes, and top speed."""
    step = np.linalg.norm(np.diff(r.pos, axis=0), axis=-1)            # (T-1, 2, 11)
    same_player = r.ids[1:] == r.ids[:-1]
    same_period = (r.period[1:] == r.period[:-1])[:, None, None]
    step = np.where(same_player & same_period, step, 0.0)
    dist: dict[int, float] = {}
    frames: dict[int, int] = {}
    for t in range(2):
        for s in range(11):
            codes = r.ids[1:, t, s]
            for code in np.unique(codes):
                m = codes == code
                dist[code] = dist.get(code, 0.0) + float(step[m, t, s].sum())
                frames[code] = frames.get(code, 0) + int(m.sum())
    full = [dist[c] / 1000.0 for c in dist if frames[c] * DT >= 80 * 60]
    speed = step / DT
    return {
        "distance_km_mean": float(np.mean(full)) if full else 0.0,
        "distance_km_max": float(np.max(full)) if full else 0.0,
        "top_speed_ms": float(speed.max()),
        "ball_in_play_min": float((r.ball_state != DEAD).sum() * DT / 60.0),
        "match_min": float(len(r.period) * DT / 60.0),
    }


def print_report(r: MatchResult, names: dict[str, str] | None = None, players: dict | None = None) -> None:
    names = names or {}
    home, away = names.get(r.home, r.home), names.get(r.away, r.away)
    print(f"\n{home} {r.score[0]} - {r.score[1]} {away}   (seed {r.seed})")
    for e in r.events:
        if e["type"] == "goal":
            scorer = players[e["player_id"]].name if players and e["player_id"] else e["player_id"]
            print(f"  ⚽ {int(e['minute']) + 1}'  {names.get(e['team'], e['team'])}: {scorer}")
    stats = team_stats(r)
    print(f"\n  {'':<22}{home[:14]:>14}{away[:14]:>14}   real range (per team)")
    for key, (lo, hi) in REAL_RANGES.items():
        a, b = stats[0][key], stats[1][key]
        fmt = (lambda v: f"{v:.1f}") if isinstance(a, float) else str
        ok = "✓" if lo <= a <= hi and lo <= b <= hi else "✗"
        print(f"  {key:<22}{fmt(a):>14}{fmt(b):>14}   {lo}-{hi} {ok}")
    phys = physical_stats(r)
    print(f"\n  distance per full-match player: {phys['distance_km_mean']:.1f} km (max {phys['distance_km_max']:.1f})  real ~9-12")
    print(f"  top speed: {phys['top_speed_ms']:.1f} m/s  real <= ~10")
    print(f"  ball in play: {phys['ball_in_play_min']:.0f} of {phys['match_min']:.0f} min  real ~55-60")
