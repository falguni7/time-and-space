"""The live interpreter: updates the match state one event at a time and emits a timed feed.

Feed messages (one JSON object each, in match order; `frame` is the 10 Hz match timeline):

    {"kind": "event",  "frame", "clock", "event": {...}, "metrics": {...}}   every validated provider event;
                                                                    passes and shots carry their quality metrics
    {"kind": "state",  "frame", "clock", "score", "teams": {...}}   running statistics, every 30 s and after goals
    {"kind": "moment", "frame", "clock", "moment_id", "type", "team", "for": [viewer ids]}   a key moment, as it happens
    {"kind": "rejected", "frame", "reason"}                        an event that failed the schema

Per-pass pressure and threat come from the engine's per-pass interpretation (the metrics cube), the
same numbers the agents explain, so the live panel and the explanations always agree.
"""

from __future__ import annotations

import math
from collections import deque

from pydantic import ValidationError

from ..mcp_server import tools as T
from .schema import MatchEvent, TeamState

STATE_EVERY = 300           # frames (30 s)
DIFFICULTY = [(0.85, "easy"), (0.6, "medium"), (0.35, "hard"), (0.0, "very hard")]   # by xPass
MOMENTUM_WINDOW = 3000      # frames (5 min)


class LiveMatch:
    def __init__(self, match_id: str, moments: list[dict] | None = None):
        idx = T._index().loc[match_id]
        self.match_id, self.home, self.away = match_id, idx.home, idx.away
        self.teams = {self.home: TeamState(), self.away: TeamState()}
        self.score = "0-0"
        self.per_event = _per_event_metrics(match_id)
        self.moments = sorted(moments or [], key=lambda m: m["frame"])
        self._next_moment = 0
        self._last_state = -STATE_EVERY
        self._recent: deque[tuple[int, str, float]] = deque()     # (frame, team, xt) for momentum
        self.frame, self.period, self.minute = 0, 1, 0.0

    def clock(self) -> str:
        return T.clock(self.minute, self.period)

    def state(self) -> dict:
        total = sum(t.possession_s for t in self.teams.values())
        momentum = {team: round(sum(x for _, t, x in self._recent if t == team), 3) for team in self.teams}
        return {"kind": "state", "frame": self.frame, "clock": self.clock(), "score": self.score,
                "teams": {team: {**s.view(total), "momentum_5min": momentum[team]} for team, s in self.teams.items()}}

    def ingest(self, raw: dict) -> list[dict]:
        """Validate and apply one incoming event; return the feed messages it causes."""
        try:
            ev = MatchEvent.model_validate(raw)
        except ValidationError as exc:
            return [{"kind": "rejected", "frame": raw.get("frame"), "reason": exc.errors()[0]["msg"]}]
        out: list[dict] = []
        # Moments whose time has come are emitted before the event that follows them.
        while self._next_moment < len(self.moments) and self.moments[self._next_moment]["frame"] <= ev.frame:
            m = self.moments[self._next_moment]
            out.append({"kind": "moment", "frame": m["frame"], "clock": m["clock"], "moment_id": m["moment_id"],
                        "type": m["type"], "team": m["team"], "for": m.get("for", [])})
            if m["type"] == "press_trap":
                self.teams[m["team"]].presses_won += 1
            self._next_moment += 1
        self.frame, self.period, self.minute = ev.frame, ev.period, ev.minute
        self._apply(ev)
        msg = {"kind": "event", "frame": ev.frame, "clock": self.clock(), "event": ev.model_dump(exclude_none=True)}
        metrics = self.quality(ev)
        if metrics:
            msg["metrics"] = metrics
        out.append(msg)
        if ev.type == "goal" or ev.frame - self._last_state >= STATE_EVERY:
            self._last_state = ev.frame
            out.append(self.state())
        return out

    def quality(self, ev: MatchEvent) -> dict:
        """Pass quality (distance, ball speed, difficulty from xPass, pressure, threat) and shot speed."""
        if ev.type not in ("pass", "shot") or None in (ev.x, ev.y, ev.end_x, ev.end_y):
            return {}
        dist = math.hypot(ev.end_x - ev.x, ev.end_y - ev.y)
        secs = (ev.end_frame - ev.frame) / 10 if ev.end_frame and ev.end_frame > ev.frame else None
        out = {"distance_m": round(dist, 1)}
        if secs:
            out["ball_speed_kmh"] = round(3.6 * dist / secs)
        pressure, xt, xpass = self.per_event.get(ev.event_id, (None, None, None))
        if ev.type == "pass" and xpass is not None:
            out |= {"xpass": round(xpass, 2), "difficulty": next(w for t, w in DIFFICULTY if xpass >= t),
                    "pressure": round(pressure, 2), "xt_added": round(xt, 4)}
        return out

    def _apply(self, ev: MatchEvent) -> None:
        team = self.teams.get(ev.team) if ev.team else None
        while self._recent and self._recent[0][0] < ev.frame - MOMENTUM_WINDOW:
            self._recent.popleft()
        if team is None:
            return
        pressure, xt, _ = self.per_event.get(ev.event_id, (None, None, None))
        if ev.type == "possession" and ev.end_frame is not None:
            team.possession_s += (ev.end_frame - ev.frame) / 10
        elif ev.type == "pass":
            team.passes += 1
            team.completed += ev.outcome == "complete"
            team.pressure_faced_sum += pressure or 0.0
        elif ev.type == "pressure":
            team.pressures += 1
        elif ev.type == "shot":
            team.shots += 1
            team.xg += ev.provider_xg or 0.0
        elif ev.type == "goal":
            team.goals += 1
            self.score = ev.score or f"{self.teams[self.home].goals}-{self.teams[self.away].goals}"
        if xt is not None and ev.type in ("pass", "carry"):
            team.xt_created += max(xt, 0.0)
            self._recent.append((ev.frame, ev.team, xt))


def _per_event_metrics(match_id: str) -> dict[int, tuple[float | None, float, float | None]]:
    """event_id -> (pressure on the passer, threat added, xPass of the chosen pass), from the engine's
    per-pass and per-carry tables."""
    from ..engine import cube
    with cube.connect_cube() as con:
        rows = con.execute("select event_id, pressure, xt_added, chosen_xpass from passes where match_id = ? "
                           "union all select event_id, null, xt_added, null from carries where match_id = ?",
                           [match_id, match_id]).fetchall()
    return {int(e): (p, float(x), q) for e, p, x, q in rows}


__all__ = ["LiveMatch"]
