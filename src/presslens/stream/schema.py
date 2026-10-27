"""The messages of the live stream: what the provider sends (MatchEvent) and what the pipeline emits.

Every incoming event is validated before anything reads it; a malformed message is rejected with
the reason instead of corrupting the live state.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

EVENT_TYPES = {
    "period_start", "period_end", "kickoff", "pass", "carry", "possession", "pressure", "recovery", "deflection",
    "tackle", "duel_won", "ball_out", "clearance", "overrun", "throw_in", "shot", "dispossessed", "interception",
    "free_kick", "foul", "goal_kick", "corner", "goal", "save", "substitution", "gk", "miscontrol",
}
PITCH_X, PITCH_Y = 52.5 + 15, 34.0 + 15   # half-lengths plus room for balls that go out of play


class MatchEvent(BaseModel):
    """One provider event, as it arrives."""

    model_config = ConfigDict(extra="ignore")

    event_id: int
    frame: int
    end_frame: int | None = None
    period: Literal[1, 2]
    minute: float
    type: str
    team: str | None = None
    player_id: str | None = None
    x: float | None = None
    y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    outcome: str | None = None
    receiver_id: str | None = None
    other_player_id: str | None = None
    restart: str | None = None
    provider_xg: float | None = None
    min_distance_m: float | None = None
    score: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _nan_is_missing(cls, v):
        return None if isinstance(v, float) and math.isnan(v) else v

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in EVENT_TYPES:
            raise ValueError(f"unknown event type {v!r}")
        return v

    @field_validator("x", "end_x")
    @classmethod
    def _on_pitch_x(cls, v):
        if v is not None and abs(v) > PITCH_X:
            raise ValueError(f"x {v} is off the pitch")
        return v

    @field_validator("y", "end_y")
    @classmethod
    def _on_pitch_y(cls, v):
        if v is not None and abs(v) > PITCH_Y:
            raise ValueError(f"y {v} is off the pitch")
        return v


class TeamState(BaseModel):
    goals: int = 0
    shots: int = 0
    xg: float = 0.0
    passes: int = 0
    completed: int = 0
    possession_s: float = 0.0
    xt_created: float = 0.0
    pressures: int = 0
    pressure_faced_sum: float = 0.0
    presses_won: int = 0

    def view(self, possession_total: float) -> dict:
        return {"goals": self.goals, "shots": self.shots, "xg": round(self.xg, 2), "passes": self.passes,
                "completion_pct": round(100 * self.completed / self.passes) if self.passes else None,
                "possession_pct": round(100 * self.possession_s / possession_total) if possession_total else 50,
                "xt_created": round(self.xt_created, 2), "pressures": self.pressures,
                "pressure_faced": round(self.pressure_faced_sum / self.passes, 2) if self.passes else None,
                "presses_won": self.presses_won}


__all__ = ["EVENT_TYPES", "MatchEvent", "TeamState"]
