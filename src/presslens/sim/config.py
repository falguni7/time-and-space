"""Clubs, player profiles and the mapping from ratings to simulation parameters.

Player profiles use 0-99 attributes in six groups (plus goalkeeping). Every attribute
feeds a concrete simulation parameter in `player_params`, so ratings change behaviour
rather than being decoration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .. import paths

CONFIG_DIR = paths.CONFIG_DIR

ATTRIBUTE_GROUPS: dict[str, list[str]] = {
    "pace": ["acceleration", "sprint_speed"],
    "shooting": ["finishing", "shot_power", "long_shots"],
    "passing": ["short_passing", "long_passing", "vision"],
    "dribbling": ["ball_control", "dribbling", "agility", "composure"],
    "defending": ["def_awareness", "interceptions", "tackling", "aggression"],
    "physical": ["stamina", "strength", "jumping"],
    "goalkeeping": ["diving", "reflexes", "handling", "gk_positioning"],
}
ATTRIBUTES: list[str] = [a for attrs in ATTRIBUTE_GROUPS.values() for a in attrs]
GROUP_ABBR = {
    "pace": "PAC", "shooting": "SHO", "passing": "PAS",
    "dribbling": "DRI", "defending": "DEF", "physical": "PHY", "goalkeeping": "GK",
}

PLAYSTYLES: dict[str, str] = {
    "Press Trigger": "Starts pressing from further away",
    "Interceptor": "Larger reach to cut out passes",
    "Line Breaker": "Prefers passes that beat defensive lines",
    "Tiki-Taka": "Prefers short, safe passes",
    "Press Resistant": "Pressure penalty on decisions halved",
    "Rapid": "+3% top speed",
    "Engine": "Tires 30% slower",
    "Clinical": "Better finishing inside the box",
}

# Position -> role used for rating templates and overall weights.
POSITION_ROLE: dict[str, str] = {
    "GK": "GK", "CB": "CB", "LB": "FB", "RB": "FB", "LWB": "WB", "RWB": "WB",
    "CDM": "DM", "CM": "CM", "CAM": "AM", "LM": "WM", "RM": "WM",
    "LW": "W", "RW": "W", "ST": "ST",
}

# Role -> weight of each group rating in the overall (OVR).
OVR_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"goalkeeping": 0.85, "physical": 0.10, "passing": 0.05},
    "CB": {"defending": 0.50, "physical": 0.25, "pace": 0.10, "passing": 0.10, "dribbling": 0.05},
    "FB": {"defending": 0.35, "pace": 0.25, "passing": 0.15, "dribbling": 0.15, "physical": 0.10},
    "WB": {"pace": 0.25, "defending": 0.25, "passing": 0.20, "dribbling": 0.20, "physical": 0.10},
    "DM": {"defending": 0.35, "passing": 0.30, "physical": 0.20, "dribbling": 0.15},
    "CM": {"passing": 0.35, "dribbling": 0.25, "defending": 0.15, "shooting": 0.10, "physical": 0.15},
    "AM": {"passing": 0.35, "dribbling": 0.30, "shooting": 0.25, "pace": 0.10},
    "WM": {"pace": 0.25, "dribbling": 0.30, "passing": 0.30, "shooting": 0.15},
    "W": {"pace": 0.30, "dribbling": 0.35, "shooting": 0.20, "passing": 0.15},
    "ST": {"shooting": 0.45, "pace": 0.20, "dribbling": 0.15, "physical": 0.15, "passing": 0.05},
}

PRESS_HEIGHTS = ("high", "mid", "low")


@dataclass(frozen=True)
class Tactics:
    formation: str
    press_intensity: float
    press_height: str
    compactness: float
    decision_quality: float
    tempo: float

    def __post_init__(self) -> None:
        if self.press_height not in PRESS_HEIGHTS:
            raise ValueError(f"press_height must be one of {PRESS_HEIGHTS}, got {self.press_height!r}")
        for name in ("press_intensity", "compactness", "decision_quality", "tempo"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class Club:
    id: str
    name: str
    tier: str
    colors: dict[str, str]
    tactics: Tactics


@dataclass
class Player:
    id: str
    club_id: str
    name: str
    number: int
    position: str
    age: int
    height_cm: int
    foot: str
    work_rate_att: str
    work_rate_def: str
    attributes: dict[str, int]
    playstyles: list[str] = field(default_factory=list)
    slot: int | None = None  # starting XI slot index in the club formation; None = bench

    @property
    def role(self) -> str:
        return POSITION_ROLE[self.position]

    @property
    def group_ratings(self) -> dict[str, int]:
        return group_ratings(self.attributes)

    @property
    def overall(self) -> int:
        return overall(self.position, self.attributes)

    def card(self) -> str:
        g = self.group_ratings
        groups = ["goalkeeping", "physical", "passing"] if self.role == "GK" else [
            "pace", "shooting", "passing", "dribbling", "defending", "physical"]
        stats = "  ".join(f"{GROUP_ABBR[k]} {g[k]}" for k in groups)
        styles = ", ".join(self.playstyles) or "none"
        return (
            f"#{self.number:<3} {self.name:<22} {self.position:<4} OVR {self.overall}   {self.club_id}\n"
            f"     Age {self.age} · {self.height_cm / 100:.2f} m · {self.foot.title()} foot · "
            f"Work rate: {self.work_rate_att.title()} / {self.work_rate_def.title()}\n"
            f"     {stats}\n"
            f"     Playstyles: {styles}"
        )


def group_ratings(attributes: dict[str, int]) -> dict[str, int]:
    return {
        group: round(sum(attributes[a] for a in attrs) / len(attrs))
        for group, attrs in ATTRIBUTE_GROUPS.items()
    }


def overall(position: str, attributes: dict[str, int]) -> int:
    g = group_ratings(attributes)
    weights = OVR_WEIGHTS[POSITION_ROLE[position]]
    return round(sum(g[k] * w for k, w in weights.items()))


@dataclass(frozen=True)
class PlayerParams:
    """Simulation parameters derived from a player's ratings (and their team's tactics)."""

    vmax: float              # top speed, m/s
    amax: float              # max acceleration, m/s^2
    fatigue: float           # fraction of top speed lost by 90 minutes
    press: float             # personal press intensity 0-1 (team setting adjusted by aggression)
    press_trigger_m: float   # distance from the carrier at which this player starts pressing
    mark_skill: float        # 0-1 marking position quality (lower = more positional error)
    intercept_reach_m: float # how far from the pass line the player can cut a pass out
    tackle: float            # 0-1 tackle success factor
    pass_error_deg: float    # angular error on short passes
    long_pass_error_deg: float
    vision_m: float          # farthest passing option the player considers
    composure: float         # 0-1, scales how much pressure degrades decisions
    dribble: float           # 0-1, tackle resistance when carrying
    finishing: float         # 0-1, shot accuracy multiplier on hidden xG
    shot_speed: float        # m/s
    long_shot_bias: float    # 0-1 willingness to shoot from distance
    strength: float          # 0-1 duels and shielding
    aerial: float            # 0-1 headers / lofted ball duels
    save: float              # 0-1 goalkeeper save factor
    run_rate: float          # forward runs per minute when in possession
    track_back: float        # 0-1 how hard the player recovers without the ball


def _lerp(lo: float, hi: float, rating: float) -> float:
    """Map a 0-99 rating linearly from 40 -> lo to 99 -> hi (clamped)."""
    t = min(max((rating - 40.0) / 59.0, 0.0), 1.0)
    return lo + (hi - lo) * t


WORK_RATE_LEVEL = {"low": 0.0, "medium": 0.5, "high": 1.0}


def player_params(player: Player, tactics: Tactics) -> PlayerParams:
    a = player.attributes
    styles = set(player.playstyles)

    vmax = _lerp(7.0, 9.7, a["sprint_speed"]) * (1.03 if "Rapid" in styles else 1.0)
    fatigue = _lerp(0.14, 0.04, a["stamina"]) * (0.7 if "Engine" in styles else 1.0)
    press = min(max(tactics.press_intensity + (a["aggression"] - 70) / 100.0, 0.0), 1.0)
    trigger = _lerp(6.0, 14.0, 100 * press) + (4.0 if "Press Trigger" in styles else 0.0)
    composure = _lerp(0.0, 1.0, a["composure"])
    if "Press Resistant" in styles:
        composure = 1.0 - (1.0 - composure) / 2.0

    return PlayerParams(
        vmax=vmax,
        amax=_lerp(2.6, 4.2, a["acceleration"]),
        fatigue=fatigue,
        press=press,
        press_trigger_m=trigger,
        mark_skill=_lerp(0.2, 1.0, a["def_awareness"]),
        intercept_reach_m=_lerp(0.6, 2.0, a["interceptions"]) * (1.3 if "Interceptor" in styles else 1.0),
        tackle=_lerp(0.1, 0.9, a["tackling"]),
        pass_error_deg=_lerp(5.0, 1.2, a["short_passing"]),
        long_pass_error_deg=_lerp(8.0, 2.5, a["long_passing"]),
        vision_m=_lerp(18.0, 55.0, a["vision"]),
        composure=composure,
        dribble=_lerp(0.1, 0.9, (a["dribbling"] + a["ball_control"] + a["agility"]) / 3.0),
        finishing=min(_lerp(0.75, 1.2, a["finishing"]) * (1.05 if "Clinical" in styles else 1.0), 1.25),
        shot_speed=_lerp(18.0, 33.0, a["shot_power"]),
        long_shot_bias=_lerp(0.0, 1.0, a["long_shots"]),
        strength=_lerp(0.1, 0.9, a["strength"]),
        aerial=_lerp(0.1, 0.9, (a["jumping"] + a["strength"]) / 2.0 + (player.height_cm - 180) / 2.0),
        save=_lerp(0.2, 0.95, (a["diving"] + a["reflexes"] + a["handling"] + a["gk_positioning"]) / 4.0),
        run_rate=0.5 + 2.0 * WORK_RATE_LEVEL[player.work_rate_att],
        track_back=0.3 + 0.7 * WORK_RATE_LEVEL[player.work_rate_def],
    )


def load_clubs(path: Path | None = None) -> dict[str, Club]:
    data = yaml.safe_load((path or CONFIG_DIR / "clubs.yaml").read_text())
    clubs = {}
    for c in data["clubs"]:
        clubs[c["id"]] = Club(
            id=c["id"], name=c["name"], tier=c["tier"], colors=c["colors"],
            tactics=Tactics(**c["tactics"]),
        )
    return clubs


def load_players(path: Path | None = None) -> dict[str, Player]:
    data = yaml.safe_load((path or CONFIG_DIR / "players.yaml").read_text())
    return {p["id"]: Player(**p) for p in data["players"]}


def squad(players: dict[str, Player], club_id: str) -> list[Player]:
    return sorted((p for p in players.values() if p.club_id == club_id),
                  key=lambda p: (p.slot is None, p.slot if p.slot is not None else 0, p.number))


def starting_xi(players: dict[str, Player], club_id: str) -> list[Player]:
    xi = sorted((p for p in players.values() if p.club_id == club_id and p.slot is not None),
                key=lambda p: p.slot)
    if len(xi) != 11:
        raise ValueError(f"{club_id} has {len(xi)} starters, expected 11")
    return xi
