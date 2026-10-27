"""On-pitch team state and the per-frame movement update for both teams."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kinematics, tactics
from .config import Club, Player, PlayerParams, player_params, starting_xi
from .pitch import from_team_frame, to_team_frame

# slot -> (target point in the global frame, speed cap in m/s)
Overrides = dict[int, tuple[np.ndarray, float]]


@dataclass
class TeamState:
    club: Club
    players: list[Player]        # on-pitch players in formation slot order (0 = GK)
    params: list[PlayerParams]
    direction: int               # +1 if the team attacks +x in the global frame
    pos: np.ndarray              # (11, 2) global
    vel: np.ndarray              # (11, 2)
    pressing: np.ndarray         # (11,) bool, from the latest update
    mark_error: np.ndarray = field(default_factory=lambda: np.zeros((11, 2)))  # team frame

    @classmethod
    def kickoff(cls, club: Club, players: dict[str, Player], direction: int) -> TeamState:
        xi = starting_xi(players, club.id)
        params = [player_params(p, club.tactics) for p in xi]
        pos = from_team_frame(tactics.kickoff_positions(club.tactics), direction)
        return cls(club, xi, params, direction, pos, np.zeros((11, 2)), np.zeros(11, dtype=bool))

    @property
    def vmax(self) -> np.ndarray:
        return np.array([p.vmax for p in self.params])

    @property
    def amax(self) -> np.ndarray:
        return np.array([p.amax for p in self.params])

    def frame(self, xy: np.ndarray) -> np.ndarray:
        """Global -> this team's frame."""
        return to_team_frame(xy, self.direction)

    def substitute(self, slot: int, player: Player) -> None:
        self.players[slot] = player
        self.params[slot] = player_params(player, self.club.tactics)

    def refresh_mark_error(self, rng: np.random.Generator) -> None:
        sd = np.array([(1.0 - p.mark_skill) * 2.5 for p in self.params])
        self.mark_error = rng.normal(0.0, 1.0, size=(11, 2)) * sd[:, None]


def step_teams(
    attack: TeamState,
    defend: TeamState,
    ball: np.ndarray,
    carrier: int | None,
    dt: float,
    rng: np.random.Generator,
    minute: float = 0.0,
    attack_overrides: Overrides | None = None,
    defend_overrides: Overrides | None = None,
    runs: dict[int, np.ndarray] | None = None,
    allow_press: bool = True,
) -> None:
    """Move both teams one frame.

    `ball` and override/run targets are global; `carrier` indexes the attacking team or is None.
    """
    att_t = tactics.attacking_targets(
        attack.frame(attack.pos), attack.frame(defend.pos), attack.frame(ball),
        carrier, attack.params, attack.club.tactics,
        runs={i: attack.frame(xy) for i, xy in (runs or {}).items()},
    )
    def_t = tactics.defending_targets(
        defend.frame(defend.pos), defend.frame(attack.pos), defend.frame(ball),
        carrier, defend.params, defend.club.tactics,
        mark_error=defend.mark_error, allow_press=allow_press,
    )
    for team, t, overrides in ((attack, att_t, attack_overrides), (defend, def_t, defend_overrides)):
        target = from_team_frame(t.xy, team.direction)
        cap = t.speed_cap.copy()
        for i, (xy, speed) in (overrides or {}).items():
            target[i] = xy
            cap[i] = speed
        fatigue = np.array([1.0 - p.fatigue * min(minute / 90.0, 1.0) for p in team.params])
        team.pos, team.vel = kinematics.step(
            team.pos, team.vel, target, team.vmax * fatigue, team.amax, dt,
            speed_cap=cap * fatigue, accel_noise=0.3, rng=rng,
        )
        team.pressing = t.pressing
