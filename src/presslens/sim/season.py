"""The season: a full double round-robin between the 8 clubs (56 matches, 14 rounds)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import load_clubs

SEASON_SEED = 1000      # match n of the season uses seed SEASON_SEED + n
HISTORY_SEED = 50000    # "historical" matches (provider model training) use seeds from here


@dataclass(frozen=True)
class Fixture:
    round: int
    home: str
    away: str
    seed: int

    @property
    def match_id(self) -> str:
        return f"R{self.round:02d}_{self.home}-{self.away}"


def schedule(club_ids: list[str] | None = None, seed0: int = SEASON_SEED) -> list[Fixture]:
    """Circle-method double round-robin: every club plays every other home and away."""
    clubs = list(club_ids or load_clubs())
    if len(clubs) % 2:
        raise ValueError("needs an even number of clubs")
    n = len(clubs)
    rotating = clubs[1:]
    first_half: list[list[tuple[str, str]]] = []
    for rnd in range(n - 1):
        order = [clubs[0], *rotating]
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            # Alternate home advantage so nobody plays many home games in a row.
            pairs.append((a, b) if (rnd + i) % 2 == 0 else (b, a))
        first_half.append(pairs)
        rotating = rotating[-1:] + rotating[:-1]
    rounds = first_half + [[(b, a) for a, b in pairs] for pairs in first_half]
    fixtures = []
    k = 0
    for rnd, pairs in enumerate(rounds, start=1):
        for home, away in pairs:
            fixtures.append(Fixture(rnd, home, away, seed0 + k))
            k += 1
    return fixtures
