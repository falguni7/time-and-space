"""Generate the 8 club squads (18 players each) with game-style ratings.

Writes configs/players.yaml (editable) and configs/players.csv (for browsing).

    bin/pl python -m presslens.sim.squads --seed 7
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import yaml

from .config import (
    ATTRIBUTE_GROUPS,
    ATTRIBUTES,
    CONFIG_DIR,
    GROUP_ABBR,
    POSITION_ROLE,
    Player,
    load_clubs,
)
from .formations import BENCH, FORMATIONS

# Mean group ratings per role for a mid-tier club.
ROLE_TEMPLATES: dict[str, dict[str, float]] = {
    "GK": {"pace": 50, "shooting": 25, "passing": 58, "dribbling": 42, "defending": 25, "physical": 66, "goalkeeping": 77},
    "CB": {"pace": 66, "shooting": 40, "passing": 63, "dribbling": 58, "defending": 78, "physical": 78, "goalkeeping": 12},
    "FB": {"pace": 79, "shooting": 50, "passing": 70, "dribbling": 72, "defending": 73, "physical": 70, "goalkeeping": 12},
    "WB": {"pace": 81, "shooting": 52, "passing": 71, "dribbling": 74, "defending": 70, "physical": 71, "goalkeeping": 12},
    "DM": {"pace": 64, "shooting": 60, "passing": 76, "dribbling": 72, "defending": 76, "physical": 77, "goalkeeping": 12},
    "CM": {"pace": 68, "shooting": 68, "passing": 78, "dribbling": 76, "defending": 66, "physical": 72, "goalkeeping": 12},
    "AM": {"pace": 74, "shooting": 75, "passing": 79, "dribbling": 80, "defending": 45, "physical": 63, "goalkeeping": 12},
    "WM": {"pace": 82, "shooting": 68, "passing": 74, "dribbling": 78, "defending": 55, "physical": 66, "goalkeeping": 12},
    "W": {"pace": 85, "shooting": 74, "passing": 72, "dribbling": 81, "defending": 40, "physical": 62, "goalkeeping": 12},
    "ST": {"pace": 80, "shooting": 79, "passing": 65, "dribbling": 75, "defending": 38, "physical": 75, "goalkeeping": 12},
}
# Groups a star is exceptional in, per role.
STAR_GROUPS: dict[str, list[str]] = {
    "GK": ["goalkeeping"], "CB": ["defending", "physical"], "FB": ["pace", "defending"],
    "WB": ["pace", "dribbling"], "DM": ["defending", "passing"], "CM": ["passing", "dribbling"],
    "AM": ["passing", "dribbling"], "WM": ["pace", "dribbling"], "W": ["pace", "dribbling"],
    "ST": ["shooting", "pace"],
}
# Attribute-specific tweaks relative to the group mean.
ATTR_TWEAKS: dict[str, dict[str, float]] = {
    "CB": {"aggression": 3, "agility": -6},
    "DM": {"aggression": 5, "interceptions": 4, "composure": 3},
    "CM": {"stamina": 6},
    "AM": {"vision": 5, "composure": 3},
    "W": {"acceleration": 3, "agility": 5},
    "ST": {"finishing": 4, "long_passing": -6},
    "GK": {"long_passing": 6},
}
TIER_OFFSET = {"strong": 4.0, "mid": 0.0, "weak": -3.0}
HEIGHT_MEAN = {"GK": 190, "CB": 187, "FB": 178, "WB": 178, "DM": 183, "CM": 179,
               "AM": 176, "WM": 177, "W": 175, "ST": 182}
WORK_RATES = {  # (attacking, defending) choices
    "GK": [("medium", "medium")],
    "CB": [("low", "high"), ("medium", "high"), ("low", "medium")],
    "FB": [("high", "high"), ("high", "medium"), ("medium", "high")],
    "WB": [("high", "high"), ("high", "medium")],
    "DM": [("medium", "high"), ("low", "high")],
    "CM": [("high", "high"), ("high", "medium"), ("medium", "medium")],
    "AM": [("high", "medium"), ("high", "low")],
    "WM": [("high", "medium"), ("high", "high")],
    "W": [("high", "low"), ("high", "medium")],
    "ST": [("high", "low"), ("medium", "low"), ("high", "medium")],
}
SHIRT_NUMBERS = {"GK": [1, 13, 31], "CB": [4, 5, 6, 15, 24], "LB": [3, 12], "RB": [2, 22],
                 "LWB": [3, 12], "RWB": [2, 22], "CDM": [6, 16, 18], "CM": [8, 14, 17, 20, 23],
                 "CAM": [10, 19], "LM": [11, 21], "RM": [7, 27], "LW": [11, 21], "RW": [7, 27],
                 "ST": [9, 19, 29, 39]}

FIRST_NAMES = [
    "Tomás", "Kofi", "Luca", "Jonah", "Mateo", "Arjun", "Elias", "Kwame", "Rafael", "Noah",
    "Idris", "Sami", "Felix", "Dario", "Yusuf", "Oskar", "Thiago", "Malik", "Bastian", "Emre",
    "Callum", "Ruben", "Hugo", "Nikolai", "Joaquín", "Tariq", "Lars", "Diego", "Kenji", "Marek",
    "Ollie", "Ibrahim", "Leon", "Andrés", "Femi", "Viktor", "Ciaran", "Matteo", "Jules", "Amadou",
    "Rory", "Sven", "Iker", "Dominik", "Zane", "Pablo", "Teo", "Aleksander", "Kai", "Bruno",
]
SURNAMES = [
    "Varela", "Asante-Rowe", "Brennick", "Okonjo", "Halvard", "Marchetti", "Delacroix", "Tamsin",
    "Ferreiro", "Lindqvist", "Adebayo-Kent", "Quillon", "Mbeki-Hart", "Castellanos", "Vossberg",
    "Okafor-Lyle", "Renwick", "Saltoro", "Duvane", "Kerrigan", "Moravec", "Ilunga", "Pashley",
    "Thorvald", "Esquivel", "Nakashima", "Brightwater", "Oyelaran", "Fennimore", "Zaragoza",
    "Holloway-Penn", "Strand", "Mbatha", "Corvino", "Eastwick", "Dalgaard", "Ruelas", "Abernethy",
    "Kowalczyk", "Ventimiglia", "Achterberg", "Sowande", "Larkhill", "Petrakis", "Gallardo",
    "Whitmore", "Bakari-Reid", "Navarrete", "Soderholm", "Mendonça", "Ashdown", "Kaczmarek",
    "Okoro", "Valdivieso", "Ravensworth", "Tchanderi-Ford", "Lundgren", "Bellamy-Cruz",
    "Arrowsmith", "Szabo", "Mwangi", "Fairbairn", "Igwe", "Montclair", "Harlowe", "Ezeani",
]


def _clip(x: float) -> int:
    return round(min(max(x, 20.0), 97.0))


def _make_attributes(role: str, tier: str, star: bool, rng: np.random.Generator) -> dict[str, int]:
    template = ROLE_TEMPLATES[role]
    tweaks = ATTR_TWEAKS.get(role, {})
    player_level = rng.normal(0.0, 3.0)  # shared quality shift for this player
    attrs = {}
    for group, names in ATTRIBUTE_GROUPS.items():
        base = template[group]
        if group != "goalkeeping" or role == "GK":
            base += TIER_OFFSET[tier] + player_level
            if star and group in STAR_GROUPS[role]:
                base += 8.0
        for name in names:
            sd = 9.0 if name in ("composure", "aggression") else 5.0
            attrs[name] = _clip(base + tweaks.get(name, 0.0) + rng.normal(0.0, sd))
    return attrs


def _playstyles(role: str, a: dict[str, int]) -> list[str]:
    candidates = []  # (margin above threshold, name)
    rules = [
        ("Press Trigger", a["aggression"] - 84, role != "GK"),
        ("Interceptor", a["interceptions"] - 84, role in ("CB", "FB", "WB", "DM", "CM")),
        ("Line Breaker", min(a["vision"], a["long_passing"]) - 82, role != "GK"),
        ("Tiki-Taka", a["short_passing"] - 84, role in ("DM", "CM", "AM")),
        ("Press Resistant", a["composure"] - 84, role != "GK"),
        ("Rapid", a["sprint_speed"] - 89, role != "GK"),
        ("Engine", a["stamina"] - 87, role != "GK"),
        ("Clinical", a["finishing"] - 85, role in ("ST", "W", "AM", "WM", "CM")),
    ]
    for name, margin, allowed in rules:
        if allowed and margin >= 0:
            candidates.append((margin, name))
    return [name for _, name in sorted(candidates, reverse=True)[:2]]


def generate(seed: int = 7) -> list[Player]:
    rng = np.random.default_rng(seed)
    clubs = load_clubs()
    names_used: set[str] = set()
    players: list[Player] = []

    def unique_name(club_surnames: set[str]) -> str:
        """Unique across the league, and no shared surname within a club."""
        while True:
            surname = str(rng.choice(SURNAMES))
            name = f"{rng.choice(FIRST_NAMES)} {surname}"
            if name not in names_used and surname not in club_surnames:
                names_used.add(name)
                club_surnames.add(surname)
                return name

    for club in clubs.values():
        club_surnames: set[str] = set()
        first_index = len(players)
        formation = club.tactics.formation
        positions = [(slot_pos, idx) for idx, (slot_pos, _, _) in enumerate(FORMATIONS[formation])]
        positions += [(pos, None) for pos in BENCH[formation]]
        # 1-2 stars among outfield starters.
        outfield_slots = [i for i in range(1, 11)]
        n_stars = 2 if club.tier == "strong" else 1
        star_slots = set(rng.choice(outfield_slots, size=n_stars, replace=False).tolist())
        numbers_used: set[int] = set()

        for k, (position, slot) in enumerate(positions):
            role = POSITION_ROLE[position]
            bench_penalty = slot is None
            tier = club.tier
            attrs = _make_attributes(role, tier, star=slot in star_slots, rng=rng)
            if bench_penalty:
                attrs = {name: _clip(v - 3) for name, v in attrs.items()}
            height = round(rng.normal(HEIGHT_MEAN[role], 5.0))
            for name in ("jumping", "strength"):
                attrs[name] = _clip(attrs[name] + (height - 180) * 0.6)
            att, dfn = WORK_RATES[role][rng.integers(len(WORK_RATES[role]))]
            left_sided = position.startswith("L")
            foot = "left" if rng.random() < (0.75 if left_sided else 0.2) else "right"
            number = next((n for n in SHIRT_NUMBERS[position] if n not in numbers_used), None)
            if number is None:
                number = next(n for n in range(30, 99) if n not in numbers_used)
            numbers_used.add(number)
            players.append(Player(
                id=f"{club.id}-{k + 1:02d}",
                club_id=club.id,
                name=unique_name(club_surnames),
                number=number,
                position=position,
                age=int(rng.integers(18, 35)),
                height_cm=height,
                foot=foot,
                work_rate_att=att,
                work_rate_def=dfn,
                attributes={name: attrs[name] for name in ATTRIBUTES},
                playstyles=_playstyles(role, attrs),
                slot=slot,
            ))
        _pick_best_eleven(players[first_index:])
    return players


def _pick_best_eleven(club_players: list[Player]) -> None:
    """Swap a bench player into the XI when he outrates the starter in the same position."""
    for bench in [p for p in club_players if p.slot is None]:
        rivals = [p for p in club_players if p.slot is not None and p.position == bench.position]
        weakest = min(rivals, key=lambda p: p.overall, default=None)
        if weakest is not None and bench.overall > weakest.overall:
            bench.slot, weakest.slot = weakest.slot, None


def write(players: list[Player]) -> None:
    yaml_rows = []
    for p in players:
        yaml_rows.append({
            "id": p.id, "club_id": p.club_id, "name": p.name, "number": p.number,
            "position": p.position, "slot": p.slot, "age": p.age, "height_cm": p.height_cm,
            "foot": p.foot, "work_rate_att": p.work_rate_att, "work_rate_def": p.work_rate_def,
            "playstyles": p.playstyles, "attributes": p.attributes,
        })
    header = (
        "# Generated by `python -m presslens.sim.squads`; safe to edit by hand.\n"
        "# slot: starting XI position in the club formation (0 = GK); null = bench.\n"
    )
    (CONFIG_DIR / "players.yaml").write_text(
        header + yaml.safe_dump({"players": yaml_rows}, sort_keys=False, allow_unicode=True,
                                width=200)
    )
    groups = ["pace", "shooting", "passing", "dribbling", "defending", "physical", "goalkeeping"]
    with open(CONFIG_DIR / "players.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "club", "name", "number", "position", "starter", "ovr",
                    *[GROUP_ABBR[g] for g in groups], "age", "height_cm", "foot",
                    "work_rate", "playstyles", *ATTRIBUTES])
        for p in players:
            g = p.group_ratings
            w.writerow([p.id, p.club_id, p.name, p.number, p.position, p.slot is not None,
                        p.overall, *[g[k] for k in groups], p.age, p.height_cm, p.foot,
                        f"{p.work_rate_att}/{p.work_rate_def}", "; ".join(p.playstyles),
                        *[p.attributes[a] for a in ATTRIBUTES]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    players = generate(args.seed)
    write(players)
    print(f"Wrote {len(players)} players to {CONFIG_DIR / 'players.yaml'} and players.csv")


if __name__ == "__main__":
    main()
