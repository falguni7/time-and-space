"""Formation templates.

Each slot is (position, depth, width) in a team's own frame:
  depth 0 = defensive line, 1 = most advanced line (scaled by the team block length)
  width -1 = left touchline side, +1 = right (scaled by the team block width)
Slot 0 is always the goalkeeper; it is positioned separately and its depth/width are unused.
"""

from __future__ import annotations

Slot = tuple[str, float, float]

FORMATIONS: dict[str, list[Slot]] = {
    "4-3-3": [
        ("GK", 0.0, 0.0),
        ("LB", 0.05, -0.90), ("CB", 0.0, -0.32), ("CB", 0.0, 0.32), ("RB", 0.05, 0.90),
        ("CDM", 0.33, 0.0), ("CM", 0.52, -0.45), ("CM", 0.52, 0.45),
        ("LW", 0.90, -0.80), ("ST", 1.0, 0.0), ("RW", 0.90, 0.80),
    ],
    "4-4-2": [
        ("GK", 0.0, 0.0),
        ("LB", 0.05, -0.90), ("CB", 0.0, -0.32), ("CB", 0.0, 0.32), ("RB", 0.05, 0.90),
        ("LM", 0.50, -0.85), ("CM", 0.45, -0.28), ("CM", 0.45, 0.28), ("RM", 0.50, 0.85),
        ("ST", 1.0, -0.22), ("ST", 1.0, 0.22),
    ],
    "4-2-3-1": [
        ("GK", 0.0, 0.0),
        ("LB", 0.05, -0.90), ("CB", 0.0, -0.32), ("CB", 0.0, 0.32), ("RB", 0.05, 0.90),
        ("CDM", 0.30, -0.25), ("CDM", 0.30, 0.25),
        ("LW", 0.70, -0.80), ("CAM", 0.68, 0.0), ("RW", 0.70, 0.80),
        ("ST", 1.0, 0.0),
    ],
    "3-5-2": [
        ("GK", 0.0, 0.0),
        ("CB", 0.0, -0.55), ("CB", 0.0, 0.0), ("CB", 0.0, 0.55),
        ("LWB", 0.45, -0.95), ("CDM", 0.33, 0.0), ("CM", 0.50, -0.40), ("CM", 0.50, 0.40),
        ("RWB", 0.45, 0.95),
        ("ST", 1.0, -0.22), ("ST", 1.0, 0.22),
    ],
}

# Bench positions (7 substitutes) per formation.
BENCH: dict[str, list[str]] = {
    "4-3-3": ["GK", "CB", "LB", "CM", "CDM", "RW", "ST"],
    "4-4-2": ["GK", "CB", "RB", "CM", "LM", "ST", "CM"],
    "4-2-3-1": ["GK", "CB", "LB", "CDM", "CAM", "LW", "ST"],
    "3-5-2": ["GK", "CB", "RWB", "CM", "CDM", "ST", "CB"],
}
