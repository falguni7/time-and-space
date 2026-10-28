"""The relevance selector: which of a match's moments each viewer gets, and why.

Plain code, no model: it is free, repeatable and explainable. A moment's score is

    importance (engine) x type weight (viewer kind) x involvement (focus club / player)

and the feed is filled greedily up to the viewer's alert budget, with goals first and a penalty
for repeating a type or crowding the same few minutes. Every pick carries its reasons, so the app
can say "your player is involved" and every viewer can see why the agents wrote about it.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..engine import moments as engine_moments
from ..mcp_server import tools as T
from .personas import Profile

TYPE_WEIGHT = {
    "casual": {"goal": 1.5, "big_chance": 1.2, "momentum_swing": 1.0, "press_broken": 0.7, "press_trap": 0.6,
               "missed_option": 0.5},
    "analyst": {"goal": 1.0, "big_chance": 1.0, "momentum_swing": 1.0, "press_broken": 1.3, "press_trap": 1.3,
                "missed_option": 1.4},
    "supporter": {"goal": 1.5, "big_chance": 1.2, "momentum_swing": 1.1, "press_broken": 0.9, "press_trap": 0.9,
                  "missed_option": 0.7},
    "player_follower": {"goal": 1.3, "big_chance": 1.1, "momentum_swing": 0.8, "press_broken": 1.0,
                        "press_trap": 1.0, "missed_option": 1.0},
}
TYPE_WORDS = {"goal": "a goal", "big_chance": "a big chance", "momentum_swing": "a momentum swing",
              "press_broken": "a press broken", "press_trap": "a press trap", "missed_option": "a missed pass option"}
GOOD_FOR_CLUB = 1.8     # the supporter's club comes out of the moment well ...
BAD_FOR_CLUB = 1.2      # ... or badly (they still want to know, but less)
TOP_MOMENTS = 5         # non-goal moments per match called "one of the biggest"
PLAYER_BOOST = 2.5      # the followed player is the actor or the pressed player
PLAYER_TEAM_BOOST = 1.2
SAME_TYPE_DECAY = 0.8   # per moment of the same type already picked
CROWD_MINUTES = 3.0     # picks this close in time ...
CROWD_DECAY = 0.7       # ... are discounted


class Pick(BaseModel):
    moment_id: str
    type: str
    period: int
    minute: float           # restarts at 45 in the second half: order by (period, minute)
    score: float
    reasons: list[str]


def good_for(card: dict, club: str) -> bool:
    """Cards credit the team that did something; a missed option is a mistake by that team."""
    return (card["team"] == club) != (card["type"] == "missed_option")


def _involves(card: dict, player_id: str) -> bool:
    return player_id in (card.get("player_id"), card.get("pressed_player_id"))


def _base(card: dict, profile: Profile, top: set[str]) -> tuple[float, list[str]]:
    kind = profile.kind if profile.kind in TYPE_WEIGHT else "casual"
    score = card["importance"] * TYPE_WEIGHT[kind].get(card["type"], 0.5)
    reasons = []
    if card["type"] == "goal":
        reasons.append("a goal")
    elif card["moment_id"] in top:
        reasons.append(f"one of the match's biggest moments ({TYPE_WORDS.get(card['type'], card['type'])})")
    if profile.focus_club and profile.focus_club in (card["team"], card["opponent"]):
        good = good_for(card, profile.focus_club)
        score *= GOOD_FOR_CLUB if good else BAD_FOR_CLUB
        reasons.append(f"{'good' if good else 'bad'} moment for {T._clubs().get(profile.focus_club)}")
    if profile.focus_player:
        if _involves(card, profile.focus_player):
            score *= PLAYER_BOOST
            reasons.append(f"{T._name(profile.focus_player)} is involved")
        elif profile.focus_player.split("-")[0] in (card["team"], card["opponent"]):
            score *= PLAYER_TEAM_BOOST
    if profile.kind == "analyst" and card["type"] in ("missed_option", "press_trap", "press_broken"):
        reasons.append("a decision under pressure worth analysing")
    return score, reasons


def select_moments(match_id: str, profile: Profile, cards: list[dict] | None = None,
                   budget: int | None = None) -> list[Pick]:
    """At most `budget` (default: the viewer's alert_budget) moments for this viewer, in match order."""
    cards = engine_moments.load_moments(match_id) if cards is None else cards
    budget = profile.alert_budget if budget is None else budget
    non_goals = sorted((c for c in cards if c["type"] != "goal"), key=lambda c: -c["importance"])
    top = {c["moment_id"] for c in non_goals[:TOP_MOMENTS]}
    scored = [(card, *_base(card, profile, top)) for card in cards]
    # Goals go first: no viewer should miss a goal for a better-scoring press trap.
    pending = sorted(scored, key=lambda x: (x[0]["type"] != "goal", -x[1]))
    picked: list[Pick] = []
    while pending and len(picked) < budget:
        best_i, best_score = 0, -1.0
        for i, (card, score, _) in enumerate(pending):
            if card["type"] == "goal":
                best_i, best_score = i, score
                break
            same = sum(p.type == card["type"] for p in picked)
            crowd = any(p.period == card["period"] and abs(p.minute - card["minute"]) < CROWD_MINUTES
                        for p in picked)
            s = score * SAME_TYPE_DECAY ** same * (CROWD_DECAY if crowd else 1.0)
            if s > best_score:
                best_i, best_score = i, s
        card, _, reasons = pending.pop(best_i)
        picked.append(Pick(moment_id=card["moment_id"], type=card["type"], period=card["period"],
                           minute=card["minute"],
                           score=round(best_score, 3), reasons=reasons or [TYPE_WORDS.get(card["type"], card["type"])]))
    return sorted(picked, key=lambda p: (p.period, p.minute))


def plan_feeds(match_id: str, profiles: list[Profile]) -> dict[str, list[Profile]]:
    """moment_id -> the viewers who get it, so each moment is explained once for all of them."""
    cards = engine_moments.load_moments(match_id)
    plan: dict[str, list[Profile]] = {}
    for profile in profiles:
        for pick in select_moments(match_id, profile, cards):
            plan.setdefault(pick.moment_id, []).append(profile)
    return dict(sorted(plan.items()))


__all__ = ["Pick", "good_for", "plan_feeds", "select_moments"]
