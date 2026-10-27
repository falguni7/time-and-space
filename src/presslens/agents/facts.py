"""Fact sheets (what may be said about a moment) and templates (what is said when no model can).

The fact sheet is built by code from the MCP tool functions, never by a model, so the Verifier
always has a trusted reference. Templates turn a fact sheet into plain sentences for the offline
backend and for answers that fail verification twice.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..engine import cube
from ..mcp_server import tools as T
from .schemas import Fact, FactSheet, PlayerRef, Version

PITCH_HALF_X = 52.5
LANE_WORDS = {"left_wing": "left wing", "left_half_space": "left half-space", "centre": "centre",
              "right_half_space": "right half-space", "right_wing": "right wing"}
OUTCOME_WORDS = {"complete": "found its man", "intercepted": "was intercepted", "lost_duel": "was lost in a duel",
                 "misplaced": "went astray", "out": "went out of play", "overhit": "was overhit",
                 "saved": "was saved", "off_target": "went wide", "blocked": "was blocked", "goal": "went in"}
PHASE_WORDS = {"build_up": "build-up", "create": "chance creation", "finish": "final-third attack",
               "counter_attack": "counter-attack", "transition": "transition", "set_piece": "set piece",
               "high_press": "high press", "mid_block": "mid block", "low_block": "low block"}


def _player(role: str, pid: str | None) -> PlayerRef | None:
    p = T._player(pid)
    if not p:
        return None
    return PlayerRef(role=role, name=p["name"], club=p.get("club", "?"), position=p.get("position"))


def _pct(x: float) -> int:
    return round(100 * float(x))


def pressure_word(p: float) -> str:
    """The cube's pressure bands (none / some / high), in words the Critic and writers share."""
    return "no pressure" if p <= 0 else "some pressure" if p <= 0.5 else "high pressure"


@lru_cache(maxsize=1)
def _league_gaps() -> np.ndarray:
    with cube.connect_cube() as con:
        return np.sort(con.execute("select decision_gap from passes").fetchnumpy()["decision_gap"])


THIRD_WORDS = {"attacking": "final third", "middle": "middle third", "defensive": "defensive third"}


@lru_cache(maxsize=16)
def _league_zone(third: str, band: str) -> dict:
    """League passes from one third of the pitch under one pressure band: threat per pass (sorted, for
    ranking a single pass) and completion."""
    with cube.connect_cube() as con:
        df = con.execute("select xt_added, outcome = 'complete' as done from passes where third = ? and "
                         "pressure_band = ?", [third, band]).df()
    return {"xt": np.sort(df.xt_added.to_numpy()), "mean_xt": float(df.xt_added.mean()),
            "completion": float(df.done.mean()), "n": len(df)}


def pass_rank(xt_added: float, third: str, band: str) -> str:
    """Where one pass's threat change sits among league passes from the same third under the same pressure."""
    xs = _league_zone(third, band)["xt"]
    below = round(100 * np.searchsorted(xs, xt_added, side="left") / len(xs))
    above = round(100 * (len(xs) - np.searchsorted(xs, xt_added, side="right")) / len(xs))
    if above >= 50:
        return f"lower than {min(above, 99)}% of them"
    return f"higher than {min(below, 99)}% of them"


def league_zone_facts(match_id: str, frame: int) -> list[Fact]:
    """The pass against the league: what passes from the same third under the same pressure usually do
    to the threat, and what they do there with nobody pressing. This is the pressure-threat curve of
    the whole method, read at the spot of the moment."""
    with cube.connect_cube() as con:
        row = con.execute("select third, pressure_band, xt_added from passes where match_id = ? and frame = ?",
                          [match_id, frame]).fetchone()
    if row is None:
        return []
    third, band, xt_added = row
    zone = THIRD_WORDS.get(third, third)
    band_word = pressure_word({"none": 0, "some": 0.5, "high": 1}[band])
    same = _league_zone(third, band)
    facts = [
        Fact(key="league_xt_zone", label=f"League average threat per pass, {zone}, {band_word}",
             value=round(same["mean_xt"], 4), unit="xT",
             meaning=f"from {same['n']} league passes in this situation; this pass was {pass_rank(xt_added, third, band)}"),
        Fact(key="league_completion_zone", label=f"League pass completion, {zone}, {band_word}",
             value=_pct(same["completion"]), unit="%"),
    ]
    if band != "none":
        free = _league_zone(third, "none")
        facts.append(Fact(key="league_xt_zone_free", label=f"League average threat per pass, {zone}, no pressure",
                          value=round(free["mean_xt"], 4), unit="xT",
                          meaning="the same spot without a presser, for comparison"))
    return facts


def gap_meaning(gap: float) -> str:
    if gap <= 0:
        return "0: the best option was chosen"
    share = min(99, round(100 * np.searchsorted(_league_gaps(), gap) / len(_league_gaps())))
    return f"the value lost by not choosing the best option, in goal probability; bigger than {share}% of league passes"


def build_fact_sheet(moment_id: str) -> FactSheet:
    card = T.get_moment(moment_id)
    m = card.get("metrics", {})
    loc = card.get("location") or {}
    where = f"{THIRD_WORDS.get(loc.get('third'), '?')}, {LANE_WORDS.get(loc.get('lane'), loc.get('lane', '?'))}" if loc else ""
    kind = card["type"]
    facts: list[Fact] = []
    players: list[PlayerRef] = []

    def add(key, label, value, unit="", meaning=""):
        facts.append(Fact(key=key, label=label, value=value, unit=unit, meaning=meaning))

    if kind in ("goal", "big_chance"):
        shooter = _player("shooter", card["player_id"])
        players.append(shooter)
        dist = math.hypot(PITCH_HALF_X - loc["ax"], loc["ay"])
        add("xg", "Chance quality (xG)", m["provider_xg"], "",
            f"a shot like this is scored about {_pct(m['provider_xg'])} times in 100")
        add("distance", "Distance from goal", round(dist, 1), "m")
        kind_word = ("penalty" if m.get("restart") == "penalty" else "free kick" if m.get("restart") == "free_kick"
                     else "header" if m.get("header") else "shot")
        situation = (f"{shooter.name} {'scored with a' if kind == 'goal' else 'had a'} {kind_word} from "
                     f"{round(dist)} m ({where}); it {OUTCOME_WORDS.get(m['outcome'], m['outcome'])}.")
    elif kind in ("press_trap", "press_broken", "missed_option"):
        passer_id = card.get("pressed_player_id") or card["player_id"]
        opts = T.pass_options(card["match_id"], card["frame"])
        chosen = next(o for o in opts["options"] if o["chosen"])
        best = next(o for o in opts["options"] if o["best"])
        players.append(_player("passer", passer_id))
        if chosen["player_id"] != best["player_id"]:
            players += [_player("chosen receiver", chosen["player_id"]), _player("best option", best["player_id"])]
        else:
            players.append(_player("receiver", chosen["player_id"]))
        add("pressure", "Pressure on the passer", opts["pressure"], "(0-1)",
            f"{pressure_word(opts['pressure'])}; 1 means a defender within 1 m, 0 means nobody within 5 m")
        add("carrier_proximity", "Nearest defender distance", opts["carrier_proximity_m"], "m")
        add("free_options", "Free teammates (no marker within 5 m)", opts["free_options"], f"of {len(opts['options'])}")
        add("chosen_xpass", "Chance the chosen pass arrives (xPass)", chosen["xpass"])
        add("chosen_marking", "Marker distance to the chosen receiver", chosen["marking_m"], "m")
        if chosen["player_id"] != best["player_id"]:
            add("best_xpass", "Chance the best pass arrives (xPass)", best["xpass"])
            add("best_marking", "Marker distance to the best option", best["marking_m"], "m")
            add("best_lane", "Where the best option was", f"{THIRD_WORDS[best['third']]}, {LANE_WORDS[best['lane']]}")
        add("decision_gap", "Decision gap (best minus chosen option value)", opts["decision_gap"], "",
            gap_meaning(opts["decision_gap"]))
        add("xt_added", "Threat added by the pass (xT)", opts["xt_added"], "",
            "change in the chance of scoring soon; negative means the ball was lost")
        facts += league_zone_facts(card["match_id"], card["frame"])
        if kind != "missed_option":
            pr = T.pressure_at(card["match_id"], card["frame"])
            if pr.get("nearest_defender"):
                players.append(_player("nearest presser", pr["nearest_defender"]["player_id"]))
                add("defenders_close", "Defenders within 5 m of the passer", pr["defenders_within_5m"])
        phase = m.get("opponent_phase") or ""
        situation = (f"{players[0].name} passed under pressure in the {where}"
                     f"{' against a ' + PHASE_WORDS[phase] if phase in PHASE_WORDS else ''}; "
                     f"the pass to {T._name(chosen['player_id'])} {OUTCOME_WORDS.get(opts['outcome'], opts['outcome'])}.")
        if kind == "missed_option":
            situation = (f"{players[0].name} passed to {T._name(chosen['player_id'])} in the {where}, while "
                         f"{T._name(best['player_id'])} was the better option; the pass "
                         f"{OUTCOME_WORDS.get(opts['outcome'], opts['outcome'])}.")
        elif kind == "press_trap" and opts["outcome"] != "complete":
            situation += f" {card['team_name']} won the ball."
    elif kind == "momentum_swing":
        add("xt_balance", "Threat balance over the last 5 minutes (xT)", m["xt_balance_5min"], "",
            "how much more threat this side created than the opponent in 5 minutes")
        situation = f"{card['team_name']} took control of the game after a spell of pressure from {card['opponent_name']}."
    else:
        situation = kind

    return FactSheet(
        moment_id=moment_id, match_id=card["match_id"], type=kind, clock=card["clock"], frame=card["frame"],
        team=card["team"], team_name=card["team_name"], opponent=card["opponent"],
        opponent_name=card["opponent_name"], score_before=card["score_before"], situation=situation,
        players=[p for p in players if p], facts=facts, importance=card["importance"],
    )


# --------------------------------------------------------------------------- templates


def _possessive(name: str) -> str:
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _f(sheet: FactSheet, key: str):
    return next((f.value for f in sheet.facts if f.key == key), None)


def template_explanation(sheet: FactSheet) -> tuple[str, str]:
    """Neutral headline and body from the fact sheet alone."""
    k = sheet.type
    if k in ("goal", "big_chance"):
        xg = _f(sheet, "xg")
        head = (f"{sheet.players[0].name} scores for {sheet.team_name}" if k == "goal"
                else f"Big chance for {sheet.team_name}")
        body = (f"{sheet.situation} Chances like this are scored about {_pct(xg)} times in 100 (xG {xg}).")
    elif k == "press_trap":
        head = f"{_possessive(sheet.team_name)} press wins the ball"
        body = (f"{sheet.situation} The pressure was {_f(sheet, 'pressure')} on a 0-1 scale, with the nearest "
                f"defender {_f(sheet, 'carrier_proximity')} m away, and the chosen pass had only a "
                f"{_pct(_f(sheet, 'chosen_xpass'))}% chance of arriving.")
    elif k == "press_broken":
        head = f"{sheet.team_name} play through the press"
        body = (f"{sheet.situation} Despite pressure of {_f(sheet, 'pressure')} on a 0-1 scale, the pass "
                f"added {_f(sheet, 'xt_added')} xT, a {round(100 * _f(sheet, 'xt_added'), 1)}% rise in the chance "
                f"of scoring soon.")
    elif k == "missed_option":
        head = "The better pass was on"
        body = (f"{sheet.situation} The best option had a {_pct(_f(sheet, 'best_xpass') or 0)}% chance of "
                f"arriving with the nearest marker {_f(sheet, 'best_marking')} m away; the decision gap was "
                f"{_f(sheet, 'decision_gap')}.")
    elif k == "momentum_swing":
        head = f"Momentum swings to {sheet.team_name}"
        body = (f"{sheet.situation} Over the last 5 minutes they created {_f(sheet, 'xt_balance')} more xT "
                f"than {sheet.opponent_name}.")
    else:
        head, body = sheet.type, sheet.situation
    return head, body


def template_version(sheet: FactSheet, profile_id: str = "neutral", issues: list[str] | None = None) -> Version:
    head, body = template_explanation(sheet)
    return Version(profile_id=profile_id, headline=head, body=f"{sheet.clock} {body}", source="template",
                   verified=True, attempts=0, issues=issues or [])
