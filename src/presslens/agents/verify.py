"""The Verifier: rule-based grounding checks on every text an agent writes.

1. Numbers: every number in the text must match a trusted number (a fact, a tool result) after
   rounding, or as a percentage of a 0-1 value. Clock times and the score are trusted too.
2. Names: no player or club may appear that is not part of the moment (catches the model
   borrowing a name from elsewhere in the league).
3. Shape: not empty, not too long.

It is deliberately not a model: a check that can itself hallucinate cannot vouch for anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from ..mcp_server import tools as T

NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:[.,]\d+)?")
CLOCK = re.compile(r"\b\d{1,3}\+\d{1,2}'|\b\d{1,3}'")
SCALE_WORDS = {0.0, 1.0, 5.0, 100.0}   # "0-1 scale", "within 5 m", "times in 100"
MAX_DIFFERENCE_TERMS = 300


@dataclass
class Check:
    ok: bool
    issues: list[str] = field(default_factory=list)

    def feedback(self) -> str:
        return "Fix these problems and answer again: " + "; ".join(self.issues)


def numbers_in(text: str) -> list[float]:
    """Numbers in free text; decimal commas (German, French, ...) are read as decimal points."""
    out = []
    for tok in NUMBER.findall(CLOCK.sub(" ", text)):
        tok = tok.replace("−", "-")
        if "," in tok and re.fullmatch(r"-?\d+,\d{1,4}", tok):
            tok = tok.replace(",", ".")
        tok = tok.replace(",", "")
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def _decimals(x: float) -> int:
    s = repr(abs(x))
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


def trusted_variants(values) -> set[float]:
    """Each trusted value plus its roundings and, for 0-1 values, its percentage forms."""
    out = set(SCALE_WORDS)
    for v in values:
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        forms = [v, abs(v)]
        if abs(v) <= 1.0:
            forms += [100 * v, 100 * abs(v)]
        for f in forms:
            for d in range(5):
                out.add(round(f, d))
    return out


def number_ok(x: float, trusted: set[float]) -> bool:
    tol = 0.5 * 10 ** (-_decimals(x)) + 1e-9
    return any(abs(abs(x) - abs(t)) <= tol for t in trusted)


@lru_cache(maxsize=1)
def _league_names() -> tuple[frozenset[str], frozenset[str]]:
    players = frozenset(p["name"] for p in T._roster().values())
    clubs = frozenset(T._clubs().values())
    return players, clubs


def check_text(text: str, trusted_numbers, allowed_names: set[str], max_words: int = 160,
               clock_and_score: str = "", differences: bool = False) -> Check:
    """differences=True also accepts the gap between any two trusted numbers ("13.6 more per match")."""
    issues = []
    if not text or not text.strip():
        return Check(False, ["the text is empty"])
    # numbers_in() skips clock tokens like 30', so read the clock and score digits directly.
    values = [*trusted_numbers, *map(float, re.findall(r"\d+", clock_and_score))]
    if differences:
        distinct = sorted(set(values))[:MAX_DIFFERENCE_TERMS]
        values += [b - a for i, a in enumerate(distinct) for b in distinct[i + 1:]]
    trusted = trusted_variants(values)
    bad = [x for x in numbers_in(text) if not number_ok(x, trusted)]
    if bad:
        issues.append("these numbers are not in the facts: " + ", ".join(f"{b:g}" for b in sorted(set(bad))))
    players, clubs = _league_names()
    stray = sorted(n for n in (players | clubs) - allowed_names if n in text)
    if stray:
        issues.append("these players/clubs are not part of this moment: " + ", ".join(stray))
    words = len(text.split())
    if words > max_words:
        issues.append(f"too long ({words} words, limit {max_words})")
    return Check(not issues, issues)


def tool_result_numbers(messages) -> list[float]:
    """Every number an agent saw in its tool results (what it may quote)."""
    out = []
    for m in messages:
        for c in getattr(m, "contents", []):
            if getattr(c, "type", None) == "function_result":
                result = c.result
                if isinstance(result, list):
                    result = " ".join(getattr(x, "text", None) or str(x) for x in result)
                out += numbers_in(str(result))
    return out


def sheet_numbers(sheet) -> list[float]:
    vals = []
    for f in sheet.all_facts():
        if isinstance(f.value, (int, float)):
            vals.append(f.value)
        else:
            vals += numbers_in(str(f.value))
        vals += numbers_in(f.unit) + numbers_in(f.meaning)
    return vals


def check_moment_text(sheet, headline: str, body: str, max_words: int = 160) -> Check:
    return check_text(f"{headline}\n{body}", sheet_numbers(sheet), sheet.names(), max_words,
                      clock_and_score=sheet.clock_and_score())
