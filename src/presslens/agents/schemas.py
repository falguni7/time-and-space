"""Messages passed between the agents. LLM output models are kept flat (strict JSON schema)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- built by code


class Fact(BaseModel):
    key: str
    label: str
    value: float | int | str
    unit: str = ""
    meaning: str = ""


class PlayerRef(BaseModel):
    role: str
    name: str
    club: str
    position: str | None = None


class FactSheet(BaseModel):
    """Everything an agent may say about one moment. The Verifier checks text against it."""
    moment_id: str
    match_id: str
    type: str
    clock: str
    frame: int
    team: str
    team_name: str
    opponent: str
    opponent_name: str
    score_before: str
    situation: str
    players: list[PlayerRef] = []
    facts: list[Fact] = []
    context_facts: list[Fact] = []
    importance: float = 0.0

    def all_facts(self) -> list[Fact]:
        return [*self.facts, *self.context_facts]

    def names(self) -> set[str]:
        return {p.name for p in self.players} | {self.team_name, self.opponent_name}

    def _scores(self) -> tuple[dict[str, int], dict[str, int] | None]:
        """(before, after) as {club name: goals}; after only for a goal."""
        m = re.fullmatch(r"(\w+) (\d+)-(\d+) (\w+)", self.score_before)
        if not m:
            return {}, None
        names = {self.team: self.team_name, self.opponent: self.opponent_name}
        before = {names.get(m[1], m[1]): int(m[2]), names.get(m[4], m[4]): int(m[3])}
        if self.type != "goal":
            return before, None
        return before, {k: v + (k == self.team_name) for k, v in before.items()}

    @staticmethod
    def _score_words(score: dict[str, int]) -> str:
        (a, x), (b, y) = score.items()
        state = "level" if x == y else f"{a if x > y else b} lead by {abs(x - y)}"
        return f"{a} {x}, {b} {y} ({state})"

    def score_lines(self) -> list[str]:
        """Spelled out: a bare "EAS 0-2 NOR" gets misread ("narrows the gap to 1-2")."""
        before, after = self._scores()
        if not before:
            return [f"Score before: {self.score_before}."]
        lines = [f"Score before the moment: {self._score_words(before)}."]
        if after:
            lines.append(f"Score after this goal: {self._score_words(after)}.")
        return lines

    def clock_and_score(self) -> str:
        """Text whose numbers (clock, scores, leads) any agent may repeat."""
        return " ".join([self.clock, self.score_before, *self.score_lines()])

    def to_prompt(self) -> str:
        lines = [f"Moment {self.moment_id}: {self.type} at {self.clock}.", *self.score_lines(),
                 f"Credited team: {self.team_name} ({self.team}); opponent: {self.opponent_name} ({self.opponent}).",
                 f"Situation: {self.situation}", "Players involved:"]
        lines += [f"- {p.role}: {p.name} ({p.club}, {p.position or '?'})" for p in self.players]
        lines.append("Facts:")
        for f in self.all_facts():
            unit = f" {f.unit}" if f.unit else ""
            meaning = f" -- {f.meaning}" if f.meaning else ""
            lines.append(f"- {f.label}: {f.value}{unit}{meaning}")
        return "\n".join(lines)


class Version(BaseModel):
    """One viewer's text for a moment, recap or answer, with where it came from."""
    profile_id: str
    headline: str
    body: str
    language: str = "en"
    source: str            # model label, or "template"
    verified: bool
    attempts: int = 1
    issues: list[str] = []      # why this is a fallback (empty when the model's own text was accepted)
    rejected: list[str] = []    # every objection raised on the way, by the Verifier or the Critic


class MomentStory(BaseModel):
    facts: FactSheet
    neutral: Version
    versions: list[Version] = []


class FeedItem(BaseModel):
    moment_id: str
    type: str
    clock: str
    reasons: list[str]      # why the selector picked it for this viewer
    version: Version
    neutral: Version        # the shared explanation the version was written from


class Feed(BaseModel):
    """One viewer's personal stream for a match, in match order."""
    profile_id: str
    match_id: str
    items: list[FeedItem] = []


# --------------------------------------------------------------------------- LLM outputs


class Explanation(BaseModel):
    headline: str = Field(description="At most 10 words.")
    body: str = Field(description="2-3 sentences.")


class ContextFact(BaseModel):
    label: str = Field(description="What the number is, e.g. 'Arrowsmith season pass completion'.")
    value: str = Field(description="The number exactly as the tool returned it (rounding allowed), with unit.")
    sample_size: int = Field(description="n reported by the tool for this number.")
    tool: str = Field(description="Name of the tool that returned it.")


class ContextFacts(BaseModel):
    facts: list[ContextFact] = Field(description="At most 3 comparisons.")


class Recap(BaseModel):
    headline: str
    body: str = Field(description="4-6 sentences.")


class AnalystAnswer(BaseModel):
    answer: str = Field(description="2-5 sentences, citing numbers with their sample size n.")


class Problem(BaseModel):
    claim: str = Field(description="The words of the text that state it.")
    reason: str = Field(description="What the facts say about it (one short sentence).")
    verdict: Literal["unsupported", "acceptable"] = Field(
        description="unsupported: the facts contradict it or do not contain it. acceptable: on reflection it is fine.")


class Critique(BaseModel):
    problems: list[Problem] = Field(description="Doubtful factual claims, each with a verdict; empty if none.")

    def unsupported(self) -> list[Problem]:
        return [p for p in self.problems if p.verdict == "unsupported"]

