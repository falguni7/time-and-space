"""The agents and their verify-retry-fallback steps. workflow.py wires them into one graph.

Each step returns a verified Version: the model's text if it passes the Verifier (after at most
one retry with the Verifier's feedback), otherwise a template built from the facts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_framework import Agent

from ..mcp_server import tools as T
from . import prompts
from .facts import template_explanation, template_version
from .llm import LLM, is_offline_miss
from .personas import Profile, brief
from .schemas import (
    AnalystAnswer,
    ContextFacts,
    Critique,
    Explanation,
    Fact,
    FactSheet,
    Recap,
    Version,
)
from .verify import (
    Check,
    check_moment_text,
    check_text,
    numbers_in,
    sheet_numbers,
    tool_result_numbers,
)

log = logging.getLogger(__name__)
MAX_ATTEMPTS = 2
TEMPERATURE = 0.3
STATS_TOOLS = ["player_profile", "query_metrics"]
ANALYST_TOOLS = ["list_metrics", "query_metrics", "list_matches", "match_summary", "player_profile"]


async def ask_structured(agent: Agent, prompt: str, model: type,
                         temperature: float = TEMPERATURE) -> tuple[Any | None, Any]:
    """Run an agent for a structured answer; (None, response) when offline or unparseable."""
    try:
        response = await agent.run(prompt, options={"response_format": model, "temperature": temperature})
    except Exception as exc:  # noqa: BLE001 - network, rate limit after retries, content filter ...
        log.warning("%s failed: %s", agent.name, exc)
        return None, None
    if is_offline_miss(response):
        return None, response
    try:
        return response.value, response
    except Exception as exc:  # noqa: BLE001 - any parse/validation failure
        # Seen in practice: the JSON object followed by a second copy. Keep the first object.
        try:
            first, _ = json.JSONDecoder().raw_decode(response.text.strip())
            return model.model_validate(first), response
        except Exception:  # noqa: BLE001
            log.warning("%s returned unparseable output: %s", agent.name, exc)
            return None, response


class Team:
    """All agents sharing one chat client (and its cache) and one MCP connection."""

    def __init__(self, llm: LLM, mcp=None):
        self.llm = llm
        self.mcp = mcp
        c = llm.client
        self.stats = Agent(client=c, name="Stats", instructions=prompts.STATS, tools=self._tools(STATS_TOOLS))
        self.explainer = Agent(client=c, name="Explainer", instructions=prompts.EXPLAINER)
        self.critic = Agent(client=c, name="Critic", instructions=prompts.CRITIC)
        self.analyst = Agent(client=c, name="Analyst", instructions=prompts.ANALYST,
                             tools=self._tools(ANALYST_TOOLS))

    def _tools(self, names: list[str]) -> list:
        """Only the MCP tools an agent needs: every tool schema is resent on each model call."""
        if self.mcp is None:
            return []
        return [f for f in self.mcp.functions if f.name in names]

    # ------------------------------------------------------------------ Stats

    async def context(self, sheet: FactSheet) -> list[Fact]:
        """Up to 3 tool-backed comparisons; any whose numbers are not in the tool results is dropped."""
        if self.mcp is None:
            return []
        value, response = await ask_structured(self.stats, sheet.to_prompt(), ContextFacts)
        if value is None:
            return []
        seen = tool_result_numbers(response.messages)
        kept = []
        for i, f in enumerate(value.facts[:3]):
            if not numbers_in(str(f.value)) or not f.sample_size:
                continue  # "not available" is not context
            text = f"{f.label} {f.value} {f.sample_size}"
            if check_text(text, seen, sheet.names(), max_words=40).ok:
                kept.append(Fact(key=f"context_{i}", label=f.label, value=f.value, unit=f"(n={f.sample_size})",
                                 meaning=f"from {f.tool.removeprefix('functions.')}"))
            else:
                log.info("Stats fact dropped (not grounded): %s", text)
        return kept

    # ------------------------------------------------------------------ Critic

    async def critique(self, facts: str, text: str, reader: str = "a neutral reader") -> list[str]:
        """The Critic's objections to a text whose numbers and names already passed the Verifier.
        It can only reject. If a live Critic cannot answer (error, rate limit), the text is not let
        through unreviewed: it counts as an objection, so the writer retries or the fallback is shown.
        The offline backend has no Critic; there the text keeps its rule check."""
        prompt = f"Facts:\n{facts}\n\nThe text is written for: {reader}\n\nText to check:\n{text}"
        value, _ = await ask_structured(self.critic, prompt, Critique, temperature=0.0)
        if value is None:
            return [] if self.llm.backend == "offline" else ["the Critic could not review this text"]
        return [f'unsupported claim "{p.claim}": {p.reason}' for p in value.unsupported()]

    async def review(self, check: Check, facts: str, text: str, reader: str = "a neutral reader") -> Check:
        """Rule check first (cheap, exact); only text that passes it goes to the Critic."""
        if not check.ok:
            return check
        problems = await self.critique(facts, text, reader)
        return Check(not problems, problems)

    # ------------------------------------------------------------------ Explainer

    async def explain(self, sheet: FactSheet, feedback: str | None = None) -> tuple[Explanation | None, str]:
        prompt = sheet.to_prompt()
        if feedback:
            prompt += f"\n\nYour previous answer was rejected. {feedback}"
        value, _ = await ask_structured(self.explainer, prompt, Explanation)
        return value, self.llm.label

    # ------------------------------------------------------------------ Persona / Localizer

    def _persona_agent(self, profile: Profile, sheet: FactSheet) -> Agent:
        instructions = prompts.PERSONA.format(language=profile.language_name, brief=self._reader(profile),
                                              max_words=profile.max_words())
        return Agent(client=self.llm.client, name=f"Persona[{profile.id}]", instructions=instructions)

    def _reader(self, profile: Profile) -> str:
        """The viewer brief, with the followed player's club so nobody gets put in the wrong team."""
        club = T._clubs().get(profile.focus_club) if profile.focus_club else None
        player = T._player(profile.focus_player) if profile.focus_player else None
        return brief(profile, club, player and player["name"],
                     player and T._clubs().get(player["club"], player["club"]))

    def _persona_names(self, profile: Profile, sheet: FactSheet) -> set[str]:
        # The followed player is not added: naming them in a moment they are not part of is an
        # invented involvement ("Saltoro and the Reds win the ball back").
        names = set(sheet.names())
        if profile.focus_club:
            names.add(T._clubs().get(profile.focus_club, profile.focus_club))
        return names

    async def personalise(self, sheet: FactSheet, neutral: Version, profile: Profile) -> Version:
        agent = self._persona_agent(profile, sheet)
        base = (f"{sheet.to_prompt()}\n\nVerified explanation to rewrite:\nHeadline: {neutral.headline}\n"
                f"Body: {neutral.body}")
        feedback, issues, rejected = None, [], []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = base + (f"\n\nYour previous answer was rejected. {feedback}" if feedback else "")
            value, _ = await ask_structured(agent, prompt, Explanation)
            if value is None:
                break
            text = f"{value.headline}\n{value.body}"
            check = check_text(text, sheet_numbers(sheet) + numbers_in(neutral.body),
                               self._persona_names(profile, sheet), profile.max_words() + 25,
                               clock_and_score=sheet.clock_and_score())
            check = await self.review(check, sheet.to_prompt(), text, self._reader(profile))
            if check.ok:
                return Version(profile_id=profile.id, headline=value.headline, body=value.body,
                               language=profile.language, source=self.llm.label, verified=True, attempts=attempt,
                               rejected=rejected)
            feedback, issues = check.feedback(), check.issues
            rejected += check.issues
        # Fallback: the verified neutral text (English), marked so the UI can say so.
        out = neutral.model_copy(update={"profile_id": profile.id, "issues": issues, "rejected": rejected})
        if profile.language != "en":
            out.issues = [*issues, f"{profile.language_name} not available; showing English"]
        return out

    # ------------------------------------------------------------------ Recap

    async def recap(self, match_id: str, sheets: list[FactSheet], profile: Profile) -> Version:
        summary = T.match_summary(match_id)
        moments = "\n\n".join(s.to_prompt() for s in sheets)
        prompt = f"Match summary (JSON):\n{json.dumps(summary)}\n\nKey moments:\n{moments}"
        names = {summary["home"]["name"], summary["away"]["name"]}
        names |= {g["scorer"] for g in summary["goals"] if g["scorer"]}
        names |= {p["name"] for p in summary["top_threat_creators"]}
        for s in sheets:
            names |= s.names()
        trusted = numbers_in(json.dumps(summary)) + [n for s in sheets for n in sheet_numbers(s)]
        clocks = " ".join([g["clock"] for g in summary["goals"]] + [s.clock for s in sheets] + [summary["score"]])
        agent = Agent(client=self.llm.client, name="Recap", instructions=prompts.RECAP.format(
            language=profile.language_name, brief=self._reader(profile)))
        feedback, issues, rejected = None, [], []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            value, _ = await ask_structured(
                agent, prompt + (f"\n\nYour previous answer was rejected. {feedback}" if feedback else ""), Recap)
            if value is None:
                break
            text = f"{value.headline}\n{value.body}"
            check = check_text(text, trusted, names, 160, clock_and_score=clocks)
            check = await self.review(check, prompt, text, self._reader(profile))
            if check.ok:
                return Version(profile_id=profile.id, headline=value.headline, body=value.body,
                               language=profile.language, source=self.llm.label, verified=True, attempts=attempt,
                               rejected=rejected)
            feedback, issues = check.feedback(), check.issues
            rejected += check.issues
        return template_recap(summary, profile.id, issues).model_copy(update={"rejected": rejected})

    # ------------------------------------------------------------------ Analyst

    async def ask(self, question: str) -> Version:
        if self.mcp is None:
            return Version(profile_id="analyst", headline="Analyst unavailable",
                           body="The metrics tools are not connected.", source="none", verified=False)
        feedback, issues, rejected = None, [], []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            prompt = question + (f"\n\nYour previous answer was rejected. {feedback}" if feedback else "")
            value, response = await ask_structured(self.analyst, prompt, AnalystAnswer)
            if value is None:
                break
            seen_text = " ".join(str(getattr(c, "result", "")) for m in response.messages for c in m.contents
                                 if getattr(c, "type", None) == "function_result")
            seen_text += " " + question  # a name the user asked about may be repeated
            names = {n for n in T._clubs().values() if n in seen_text}
            names |= {p["name"] for p in T._roster().values() if p["name"] in seen_text}
            check = check_text(value.answer, tool_result_numbers(response.messages) + numbers_in(question), names, 200,
                               differences=True)
            if check.ok:
                return Version(profile_id="analyst", headline=question, body=value.answer, source=self.llm.label,
                               verified=True, attempts=attempt, rejected=rejected)
            feedback, issues = check.feedback(), check.issues
            rejected += check.issues
        return Version(profile_id="analyst", headline=question, verified=False, source="none", issues=issues,
                       rejected=rejected,
                       body="I could not give a verified answer to this question from the metrics cube.")


def check_explanation(sheet: FactSheet, exp: Explanation) -> Check:
    return check_moment_text(sheet, exp.headline, exp.body, max_words=95)


def template_recap(summary: dict, profile_id: str, issues: list[str] | None = None) -> Version:
    home, away = summary["home"], summary["away"]
    goals = "; ".join(f"{g['scorer']} {g['clock']}" for g in summary["goals"]) or "no goals"
    t = summary["teams"]
    body = (f"{home['name']} {summary['score']} {away['name']}. Goals: {goals}. "
            f"Chance quality (xG): {home['name']} {t[home['club_id']]['provider_xg']}, "
            f"{away['name']} {t[away['club_id']]['provider_xg']}. Threat created (xT): "
            f"{home['name']} {t[home['club_id']]['xt_generated']}, {away['name']} {t[away['club_id']]['xt_generated']}.")
    return Version(profile_id=profile_id, headline=f"{home['name']} {summary['score']} {away['name']}", body=body,
                   source="template", verified=True, attempts=0, issues=issues or [])


__all__ = ["Team", "check_explanation", "template_explanation", "template_recap", "template_version"]
