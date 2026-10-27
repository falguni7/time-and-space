"""The agent team and orchestrator, offline: a scripted fake model stands in for Azure."""

import asyncio
import json

import pytest
from agent_framework import Agent, ChatResponse, Message

from presslens.agents import llm as L
from presslens.agents import verify as V
from presslens.agents.personas import Profile
from presslens.agents.schemas import Explanation
from presslens.agents.team import ask_structured
from presslens.agents.workflow import Orchestrator
from presslens.mcp_server import server as S
from presslens.mcp_server import tools as T

MOMENT = "R01_EAS-NOR-M008"   # Ezeani's goal at 30'
GOOD = {"headline": "A goal for the visitors", "body": "At 30' the striker finished a good chance."}
BAD = {"headline": "A goal", "body": "Scored with a 99.9% chance."}   # 99.9 is in no fact
NO_PROBLEMS = {"problems": []}
INVENTED = {"problems": [{"claim": "the striker finished", "reason": "the facts do not say who shot",
                          "verdict": "unsupported"},
                         {"claim": "a good chance", "reason": "xG 0.19 is decent", "verdict": "acceptable"}]}
PROFILES = [Profile(id="casual", kind="casual"), Profile(id="fan_de", kind="supporter", focus_club="NOR", language="de")]


class ScriptedClient(L.OfflineChatClient):
    """Answers by agent: `script[agent_name]` is a list of JSON answers, the last one repeats.
    The Critic has no objections unless scripted."""

    def __init__(self, script: dict[str, list], **kwargs):
        super().__init__(**kwargs)
        self.script = {"critic": [NO_PROBLEMS], **script}
        self.calls: list[str] = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        who = next(k for k in self.script if options.get("instructions", "").find(_ROLE[k]) >= 0)
        self.calls.append(who)
        answers = self.script[who]
        answer = answers[min(self.calls.count(who), len(answers)) - 1]
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return ChatResponse(messages=[Message(role="assistant", contents=[text])], model=self.model)


_ROLE = {"explainer": "You are the Explainer", "persona": "You are the Persona", "recap": "You are the Recap",
         "critic": "You are the Critic"}


def _orch(script) -> tuple[Orchestrator, ScriptedClient]:
    client = ScriptedClient(script)
    return Orchestrator(L.LLM(client=client, backend="test", model="fake", cache=None), use_mcp=False), client


def _explain(script, profiles=PROFILES):
    orch, client = _orch(script)

    async def run():
        async with orch:
            return await orch.explain(MOMENT, profiles)
    return asyncio.run(run()), client


# ------------------------------------------------------------------ Verifier


def test_clock_minutes_and_score_are_trusted():
    assert V.check_text("At 30' and in the 30th minute, 2-1.", [], set(), clock_and_score="30' EAS 2-1 NOR").ok
    assert not V.check_text("In the 31st minute.", [], set(), clock_and_score="30' EAS 2-1 NOR").ok


def test_differences_only_when_allowed():
    text = "345.0 against 331.36, a gap of 13.64 per match."
    assert not V.check_text(text, [345.0, 331.3571], set()).ok
    assert V.check_text(text, [345.0, 331.3571], set(), differences=True).ok


def test_decimal_commas_and_percentages():
    assert V.check_text("Eine Chance von 19 % (xG 0,191).", [0.191], set()).ok


# ------------------------------------------------------------------ structured answers


def test_doubled_json_answer_is_recovered():
    doubled = json.dumps(GOOD) + "\n" + json.dumps(GOOD)
    client = ScriptedClient({"explainer": [doubled]})
    agent = Agent(client=client, name="Explainer", instructions=_ROLE["explainer"])
    value, _ = asyncio.run(ask_structured(agent, "x", Explanation))
    assert value == Explanation(**GOOD)


# ------------------------------------------------------------------ orchestrator


def test_verified_story_for_every_profile():
    story, client = _explain({"explainer": [GOOD], "persona": [GOOD]})
    assert story.neutral.verified and story.neutral.source == "fake (test)" and story.neutral.attempts == 1
    assert [v.profile_id for v in story.versions] == ["casual", "fan_de"]
    assert all(v.verified and v.language == p.language for v, p in zip(story.versions, PROFILES))
    assert client.calls.count("explainer") == 1 and client.calls.count("persona") == 2
    assert client.calls.count("critic") == 3      # the neutral draft and each persona version


def test_critic_objection_sends_the_draft_back():
    story, client = _explain({"explainer": [GOOD], "critic": [INVENTED, NO_PROBLEMS]}, profiles=[])
    assert story.neutral.source == "fake (test)" and story.neutral.attempts == 2
    assert client.calls == ["explainer", "critic", "explainer", "critic"]
    assert story.neutral.rejected == ['unsupported claim "the striker finished": the facts do not say who shot']


def test_critic_rejecting_twice_gives_the_template():
    story, _ = _explain({"explainer": [GOOD], "critic": [INVENTED]}, profiles=[])
    assert story.neutral.source == "template"
    assert story.neutral.issues == ['unsupported claim "the striker finished": the facts do not say who shot']


def test_critic_only_sees_text_that_passed_the_rules():
    _, client = _explain({"explainer": [BAD], "persona": [BAD]})
    assert "critic" not in client.calls


def test_rejected_draft_is_retried_with_feedback():
    story, client = _explain({"explainer": [BAD, GOOD], "persona": [GOOD]}, profiles=[])
    assert story.neutral.source == "fake (test)" and story.neutral.attempts == 2
    assert client.calls.count("explainer") == 2


def test_template_after_two_rejections_and_persona_fallback():
    story, _ = _explain({"explainer": [BAD], "persona": [BAD]})
    assert story.neutral.source == "template" and "99.9" in story.neutral.issues[0]
    assert "Viktor Ezeani" in story.neutral.headline
    casual, fan_de = story.versions
    assert casual.body == story.neutral.body          # falls back to the verified neutral text
    assert fan_de.issues[-1] == "German not available; showing English"


def test_offline_misses_fall_back_to_templates():
    orch = Orchestrator(L.LLM(client=L.OfflineChatClient(), backend="offline", model="none", cache=None),
                        use_mcp=False)

    async def run():
        async with orch:
            story = await orch.explain(MOMENT, PROFILES)
            recap = await orch.recap("R01_EAS-NOR", PROFILES[:1])
            answer = await orch.ask("Who presses most?")
            return story, recap, answer
    story, recap, answer = asyncio.run(run())
    assert story.neutral.source == "template" and len(story.versions) == 2
    assert recap[0].source == "template" and recap[0].headline == "Eastvale Whites 0-4 Northgate Blues"
    assert not answer.verified


def test_recap_is_verified_against_summary_and_moments():
    orch, _ = _orch({"recap": [{"headline": "Northgate Blues win", "body": "Viktor Ezeani scored four, at 30' among others."}]})

    async def run():
        async with orch:
            return await orch.recap("R01_EAS-NOR", PROFILES[:1], top_moments=3)
    (recap,) = asyncio.run(run())
    assert recap.verified and recap.source == "fake (test)"


# ------------------------------------------------------------------ MCP tools the agents rely on


def test_team_totals_are_compared_per_match_with_names():
    rows = T.query_metrics("pressures_applied", ["team"])["rows"]
    assert all(r["per_match"] == pytest.approx(r["value"] / r["matches"], abs=1e-3) for r in rows)
    assert rows == sorted(rows, key=lambda r: -r["per_match"]) and rows[0]["team_name"]


def test_players_and_clubs_by_name():
    pid = "NOR-18"
    name = T._name(pid)
    assert T.player_profile(name.upper())["player_id"] == pid
    by_name = T.query_metrics("passes", None, {"player_id": name, "team": T._clubs()["NOR"]})["rows"]
    assert by_name == T.query_metrics("passes", None, {"player_id": pid, "team": "NOR"})["rows"]


def test_server_returns_argument_errors_to_the_model():
    wrapped = S._explain_errors(T.query_metrics)
    out = wrapped("passes", ["pressure_level"])
    assert "allowed" in out["error"] and "pressure_band" in out["error"]


def test_followed_player_may_not_be_named_in_a_moment_without_them():
    from presslens.agents.facts import build_fact_sheet
    from presslens.agents.team import Team
    sheet = build_fact_sheet("R01_RIV-STO-M044")          # Saltoro is not part of this press trap
    names = Team._persona_names(None, Profile(id="f", kind="player_follower", focus_player="RIV-10"), sheet)
    assert "Matteo Saltoro" not in names
    assert V.check_text("Matteo Saltoro and the Reds win the ball back", [], names).issues
