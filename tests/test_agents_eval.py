"""Evaluation report and demo cache replay (offline, scripted model)."""

import asyncio

from test_agents_team import GOOD, INVENTED, NO_PROBLEMS, ScriptedClient

from presslens.agents import evaluate as E
from presslens.agents import llm as L
from presslens.agents.demo import _texts, run_demo
from presslens.agents.workflow import Orchestrator

MATCH = "R01_EAS-NOR"
RECAP = {"headline": "Northgate Blues win", "body": "Viktor Ezeani scored four."}


def _llm(client):
    return L.LLM(client=client, backend="test", model="gpt-4.1-mini", cache=None)


def test_report_counts_outcomes_and_objections():
    client = ScriptedClient({"explainer": [GOOD], "persona": [GOOD], "critic": [INVENTED, NO_PROBLEMS]})
    orch = Orchestrator(_llm(client), use_mcp=False)
    profiles = [p.model_copy(update={"alert_budget": 3}) for p in E.profiles_for(MATCH)]

    async def run():
        async with orch:
            return await orch.feeds(MATCH, profiles)
    feeds = asyncio.run(run())
    r = E.summarise({MATCH: feeds}, {MATCH: profiles}, {MATCH: 1.0})
    assert r["viewer_texts"] == 15 and r["moments_explained"] <= 15
    for rates in (r["neutral"], r["versions"], *r["by_kind"].values()):
        assert abs(rates["first_try"] + rates["after_retry"] + rates["fallback"] - 1) < 0.01
    assert r["objections"]["critic"] >= 1 and "verifier" not in r["objections"]
    assert set(r["by_language"]) == {"en", "de", "es"}
    assert 0 <= r["feed_overlap_between_viewers"] <= 1


def test_outcome_and_checker_labels():
    from presslens.agents.schemas import Version
    v = Version(profile_id="x", headline="h", body="b", source="m", verified=True)
    assert E.outcome(v) == "first_try"
    assert E.outcome(v.model_copy(update={"attempts": 2})) == "after_retry"
    assert E.outcome(v.model_copy(update={"issues": ["German not available; showing English"]})) == "fallback"
    assert E.checker('unsupported claim "x": y') == "critic"
    assert E.verifier_kind("these numbers are not in the facts: 9") == "number not in facts"


def test_demo_replays_offline_from_the_recorded_cache(tmp_path):
    cache = tmp_path / "llm_cache.sqlite"
    script = {"explainer": [GOOD], "persona": [GOOD], "recap": [RECAP]}
    recorder = ScriptedClient(script, middleware=[L.CacheMiddleware(L.ResponseCache(cache), "gpt-4.1-mini")])
    recorded = asyncio.run(run_demo(L.LLM(recorder, "test", "gpt-4.1-mini", recorder.chat_middleware[0]),
                                    matches=[MATCH], questions=[], use_mcp=False))
    mw = L.CacheMiddleware(L.ResponseCache(cache), "gpt-4.1-mini")
    offline = L.OfflineChatClient(middleware=[mw])
    replay = asyncio.run(run_demo(L.LLM(offline, "offline", "gpt-4.1-mini", mw), matches=[MATCH], questions=[],
                                  use_mcp=False))
    assert replay["calls"]["offline_misses"] == 0 and replay["calls"]["hits"] == recorded["calls"]["misses"] > 0
    assert _texts(replay) == _texts(recorded)
