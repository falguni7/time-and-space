"""The relevance selector and personal feeds (offline, scripted model)."""

import asyncio

from test_agents_team import GOOD, ScriptedClient

from presslens.agents import llm as L
from presslens.agents.personas import Profile, brief
from presslens.agents.selector import good_for, plan_feeds, select_moments
from presslens.agents.workflow import Orchestrator
from presslens.engine import moments as engine_moments

MATCH = "R01_EAS-NOR"   # Eastvale Whites 0-4 Northgate Blues, all four goals by Ezeani (NOR-18)
CASUAL = Profile(id="casual", kind="casual")
ANALYST = Profile(id="analyst", kind="analyst", detail="deep")
NOR_FAN = Profile(id="nor", kind="supporter", focus_club="NOR")
EAS_FAN = Profile(id="eas", kind="supporter", focus_club="EAS", language="de")
EZEANI = Profile(id="ezeani", kind="player_follower", focus_player="NOR-18")


def _cards():
    return {c["moment_id"]: c for c in engine_moments.load_moments(MATCH)}


def test_budget_order_and_goals_first():
    cards = _cards()
    goals = {m for m, c in cards.items() if c["type"] == "goal"}
    for p in (CASUAL, ANALYST, NOR_FAN, EAS_FAN, EZEANI):
        picks = select_moments(MATCH, p)
        assert len(picks) == p.alert_budget
        assert [(k.period, k.minute) for k in picks] == sorted((k.period, k.minute) for k in picks)
        assert goals <= {k.moment_id for k in picks}
        assert all(k.reasons for k in picks)
    assert len(select_moments(MATCH, CASUAL, budget=2)) == 2
    assert {k.type for k in select_moments(MATCH, CASUAL, budget=2)} == {"goal"}


def test_viewers_get_different_feeds():
    feeds = {p.id: [k.moment_id for k in select_moments(MATCH, p)] for p in (CASUAL, ANALYST, NOR_FAN, EAS_FAN)}
    assert len({tuple(f) for f in feeds.values()}) == 4
    cards = _cards()
    eas_good = sum(good_for(cards[m], "EAS") for m in feeds["eas"])
    nor_good = sum(good_for(cards[m], "NOR") for m in feeds["nor"])
    assert eas_good >= 4 and nor_good == 8     # the losing side's fan still gets their best moments
    analyst_types = {cards[m]["type"] for m in feeds["analyst"]}
    assert analyst_types & {"missed_option", "press_trap", "press_broken"}


def test_followed_player_is_named_as_the_reason():
    picks = select_moments(MATCH, EZEANI)
    involved = [k for k in picks if "Viktor Ezeani is involved" in k.reasons]
    assert len(involved) >= 5     # four goals and a big chance
    assert any("good moment for Eastvale Whites" in r for k in select_moments(MATCH, EAS_FAN) for r in k.reasons)


def test_plan_explains_each_moment_once():
    plan = plan_feeds(MATCH, [CASUAL, NOR_FAN, EAS_FAN])
    assert sum(len(v) for v in plan.values()) == 24 and len(plan) < 24


def test_note_reaches_the_persona_brief():
    note = NOR_FAN.model_copy(update={"note": "a goal; good moment for Northgate Blues"})
    assert "picked for them because: a goal; good moment" in brief(note, "Northgate Blues")


def test_feeds_share_explanations():
    client = ScriptedClient({"explainer": [GOOD], "persona": [GOOD]})
    orch = Orchestrator(L.LLM(client=client, backend="test", model="fake", cache=None), use_mcp=False)
    profiles = [CASUAL.model_copy(update={"alert_budget": 3}), EAS_FAN.model_copy(update={"alert_budget": 3})]

    async def run():
        async with orch:
            return await orch.feeds(MATCH, profiles)
    feeds = asyncio.run(run())
    assert [f.profile_id for f in feeds] == ["casual", "eas"]
    assert all(len(f.items) == 3 and all(i.version.verified for i in f.items) for f in feeds)
    distinct = {i.moment_id for f in feeds for i in f.items}
    assert client.calls.count("explainer") == len(distinct) < 6
    assert client.calls.count("persona") == 6
    assert feeds[1].items[0].version.language == "de" and feeds[1].items[0].reasons


def test_scores_are_spelled_out_for_the_writers():
    from presslens.agents.facts import build_fact_sheet
    sheet = build_fact_sheet("R01_EAS-NOR-M013")      # Ezeani's third, at 46'
    assert sheet.score_lines() == [
        "Score before the moment: Eastvale Whites 0, Northgate Blues 2 (Northgate Blues lead by 2).",
        "Score after this goal: Eastvale Whites 0, Northgate Blues 3 (Northgate Blues lead by 3).",
    ]
    assert "lead by 3" in sheet.to_prompt()


def test_pressing_moment_is_compared_with_the_league_in_its_zone():
    from presslens.agents.facts import build_fact_sheet
    sheet = build_fact_sheet("R01_RIV-STO-M045")      # Holloway-Penn plays through Stonebridge's press, 81'
    facts = {f.key: f for f in sheet.facts}
    assert {"league_xt_zone", "league_completion_zone", "league_xt_zone_free"} <= set(facts)
    assert "middle third, high pressure" in facts["league_xt_zone"].label
    # a pass that added 0.04 xT under high pressure beats almost every league pass in that spot
    assert facts["league_xt_zone"].value < 0 < facts["xt_added"].value
    assert "higher than 99%" in facts["league_xt_zone"].meaning
