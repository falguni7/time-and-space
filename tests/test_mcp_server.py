import asyncio
import json

import pytest

from presslens.engine import data
from presslens.mcp_server import tools as T

BUILT = (data.MATCHES / "index.parquet").exists() and (data.MODELS / "moments.parquet").exists()
needs_engine = pytest.mark.skipif(not BUILT, reason="season or engine outputs not built")
HIDDEN = ("tactics", "attributes", "press_intensity", "decision_quality", "composure", "truth")


def _no_hidden_truth(out):
    text = json.dumps(out)
    assert not any(word in text for word in HIDDEN), "a tool leaked simulation ground truth"


def test_broadcast_clock():
    assert [T.clock(*a) for a in [(0.2, 1), (29.92, 1), (45.0, 1), (46.08, 1), (45.3, 2), (92.4, 2)]] == \
        ["1'", "30'", "45'", "45+2'", "46'", "90+3'"]


def test_server_registers_every_tool_read_only():
    from presslens.mcp_server.server import build_server
    listed = asyncio.run(build_server().list_tools())
    assert {t.name for t in listed} == {f.__name__ for f in T.TOOLS}
    assert all(t.annotations.read_only_hint for t in listed)


@needs_engine
def test_moment_tools_chain_to_the_situation():
    mid = T.list_matches()["matches"][0]["match_id"]
    listed = T.list_moments(mid, limit=60)["moments"]
    assert listed and [m["frame"] for m in listed] == sorted(m["frame"] for m in listed)
    trap = next((m for m in listed if m["type"] in ("press_trap", "missed_option")), None)
    if trap is None:
        pytest.skip("no pass moment in this match")
    card = T.get_moment(trap["moment_id"])
    assert card["clock"] == trap["clock"] and card["score_before"].count("-") == 1
    opts = T.pass_options(mid, card["frame"])
    assert opts["frame"] == card["frame"]
    assert sum(o["chosen"] for o in opts["options"]) == 1 and sum(o["best"] for o in opts["options"]) == 1
    assert all(0 < o["xpass"] < 1 for o in opts["options"])
    p = T.pressure_at(mid, card["frame"])
    assert p["carrier"]["player_id"] == opts["passer"]["player_id"] and 0 <= p["pressure"] <= 1
    for out in (listed, card, opts, p):
        _no_hidden_truth(out)


@needs_engine
def test_summary_and_profile_carry_sample_sizes_and_no_truth():
    mid = T.list_matches()["matches"][0]["match_id"]
    s = T.match_summary(mid)
    home, away = (int(x) for x in s["score"].split("-"))
    assert len(s["goals"]) == home + away and set(s["teams"]) == {s["home"]["club_id"], s["away"]["club_id"]}
    pid = s["top_threat_creators"][0]["player_id"]
    prof = T.player_profile(pid)
    assert all("n" in v for v in prof["metrics"].values() if isinstance(v, dict))
    assert "ci_low" in prof["metrics"]["pass_completion_pct"]
    for out in (s, prof, T.list_metrics()):
        _no_hidden_truth(out)


@needs_engine
def test_bad_arguments_explain_the_allowed_values():
    with pytest.raises(ValueError, match="choose from"):
        T.query_metrics("goals_scored_by_vibes")
    with pytest.raises(ValueError, match="allowed"):
        T.query_metrics("shots", ["pressure_band"])
    with pytest.raises(ValueError, match="list_moments"):
        T.get_moment("R99_XXX-YYY-M000")
    with pytest.raises(ValueError, match="choose from"):
        T.list_moments("R01_EAS-NOR", ["own_goal"])


@needs_engine
def test_agent_framework_reaches_the_server_over_stdio():
    from presslens.mcp_server.connect import metrics_tool

    async def call():
        async with metrics_tool(allowed_tools=["list_matches"]) as mcp:
            assert [f.name for f in mcp.functions] == ["list_matches"]
            return await mcp.functions[0].invoke(arguments={"team": "RIV"})

    out = json.loads(asyncio.run(call())[0].text)
    assert len(out["matches"]) == 14 and all("RIV" in (m["home"], m["away"]) for m in out["matches"])


def test_pressure_is_read_as_the_pressure_band():
    by_band = T.query_metrics("pass_completion_pct", group_by=["pressure"], filters={"team": "RIV"})
    assert {r["pressure_band"] for r in by_band["rows"]} == {"none", "some", "high"}
