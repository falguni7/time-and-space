"""The live stream: schema checks and an interpreter whose final state matches the engine."""

from presslens.engine import cube
from presslens.stream.live import LiveMatch
from presslens.stream.replay import provider_events, record_feed

MATCH = "R01_RIV-STO"


def test_bad_events_are_rejected_with_a_reason():
    live = LiveMatch(MATCH)
    assert live.ingest({"event_id": 1, "frame": 5, "period": 3, "minute": 0.0, "type": "pass"})[0]["kind"] == "rejected"
    out = live.ingest({"event_id": 2, "frame": 5, "period": 1, "minute": 0.0, "type": "teleport"})
    assert out[0]["kind"] == "rejected" and "unknown event type" in out[0]["reason"]


def test_live_state_ends_where_the_engine_does():
    feed = record_feed(MATCH, moments=[{"moment_id": "x", "frame": 100, "clock": "1'", "type": "press_trap",
                                        "team": "RIV", "for": ["casual"]}])
    final = feed[-1]
    assert final["score"] == "4-1" and not any(m["kind"] == "rejected" for m in feed)
    assert sum(m["kind"] == "event" for m in feed) == sum(1 for _ in provider_events(MATCH))
    xg = cube.query_metrics("provider_xg", ["team"], {"match_id": MATCH}).set_index("team")["value"]
    passes = cube.query_metrics("passes", ["team"], {"match_id": MATCH}).set_index("team")["value"]
    for team in ("RIV", "STO"):
        assert abs(final["teams"][team]["xg"] - xg[team]) < 0.01
        assert final["teams"][team]["passes"] == passes[team]
    assert final["teams"]["RIV"]["presses_won"] == 1
    moment = next(m for m in feed if m["kind"] == "moment")
    assert moment["for"] == ["casual"] and moment["frame"] == 100
