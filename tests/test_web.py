"""The web app serves the recorded demo from files alone."""

import json
import subprocess
import sys

from fastapi.testclient import TestClient

from presslens.paths import ROOT
from presslens.web.server import create_app


def test_page_demo_and_clips_are_served():
    c = TestClient(create_app(ROOT / "demo"))
    assert c.get("/").status_code == 200 and c.get("/static/app.js").status_code == 200
    demo = c.get("/api/demo").json()
    match = demo["matches"]["R01_RIV-STO"]
    assert len(match["viewers"]) == 5 and len(match["key_moments"]) == 5
    for m in match["moments"]:                       # every moment on the timeline has its clip
        clip = c.get(f"/api/clips/{m['clip']}").json()
        assert any(k["moment_id"] == m["moment_id"] for k in clip["moments"])
    assert c.get("/api/clips/..%2Fapp").status_code == 404


def test_every_viewer_has_a_text_for_every_key_moment():
    match = json.loads((ROOT / "demo" / "app.json").read_text())["matches"]["R01_RIV-STO"]
    for story in match["stories"].values():
        assert {v["profile_id"] for v in story["versions"]} == {v["id"] for v in match["viewers"]}


def test_server_needs_no_engine_or_model_code():
    """The slim Docker image has only FastAPI: importing the server must not pull in the pipeline."""
    code = ("import sys, presslens.web.server; "
            "print([m for m in ('numpy', 'pandas', 'duckdb', 'agent_framework', 'presslens.engine', "
            "'presslens.agents', 'presslens.mcp_server') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_live_feed_is_served_as_file_and_as_a_stream():
    c = TestClient(create_app(ROOT / "demo"))
    assert c.get("/data/app.json").json() == c.get("/api/demo").json()
    feed = [json.loads(line) for line in c.get("/data/live/R01_RIV-STO.jsonl").text.splitlines()]
    assert feed[-1]["kind"] == "state" and feed[-1]["score"] == "4-1"
    assert sum(m["kind"] == "moment" for m in feed) == 17 and not any(m["kind"] == "rejected" for m in feed)
    with c.stream("GET", "/api/live/R01_RIV-STO/stream", params={"speed": 1000, "start_frame": 56900}) as r:
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert json.loads(lines[-1][6:])["score"] == "4-1"
    assert c.get("/api/live/NOPE/stream").status_code == 404
