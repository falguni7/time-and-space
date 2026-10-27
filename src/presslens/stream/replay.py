"""Replay a simulated match as a live provider stream, and record the pipeline's timed feed.

    python -m presslens.stream replay R01_RIV-STO --speed 20     # watch it live in the terminal
    python -m presslens.stream feed                               # record the demo feed -> demo/live/

The source sends the provider's public events one by one, at match speed times `speed`. The live
interpreter validates each one and updates the state; key moments are announced as their time comes,
with the viewers whose feed holds them. The recorded feed (JSON Lines) is what the web app's Live mode
plays, and what a broadcast graphics system would consume as a synchronised overlay feed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pandas as pd

from ..engine import data
from .live import LiveMatch

FPS = 10


def provider_events(match_id: str) -> Iterator[dict]:
    """The match's public events in the order a provider sends them."""
    df = pd.read_parquet(data.MATCHES / match_id / "events.parquet").sort_values(["frame", "event_id"])
    for rec in df.to_dict("records"):
        yield {k: v for k, v in rec.items() if v is not None}


async def live_events(match_id: str, speed: float = 1.0) -> AsyncIterator[dict]:
    """Provider events, released in match time (speed 1 = real time)."""
    start, first = time.monotonic(), None
    for ev in provider_events(match_id):
        first = ev["frame"] if first is None else first
        wait = (ev["frame"] - first) / FPS / speed - (time.monotonic() - start)
        if wait > 0:
            await asyncio.sleep(wait)
        yield ev


def record_feed(match_id: str, moments: list[dict]) -> list[dict]:
    """Run the whole match through the live interpreter (as fast as possible) and keep every message."""
    live = LiveMatch(match_id, moments)
    feed: list[dict] = []
    for ev in provider_events(match_id):
        feed += live.ingest(ev)
    feed.append(live.state())
    return feed


def demo_moments(app: dict, match_id: str) -> list[dict]:
    """The demo's moments with the viewers each one is for (key moments: everyone; others: their feed)."""
    match = app["matches"][match_id]
    viewers = [v["id"] for v in match["viewers"]]
    out = []
    for m in match["moments"]:
        for_ = viewers if m["moment_id"] in match["key_moments"] else [
            v for v in viewers if any(i["moment_id"] == m["moment_id"] for i in match["feeds"].get(v, []))]
        out.append({**m, "for": for_})
    return out


def write_feed(feed: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(m, separators=(",", ":"), ensure_ascii=False) + "\n" for m in feed))


async def watch(match_id: str, speed: float, moments: list[dict]) -> None:
    """Print the live feed in the terminal as the match plays."""
    live = LiveMatch(match_id, moments)
    async for ev in live_events(match_id, speed):
        for msg in live.ingest(ev):
            if msg["kind"] == "moment":
                print(f"{msg['clock']:>7}  ** {msg['type']} ({msg['team']})  {msg['moment_id']}  "
                      f"-> {', '.join(msg['for']) or 'no viewer'}", flush=True)
            elif msg["kind"] == "event" and msg["event"]["type"] in ("goal", "shot", "substitution"):
                e = msg["event"]
                print(f"{msg['clock']:>7}  {e['type']:<12} {e.get('team', '')} {e.get('player_id', '')}", flush=True)
            elif msg["kind"] == "state":
                h, a = (msg["teams"][t] for t in (live.home, live.away))
                print(f"{msg['clock']:>7}  {live.home} {msg['score']} {live.away} | possession {h['possession_pct']}-"
                      f"{a['possession_pct']} | xT {h['xt_created']}-{a['xt_created']} | pressure faced "
                      f"{h['pressure_faced']}-{a['pressure_faced']}", flush=True)
            elif msg["kind"] == "rejected":
                print(f"         rejected: {msg['reason']}", flush=True)


__all__ = ["demo_moments", "live_events", "provider_events", "record_feed", "watch", "write_feed"]
