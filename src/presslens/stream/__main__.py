"""python -m presslens.stream replay <match_id> [--speed N]   |   python -m presslens.stream feed"""

from __future__ import annotations

import argparse
import asyncio
import json

from ..paths import ROOT
from .replay import demo_moments, record_feed, watch, write_feed

DEMO_DIR = ROOT / "demo"


def main() -> None:
    ap = argparse.ArgumentParser(description="Live stream: replay a match event by event")
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("replay", help="play a match live in the terminal")
    rp.add_argument("match_id")
    rp.add_argument("--speed", type=float, default=20.0, help="match seconds per real second")
    sub.add_parser("feed", help="record the demo matches' live feeds -> demo/live/<match>.jsonl")
    args = ap.parse_args()
    app = json.loads((DEMO_DIR / "app.json").read_text())
    if args.command == "replay":
        moments = demo_moments(app, args.match_id) if args.match_id in app["matches"] else []
        asyncio.run(watch(args.match_id, args.speed, moments))
        return
    for match_id in app["matches"]:
        feed = record_feed(match_id, demo_moments(app, match_id))
        path = DEMO_DIR / "live" / f"{match_id}.jsonl"
        write_feed(feed, path)
        kinds = {k: sum(m["kind"] == k for m in feed) for k in ("event", "state", "moment", "rejected")}
        print(f"{match_id}: {kinds} -> {path}")


if __name__ == "__main__":
    main()
