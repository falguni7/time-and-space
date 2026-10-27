"""The demo cache: every model answer the demo needs, recorded once, replayed offline for free.

    bin/pl python -m presslens.agents.demo build --backend azure   # paid, once
    bin/pl python -m presslens.agents.demo check                   # offline replay, must make no new call

`build` runs the feeds (five viewers), the pressing key moments for every viewer, recaps and analyst questions for the demo matches and
records every model call in demo/llm_cache.sqlite, plus the finished texts in demo/demo.json.
`check` runs the same requests on the offline backend: each answer must come from the cache and
match the recorded text, so the judges' build shows exactly what the video shows, without a key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ..paths import ROOT
from .evaluate import profiles_for
from .llm import LLM, chat_client
from .workflow import Orchestrator

DEMO_DIR = ROOT / "demo"
DEMO_CACHE = DEMO_DIR / "llm_cache.sqlite"
DEMO_MATCHES = ["R01_RIV-STO"]
# The pressing story of the demo match, explained for every viewer whatever their feed holds:
# Stonebridge play through Riverside's press (1'), Riverside play through Stonebridge's (30'), and the
# 81' chain - Riverside's press wins the ball, the next pass breaks Stonebridge's press, Saltoro scores.
DEMO_KEY_MOMENTS = {"R01_RIV-STO": ["R01_RIV-STO-M000", "R01_RIV-STO-M017", "R01_RIV-STO-M044",
                                    "R01_RIV-STO-M045", "R01_RIV-STO-M046"]}
KEY_MOMENT_NOTE = "a key pressing moment of the match"
DEMO_QUESTIONS = [
    "How does pressure on the passer change the threat a pass adds in each third of the pitch?",
    "Which team loses the most threat per match through turnovers under high pressure?",
    "Against which team's press do opponents make the biggest decision gap under high pressure?",
    "Is Riverside Reds' pass completion percentage under high pressure higher than the league's?",
    "Which team creates the most threat per match, and how much pressure do they face per pass?",
]


async def run_demo(llm: LLM, matches: list[str] = DEMO_MATCHES, questions: list[str] = DEMO_QUESTIONS,
                   use_mcp: bool = True) -> dict:
    out: dict = {"model": llm.label, "matches": {}, "answers": []}
    async with Orchestrator(llm, use_mcp=use_mcp) as orch:
        for m in matches:
            profiles = profiles_for(m)
            feeds = await orch.feeds(m, profiles)
            recaps = await orch.recap(m, profiles)
            noted = [p.model_copy(update={"note": KEY_MOMENT_NOTE}) for p in profiles]
            key = [await orch.explain(k, noted) for k in DEMO_KEY_MOMENTS.get(m, [])]
            out["matches"][m] = {"profiles": [p.model_dump() for p in profiles],
                                 "feeds": [f.model_dump() for f in feeds],
                                 "key_moments": [s.model_dump() for s in key],
                                 "recaps": [r.model_dump() for r in recaps]}
            print(f"  {m}: {sum(len(f.items) for f in feeds)} feed texts, {len(recaps)} recaps", flush=True)
        for q in questions:
            out["answers"].append((await orch.ask(q)).model_dump())
    out["calls"] = {"hits": llm.cache.stats.hits, "misses": llm.cache.stats.misses,
                    "offline_misses": llm.cache.stats.offline_misses}
    return out


def _texts(demo: dict) -> list[str]:
    """Everything a viewer reads, in a fixed order, for comparing a replay with the recording."""
    texts = []
    for m in demo["matches"].values():
        texts += [f"{i['version']['headline']}|{i['version']['body']}" for f in m["feeds"] for i in f["items"]]
        texts += [f"{v['headline']}|{v['body']}" for s in m["key_moments"] for v in [s["neutral"], *s["versions"]]]
        texts += [f"{r['headline']}|{r['body']}" for r in m["recaps"]]
    return texts + [a["body"] for a in demo["answers"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build or check the offline demo cache")
    ap.add_argument("command", choices=["build", "check"])
    ap.add_argument("--backend", default=None, help="build: the live backend (default: from .env)")
    ap.add_argument("--dir", type=Path, default=DEMO_DIR)
    args = ap.parse_args()
    cache, record = args.dir / "llm_cache.sqlite", args.dir / "demo.json"
    if args.command == "build":
        llm = chat_client(args.backend, cache_path=cache)
        demo = asyncio.run(run_demo(llm))
        args.dir.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(demo, indent=1, ensure_ascii=False))
        print(f"recorded {len(_texts(demo))} texts, calls {demo['calls']} -> {args.dir}")
        return
    llm = chat_client("offline", cache_path=cache)
    replay = asyncio.run(run_demo(llm))
    recorded = json.loads(record.read_text())
    same = _texts(replay) == _texts(recorded)
    print(f"offline replay: calls {replay['calls']}, identical texts: {same}")
    sys.exit(0 if same and replay["calls"]["offline_misses"] == 0 else 1)


if __name__ == "__main__":
    main()
