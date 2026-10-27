"""Evaluation: run the whole agent team on matches it was not tuned on and measure it.

    bin/pl python -m presslens.agents.evaluate --backend azure --rounds 11-14 --max-matches 8

Per match, five viewers (casual, analyst, home supporter, away supporter in German, follower of the
match's top threat creator in Spanish with plain language) get their feeds. The report measures
how often each text passed first time, after a retry, or fell back; which checker objected; how
different the personal versions are; how long they are; and what the run cost. Written to
data/eval/report.json and report.md.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from ..engine.data import TEST_ROUNDS
from ..mcp_server import tools as T
from ..paths import DATA_DIR
from .llm import LLM, chat_client
from .personas import Profile
from .schemas import Feed, Version
from .workflow import Orchestrator

EVAL_DIR = DATA_DIR / "eval"
# USD per million tokens (input, output), list price of Azure OpenAI global deployments.
PRICE_PER_M = {"gpt-4.1-mini": (0.40, 1.60)}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def profiles_for(match_id: str) -> list[Profile]:
    """The same five kinds of viewer for every match, attached to that match's clubs and star."""
    s = T.match_summary(match_id)
    star = s["top_threat_creators"][0]["player_id"]
    return [
        Profile(id="casual", kind="casual"),
        Profile(id="analyst", kind="analyst", detail="deep"),
        Profile(id="home_fan", kind="supporter", focus_club=s["home"]["club_id"]),
        Profile(id="away_fan_de", kind="supporter", focus_club=s["away"]["club_id"], language="de"),
        Profile(id="star_follower_es", kind="player_follower", focus_player=star, language="es",
                accessibility="plain"),
    ]


def checker(issue: str) -> str:
    return "critic" if issue.startswith("unsupported claim") else "verifier"


def verifier_kind(issue: str) -> str:
    for key, kind in (("numbers", "number not in facts"), ("players/clubs", "name not in moment"),
                      ("too long", "too long"), ("empty", "empty text"), ("no model", "no answer")):
        if key in issue:
            return kind
    return "other"


def outcome(v: Version) -> str:
    """first_try | after_retry | fallback (the neutral text or a template stands in)."""
    if v.source == "template" or v.issues:
        return "fallback"
    return "first_try" if v.attempts <= 1 else "after_retry"


def _words(text: str) -> set[str]:
    return {w.lower() for w in WORD.findall(text)}


def jaccard(a: str, b: str) -> float:
    x, y = _words(a), _words(b)
    return len(x & y) / len(x | y) if x | y else 1.0


def _rates(outcomes: list[str]) -> dict:
    c = Counter(outcomes)
    n = max(len(outcomes), 1)
    return {"n": len(outcomes), **{k: round(c[k] / n, 3) for k in ("first_try", "after_retry", "fallback")}}


def _new_tokens(cache_path: Path, since: float) -> tuple[int, int, int]:
    with sqlite3.connect(cache_path) as con:
        n, p, c = con.execute("select count(*), coalesce(sum(prompt_tokens), 0), coalesce(sum(completion_tokens), 0)"
                              " from responses where created >= ?", (since,)).fetchone()
    return n, p, c


def summarise(feeds_by_match: dict[str, list[Feed]], profiles_by_match: dict[str, list[Profile]],
              seconds: dict[str, float]) -> dict:
    kinds = {(m, p.id): p for m, ps in profiles_by_match.items() for p in ps}
    versions, neutrals = [], {}
    for m, feeds in feeds_by_match.items():
        for f in feeds:
            p = kinds[(m, f.profile_id)]
            for i in f.items:
                versions.append((m, p, i))
                neutrals[i.moment_id] = (i.type, i.neutral)

    objections = Counter()
    verifier_kinds = Counter()
    for v in [i.version for _, _, i in versions] + [n for _, n in neutrals.values()]:
        for issue in v.rejected:
            objections[checker(issue)] += 1
            if checker(issue) == "verifier":
                verifier_kinds[verifier_kind(issue)] += 1

    by = defaultdict(list)
    for _, p, i in versions:
        o = outcome(i.version)
        by[("kind", p.kind)].append(o)
        by[("language", p.language)].append(o)
        by[("moment", i.type)].append(o)
    lang_fallbacks = sum(any("not available" in s for s in i.version.issues) for _, _, i in versions)

    # How personal are the versions? Word overlap between viewers' texts of the same moment
    # (English ones, so language does not inflate the difference), and with the shared neutral text.
    per_moment = defaultdict(list)
    for _, p, i in versions:
        if p.language == "en" and outcome(i.version) != "fallback":
            per_moment[i.moment_id].append(i.version.body)
    pair_sims = [jaccard(a, b) for bodies in per_moment.values() for a, b in itertools.combinations(bodies, 2)]
    neutral_sims = [jaccard(i.version.body, i.neutral.body) for _, p, i in versions
                    if p.language == "en" and outcome(i.version) != "fallback"]

    lengths = defaultdict(list)
    for _, p, i in versions:
        if outcome(i.version) != "fallback":
            lengths[p.id].append(len(i.version.body.split()))
    feed_overlap = []
    for feeds in feeds_by_match.values():
        sets = [{i.moment_id for i in f.items} for f in feeds]
        feed_overlap += [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    return {
        "matches": list(feeds_by_match),
        "viewers_per_match": len(next(iter(profiles_by_match.values()))),
        "moments_explained": len(neutrals),
        "viewer_texts": len(versions),
        "neutral": _rates([outcome(n) for _, n in neutrals.values()]),
        "versions": _rates([outcome(i.version) for _, _, i in versions]),
        "language_fallbacks": lang_fallbacks,
        "objections": dict(objections),
        "verifier_objections": dict(verifier_kinds),
        "by_kind": {k: _rates(v) for (dim, k), v in sorted(by.items()) if dim == "kind"},
        "by_language": {k: _rates(v) for (dim, k), v in sorted(by.items()) if dim == "language"},
        "by_moment_type": {k: _rates(v) for (dim, k), v in sorted(by.items()) if dim == "moment"},
        "similarity_between_viewers": mean(pair_sims),
        "similarity_to_neutral": mean(neutral_sims),
        "feed_overlap_between_viewers": mean(feed_overlap),
        "mean_words_by_viewer": {k: mean(v) for k, v in sorted(lengths.items())},
        "seconds_per_match": mean(list(seconds.values())),
    }


async def evaluate(match_ids: list[str], llm: LLM, parallel: int = 4) -> tuple[dict, dict]:
    t0 = time.time()
    feeds, profiles, seconds = {}, {}, {}
    async with Orchestrator(llm) as orch:
        for m in match_ids:
            t = time.time()
            profiles[m] = profiles_for(m)
            feeds[m] = await orch.feeds(m, profiles[m], parallel=parallel)
            seconds[m] = time.time() - t
            print(f"  {m}: {seconds[m]:.0f}s", flush=True)
    report = summarise(feeds, profiles, seconds)
    calls, p_tok, c_tok = _new_tokens(llm.cache.cache.path, t0)
    price_in, price_out = PRICE_PER_M.get(llm.model, (0.0, 0.0))
    report["run"] = {"backend": llm.label, "new_model_calls": calls, "cache_hits": llm.cache.stats.hits,
                     "new_prompt_tokens": p_tok, "new_completion_tokens": c_tok,
                     "cost_usd": round((p_tok * price_in + c_tok * price_out) / 1e6, 4),
                     "minutes": round((time.time() - t0) / 60, 1)}
    raw = {m: [f.model_dump() for f in fs] for m, fs in feeds.items()}
    return report, raw


def to_markdown(r: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.0f}%"

    def rate_rows(title, d):
        rows = [f"| {title} | n | first try | after retry | fallback |", "|---|---|---|---|---|"]
        rows += [f"| {k} | {v['n']} | {pct(v['first_try'])} | {pct(v['after_retry'])} | {pct(v['fallback'])} |"
                 for k, v in d.items()]
        return "\n".join(rows)

    run = r.get("run", {})
    intro = (f"{len(r['matches'])} held-out matches ({', '.join(r['matches'])}), {r['viewers_per_match']} viewers "
             f"each: {r['moments_explained']} moments explained, {r['viewer_texts']} personal texts.")
    objections = (f"Objections raised: {r['objections']} (Verifier detail: {r['verifier_objections']}). "
                  f"Language fallbacks: {r['language_fallbacks']}.")
    personal = (f"Personalisation: word overlap between two viewers' versions of the same moment "
                f"{pct(r['similarity_between_viewers'])}, with the neutral text {pct(r['similarity_to_neutral'])}; "
                f"moments shared by two viewers' feeds {pct(r['feed_overlap_between_viewers'])}.")
    cost = (f"Run: {run.get('backend')}, {run.get('new_model_calls')} new model calls "
            f"({run.get('new_prompt_tokens')} + {run.get('new_completion_tokens')} tokens, ${run.get('cost_usd')}), "
            f"{r['seconds_per_match']} s per match.")
    return "\n\n".join([
        "# Agent evaluation", intro,
        rate_rows("text", {"neutral explanation": r["neutral"], "viewer version": r["versions"]}),
        rate_rows("viewer kind", r["by_kind"]),
        rate_rows("language", r["by_language"]),
        rate_rows("moment type", r["by_moment_type"]),
        objections, personal, f"Mean words: {r['mean_words_by_viewer']}.", cost,
    ]) + "\n"


def held_out_matches(rounds: range = TEST_ROUNDS, max_matches: int | None = None) -> list[str]:
    idx = T._index().reset_index()
    ids = idx[idx["round"].isin(list(rounds))].match_id.tolist()
    return ids[:max_matches] if max_matches else ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the agent team on held-out matches")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--cache", type=Path, default=EVAL_DIR / "llm_cache.sqlite")
    ap.add_argument("--rounds", default=f"{TEST_ROUNDS.start}-{TEST_ROUNDS.stop - 1}")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--matches", nargs="*", help="explicit match ids instead of --rounds")
    ap.add_argument("--out", type=Path, default=EVAL_DIR)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.rounds.split("-"))
    matches = args.matches or held_out_matches(range(lo, hi + 1), args.max_matches)
    llm = chat_client(args.backend, cache_path=args.cache)
    print(f"evaluating {len(matches)} matches with {llm.label}", flush=True)
    report, raw = asyncio.run(evaluate(matches, llm))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "feeds.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    (args.out / "report.md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
