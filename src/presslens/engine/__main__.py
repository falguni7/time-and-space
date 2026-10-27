"""Engine command line.

    bin/pl python -m presslens.engine build       # fit xT + xPass, decisions, cube, moments, validation
    bin/pl python -m presslens.engine query xt_generated --by team --where third=attacking
"""

from __future__ import annotations

import argparse
import time

import pandas as pd


def cmd_build() -> None:
    from . import cube, moments, validate, xpass, xt
    from .data import MODELS, TRAIN_ROUNDS, connect, match_ids
    from .decisions import all_decisions

    t0 = time.time()
    train = match_ids(TRAIN_ROUNDS)
    xt.fit(connect(), train).save()
    print(f"xT grid fitted on {len(train)} matches")
    xpass.fit(xpass.pass_tables(train)).save()
    print("xPass model fitted")
    decisions = all_decisions(match_ids())
    decisions.to_parquet(MODELS / "decisions.parquet", index=False)
    print(f"{len(decisions)} pass decisions")
    cube.build(decisions).close()
    print(f"metrics cube -> {cube.CUBE_PATH}")
    m = moments.build()
    print(f"{len(m)} moment cards -> {moments.MOMENTS_PATH}")
    validate.main()
    print(f"\nEngine built in {time.time() - t0:.0f} s")


def cmd_query(metric: str, by: list[str], where: list[str]) -> None:
    from .cube import query_metrics
    filters = dict(w.split("=", 1) for w in where)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(query_metrics(metric, by, filters).round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="PressLens engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="fit models and build decisions, cube, moments, validation")
    q = sub.add_parser("query", help="query the metrics cube")
    q.add_argument("metric")
    q.add_argument("--by", nargs="*", default=[])
    q.add_argument("--where", nargs="*", default=[], help="dimension=value filters")
    args = parser.parse_args()
    if args.cmd == "build":
        cmd_build()
    else:
        cmd_query(args.metric, args.by, args.where)


if __name__ == "__main__":
    main()
