"""Simulation command line.

    bin/pl python -m presslens.sim fit-provider-xg            # fit the provider's shot xG model
    bin/pl python -m presslens.sim generate                   # simulate and write the full season
    bin/pl python -m presslens.sim generate --limit 4         # first 4 fixtures only (quick check)
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .io import DATA_DIR
from .season import schedule


def _simulate_and_write(job) -> dict:
    fixture, out_dir = job
    from .io import write_match
    from .match import Match
    result = Match(fixture.home, fixture.away, seed=fixture.seed, match_id=fixture.match_id).run()
    return write_match(result, fixture, out_dir)


def cmd_generate(out_dir: Path, workers: int, limit: int | None) -> None:
    fixtures = schedule()[:limit] if limit else schedule()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, len(fixtures))) as ex:
        rows = list(ex.map(_simulate_and_write, [(f, out_dir) for f in fixtures]))
    index = pd.DataFrame(rows).sort_values(["round", "match_id"]).reset_index(drop=True)
    index.to_parquet(out_dir / "index.parquet", index=False)
    goals = (index["home_goals"] + index["away_goals"]).mean()
    print(f"Wrote {len(index)} matches to {out_dir} in {time.time() - t0:.0f} s "
          f"({goals:.2f} goals per match)")


def cmd_fit_provider(matches: int, workers: int) -> None:
    from .provider import COEF_PATH, fit
    summary = fit(matches, workers)
    print(f"Fitted on {summary['trained_on']['shots']} shots from {matches} historical matches -> {COEF_PATH}")
    for k, v in summary["coefficients"].items():
        print(f"  {k:<14}{v:>9.4f}")
    print("  check:", summary["check"])


def main() -> None:
    parser = argparse.ArgumentParser(description="PressLens simulation")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="simulate and write the season")
    g.add_argument("--out", type=Path, default=DATA_DIR)
    g.add_argument("--workers", type=int, default=56)
    g.add_argument("--limit", type=int, default=None)
    f = sub.add_parser("fit-provider-xg", help="fit the provider's shot xG model on historical matches")
    f.add_argument("--matches", type=int, default=56)
    f.add_argument("--workers", type=int, default=56)
    args = parser.parse_args()
    if args.cmd == "generate":
        cmd_generate(args.out, args.workers, args.limit)
    else:
        cmd_fit_provider(args.matches, args.workers)


if __name__ == "__main__":
    main()
