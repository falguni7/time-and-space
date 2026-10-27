"""Write and read simulated matches.

data/matches/
  index.parquet                       one row per match
  R01_RIV-NOR/
    meta.json  tracking.parquet  events.parquet  phases.parquet
    truth/  decisions.parquet  events.parquet  settings.json      (hidden: validation only)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import paths
from .config import load_clubs, load_players
from .export import build_decisions, build_events, build_meta, build_settings, build_tracking
from .phases import build_phases

DATA_DIR = paths.DATA_DIR / "matches"
COMPRESSION = "zstd"


def _write_df(df: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression=COMPRESSION)


def write_match(result, fixture, out_dir: Path = DATA_DIR) -> dict:
    """Write one match's public and truth files; returns its index row."""
    clubs, players = load_clubs(), load_players()
    folder = out_dir / fixture.match_id
    (folder / "truth").mkdir(parents=True, exist_ok=True)

    meta = build_meta(result, fixture, clubs, players)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    pq.write_table(build_tracking(result), folder / "tracking.parquet", compression=COMPRESSION)
    events, truth_events = build_events(result)
    _write_df(events, folder / "events.parquet")
    _write_df(build_phases(result), folder / "phases.parquet")

    _write_df(build_decisions(result), folder / "truth" / "decisions.parquet")
    _write_df(truth_events, folder / "truth" / "events.parquet")
    (folder / "truth" / "settings.json").write_text(
        json.dumps(build_settings(result, clubs, players), indent=2) + "\n")
    return {"match_id": fixture.match_id, "round": fixture.round, "home": result.home, "away": result.away,
            "home_goals": result.score[0], "away_goals": result.score[1], "seed": result.seed,
            "frames": len(result.period)}


@dataclass
class MatchData:
    match_id: str
    meta: dict
    tracking: pd.DataFrame
    events: pd.DataFrame
    phases: pd.DataFrame


def load_match(match_id: str, data_dir: Path = DATA_DIR, columns: list[str] | None = None) -> MatchData:
    """Load the public files of one match (never the truth folder)."""
    folder = data_dir / match_id
    return MatchData(
        match_id=match_id,
        meta=json.loads((folder / "meta.json").read_text()),
        tracking=pq.read_table(folder / "tracking.parquet", columns=columns).to_pandas(),
        events=pd.read_parquet(folder / "events.parquet"),
        phases=pd.read_parquet(folder / "phases.parquet"),
    )


def load_truth(match_id: str, data_dir: Path = DATA_DIR) -> dict:
    """Hidden ground truth of one match. For validation only: the engine must not call this."""
    folder = data_dir / match_id / "truth"
    return {
        "decisions": pd.read_parquet(folder / "decisions.parquet"),
        "events": pd.read_parquet(folder / "events.parquet"),
        "settings": json.loads((folder / "settings.json").read_text()),
    }


def load_index(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "index.parquet")
