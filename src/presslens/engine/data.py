"""The engine's only access to match data: the PUBLIC provider files.

Anything under a match's truth/ folder is refused here, so engine code cannot read hidden
ground truth by accident. Validation code (engine/validate.py) reads truth separately.

Coordinates: tracking and events are in the global frame (origin centre spot). Helpers give
everything in the acting team's attacking frame (attacking towards x = +52.5).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import paths

MATCHES = paths.DATA_DIR / "matches"
MODELS = paths.DATA_DIR / "engine"
TRAIN_ROUNDS = range(1, 11)   # models are fitted on rounds 1-10 ...
TEST_ROUNDS = range(11, 15)   # ... and validated on rounds 11-14


def single_threaded() -> None:
    """Worker-process initializer: one numerical thread per process (we parallelise over matches).

    Without it every worker starts its own OpenMP/BLAS pool sized to the whole node and
    56 workers end up fighting over the cores.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _public(path: Path) -> Path:
    if "truth" in path.parts:
        raise PermissionError(f"the engine may not read ground truth: {path}")
    return path


def connect(matches_dir: Path = MATCHES) -> duckdb.DuckDBPyConnection:
    """DuckDB with views over the public files of every match: events, phases, idx, dirs."""
    con = duckdb.connect()
    for name, pattern in (("events", "*/events.parquet"), ("phases", "*/phases.parquet")):
        glob = _public(matches_dir / pattern)
        con.execute(f"""
            create view {name} as
            select *, regexp_extract(filename, '(R\\d+_[A-Z]+-[A-Z]+)', 1) as match_id
            from read_parquet('{glob}', filename = true)""")
    con.execute(f"create view idx as select * from read_parquet('{_public(matches_dir / 'index.parquet')}')")
    rows = []
    for meta_path in sorted(matches_dir.glob("*/meta.json")):
        meta = json.loads(_public(meta_path).read_text())
        for p in meta["periods"]:
            for team, d in p["directions"].items():
                rows.append({"match_id": meta["match_id"], "period": p["period"], "team": team, "direction": d})
    con.register("dirs_df", pd.DataFrame(rows))
    con.execute("create table dirs as select * from dirs_df")
    # Events with coordinates in the acting team's attacking frame.
    con.execute("""
        create view events_a as
        select e.*, d.direction,
               e.x * d.direction as ax, e.y * d.direction as ay,
               e.end_x * d.direction as end_ax, e.end_y * d.direction as end_ay
        from events e left join dirs d on d.match_id = e.match_id and d.period = e.period and d.team = e.team""")
    return con


def match_ids(rounds=None, matches_dir: Path = MATCHES) -> list[str]:
    idx = pd.read_parquet(_public(matches_dir / "index.parquet"))
    if rounds is not None:
        idx = idx[idx["round"].isin(list(rounds))]
    return idx["match_id"].tolist()


@dataclass
class MatchArrays:
    """One match's tracking as arrays. Objects per frame: 11 home slots, 11 away slots, ball."""

    match_id: str
    meta: dict
    xy: np.ndarray          # (T, 23, 2)
    z: np.ndarray           # (T,) ball height
    vel: np.ndarray         # (T, 23, 2)
    ids: np.ndarray         # (T, 23) object codes into `names`
    names: list[str]
    period: np.ndarray      # (T,)
    in_play: np.ndarray     # (T,) bool
    possession: np.ndarray  # (T,) 0 home, 1 away, -1 none/dead
    carrier: np.ndarray     # (T,) object index 0..21 of the ball carrier, -1 if none

    @property
    def home(self) -> str:
        return self.meta["home"]["id"]

    @property
    def away(self) -> str:
        return self.meta["away"]["id"]

    def team_index(self, club: str) -> int:
        return 0 if club == self.home else 1

    @cached_property
    def directions(self) -> np.ndarray:
        """(n_periods + 1, 2) attack direction per period and team (index by period number)."""
        out = np.ones((len(self.meta["periods"]) + 1, 2), dtype=int)
        for p in self.meta["periods"]:
            out[p["period"]] = [p["directions"][self.home], p["directions"][self.away]]
        return out

    def direction(self, frame: int, team: int) -> int:
        return int(self.directions[self.period[frame], team])

    def player_index(self, frame: int, player_id: str) -> int:
        code = self.names.index(player_id)
        hits = np.where(self.ids[frame, :22] == code)[0]
        return int(hits[0]) if len(hits) else -1


def load_arrays(match_id: str, matches_dir: Path = MATCHES) -> MatchArrays:
    folder = matches_dir / match_id
    meta = json.loads(_public(folder / "meta.json").read_text())
    table = pq.read_table(_public(folder / "tracking.parquet"),
                          columns=["object_id", "x", "y", "z", "vx", "vy", "period", "ball_state",
                                   "possession_team", "is_carrier"])
    n = table.num_rows
    T = n // 23
    obj = table.column("object_id").combine_chunks()
    codes = obj.indices.to_numpy().reshape(T, 23)
    names = obj.dictionary.to_pylist()
    xy = np.stack([table.column("x").to_numpy(), table.column("y").to_numpy()], axis=1).reshape(T, 23, 2)
    vel = np.stack([table.column("vx").to_numpy(), table.column("vy").to_numpy()], axis=1).reshape(T, 23, 2)
    z = table.column("z").to_numpy().reshape(T, 23)[:, 22]
    period = table.column("period").to_numpy().reshape(T, 23)[:, 0]
    in_play = (table.column("ball_state").to_pandas().to_numpy().reshape(T, 23)[:, 0] == "in_play")
    poss_raw = table.column("possession_team").to_pandas().to_numpy().reshape(T, 23)[:, 0]
    home = meta["home"]["id"]
    possession = np.where(pd.isna(poss_raw), -1, np.where(poss_raw == home, 0, 1)).astype(int)
    is_carrier = table.column("is_carrier").to_numpy(zero_copy_only=False).reshape(T, 23)[:, :22]
    carrier = np.where(is_carrier.any(axis=1), is_carrier.argmax(axis=1), -1)
    return MatchArrays(match_id, meta, xy, z, vel, codes, names, period, in_play, possession, carrier)


def attack_frame(xy: np.ndarray, direction: int) -> np.ndarray:
    return np.asarray(xy, dtype=float) * direction
