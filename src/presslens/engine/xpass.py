"""xPass: the chance a pass is completed, learned from public events + tracking."""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MODELS, connect, load_arrays, single_threaded
from .features import FEATURES, option_features

XPASS_PATH = MODELS / "xpass.pkl"


def pass_table(match_id: str) -> pd.DataFrame:
    """One row per pass in the match: features at the moment of the pass + outcome."""
    con = connect()
    passes = con.execute(f"""
        select event_id, frame, period, team, player_id, intended_id, lofted, restart, outcome
        from events where match_id = '{match_id}' and type = 'pass' and intended_id is not null""").df()
    m = load_arrays(match_id)
    rows = []
    for p in passes.itertuples():
        team = m.team_index(p.team)
        passer = m.player_index(p.frame, p.player_id)
        receiver = m.player_index(p.frame, p.intended_id)
        if passer < 0 or receiver < 0:
            continue
        f = option_features(m, p.frame, team, passer, receiver, bool(p.lofted), p.restart is not None)
        rows.append([match_id, p.event_id, p.frame, p.team, p.player_id, p.intended_id,
                     p.outcome == "complete", *f])
    return pd.DataFrame(rows, columns=["match_id", "event_id", "frame", "team", "player_id", "intended_id",
                                       "complete", *FEATURES])


def pass_tables(match_ids: list[str], workers: int = 32) -> pd.DataFrame:
    with ProcessPoolExecutor(min(workers, len(match_ids)), initializer=single_threaded) as ex:
        return pd.concat(list(ex.map(pass_table, match_ids)), ignore_index=True)


@dataclass
class XPassModel:
    model: object
    trained_on: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def save(self, path: Path = XPASS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = XPASS_PATH) -> XPassModel:
        with open(path, "rb") as f:
            return pickle.load(f)


def fit(table: pd.DataFrame) -> XPassModel:
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(table[FEATURES].to_numpy(), table["complete"].to_numpy())
    return XPassModel(model, sorted(table["match_id"].unique()))
