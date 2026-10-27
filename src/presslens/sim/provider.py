"""The data provider's shot xG model (shipped in the public events as `provider_xg`).

Like a real provider, it is fitted on historical matches (simulated with seeds outside the
season) using only what a provider can see: shot distance, angle, header and situation.
It knows nothing about pressure, defenders in the shooting cone, the shooter's finishing
or the keeper, so it is right on average but imperfect for individual shots. The true xG
stays in truth/events.parquet.

    bin/pl python -m presslens.sim fit-provider-xg --matches 56
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import CONFIG_DIR
from .pitch import HALF_L
from .truth import goal_angle

COEF_PATH = CONFIG_DIR / "provider_xg.json"
PROVIDER_PENALTY_XG = 0.78  # providers use a fixed value for penalties
FEATURES = ["intercept", "log_distance", "angle", "header", "free_kick"]


def shot_features(x: float, y: float, header: bool, free_kick: bool) -> np.ndarray:
    """Features in the shooting team's attacking frame (goal at x = +52.5)."""
    dist = max(float(np.hypot(HALF_L - x, y)), 0.5)
    angle = float(goal_angle(np.array([[x, y]]))[0])
    return np.array([1.0, np.log(dist), angle, float(header), float(free_kick)])


def load_coefficients() -> np.ndarray:
    data = json.loads(COEF_PATH.read_text())
    return np.array([data["coefficients"][f] for f in FEATURES])


def provider_xg(x: float, y: float, header: bool, restart: str | None, coef: np.ndarray) -> float:
    if restart == "penalty":
        return PROVIDER_PENALTY_XG
    z = float(shot_features(x, y, header, restart == "free_kick") @ coef)
    return float(1.0 / (1.0 + np.exp(-z)))


def _history_shots(fixture) -> list[tuple[np.ndarray, int, float]]:
    from .match import Match
    r = Match(fixture.home, fixture.away, seed=fixture.seed).run()
    rows = []
    for e in r.events:
        if e["type"] != "shot" or e.get("restart") == "penalty":
            continue
        team = 0 if e["team"] == r.home else 1
        d = r.directions[e["period"] - 1][team]
        f = shot_features(e["x"] * d, e["y"] * d, bool(e.get("header")), e.get("restart") == "free_kick")
        rows.append((f, int(e["outcome"] == "goal"), float(e["xg"])))
    return rows


def fit(n_matches: int = 56, workers: int = 56) -> dict:
    """Simulate `n_matches` historical matches, fit the logistic model, save the coefficients."""
    from sklearn.linear_model import LogisticRegression

    from .season import HISTORY_SEED, schedule
    fixtures = schedule(seed0=HISTORY_SEED)
    fixtures = (fixtures * (n_matches // len(fixtures) + 1))[:n_matches]
    fixtures = [type(f)(f.round, f.home, f.away, HISTORY_SEED + k) for k, f in enumerate(fixtures)]
    with ProcessPoolExecutor(min(workers, len(fixtures))) as ex:
        rows = [row for res in ex.map(_history_shots, fixtures) for row in res]
    X = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    true_xg = np.array([r[2] for r in rows])
    model = LogisticRegression(fit_intercept=False, C=10.0, max_iter=1000).fit(X, y)
    coef = model.coef_[0]
    pred = 1.0 / (1.0 + np.exp(-(X @ coef)))
    summary = {
        "features": FEATURES,
        "coefficients": dict(zip(FEATURES, coef.round(5).tolist())),
        "penalty_xg": PROVIDER_PENALTY_XG,
        "trained_on": {"matches": n_matches, "shots": len(y), "goals": int(y.sum()),
                       "seeds": [HISTORY_SEED, HISTORY_SEED + n_matches - 1]},
        "check": {
            "goals_per_provider_xg": round(float(y.sum() / pred.sum()), 3),
            "corr_with_true_xg": round(float(np.corrcoef(pred, true_xg)[0, 1]), 3),
            "mean_abs_error_vs_true_xg": round(float(np.mean(np.abs(pred - true_xg))), 4),
        },
    }
    COEF_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
