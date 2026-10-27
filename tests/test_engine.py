import numpy as np
import pytest

from presslens.engine import data
from presslens.engine.decisions import pressure_index

SEASON = (data.MATCHES / "index.parquet").exists()
MODELS = (data.MODELS / "xt.json").exists() and (data.MODELS / "xpass.pkl").exists()


def test_engine_refuses_ground_truth():
    with pytest.raises(PermissionError):
        data._public(data.MATCHES / "R01_RIV-STO" / "truth" / "events.parquet")


def test_pressure_index_range():
    np.testing.assert_allclose(pressure_index([0.5, 1.0, 3.0, 5.0, 9.0]), [1.0, 1.0, 0.5, 0.0, 0.0])


def test_train_and_test_rounds_are_disjoint():
    assert not set(data.TRAIN_ROUNDS) & set(data.TEST_ROUNDS)


@pytest.mark.skipif(not SEASON, reason="season not generated")
def test_match_arrays_line_up_with_events():
    con = data.connect()
    mid = data.match_ids()[0]
    m = data.load_arrays(mid)
    spells = con.execute(f"select frame, player_id from events where match_id = '{mid}' "
                         f"and type = 'possession' limit 50").fetchall()
    for frame, pid in spells:
        assert m.names[m.ids[frame, m.carrier[frame]]] == pid


@pytest.mark.skipif(not (SEASON and MODELS), reason="models not fitted")
def test_xt_rises_towards_goal_and_xpass_is_a_probability():
    from presslens.engine.features import FEATURES
    from presslens.engine.xpass import XPassModel, pass_table
    from presslens.engine.xt import XTModel
    xt = XTModel.load()
    assert xt.value(-40, 0) < xt.value(0, 0) < xt.value(30, 0) < xt.value(46, 0)
    table = pass_table(data.match_ids()[0])
    p = XPassModel.load().predict(table[FEATURES].to_numpy())
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.skipif(not (SEASON and MODELS), reason="models not fitted")
def test_decision_rows_are_consistent():
    from presslens.engine.decisions import match_decisions
    d = match_decisions(data.match_ids()[0])
    assert (d.decision_gap >= -1e-12).all()
    assert (d.best_value >= d.chosen_value - 1e-12).all()
    assert d.pressure.between(0, 1).all() and d.free_options.between(0, 10).all()
    assert (d.chose_best == (d.decision_gap < 1e-12)).mean() > 0.99


CUBE = SEASON and (data.MODELS / "metrics.duckdb").exists()


@pytest.mark.skipif(not CUBE, reason="cube not built")
def test_query_metrics_whitelists_names_and_reports_uncertainty():
    from presslens.engine.cube import query_metrics
    with pytest.raises(ValueError):
        query_metrics("drop table passes", ["team"])
    with pytest.raises(ValueError):
        query_metrics("xt_generated", ["1; drop table passes"])
    with pytest.raises(ValueError):
        query_metrics("shots", ["pressure_band"])  # not a dimension of the shots table
    df = query_metrics("decision_gap", ["pressure_band"], {"team": "RIV"})
    assert {"value", "n", "ci_low", "ci_high"} <= set(df.columns)
    assert (df.ci_low <= df.value).all() and (df.value <= df.ci_high).all()
    teams = query_metrics("xt_generated", ["team"])
    assert len(teams) == 8 and (teams.value > 0).all()


@pytest.mark.skipif(not (CUBE and (data.MODELS / "moments.parquet").exists()), reason="moments not built")
def test_moment_cards_are_well_formed():
    from presslens.engine.moments import BASE_IMPORTANCE, load_moments
    cards = load_moments(data.match_ids()[0])
    assert cards and {c["type"] for c in cards} <= set(BASE_IMPORTANCE)
    for c in cards:
        assert 0.0 < c["importance"] <= 1.0 and c["team"] != c["opponent"]
        assert {"moment_id", "frame", "minute", "metrics"} <= set(c)
