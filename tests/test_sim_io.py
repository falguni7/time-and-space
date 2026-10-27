from collections import Counter

import numpy as np
import pandas as pd
import pytest

from presslens.sim import provider
from presslens.sim.export import PASS_OUTCOMES, PUBLIC_EVENT_COLUMNS, TRUTH_FIELDS
from presslens.sim.io import load_match, load_truth, write_match
from presslens.sim.match import Match
from presslens.sim.pitch import HALF_L
from presslens.sim.season import Fixture, schedule


def test_schedule_is_a_double_round_robin():
    fixtures = schedule()
    assert len(fixtures) == 56 and len({f.round for f in fixtures}) == 14
    assert set(Counter((f.home, f.away) for f in fixtures).values()) == {1}
    assert set(Counter(f.home for f in fixtures).values()) == {7}
    for rnd in range(1, 15):
        teams = [t for f in fixtures if f.round == rnd for t in (f.home, f.away)]
        assert len(teams) == len(set(teams)) == 8
    assert len({f.seed for f in fixtures}) == 56


def test_provider_xg_behaves_like_a_provider_model():
    coef = provider.load_coefficients()
    close = provider.provider_xg(HALF_L - 8, 0.0, False, None, coef)
    far = provider.provider_xg(HALF_L - 25, 0.0, False, None, coef)
    header = provider.provider_xg(HALF_L - 8, 0.0, True, None, coef)
    assert 0.0 < far < close < 0.6 and header < close
    assert provider.provider_xg(HALF_L - 11, 0.0, False, "penalty", coef) == provider.PROVIDER_PENALTY_XG


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("matches")
    fixture = Fixture(1, "RIV", "EAS", seed=77)
    result = Match(fixture.home, fixture.away, seed=fixture.seed, match_id=fixture.match_id,
                   period_minutes=6).run()
    row = write_match(result, fixture, out)
    return out, fixture, result, row


def test_all_files_written_and_index_row(written):
    out, fixture, result, row = written
    folder = out / fixture.match_id
    for name in ("meta.json", "tracking.parquet", "events.parquet", "phases.parquet",
                 "truth/decisions.parquet", "truth/events.parquet", "truth/settings.json"):
        assert (folder / name).exists(), name
    assert row["home_goals"] == result.score[0] and row["frames"] == len(result.period)


def test_tracking_round_trip(written):
    out, fixture, result, _ = written
    m = load_match(fixture.match_id, out)
    T = len(result.period)
    assert len(m.tracking) == T * 23
    ball = m.tracking[m.tracking.object_id == "ball"]
    np.testing.assert_allclose(ball["x"].to_numpy(), result.ball[:, 0], atol=1e-4)
    carriers = m.tracking[m.tracking.is_carrier]
    assert carriers.groupby("frame").size().max() == 1
    assert m.meta["total_frames"] == T and m.meta["score"][result.home] == result.score[0]


def test_public_events_hold_no_truth(written):
    out, fixture, _, _ = written
    m = load_match(fixture.match_id, out)
    assert list(m.events.columns) == PUBLIC_EVENT_COLUMNS
    assert not set(TRUTH_FIELDS) & set(m.events.columns)
    assert not any(c.startswith("true_") for c in m.events.columns)
    passes = m.events[m.events.type == "pass"]
    assert set(passes.outcome) <= set(PASS_OUTCOMES)
    shots = m.events[m.events.type == "shot"]
    assert shots.provider_xg.between(0.0, 1.0).all()


def test_truth_links_to_public_events(written):
    out, fixture, _, _ = written
    m = load_match(fixture.match_id, out)
    truth = load_truth(fixture.match_id, out)
    linked = truth["events"].merge(m.events[["event_id", "type"]], on="event_id", suffixes=("", "_pub"))
    assert len(linked) == len(truth["events"]) and (linked.type == linked.type_pub).all()
    shots = truth["events"][truth["events"].type == "shot"]
    assert shots.true_xg.notna().all()
    d = truth["decisions"]
    assert (d.groupby("decision_id").chosen.sum() == 1).all()
    assert {"tactics", "players"} <= set(truth["settings"]["RIV"])


def test_events_and_possession_spells_match_tracking(written):
    out, fixture, result, _ = written
    m = load_match(fixture.match_id, out)
    T = len(result.period)
    assert m.events.frame.between(0, T - 1).all() and (m.events.end_frame >= m.events.frame).all()
    spells = m.events[m.events.type == "possession"]
    carrier = m.tracking[m.tracking.is_carrier].set_index("frame")["object_id"]
    for _, s in spells.sample(min(40, len(spells)), random_state=0).iterrows():
        assert carrier.loc[s.frame] == s.player_id and carrier.loc[s.end_frame] == s.player_id
    pressure = m.events[m.events.type == "pressure"]
    assert (pressure.min_distance_m <= 3.0).all()


def test_phases_cover_live_play(written):
    out, fixture, result, _ = written
    m = load_match(fixture.match_id, out)
    in_play = np.where(result.ball_state != 3)[0]
    for flag in (True, False):
        ph = m.phases[m.phases.in_possession == flag]
        covered = np.zeros(len(result.period), dtype=bool)
        for _, p in ph.iterrows():
            covered[p.start_frame:p.end_frame + 1] = True
        assert covered[in_play].mean() > 0.995
    assert set(m.phases.phase) <= {"build_up", "create", "finish", "transition", "counter_attack", "set_piece",
                                   "high_press", "mid_block", "low_block"}


def test_same_fixture_writes_identical_events(written, tmp_path):
    out, fixture, _, _ = written
    again = Match(fixture.home, fixture.away, seed=fixture.seed, match_id=fixture.match_id,
                  period_minutes=6).run()
    write_match(again, fixture, tmp_path)
    pd.testing.assert_frame_equal(load_match(fixture.match_id, out).events,
                                  load_match(fixture.match_id, tmp_path).events)
