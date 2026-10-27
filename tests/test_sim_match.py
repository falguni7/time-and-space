import numpy as np
import pytest

from presslens.sim import decisions
from presslens.sim.config import load_clubs, load_players, player_params
from presslens.sim.match import DEAD, DT, FLIGHT, HELD, Match
from presslens.sim.pitch import HALF_L, HALF_W
from presslens.sim.tactics import offside_line
from presslens.sim.truth import defenders_in_cone, pass_completion, true_value, true_xg

# ---------------------------------------------------------------- truth models


def test_true_value_rises_towards_goal_and_is_symmetric():
    xs = np.array([[-40.0, 0.0], [0.0, 0.0], [30.0, 0.0], [45.0, 0.0]])
    v = true_value(xs)
    assert np.all(np.diff(v) > 0)
    assert true_value(np.array([[30.0, 10.0]]))[0] == pytest.approx(true_value(np.array([[30.0, -10.0]]))[0])


def test_xg_falls_with_distance_and_pressure():
    near = true_xg(np.array([44.0, 0.0]), 0.0, 1.0)
    far = true_xg(np.array([25.0, 0.0]), 0.0, 1.0)
    pressed = true_xg(np.array([44.0, 0.0]), 1.0, 1.0)
    assert near > far and near > pressed
    assert 0.0 < far < near <= 0.6


def _pass(defenders):
    n = len(defenders)
    return pass_completion(
        np.array([0.0, 0.0]), np.array([12.0, 0.0]), np.array([12.0, 0.0]), 8.0, 0.5,
        np.asarray(defenders, dtype=float), np.full(n, 8.0), np.full(n, 1.2), np.full(n, 0.5),
        error_deg=2.5, lofted=False,
    )


def test_free_short_pass_is_safe_and_a_defender_in_the_lane_is_not():
    free = _pass([[0.0, 25.0], [30.0, -20.0]])["p_complete"]
    blocked = _pass([[6.0, 0.3], [30.0, -20.0]])["p_complete"]
    assert free > 0.9
    assert blocked < 0.6


def test_shooting_cone_counts_blockers():
    defs = np.array([[45.0, 0.0], [40.0, 0.5], [30.0, 0.0], [45.0, 15.0]])
    assert defenders_in_cone(np.array([35.0, 0.0]), defs) == 2


# ---------------------------------------------------------------- decisions


@pytest.fixture(scope="module")
def ctx_parts():
    clubs, players = load_clubs(), load_players()
    from presslens.sim.config import starting_xi
    own = [player_params(p, clubs["RIV"].tactics) for p in starting_xi(players, "RIV")]
    opp = [player_params(p, clubs["NOR"].tactics) for p in starting_xi(players, "NOR")]
    return own, opp


def test_offside_receivers_are_not_offered(ctx_parts):
    own_p, opp_p = ctx_parts
    own = np.array([[-45.0, 0.0]] + [[-10.0 + 3 * i, -20.0 + 4 * i] for i in range(10)])
    own[9] = [30.0, 0.0]  # far beyond the defence
    opp = np.array([[50.0, 0.0]] + [[10.0, -20.0 + 4 * i] for i in range(10)])
    ctx = decisions.Context(carrier=5, own=own, own_vel=np.zeros((11, 2)), own_params=own_p,
                            opp=opp, opp_params=opp_p, ball=own[5].copy(), decision_quality=0.8)
    assert offside_line(opp, ctx.ball) == pytest.approx(10.0)
    receivers = {o.receiver for o in decisions.options(ctx) if o.kind == "pass"}
    assert 9 not in receivers and len(receivers) > 3


def test_pressure_raises_decision_noise_and_composure_dampens_it():
    calm = decisions.temperature(0.8, 0.0, 0.5)
    pressed = decisions.temperature(0.8, 1.0, 0.5)
    composed = decisions.temperature(0.8, 1.0, 1.0)
    assert pressed > calm and composed == pytest.approx(calm)


# ---------------------------------------------------------------- full match loop


@pytest.fixture(scope="module")
def short_match():
    return Match("KIN", "ASH", seed=11, period_minutes=4).run()


def test_match_is_deterministic(short_match):
    again = Match("KIN", "ASH", seed=11, period_minutes=4).run()
    assert again.score == short_match.score
    assert len(again.events) == len(short_match.events)
    np.testing.assert_array_equal(again.ball, short_match.ball)


def test_recorded_arrays_are_consistent(short_match):
    r = short_match
    n = len(r.period)
    assert n == sum(r.period_frames)
    assert r.pos.shape == (n, 2, 11, 2) and r.ids.shape == (n, 2, 11)
    assert set(np.unique(r.period)) == {1, 2}
    assert np.all(np.abs(r.pos[..., 0]) <= HALF_L) and np.all(np.abs(r.pos[..., 1]) <= HALF_W)
    held = r.ball_state == HELD
    assert np.all(r.carrier[held] >= 0) and np.all(r.carrier[~held] == -1)
    assert np.all(r.possession[r.ball_state == DEAD] == -1)


def test_players_respect_speed_limits(short_match):
    r = short_match
    same = (r.ids[1:] == r.ids[:-1]) & (r.period[1:] == r.period[:-1])[:, None, None]
    speed = np.linalg.norm(np.diff(r.pos, axis=0), axis=-1) / DT
    assert speed[same].max() < 10.5


def test_ball_moves_continuously_in_play(short_match):
    r = short_match
    live = (r.ball_state[1:] != DEAD) & (r.ball_state[:-1] != DEAD) & (r.period[1:] == r.period[:-1])
    flight_or_held = np.isin(r.ball_state[1:], (FLIGHT, HELD))
    step = np.linalg.norm(np.diff(r.ball[:, :2], axis=0), axis=1)
    assert np.percentile(step[live & flight_or_held], 99) < 3.5  # < 35 m/s


def test_events_are_well_formed(short_match):
    r = short_match
    clubs = {r.home, r.away}
    for e in r.events:
        assert {"frame", "period", "type", "team"} <= set(e)
        if e["type"] == "pass":
            assert e["team"] in clubs and e["outcome"] in ("complete", "intercepted", "lost_duel", "misplaced", "out", "overhit")
            if e["outcome"] == "complete":
                assert e["receiver_id"].split("-")[0] == e["team"]
        if e["type"] == "shot":
            assert 0.0 < e["xg"] <= 0.6
    goals = [e for e in r.events if e["type"] == "goal"]
    assert len(goals) == sum(r.score)


def test_every_decision_logs_its_options(short_match):
    for d in short_match.decisions:
        assert 0 <= d["chosen"] < len(d["options"])
        assert 0.0 <= d["pressure"] <= 1.0 and d["temperature"] > 0.0


def test_foul_in_the_box_gives_a_penalty_and_outside_a_free_kick():
    m = Match("RIV", "NOR", seed=3, period_minutes=1)
    m._start_period(1)
    att = m.teams[0]
    m.poss, m.state, m.carrier = 0, HELD, 9
    m.restart = None
    in_box = np.array([HALF_L - 8.0, 3.0]) * att.direction
    m.ball[:2] = in_box
    att.pos[9] = in_box
    m._foul(defender=4)
    assert m.restart.kind == "penalty"
    np.testing.assert_allclose(m.restart.spot, np.array([HALF_L - 11.0, 0.0]) * att.direction)
    m.state, m.carrier, m.restart = HELD, 9, None
    outside = np.array([10.0, 3.0]) * att.direction
    m.ball[:2] = outside
    att.pos[9] = outside
    m._foul(defender=4)
    assert m.restart.kind == "free_kick"
