import numpy as np
import pytest

from presslens.sim import kinematics
from presslens.sim.checks import proximity_summary, run_shape
from presslens.sim.match import DT
from presslens.sim.pitch import HALF_L, HALF_W


def test_step_respects_speed_and_acceleration_limits():
    pos = np.zeros((3, 2))
    vel = np.zeros((3, 2))
    target = np.array([[40.0, 0.0], [0.0, 30.0], [-30.0, -20.0]])
    vmax = np.array([9.0, 8.0, 7.5])
    amax = np.array([4.0, 3.0, 2.8])
    prev_vel = vel
    for _ in range(200):
        pos, vel = kinematics.step(pos, vel, target, vmax, amax, DT)
        assert np.all(np.linalg.norm(vel, axis=1) <= vmax + 1e-9)
        assert np.all(np.linalg.norm(vel - prev_vel, axis=1) <= amax * DT + 1e-9)
        prev_vel = vel
    np.testing.assert_allclose(pos, target, atol=0.5)


def test_players_stay_on_pitch():
    pos = np.array([[HALF_L - 0.5, 0.0], [0.0, -HALF_W + 0.5]])
    vel = np.array([[9.0, 0.0], [0.0, -9.0]])
    target = np.array([[HALF_L + 20, 0.0], [0.0, -HALF_W - 20]])
    for _ in range(50):
        pos, vel = kinematics.step(pos, vel, target, np.array([9.0, 9.0]), np.array([4.0, 4.0]), DT)
    assert np.all(np.abs(pos[:, 0]) <= HALF_L) and np.all(np.abs(pos[:, 1]) <= HALF_W)


@pytest.fixture(scope="module")
def runs():
    return run_shape("EAS", "RIV", minutes=6), run_shape("RIV", "EAS", minutes=6)


def test_live_play_is_physically_plausible(runs):
    for run in runs:
        r = run.result
        same = (r.ids[1:] == r.ids[:-1]) & (r.period[1:] == r.period[:-1])[:, None, None]
        speed = np.linalg.norm(np.diff(r.pos, axis=0), axis=-1) / DT
        assert speed[same].max() < 10.5
        assert np.all(np.abs(r.pos[..., 0]) <= HALF_L) and np.all(np.abs(r.pos[..., 1]) <= HALF_W)


def test_high_press_stays_closer_than_low_block(runs):
    vs_high, vs_low = (proximity_summary(r) for r in runs)
    for third in ("defensive", "middle"):
        assert vs_high[third]["carrier_m"] < vs_low[third]["carrier_m"]
        assert vs_high[third]["options_m"] < vs_low[third]["options_m"]


def test_same_seed_same_run():
    a = run_shape("KIN", "MIL", seed=5, minutes=2)
    b = run_shape("KIN", "MIL", seed=5, minutes=2)
    np.testing.assert_array_equal(a.result.pos, b.result.pos)
