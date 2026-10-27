from collections import Counter

import pytest

from presslens.sim import squads
from presslens.sim.config import (
    ATTRIBUTES,
    PLAYSTYLES,
    Tactics,
    load_clubs,
    load_players,
    player_params,
    squad,
    starting_xi,
)
from presslens.sim.formations import FORMATIONS


@pytest.fixture(scope="module")
def clubs():
    return load_clubs()


@pytest.fixture(scope="module")
def players():
    return load_players()


def test_eight_clubs_with_valid_tactics(clubs):
    assert len(clubs) == 8
    assert {c.tactics.formation for c in clubs.values()} <= set(FORMATIONS)


def test_each_squad_has_18_players_two_keepers_and_full_xi(clubs, players):
    for club in clubs.values():
        sq = squad(players, club.id)
        assert len(sq) == 18
        assert sum(p.position == "GK" for p in sq) == 2
        xi = starting_xi(players, club.id)
        expected = [pos for pos, _, _ in FORMATIONS[club.tactics.formation]]
        assert [p.position for p in xi] == expected
        assert len({p.number for p in sq}) == 18


def test_ratings_and_profiles_are_in_range(players):
    names = Counter(p.name for p in players.values())
    assert max(names.values()) == 1
    for p in players.values():
        assert set(p.attributes) == set(ATTRIBUTES)
        assert all(20 <= v <= 97 for v in p.attributes.values())
        assert 40 <= p.overall <= 97
        assert set(p.playstyles) <= set(PLAYSTYLES) and len(p.playstyles) <= 2


def test_tier_order_of_starting_elevens(clubs, players):
    avg = {}
    for club in clubs.values():
        xi = starting_xi(players, club.id)
        avg.setdefault(club.tier, []).append(sum(p.overall for p in xi) / 11)
    assert min(avg["strong"]) > max(avg["mid"]) > 0
    assert min(avg["mid"]) > max(avg["weak"])


def test_generator_is_deterministic():
    a = squads.generate(seed=3)
    b = squads.generate(seed=3)
    assert [(p.name, p.attributes) for p in a] == [(p.name, p.attributes) for p in b]


def test_attributes_drive_params(players):
    tactics = Tactics("4-3-3", 0.5, "mid", 0.5, 0.5, 0.5)
    p = next(iter(players.values()))
    fast = type(p)(**{**p.__dict__, "attributes": {**p.attributes, "sprint_speed": 95}, "playstyles": []})
    slow = type(p)(**{**p.__dict__, "attributes": {**p.attributes, "sprint_speed": 55}, "playstyles": []})
    assert player_params(fast, tactics).vmax > player_params(slow, tactics).vmax
    calm = type(p)(**{**p.__dict__, "attributes": {**p.attributes, "composure": 90}, "playstyles": []})
    nervy = type(p)(**{**p.__dict__, "attributes": {**p.attributes, "composure": 50}, "playstyles": []})
    assert player_params(calm, tactics).composure > player_params(nervy, tactics).composure
    pressing = Tactics("4-3-3", 0.9, "high", 0.5, 0.5, 0.5)
    assert player_params(p, pressing).press > player_params(p, tactics).press


def test_invalid_tactics_rejected():
    with pytest.raises(ValueError):
        Tactics("4-3-3", 1.5, "mid", 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        Tactics("4-3-3", 0.5, "sky", 0.5, 0.5, 0.5)
