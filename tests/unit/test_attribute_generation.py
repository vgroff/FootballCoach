from __future__ import annotations

import numpy as np

from footballcoach.generation.attributes import generate_attributes, generate_squad


def test_generated_attributes_within_bounds():
    rng = np.random.default_rng(1)
    for _ in range(200):
        attrs = generate_attributes(tier="generic", rng=rng)
        for value in (
            attrs.top_speed, attrs.acceleration, attrs.stamina, attrs.kick_precision,
            attrs.kick_power, attrs.dribbling, attrs.ball_control, attrs.tackling,
        ):
            assert 0.0 <= value <= 1.0


def test_premier_league_tier_lands_in_expected_band():
    rng = np.random.default_rng(2)
    squad = generate_squad(200, tier="premier_league", rng=rng)
    means = {
        "top_speed": np.mean([p.top_speed for p in squad]),
        "tackling": np.mean([p.tackling for p in squad]),
    }
    for name, mean in means.items():
        assert 0.55 <= mean <= 0.9, f"{name} mean {mean} outside expected PL band"


def test_league_three_tier_lands_lower_than_premier_league():
    rng = np.random.default_rng(3)
    pl_squad = generate_squad(200, tier="premier_league", rng=rng)
    l3_squad = generate_squad(200, tier="league_three", rng=rng)
    pl_mean = np.mean([p.top_speed for p in pl_squad])
    l3_mean = np.mean([p.top_speed for p in l3_squad])
    assert l3_mean < pl_mean
    assert l3_mean > 0.05  # still "competent", not near zero


def test_top_speed_and_acceleration_are_positively_correlated():
    rng = np.random.default_rng(4)
    squad = generate_squad(500, tier="generic", rng=rng)
    top_speeds = np.array([p.top_speed for p in squad])
    accels = np.array([p.acceleration for p in squad])
    corr = np.corrcoef(top_speeds, accels)[0, 1]
    assert corr > 0.3


def test_tackling_and_dribbling_are_negatively_correlated():
    rng = np.random.default_rng(5)
    squad = generate_squad(500, tier="generic", rng=rng)
    tackling = np.array([p.tackling for p in squad])
    dribbling = np.array([p.dribbling for p in squad])
    corr = np.corrcoef(tackling, dribbling)[0, 1]
    assert corr < 0.0
