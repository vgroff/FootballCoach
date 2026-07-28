from __future__ import annotations

import random

from footballcoach.engine.tackling import TacklingParams, attempt_tackle


def test_tackle_deterministic_at_rng_reduction_one_tackler_wins():
    params = TacklingParams.from_config()
    # tackling=0.5 * 1.2 = 0.6 > dribbling=0.5 => tackler should win deterministically.
    result = attempt_tackle(0.5, 0.5, rng_reduction=1.0, rng=random.Random(1), params=params)
    assert result.tackler_won is True


def test_tackle_deterministic_at_rng_reduction_one_dribbler_wins():
    params = TacklingParams.from_config()
    # tackling=0.3 * 1.2 = 0.36 < dribbling=0.9 => dribbler should win deterministically.
    result = attempt_tackle(0.3, 0.9, rng_reduction=1.0, rng=random.Random(1), params=params)
    assert result.tackler_won is False


def test_tackler_boost_helps_equal_attributes():
    params = TacklingParams.from_config()
    wins = sum(
        1 for seed in range(2000)
        if attempt_tackle(0.5, 0.5, rng_reduction=0.3, rng=random.Random(seed), params=params).tackler_won
    )
    win_rate = wins / 2000
    # With equal attributes and a 1.2x tackler boost, tackler should win
    # somewhat more than half the time.
    assert win_rate > 0.5
